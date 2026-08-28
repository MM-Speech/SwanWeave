import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Tokenizer, 
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model, \
    Qwen3DecoderLayer, Qwen3Attention, Qwen3MLP, Qwen3RMSNorm, Qwen3PreTrainedModel
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack

from modules.commons.hf.transformer import TransformerDecoderModel, TransformerEncoderModel, CrossAttention
from modules.commons.hf.transformer_config import TransformerConfig
from modules.commons.hf.transformer_dit import TransformerDiTModel, TimestepEmbedding
from modules.commons.hf.transformer_dit_config import TransformerDiTConfig
from modules.asr.nepa.nepa import build_nepa_model
from modules.flow_matching.mask_cfm import MaskFlowMatching

from utils.nn.seq_utils import sequence_mask
from utils.nn.embedding import resize_embedding_layer
from utils.commons.tensor_utils import all_gather_varlen_tensor_stack, all_gather_varlen_tensor
from utils.commons.hparams import set_hparams
from utils.commons.ckpt_utils import load_ckpt


def build_text_tokenizer(hparams, model_name="Qwen/Qwen3-0.6B"):

    timestamp_start = float(hparams.get('timestamp_start', 0.0))
    timestamp_end = float(hparams.get('timestamp_end', 300.0))
    timestamp_step = float(hparams.get('timestamp_step', 0.08))

    n_steps = int(round((timestamp_end - timestamp_start) / timestamp_step))
    timestamps = [timestamp_start + i * timestamp_step for i in range(n_steps + 1)]
    timestamp_tokens = [f"<|TS{t:.2f}|>" for t in timestamps]
    special_tokens = ['<|MASK|>'] + timestamp_tokens

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.add_tokens(special_tokens, special_tokens=True)
    
    tokenizer.timestamp_start = timestamp_start
    tokenizer.timestamp_end = timestamp_end
    tokenizer.timestamp_step = timestamp_step
    tokenizer.timestamp_start_id = tokenizer.encode(f"<|TS{timestamp_start:.2f}|>")[0]
    tokenizer.timestamp_end_id = tokenizer.encode(f"<|TS{timestamp_end:.2f}|>")[0]

    return tokenizer

def build_aligner_model(hparams, attn_implementation="flash_attention_2", init_pretrained=True):
    

    config = ModelArgs(
        sample_rate=hparams['audio_sample_rate'],

        vocab_size=len(tokenizer),
        mask_id=tokenizer.encode('<|MASK|>')[0],

        timestamp_start=float(getattr(tokenizer, 'timestamp_start', hparams.get('timestamp_start', 0.0))),
        timestamp_end=float(getattr(tokenizer, 'timestamp_end', hparams.get('timestamp_end', 300.0))),
        timestamp_step=float(getattr(tokenizer, 'timestamp_step', hparams.get('timestamp_step', 0.08))),
        timestamp_start_id=int(getattr(tokenizer, 'timestamp_start_id', tokenizer.encode('<|TS0.00|>')[0])),
        timestamp_end_id=int(getattr(tokenizer, 'timestamp_end_id', tokenizer.encode('<|TS300.00|>')[0])),

        mono_loss_weight=float(hparams.get('mono_loss_weight', 0.0)),
        mono_min_gap=float(hparams.get('mono_min_gap', 0.0)),
        mono_pos_weight_masked=float(hparams.get('mono_pos_weight_masked', 1.0)),
        mono_pos_weight_unmasked=float(hparams.get('mono_pos_weight_unmasked', 0.25)),

        self_conditioning=hparams.get('self_conditioning', False),
        self_conditioning_prob=float(hparams.get('self_conditioning_prob', 0.5)),
        self_conditioning_mode=hparams.get('self_conditioning_mode', 'replace_masked'),

        nepa_ckpt=hparams.get('nepa_ckpt', 'checkpoints/260213_nepa_base'),
        init_pretrained=init_pretrained,
        freeze_nepa=hparams.get('freeze_nepa', False),

        attn_implementation=attn_implementation,
        gradient_checkpointing=hparams.get('gradient_checkpointing', False),
    )


@dataclass
class ModelArgs:
    # audio
    sample_rate: int = 16000
    hop_size: int = None  # will be filled from nepa.config.hop_size

    # text
    vocab_size: int = None
    mask_id: int = None

    # timestamp token space (for monotonic regularization / decoding)
    timestamp_start: float = 0.0
    timestamp_end: float = 300.0
    timestamp_step: float = 0.08
    timestamp_start_id: int = None
    timestamp_end_id: int = None

    # monotonic regularization (training)
    mono_loss_weight: float = 0.0
    mono_min_gap: float = 0.0
    mono_pos_weight_masked: float = 1.0
    mono_pos_weight_unmasked: float = 0.25

    # self-conditioning (training)
    self_conditioning: bool = False
    self_conditioning_prob: float = 0.5
    self_conditioning_mode: str = 'replace_masked'

    # pretrained audio encoder
    audio_encoder_type: str = 'nepa'
    nepa_ckpt: str = None
    init_pretrained: bool = True
    freeze_nepa: bool = True

    attn_implementation: str = 'flash_attention_2'
    gradient_checkpointing: bool = False


class Qwen3DecoderLayerWithCrossAttention(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.cross_attn = CrossAttention(config=config, layer_idx=layer_idx)
        self.cross_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_gating_proj = nn.Linear(config.hidden_size, config.hidden_size)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        encoder_hidden_states: Optional[torch.Tensor] = None,      # [bsz, src_len, hidden]
        encoder_attention_mask: Optional[torch.Tensor] = None,     # [cross-attn mask [bsz, 1, tgt_len, src_len]
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.cross_attention_layernorm(hidden_states)
        cross_attn_output, _ = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **kwargs,
        )
        cross_gate = torch.sigmoid(self.cross_gating_proj(hidden_states))
        hidden_states = residual + cross_gate.to(cross_attn_output) * cross_attn_output

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen3ModelWithCrossAttention(Qwen3Model):
    def __init__(self, config: Qwen3Config):
        Qwen3PreTrainedModel.__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayerWithCrossAttention(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

class ForcedAlignerBackbone(Qwen3ForCausalLM):
    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(config)
        self.config = config
        self.model = Qwen3ModelWithCrossAttention(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class ForcedAligner(MaskFlowMatching):
    def __init__(self, config: ModelArgs):
        model_name="Qwen/Qwen3-0.6B"
        tokenizer: Qwen2Tokenizer = build_text_tokenizer(hparams, model_name)
        if config.init_pretrained:
            backbone = ForcedAlignerBackbone.from_pretrained(model_name, attn_implementation=config.attn_implementation)
        else:
            backbone = Qwen3ForCausalLM._from_config(Qwen3Config.from_pretrained(model_name), attn_implementation=attn_implementation, torch_dtype=dtype)
        resize_embedding_layer(backbone, len(tokenizer))
        # TODO: resize and load lm head

        super().__init__(
            vocab_size=config.vocab_size,
            mask_id=config.mask_id,
            backbone=backbone,
            schedule="cosine",            # 'linear' | 'cosine' | 'quadratic'
            t_eps=0.05,
        )
        self.config = config

        if config.nepa_ckpt.endswith('.ckpt'):
            nepa_hp = set_hparams(os.path.join(Path(config.nepa_ckpt).parent, 'config.yaml'), global_hparams=False)
        else:
            nepa_hp = set_hparams(os.path.join(config.nepa_ckpt, 'config.yaml'), global_hparams=False)
        self.nepa = build_nepa_model(nepa_hp, attn_implementation=config.attn_implementation)
        if config.init_pretrained:
            load_ckpt(self.nepa, config.nepa_ckpt, strict=False)
        self.config.hop_size = int(self.nepa.config.hop_size)
        if config.freeze_nepa:
            self.nepa.eval()
            for p in self.nepa.parameters():
                p.requires_grad = False
        self.audio_proj = nn.Linear(self.nepa.config.hidden_size, backbone.config.hidden_size, bias=False)
        
        self.time_embed = TimestepEmbedding(config.hidden_size)
    
        if config.gradient_checkpointing and self.training:
            self.nepa.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward_encoder(self, wavs, wav_lens):
        if self.config.freeze_nepa:
            with torch.no_grad():
                feat, audio_mask = self.nepa(wavs, wav_lens)  # eval-mode => Tensor[B, T', C]
        else:
            feat, audio_mask = self.nepa(wavs, wav_lens)
        x = self.audio_proj(feat)
        return x, audio_mask

    def _time_embed(self, t: torch.Tensor, *, batch: int) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise TypeError("t must be a torch.Tensor")

        if t.ndim == 0:
            t = t.view(1)
        if t.shape[0] == 1 and batch != 1:
            t = t.expand(batch)
        elif t.shape[0] != batch:
            raise ValueError(f"timesteps batch mismatch: got {t.shape[0]}, expected {batch}")

        device_type = t.device.type
        enabled = device_type == "cuda"
        with torch.amp.autocast(device_type=device_type, dtype=torch.float32, enabled=enabled):
            t_emb = self.time_embed(t)
        return t_emb

