'''
这里写音频编辑的 DiT 和配套的 DiTBuildModelMixin
参考 modules/tts/scriptspeech/dit_prompt.py modules/tts/scriptspeech/build_model_utils.py
'''
import os
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
import torch
import json
from pathlib import Path

from utils.commons.hparams import hparams, set_hparams
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.io import print_once

from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy

import logging
import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import torch
from torch import nn
import torchdiffeq
from modules.tts.llama_dit.llama_prompt import LLaMa
from modules.commons.engram import NGramEngram
from utils.nn.seq_utils import sequence_mask

# 倒入相应的函数，构造模型需要的，可能需要重新改写
from modules.tts.scriptspeech.build_model_utils import build_vae, \
shard_model_in_node, get_class_from_module, build_qwen3



class DiTBuildModelMixin:
    def build_model(self):  # interface to BaseTask()
        self._build_model()
        if hparams.get('trained_module', 'all') == 'crossatt':
            self.training_crossatt()
        return {'trainable': [self.dit], 'others': [self.vae]}
    
    def _build_model(self):  # interface to infer
        self.vae, self.hp_vae = build_vae(hparams.get('vae_ckpt'))
        self.dit_text_tokenizer, self.dit_vocab_size = self.build_dit_text_tokenizer()

        self.build_dit(hparams)
        if hparams.get('load_sd_text_encoder', False):
            if 'goku' in hparams.get('model_size', 'base'):
                self.build_goku_text_encoder(hparams)

    def training_crossatt(self):
        for param in self.dit.parameters():
            param.requires_grad = False

        # 仅解冻特定模块
        for name, param in self.dit.named_parameters():
            if (
                    'caption_proj' in name or
                    'cross_attention' in name or
                    'cross_attention_norm' in name
            ):
                param.requires_grad = True

    def build_dit(self, hparams):
        # from modules.tts.scriptspeech.dit_prompt import Diffusion, ModelArgs, PostTrainNFTWrapper
        from modules.tts.scriptspeech.dit_edit import Diffusion, ModelArgs
        
        print(f'| Building DIT model version {hparams.get("dit_version", 1)}')
        config = ModelArgs()
        config.do_checkpoint = hparams.get('do_checkpoint', False)
        if hparams.get('model_size', 'base') == 'small':
            print('| use small model')
            config.encoder_n_layers = 12
            config.encoder_n_heads = 12
            config.encoder_dim = 768 
        elif hparams.get('model_size', 'base') == '1b':
            print('| use base model')
            config.encoder_n_layers = 28
            config.encoder_n_heads = 16
            config.encoder_dim = 1536
        elif hparams.get('model_size', 'base') == 'seedance_7b':
            print('| use seedance_7b model')
            config.encoder_n_layers = 32
            config.encoder_n_heads = 24
            config.encoder_dim = 1536
        elif hparams.get('model_size', 'base') == 'goku_2':
            print('| use goku_2 model')
            config.encoder_n_layers = 16
            config.encoder_n_heads = 24
            config.encoder_dim = 1536
        elif hparams.get('model_size', 'base') == 'goku_2_large':
            print('| use goku_2_large model')
            config.encoder_n_layers = 32
            config.encoder_n_heads = 24
            config.encoder_dim = 1536
        else:
            print('| use base model')

        config.in_channels = config.out_channels = hparams['latent_dim']
        config.vocab_size = self.dit_vocab_size
        # config.audio_tokenizer = hparams.get('audio_tokenizer', 'glm4v')
        # config.audio_vocab_size = self.audio_vocab_size
        config.caption_dim = hparams.get('caption_encoder_dim', 5120 + 1)
        config.crossattn_n_layers = hparams.get('caption_crossattn_n_layers', 16)
        config.use_sparse_dur = hparams.get('use_sparse_dur', False)
        config.use_caption_encoder = hparams.get('use_caption_encoder', False)
        config.cfg_mask_text_token = self.cfg_mask_text_token = self.dit_text_tokenizer.encode('<MASK>')[0]
        # config.cfg_mask_audio_token = self.cfg_mask_audio_token = config.audio_vocab_size - 1
        config.cfg_mask_ph_token = self.cfg_mask_ph_token = 302 - 1
        config.text_fill_token = self.text_fill_token = self.dit_text_tokenizer.encode('<FILL>')[0]
        config.ph_fill_token = self.ph_fill_token = 302 - 2
        config.do_checkpoint = hparams.get('gradient_checkpointing', False)
        config.use_dur = hparams.get('use_dur', False)
        config.use_caption_text_mark = hparams.get('use_caption_text_mark', False)
        config.use_caption_pool_in_adaln = hparams.get('use_caption_pool_in_adaln', False)
        config.use_dynamic_cross_gate = hparams.get('use_dynamic_cross_gate', False)
        config.use_qk_norm = hparams.get('use_qk_norm', False)
        config.use_spk_mask = hparams.get('use_spk_mask', False)
        config.use_gated_attention = hparams.get('use_gated_attention', False)

        config.use_engram = hparams.get('use_engram', False)
        config.engram_target = hparams.get('engram_target', 'caption')
        engram_orders = hparams.get('engram_orders', (1, 2, 3))
        if isinstance(engram_orders, list):
            engram_orders = tuple(engram_orders)
        config.engram_orders = engram_orders
        config.engram_num_heads = hparams.get('engram_num_heads', 4)
        config.engram_table_size = hparams.get('engram_table_size', 262144)
        config.engram_head_dim = hparams.get('engram_head_dim', 64)
        config.engram_dropout = hparams.get('engram_dropout', 0.0)
        config.engram_gate_bias = hparams.get('engram_gate_bias', -4.0)
        config.engram_conv_kernel = hparams.get('engram_conv_kernel', 3)
        config.engram_special_replace_id = hparams.get('engram_special_replace_id', 1)

        config.use_moe_ffn = hparams.get('use_moe_ffn', False)
        config.decoder_sparse_step = int(hparams.get('decoder_sparse_step', hparams.get('moe_sparse_step', 1)) or 1)
        mlp_only_layers = hparams.get('mlp_only_layers', ())
        if isinstance(mlp_only_layers, list):
            mlp_only_layers = tuple(mlp_only_layers)
        config.mlp_only_layers = mlp_only_layers
        config.moe_p = hparams.get('moe_p', 0.7)
        config.moe_num_routed = hparams.get('moe_num_routed', 8)
        config.moe_num_shared = hparams.get('moe_num_shared', 2)
        config.moe_num_null = hparams.get('moe_num_null', 4)
        config.moe_use_gumbel = hparams.get('moe_use_gumbel', False)
        config.moe_gumbel_tau_start = hparams.get('moe_gumbel_tau_start', 1.0)
        config.moe_gumbel_tau_end = hparams.get('moe_gumbel_tau_end', 0.3)
        config.moe_gumbel_tau_anneal_steps = hparams.get('moe_gumbel_tau_anneal_steps', 200_000)
        config.moe_expert_dropout = hparams.get('moe_expert_dropout', 0.0)
        # config.moe_max_experts_per_token = hparams.get('moe_max_experts_per_token', 4)
        config.moe_load_balance_loss_coef = hparams.get('moe_load_balance_loss_coef', 1e-2)
        config.moe_router_z_loss_coef = hparams.get('moe_router_z_loss_coef', 1e-3)
        config.moe_capacity_factor_min = hparams.get('moe_capacity_factor_min', 1.0)
        config.moe_capacity_factor_max = hparams.get('moe_capacity_factor_max', 2.0)
        config.moe_overflow_drop = hparams.get('moe_overflow_drop', True)
        config.moe_use_t_budget = hparams.get('moe_use_t_budget', True)
        config.moe_p_min = hparams.get('moe_p_min', 0.4)
        config.moe_p_max = hparams.get('moe_p_max', 0.95)
        config.moe_null_logit_bias_min = hparams.get('moe_null_logit_bias_min', -2.0)
        config.moe_null_logit_bias_max = hparams.get('moe_null_logit_bias_max', 0.0)
        config.moe_load_balance_loss_coef = hparams.get('moe_load_balance_loss_coef', 1e-2)
        config.moe_router_z_loss_coef = hparams.get('moe_router_z_loss_coef', 1e-3)
        config.use_bgm_flag = hparams.get('use_bgm_flag', False)

        if hparams.get('is_posttrain', False):
            self.dit = PostTrainNFTWrapper(config)
        else:
            self.dit = Diffusion(config)
        self.dit.text_tokenizer = self.dit_text_tokenizer
        self.vae_stride = hparams.get('vae_stride', 8)
        return self.dit

    def build_dit_text_tokenizer(self):
        if hparams.get('use_cosyvoice2_text_tokenizer',False):
            from utils.text.cosyvoice2_tokenizer import get_tokenizer
            text_tokenizer = get_tokenizer(multilingual=True, num_languages=100)
            vocab_size = text_tokenizer.encoding.n_vocab
            return text_tokenizer, vocab_size
        else:
            from transformers import AutoTokenizer, Qwen2Tokenizer
            text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            text_tokenizer.add_tokens([
                '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>', 
                '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>', 
                '<S1>', '</S1>','<S2>', '</S2>', 
                '<Audio>', '</Audio>','<BGM>','</BGM>', 
                '<W>', '</W>', '<FILL>', '<PAD>',
                '<S3>', '</S3>','<S4>', '</S4>', 
                '<|laughter|>', '<|breathe|>', '<|cry|>',
                '<ENV>','</ENV>', '<|sp|>' 
            ], special_tokens=True)
            vocab_size = len(text_tokenizer)
            return text_tokenizer, vocab_size

    def build_goku_text_encoder(self, config):
        from transformers import T5EncoderModel, T5Tokenizer
        dtype = torch.bfloat16

        if config['use_qwen']:
            qwen_name = config['pretrained_text_encoder_qwen']

            # 分支1：包含 Qwen2 -> 保持现有逻辑不变
            if 'qwen2' in qwen_name.lower():
                from transformers import Qwen2Model, AutoTokenizer
                self.goku_text_encoder = Qwen2Model.from_pretrained(qwen_name, torch_dtype=dtype)
                self.goku_text_encoder.requires_grad_(False).eval()

                self.goku_tokenizer = AutoTokenizer.from_pretrained(
                    qwen_name,
                    max_length=config['text_max_token_length'],
                    revision=None
                )
                vocal_size = len(self.goku_tokenizer)
                self.goku_special_token_ids = [vocal_size + 11, vocal_size + 12]
                self.goku_tokenizer.add_tokens([
                    '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>',
                    '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>',
                    '<S1>', '</S1>', '<S2>', '</S2>',
                    '<Audio>', '</Audio>', '<BGM>', '</BGM>',
                    '<W>', '</W>', '<FILL>', '<PAD>',
                    '<S3>', '</S3>', '<S4>', '</S4>',
                    '<|laughter|>', '<|breathe|>', '<|cry|>',
                    '<ENV>','</ENV>','<|sp|>', 
                ], special_tokens=True)

                transformer_cls = get_class_from_module(
                    "transformers.models.qwen2.modeling_qwen2",
                    "Qwen2DecoderLayer"
                )
                if int(os.environ.get("WORLD_SIZE", 1)) > 1:
                    self.goku_text_encoder = shard_model_in_node(
                        self.goku_text_encoder,
                        transformer_cls,
                        int(os.environ.get("LOCAL_RANK", 0))
                    )

            # 分支2：包含 Qwen3 -> 使用已有的 build_qwen3 逻辑
            elif 'qwen3' in qwen_name.lower():
                # build_qwen3 内部已根据环境优先选择 FA3/FA2/SDPA，并完成必要的 tokenizer 加 special tokens 与 embedding resize
                lm, tok = build_qwen3(hparams, qwen_name, dtype=dtype)
                # 仅取解码器主体做为 encoder 使用（与其他代码保持一致）
                self.goku_text_encoder = lm.model
                self.goku_text_encoder.requires_grad_(False).eval()
                self.goku_tokenizer = tok

                # 保留原有的 special token 索引占位逻辑
                vocal_size = len(self.goku_tokenizer)
                self.goku_special_token_ids = [vocal_size + 11, vocal_size + 12]

                transformer_cls = get_class_from_module(
                    "transformers.models.qwen3.modeling_qwen3",
                    "Qwen3DecoderLayer"
                )
                if int(os.environ.get("WORLD_SIZE", 1)) > 1:
                    self.goku_text_encoder = shard_model_in_node(
                        self.goku_text_encoder,
                        transformer_cls,
                        int(os.environ.get("LOCAL_RANK", 0))
                    )

            else:
                raise ValueError(
                    f"Unknown Qwen backbone: {qwen_name}. Expect the name to contain 'Qwen2' or 'Qwen3'."
                )

        else:
            # 保持原 T5 逻辑不变
            self.goku_text_encoder = T5EncoderModel.from_pretrained(config['pretrained_text_encoder'], torch_dtype=dtype)
            self.goku_text_encoder.requires_grad_(False).eval()
            self.goku_tokenizer = T5Tokenizer.from_pretrained(
                config['pretrained_text_encoder'],
                max_length=config['text_max_token_length']
            )
            vocal_size = len(self.goku_tokenizer)
            self.goku_special_token_ids = [vocal_size + 11, vocal_size + 12]
            self.goku_tokenizer.add_tokens([
                '<BOT>', '<EOT>', '<BOS>', '<EOS>', '<TAG>', '</TAG>',
                '<GPROMPT>', '</GPROMPT>', '<SPK>', '</SPK>', '<MASK>',
                '<S1>', '</S1>', '<S2>', '</S2>',
                '<Audio>', '</Audio>', '<BGM>', '</BGM>',
                '<W>', '</W>', '<FILL>', '<PAD>',
                '<S3>', '</S3>', '<S4>', '</S4>',
                '<|laughter|>', '<|breathe|>', '<|cry|>',
                '<ENV>','</ENV>','<|sp|>', 
            ], special_tokens=True)


logger = logging.getLogger(__name__)

@dataclass
class ModelArgs:
    # text
    vocab_size: int = None
    text_dim: int = 1024

    # audio
    audio_vocab_size: int = None
    audio_tokenizer: str = 'glm4v'
    
    # llama
    encoder_dim: int = 1024
    encoder_n_layers: int = 24
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = None
    mlp_extend: float = None
    max_seq_len: int = 16384
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = 4
    use_causal_attn: bool = False

    caption_dim: int = 3584 + 1 # dim of seedance text encoder + content mask

    in_channels: int = 16
    out_channels: int = 16

    # trainging
    do_checkpoint: bool = False
    use_qk_norm: bool = False

    cfg_mask_text_token: int = None
    text_fill_token: int = None
    use_caption_pool_in_adaln: bool = False
    use_caption_text_mark: bool = False
    use_spk_mask: bool = False 

    use_gated_attention: bool = False
    use_dynamic_cross_gate: bool = False

    use_engram: bool = False
    engram_target: Literal["caption", "text", "both"] = "caption"
    engram_orders: Tuple[int, ...] = (1, 2, 3)
    engram_num_heads: int = 4
    engram_table_size: int = 262144
    engram_head_dim: int = 64
    engram_dropout: float = 0.0
    engram_gate_bias: float = -4.0
    engram_conv_kernel: int = 3
    engram_special_replace_id: int = 1

    use_moe_ffn: bool = False
    moe_p: float = 0.7                # Top-P routing threshold
    moe_num_routed: int = 8           # routed experts count
    moe_num_shared: int = 2           # shared experts count (always-on)
    moe_num_null: int = 4             # null experts count (no compute)
    moe_aux_loss_weight: float = 0.01 # training-time weight (you can anneal in loop)
    moe_use_gumbel: bool = False
    moe_gumbel_tau_start: float = 1.0       # 初始温度（训练早期）
    moe_gumbel_tau_end: float = 0.3         # 最终温度（训练后期）
    moe_gumbel_tau_anneal_steps: int = 200_000  # 退火步数
    moe_expert_dropout: float = 0.0   # 每个 step 随机屏蔽部分专家（0~1）
    moe_max_experts_per_token: int = 4  # 每个 token 最多用几个 routed experts
    moe_use_bias_balance: bool = True
    moe_bias_update_rate: float = 0.05
    moe_bias_momentum: float = 0.9
    moe_bias_clamp: float = 5.0
    moe_capacity_factor_min: float = 1.0
    moe_capacity_factor_max: float = 2.0
    moe_overflow_drop: bool = True
    moe_use_t_budget: bool = True
    moe_p_min: float = 0.4
    moe_p_max: float = 0.95
    moe_null_logit_bias_min: float = -2.0
    moe_null_logit_bias_max: float = 0.0
    decoder_sparse_step: int = 1
    mlp_only_layers: Tuple[int, ...] = ()

class Diffusion(nn.Module):
    def __init__(self, hp: ModelArgs):
        super().__init__()
        self.hp = hp

        self.encoder = LLaMa(hp)
        self.prenet = nn.Linear(self.hp.encoder_dim * 2 , self.hp.encoder_dim)

        self.lat_proj = nn.Linear(self.hp.in_channels, self.hp.encoder_dim)
        self.ctx_proj = nn.Linear(self.hp.in_channels, self.hp.encoder_dim)
        self.ctx_mask_proj = nn.Linear(1, self.hp.encoder_dim) # ctx_mask embedding
        self.postnet = nn.Linear(hp.encoder_dim, hp.out_channels)
        self.caption_proj = nn.Linear(self.hp.caption_dim, self.hp.encoder_dim)
        if hp.use_caption_text_mark:
            self.caption_text_mark_embed = nn.Embedding(5, self.hp.encoder_dim)

        from modules.tts.llama_dit.vp_cfm import ConditionalFlowMatcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        from modules.tts.f5_dit.f5_modules import TimestepEmbedding
        self.f5_time_embed = TimestepEmbedding(hp.encoder_dim)

        from modules.asr.llama.llama import LLaMa as LLaMaSmall, ModelArgs as ModelArgsSmall

        self.text_embedder = nn.Embedding(hp.vocab_size, hp.encoder_dim)
        self.text_encoder = LLaMaSmall(ModelArgsSmall(
            dim=hp.encoder_dim,
            n_layers=8, n_heads=16,
            use_causal_attn=False, 
        ))

        if hp.use_spk_mask:
            self.spk_mask_embedder = nn.Embedding(5, hp.encoder_dim)  # 0=none, 1..4=S1..S4

        self.use_engram = bool(getattr(hp, "use_engram", False))
        self.engram_target = getattr(hp, "engram_target", "caption")
        if self.use_engram and (self.engram_target in ("caption", "both")):
            self.caption_engram = NGramEngram(
                model_dim=self.hp.encoder_dim,
                orders=tuple(getattr(hp, "engram_orders", (1, 2, 3))),
                num_heads=int(getattr(hp, "engram_num_heads", 4)),
                table_size=int(getattr(hp, "engram_table_size", 262144)),
                head_dim=int(getattr(hp, "engram_head_dim", 64)),
                dropout=float(getattr(hp, "engram_dropout", 0.0)),
                gate_bias=float(getattr(hp, "engram_gate_bias", -4.0)),
                conv_kernel=int(getattr(hp, "engram_conv_kernel", 3)),
            )
        else:
            self.caption_engram = None

        if self.use_engram and (self.engram_target in ("text", "both")):
            self.text_engram = NGramEngram(
                model_dim=self.hp.encoder_dim,
                orders=tuple(getattr(hp, "engram_orders", (2, 3))),
                num_heads=int(getattr(hp, "engram_num_heads", 4)),
                table_size=int(getattr(hp, "engram_table_size", 262144)),
                head_dim=int(getattr(hp, "engram_head_dim", 64)),
                dropout=float(getattr(hp, "engram_dropout", 0.0)),
                gate_bias=float(getattr(hp, "engram_gate_bias", -4.0)),
                conv_kernel=int(getattr(hp, "engram_conv_kernel", 3)),
            )
        else:
            self.text_engram = None
            
        if hp.use_bgm_flag:
            self.bgm_flag_embed = nn.Embedding(3, hp.encoder_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # Linear and Embedding layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.hp.encoder_n_layers))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.hp.encoder_n_layers)
                )
        # Time embedding MLP
        nn.init.normal_(self.f5_time_embed.time_mlp[0].weight, std=0.02)
        nn.init.normal_(self.f5_time_embed.time_mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks
        for block in self.encoder.layers:
            nn.init.zeros_(block.attention_norm.linear.weight)
            nn.init.zeros_(block.attention_norm.linear.bias)

        # Zero-out output layers
        nn.init.zeros_(self.encoder.norm.linear.weight)
        nn.init.zeros_(self.encoder.norm.linear.bias)
        nn.init.zeros_(self.encoder.out_proj.weight)
        nn.init.zeros_(self.encoder.out_proj.bias)

    def forward_text_encoder(self, inputs, x_mask):
        tgt_len = x_mask.shape[1]
        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        bsz = txt_tokens.shape[0]

        txt_tokens = inputs["txt_tokens"]
        txt_mask = inputs["txt_mask"]
        # ===== 1) build filled text ids: [B, tgt_len] =====
        x_txt_ids = torch.full((bsz, tgt_len), self.hp.text_fill_token, device=txt_tokens.device, dtype=txt_tokens.dtype)
        fill_pos = sequence_mask(txt_mask.long().sum(1), tgt_len)  # [B, tgt_len]
        x_txt_ids[fill_pos] = txt_tokens[txt_mask]

        token_emb = self.text_embedder(x_txt_ids.long())  # [B, tgt_len, C]

        # ===== 2) add spk_mask embedding if enabled =====
        if self.hp.use_spk_mask and ('spk_mask' in inputs) and (inputs['spk_mask'] is not None):
            spk_mask = inputs['spk_mask']  # expected shape [B, T_txt] with same txt_mask
            # 对齐到和 x_txt_ids 一样的“前缀填充”布局
            spk_ids = torch.zeros((bsz, tgt_len), device=txt_tokens.device, dtype=torch.long)
            spk_ids[fill_pos] = spk_mask[txt_mask].long().clamp_(0, 4)
            token_emb = token_emb + self.spk_mask_embedder(spk_ids)  # [B, tgt_len, C]

        if self.text_engram is not None:
            special_ids = [
                int(getattr(self.hp, "text_fill_token", -1)),
                int(getattr(self.hp, "cfg_mask_text_token", -1)),
            ]
            token_emb = token_emb + self.text_engram(
                x_txt_ids,
                x_mask,
                token_emb,
                special_ids=[s for s in special_ids if s >= 0],
                special_replace_id=int(getattr(self.hp, "engram_special_replace_id", 1)),
            )

        x_txt = self.text_encoder(token_emb, x_mask)  # [B, tgt_len, C]
        return x_txt

    def _encode_caption_context(self, caption_emb, caption_lens, caption_ids=None, caption_text_mark=None):
        if (caption_emb is None) or (caption_lens is None):
            return None, None

        caption_ctx = self.caption_proj(caption_emb)
        max_ctx_len = int(caption_ctx.shape[1])

        cap_lens = caption_lens.to(device=caption_ctx.device, dtype=torch.long)
        if cap_lens.numel() == 0:
            return None, None
        eff_len = int(cap_lens.max().item())
        eff_len = min(eff_len, max_ctx_len)
        if eff_len <= 0:
            return None, None

        caption_ctx = caption_ctx[:, :eff_len]
        cap_lens = cap_lens.clamp(min=0, max=eff_len)
        caption_mask = sequence_mask(cap_lens, maxlen=eff_len)

        if (self.caption_engram is not None) and (caption_ids is not None):
            cap_ids = caption_ids.to(device=caption_ctx.device)
            if int(cap_ids.shape[1]) < eff_len:
                cap_ids = torch.nn.functional.pad(cap_ids, (0, eff_len - int(cap_ids.shape[1])), value=0)
            cap_ids = cap_ids[:, :eff_len]
            caption_ctx = caption_ctx + self.caption_engram(
                cap_ids.to(torch.long),
                caption_mask,
                caption_ctx,
                special_replace_id=int(getattr(self.hp, "engram_special_replace_id", 1)),
            )

        if self.hp.use_caption_text_mark and (caption_text_mark is not None):
            mark = self.caption_text_mark_embed(caption_text_mark.to(device=caption_ctx.device, dtype=torch.long))
            caption_ctx = caption_ctx + mark[:, :eff_len]

        return caption_ctx, cap_lens

    def forward(self, inputs, sigmas=None, x0=None, t=None):
        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask
        x = inputs['lat']
        x_mask = sequence_mask(inputs['lat_lens'], maxlen=x.shape[1])

        if x0 is None:
            x0 = torch.randn_like(x)
        if t is None:
            t = self.flow_matcher.time_sampler.sample([x0.shape[0]], x0.device).type_as(x0)

        # CFM: x is x1
        xt = t[:, None, None] * x + (1 - t[:, None, None]) * x0
        ut = x - x0

        with torch.amp.autocast('cuda', dtype=torch.float32):
            t_emb = self.f5_time_embed(t)
        x_noisy = (xt * (1 - ctx_mask)).bfloat16()
        target = ut

        bgm_flag = inputs.get('bgm_flag', None)
        if (bgm_flag is not None) and hasattr(self, 'bgm_flag_embed'):
            if not torch.is_tensor(bgm_flag):
                bgm_flag = torch.as_tensor(bgm_flag, device=t_emb.device)
            bgm_flag = bgm_flag.to(device=t_emb.device, dtype=torch.long).view(-1)
            bgm_flag = bgm_flag.clamp(0, 2)
            if bgm_flag.numel() == 1 and t_emb.shape[0] != 1:
                bgm_flag = bgm_flag.expand(t_emb.shape[0])
            t_emb = t_emb + self.bgm_flag_embed(bgm_flag)

        x_txt = self.forward_text_encoder(inputs, x_mask)

        caption_embs, caption_lens = self._encode_caption_context(
            inputs.get('caption_emb', None),
            inputs.get('caption_lens', None),
            inputs.get('caption_ids', None),
            inputs.get('caption_text_mark', None),
        )

        x_noisy = self.lat_proj(x_noisy) + self.ctx_proj(ctx_feature) + self.ctx_mask_proj(ctx_mask)
        x_noisy = self.prenet(torch.cat([x_noisy, x_txt], dim=-1))

        use_moe = bool(getattr(self.hp, "use_moe_ffn", False))

        if use_moe:
            encoder_out, moe_aux = self.encoder(
                x_noisy, t_emb, attn_mask=x_mask,
                do_checkpoint=self.hp.do_checkpoint,
                context=caption_embs,
                context_lens=caption_lens,
            )
        else:
            encoder_out = self.encoder(
                x_noisy, t_emb, attn_mask=x_mask,
                do_checkpoint=self.hp.do_checkpoint,
                context=caption_embs,
                context_lens=caption_lens,
            )
            moe_aux = None

        pred = self.postnet(encoder_out)

        if use_moe:
            return pred, target, moe_aux
        return pred, target

    def _forward(self, x, cond, timesteps, seq_cfg_w=[1.5, 3.0], timestep_annealing_w=(0.6, 0.6, 1.0)):
        """When we use torchdiffeq, we need to include the CFG process inside _forward()."""
        ctx = cond['ctx']
        ctx_mask = cond['ctx_mask']
        attn_mask = cond['attn_mask']
        x_txt = cond['x_txt']

        caption_embs = cond.get('_caption_ctx', None)
        caption_lens = cond.get('_caption_ctx_lens', None)
        if caption_embs is None:
            # backward compat: allow precomputed non-underscored keys
            caption_embs = cond.get('caption_ctx', None)
            caption_lens = cond.get('caption_ctx_lens', None)

        if caption_embs is None:
            caption_embs, caption_lens = self._encode_caption_context(
                cond.get('caption_emb', None),
                cond.get('caption_lens', None),
                cond.get('caption_ids', None),
                cond.get('caption_text_mark', None),
            )
            # cache to avoid recompute per ODE step
            cond['_caption_ctx'] = caption_embs
            cond['_caption_ctx_lens'] = caption_lens

        x = x * (1 - ctx_mask)
        x = self.lat_proj(x) + self.ctx_proj(ctx) + self.ctx_mask_proj(ctx_mask)
        x = self.prenet(torch.cat([x, x_txt], dim=-1))

        with torch.amp.autocast('cuda', dtype=torch.float32):
            t_emb = self.f5_time_embed(timesteps)

        if (t_emb is not None) and (t_emb.ndim == 2) and (t_emb.shape[0] != x.shape[0]):
            if t_emb.shape[0] == 1:
                t_emb = t_emb.expand(x.shape[0], -1)
            elif x.shape[0] % t_emb.shape[0] == 0:
                t_emb = t_emb.repeat(x.shape[0] // t_emb.shape[0], 1)
            else:
                raise ValueError(f"t_emb batch mismatch: got {t_emb.shape[0]}, expected {x.shape[0]}")

        bgm_flag = cond.get('bgm_flag', None)
        if (bgm_flag is not None) and hasattr(self, 'bgm_flag_embed'):
            if not torch.is_tensor(bgm_flag):
                bgm_flag = torch.as_tensor(bgm_flag, device=t_emb.device)
            bgm_flag = bgm_flag.to(device=t_emb.device, dtype=torch.long).view(-1)
            bgm_flag = bgm_flag.clamp(0, 2)
            if (x.shape[0] % 3 == 0) and (bgm_flag.numel() * 3 == x.shape[0]):
                bgm_flag = torch.cat([
                    bgm_flag,
                    bgm_flag,
                    torch.full_like(bgm_flag, 2),
                ], dim=0)
            elif bgm_flag.numel() == 1 and x.shape[0] != 1:
                bgm_flag = bgm_flag.expand(x.shape[0])
            if bgm_flag.numel() == x.shape[0]:
                t_emb = t_emb + self.bgm_flag_embed(bgm_flag)

        use_moe = bool(getattr(self.hp, "use_moe_ffn", False))
        pred_v = self.encoder(
            x, t_emb, attn_mask=attn_mask,
            context=caption_embs,
            context_lens=caption_lens,
            do_checkpoint=self.hp.do_checkpoint
        )
        if use_moe:
            pred_v = pred_v[0]

        pred = self.postnet(pred_v)

        if isinstance(timesteps, torch.Tensor) and timesteps.ndim > 0 and timesteps.shape[0] > pred.shape[0] // 3:
            timesteps, _, _ = timesteps.chunk(3)
            if timesteps.ndim == 1:
                timesteps = timesteps[:, None, None]
        a, b, p = timestep_annealing_w
        gamma_t = a + b * torch.pow(1 - timesteps, p)
        seq_cfg_w = [gamma_t * w for w in seq_cfg_w]

        cond_all, cond_txt, uncond = pred.chunk(3)

        pred = (
            uncond + 
            seq_cfg_w[0] * (cond_txt - uncond) + 
            seq_cfg_w[1] * (cond_all - cond_txt)
        )

        return pred

    @torch.no_grad()
    def inference(
        self, inputs, timesteps=20, seq_cfg_w=[1.5, 3.0], timestep_annealing_w=(0.6, 0.6, 1.0), 
        use_amo_sampler=False, use_sway=True, return_timesteps=False, **kwargs
    ):
        tgt_len = inputs['tgt_len']     # reference + target
        x_mask = sequence_mask(tgt_len, maxlen=inputs['ctx_mask'].shape[1])
        
        x_txt = self.forward_text_encoder(inputs, x_mask)

        (bsz, tgt_len, _), device = x_txt.shape, x_txt.device
        bsz = bsz // 3

        ctx_mask = inputs['ctx_mask']
        ctx_feature = inputs['lat_ctx'] * ctx_mask

        caption_ctx, caption_ctx_lens = self._encode_caption_context(
            inputs.get('caption_emb', None),
            inputs.get('caption_lens', None),
            inputs.get('caption_ids', None),
            inputs.get('caption_text_mark', None),
        )

        bgm_flag = inputs.get('bgm_flag', None)
        if (bgm_flag is not None) and hasattr(self, 'bgm_flag_embed'):
            if not torch.is_tensor(bgm_flag):
                bgm_flag = torch.as_tensor(bgm_flag, device=device)
            bgm_flag = bgm_flag.to(device=device, dtype=torch.long).view(-1)
            bgm_flag = bgm_flag.clamp(0, 2)
            if (x_txt.shape[0] % 3 == 0) and (bgm_flag.numel() * 3 == x_txt.shape[0]):
                bgm_flag = torch.cat([
                    bgm_flag,
                    bgm_flag,
                    torch.full_like(bgm_flag, 2),
                ], dim=0)
            elif bgm_flag.numel() == 1 and x_txt.shape[0] != 1:
                bgm_flag = bgm_flag.expand(x_txt.shape[0])
        else:
            bgm_flag = None

        cond = {
            'ctx': ctx_feature,
            'ctx_mask': ctx_mask,
            'attn_mask': x_mask,
            'x_txt': x_txt,
            'txt_lens': inputs.get('txt_lens', inputs['txt_mask'].sum(1)),
            '_caption_ctx': caption_ctx,
            '_caption_ctx_lens': caption_ctx_lens,
            'bgm_flag': bgm_flag,
        }

        ''' Euler ODE solver '''
        sway_sampling_coef = -1.0
        t_schedule = torch.linspace(0, 1, timesteps + 1).to(device)
        if use_sway:
            t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)

        if use_amo_sampler:

            def amo_sampling(sample, sigma, sigma_next, pred_v):
                # sample: [1,T,C] ; pred_v: [1,T,C]（CFG 后）
                t = sigma
                s = sigma_next
                x_t = sample

                c = 3.0
                o = torch.clamp(s + c * (s - t), max=1.0)

                pred_x_o = x_t + (o - t) * pred_v
                a = s / o
                b = torch.sqrt(torch.clamp_min((1 - s) ** 2 - (a * (1 - o)) ** 2, 0.0))

                noises = torch.randn_like(x_t)
                prev_sample = a * pred_x_o + b * noises
                prev_sample = prev_sample.to(pred_v.dtype)
                return prev_sample

            x = torch.randn([bsz, tgt_len, self.hp.out_channels], device=device)
            for step_index in range(timesteps):
                sigma = t_schedule[step_index].to(x_txt.dtype)
                sigma_next = t_schedule[step_index + 1].to(x_txt.dtype)

                model_out = self._forward(
                    torch.cat([x] * 3),
                    cond,
                    timesteps=sigma.unsqueeze(0),
                    seq_cfg_w=seq_cfg_w,
                    timestep_annealing_w=timestep_annealing_w,
                )
                x = amo_sampling(x, sigma, sigma_next, model_out)

            return x

        else:
            traj = torchdiffeq.odeint(
                lambda t, x: self._forward(
                    torch.cat([x] * 3),
                    cond,
                    timesteps=t.unsqueeze(0),
                    seq_cfg_w=seq_cfg_w,
                    timestep_annealing_w=timestep_annealing_w,
                ),
                torch.randn([bsz, tgt_len, self.hp.out_channels], device=device),
                t_schedule,
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )

            x = traj[-1]

        if return_timesteps:
            return x, t_schedule
        
        return x
