from typing import Callable, Optional, Union
import math

import torch
from torch import nn

from ._compat import (
    ALL_ATTENTION_FUNCTIONS,
    BaseModelOutputWithPast,
    GradientCheckpointingLayer,
    PreTrainedModel,
    TransformersKwargs,
    Unpack,
    auto_docstring,
    can_return_tuple,
    check_model_inputs,
)

from .transformer_dit_config import TransformerDiTConfig
from .transformer import Attention, MLP, RMSNorm, CrossAttention, RotaryEmbedding, make_padding_mask_4d, make_sliding_window_noncausal_mask_4d


class AdaLayerNormZero(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, hidden_size * 6)

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)

    def forward(self, x, emb=None):
        dtype = x.dtype
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb.to(dtype), 6, dim=1)

        x = self.norm(x.float()).to(dtype) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x.to(dtype), gate_msa.to(dtype), shift_mlp.to(dtype), scale_mlp.to(dtype), gate_mlp.to(dtype)


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, hidden_size * 2)

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)

    def forward(self, x, emb):
        dtype = x.dtype
        with torch.amp.autocast('cuda', dtype=torch.float32):
            emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb.to(dtype), 2, dim=1)

        x = self.norm(x.float()).to(dtype) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x.to(dtype)
    

class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor):  # noqa: F821
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time


class EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TransformerDiTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        # encoder self-attention is non-causal, no sliding window
        self.self_attn.is_causal = False
        # self.self_attn.sliding_window = None

        self.mlp = MLP(config)
        self.input_layernorm = AdaLayerNormZero(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
        
        self.layer_idx = layer_idx
        if layer_idx < config.num_cross_attention_layers:
            self.cross_attn = CrossAttention(config=config, layer_idx=layer_idx)
            self.cross_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_dynamic_cross_gate:
                self.cross_gating_proj = nn.Linear(config.hidden_size, config.hidden_size)
            else:
                self.cross_gate = nn.Parameter(torch.zeros(config.hidden_size))

        if config.use_mhc:
            from modules.commons.mhc import MHCResidual
            self.attn_mhc = MHCResidual(
                hidden_size=config.hidden_size, n_streams=config.mhc_num_streams, sinkhorn_iters=config.mhc_sinkhorn_iters,
                use_dynamic=config.mhc_use_dynamic, rms_norm_eps=RMSNorm,
            )
            self.mlp_mhc = MHCResidual(
                hidden_size=config.hidden_size, n_streams=config.mhc_num_streams, sinkhorn_iters=config.mhc_sinkhorn_iters,
                use_dynamic=config.mhc_use_dynamic, rms_norm_eps=RMSNorm,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_step: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # [bsz, 1, seq_len, seq_len] additive
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,      # [bsz, src_len, hidden]
        encoder_attention_mask: Optional[torch.Tensor] = None,     # [cross-attn mask [bsz, 1, tgt_len, src_len]
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states

        hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.input_layernorm(hidden_states, time_step)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            # use_cache=False,
            cache_position=None,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if self.config.use_mhc:
            hidden_states = self.attn_mhc(residual, gate_msa[:, None, :] * hidden_states)
        else:
            hidden_states = residual + gate_msa[:, None, :] * hidden_states
        
        # Cross Attention
        if self.layer_idx < self.config.num_cross_attention_layers and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attention_layernorm(hidden_states)
            cross_attn_output, _ = self.cross_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **kwargs,
            )
            if self.config.use_dynamic_cross_gate:
                cross_gate = torch.sigmoid(self.cross_gating_proj(hidden_states))
                hidden_states = residual + cross_gate.to(cross_attn_output) * cross_attn_output 
            else:
                hidden_states = residual + self.cross_gate.to(cross_attn_output) * cross_attn_output

        # Feed-forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None, :]
        hidden_states = self.mlp(hidden_states)

        if self.config.use_mhc:
            hidden_states = self.mlp_mhc(residual, gate_mlp[:, None] * hidden_states)
        else:
            hidden_states = residual + gate_mlp[:, None] * hidden_states

        return hidden_states


@auto_docstring
class TransformerPreTrainedModel(PreTrainedModel):
    config: TransformerDiTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["EncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": EncoderLayer,
        "attentions": Attention,
    }


@auto_docstring
class TransformerDiTModel(TransformerPreTrainedModel):
    """
    Encoder-only transformer model with RoPE and non-causal self-attention.
    """

    def __init__(self, config: TransformerDiTConfig):
        super().__init__(config)
        self.padding_idx = getattr(config, "pad_token_id", None)

        self.layers = nn.ModuleList(
            [EncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = AdaLayerNormZero_Final(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        if self.config.num_cross_attention_layers > 0 and config.use_caption_pool_in_adaln:
            self.cap_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.cap_gate = nn.Linear(config.hidden_size, config.hidden_size)

        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [bsz, seq_len] (1 for keep, 0 for pad)
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,  # [bsz, src_len, hidden]
        encoder_attention_mask: Optional[torch.Tensor] = None, # [bsz, src_len]
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        
        bsz, seq_len, _ = inputs_embeds.size()
        time_step = kwargs.pop('time_step')

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)

        if (
            self.config.num_cross_attention_layers > 0 and 
            self.config.use_caption_pool_in_adaln and 
            encoder_hidden_states is not None and 
            encoder_attention_mask is not None
        ):
            cap_pool = (encoder_hidden_states * encoder_attention_mask[..., None]).sum(dim=1) / encoder_attention_mask[..., None].sum(dim=1).clamp(min=1.0)
            cap_emb = self.cap_proj(cap_pool)
            cap_gate = torch.tanh(self.cap_gate(time_step + cap_emb.to(time_step)))
            time_step = time_step + cap_emb.to(time_step) * cap_gate

        hidden_states = inputs_embeds

        # shared RoPE embeddings across all layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones(bsz, seq_len, device=inputs_embeds.device, dtype=torch.long)
            
        mask_mapping = {'full_attention': attention_mask}
        if self.has_sliding_layers:
            mask_mapping['sliding_attention'] = attention_mask
            
        if (
            attention_mask.dim() == 2 and 
            self.config._attn_implementation not in ['flash_attention_2', 'flash_attention_3']
        ):
            mask_mapping['full_attention'] = make_padding_mask_4d(
                attention_mask=attention_mask,
                dtype=inputs_embeds.dtype,
                q_len=seq_len,
                k_len=seq_len,
            )

            if self.has_sliding_layers:
                mask_mapping["sliding_attention"] = make_sliding_window_noncausal_mask_4d(
                    attention_mask=attention_mask,
                    window_size=self.config.sliding_window,
                    dtype=inputs_embeds.dtype,
                )
                
        if self.config.num_cross_attention_layers > 0 and encoder_hidden_states is not None:
            
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(bsz, encoder_hidden_states.shape[1], device=encoder_hidden_states.device, dtype=torch.long)
                
            if encoder_attention_mask.dim() == 2 and self.config._attn_implementation not in ['flash_attention_2', 'flash_attention_3']:
                encoder_attention_mask = make_padding_mask_4d(
                    attention_mask=encoder_attention_mask,
                    dtype=encoder_hidden_states.dtype,
                    q_len=seq_len,
                    k_len=encoder_hidden_states.shape[1],
                )

        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                time_step,
                attention_mask=mask_mapping[encoder_layer.attention_type],
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states, time_step)

        # we reuse BaseModelOutputWithPast for API uniformity (past_key_values is always None here)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )
    