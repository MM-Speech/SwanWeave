from typing import Callable, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from ._compat import (
    ACT2FN,
    ALL_ATTENTION_FUNCTIONS,
    BaseModelOutputWithPast,
    Cache,
    CausalLMOutputWithPast,
    DynamicCache,
    FlashAttentionKwargs,
    GenerationMixin,
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
    PreTrainedModel,
    ROPE_INIT_FUNCTIONS,
    TransformersKwargs,
    Unpack,
    auto_docstring,
    can_return_tuple,
    check_model_inputs,
    create_causal_mask,
    create_sliding_window_causal_mask,
    deprecate_kwarg,
    dynamic_rope_update,
    use_kernel_forward_from_hub,
)

from .transformer_config import TransformerConfig

@use_kernel_forward_from_hub("RMSNorm")
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
    
class MLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def make_padding_mask_4d(
    attention_mask: torch.Tensor,  # [bsz, seq_len], 1 for keep, 0 for pad
    dtype: torch.dtype,
    q_len: Optional[int] = None,
    k_len: Optional[int] = None,
    additive_mask=False
) -> torch.Tensor:
    """
    [bsz, seq_len] -> [bsz, 1, q_len, k_len]
    """
    if attention_mask.dim() != 2:
        raise ValueError(f"Expected 2D attention_mask, got shape {attention_mask.shape}")

    bsz, src_len = attention_mask.shape
    if k_len is None:
        k_len = src_len
    if q_len is None:
        q_len = k_len

    # [bsz, 1, 1, src_len]
    mask = attention_mask[:, None, None, :]  # 1 for keep, 0 for pad

    # broadcast [bsz, 1, q_len, k_len]
    if k_len != src_len:
        # 截断或 pad，这里简单截断到 k_len
        mask = mask[:, :, :, :k_len]
    mask = mask.expand(bsz, 1, q_len, k_len)

    if additive_mask:
        # 转成 additive mask
        # 1 -> 0.0; 0 -> -inf
        mask = (1.0 - mask.to(dtype)) * torch.finfo(dtype).min
    else:
        mask = mask.to(torch.bool)
        
    return mask


def make_sliding_window_noncausal_mask_4d(
    attention_mask: torch.Tensor,  # [bsz, seq_len], 1 for keep, 0 for pad
    window_size: int,
    dtype: torch.dtype,
    additive_mask=False
) -> torch.Tensor:
    """
    构造非因果的 sliding window + padding 的 4D additive mask:
    - 形状: [bsz, 1, seq_len, seq_len]
    - 对于 |i - j| > window_size - 1 的位置置为 -inf
    - 总的 window size 为 2 * window_size - 1
    - 对于 padding 的 key 位置也置为 -inf
    """
    if attention_mask.dim() != 2:
        raise ValueError(f"Expected 2D attention_mask, got shape {attention_mask.shape}")

    bsz, seq_len = attention_mask.shape
    device = attention_mask.device

    # 1. 基于 window 的局部 mask: 1 表示可见，0 表示被 mask（距离太远）
    idxs = torch.arange(seq_len, device=device)
    distance = idxs[None, :] - idxs[:, None]  # [seq_len, seq_len]
    local = (distance.abs() <= window_size - 1).to(dtype)  # [seq_len, seq_len]

    # [1, 1, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
    local = local[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    
    # 2. 对于 padding 的 query（attention_mask=0），不施加 window 限制
    #    即 local[row, :, :] = True，让它可以看见所有位置（由 pad_mask 再去屏蔽 pad key）
    query_mask = attention_mask[:, None, :, None].bool()      # [bsz,1,L,1], True 表示真实 token
    # query_mask = 1 时保留 window，=0 时把这一行都设为 True
    local = torch.where(query_mask, local, torch.ones_like(local))

    # 3. padding mask: 利用你已有的函数
    pad_mask = make_padding_mask_4d(
        attention_mask=attention_mask,
        dtype=dtype,
        q_len=seq_len,
        k_len=seq_len,
        additive_mask=additive_mask
    )
    # pad_mask: 0 for keep, -inf for pad

    if additive_mask:
        # 4. window mask 转成 additive 形式：1->0, 0->-inf
        window_mask = (1.0 - local) * torch.finfo(dtype).min

        # 5. 综合：padding + window。保持 additive mask 语义（被 mask 的地方都是一个很大的负数）
        final_mask = pad_mask + window_mask
    
    else:
        window_mask = local.to(torch.bool)
        final_mask = pad_mask & window_mask
        
    return final_mask


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        
        if config.use_gated_attention:
            self.attn_gate = nn.Linear(
                config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
            )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        
        if self.config.use_gated_attention:
            attn_output = attn_output * torch.sigmoid(self.attn_gate(hidden_states))
        
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CrossAttention(nn.Module):
    """
    Simple cross attention module:
    - queries come from hidden_states (decoder/encoder)
    - keys/values come from encoder_hidden_states (memory)
    - no causal masking, no RoPE by default
    """

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  # cross attention is never causal

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        if config.use_gated_attention:
            self.attn_gate = nn.Linear(
                config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
            )

    def forward(
        self,
        hidden_states: torch.Tensor,              # [bsz, tgt_len, hidden_size]
        encoder_hidden_states: torch.Tensor,      # [bsz, src_len, hidden_size]
        attention_mask: Optional[torch.Tensor] = None,  # [bsz, src_len]
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, tgt_len, _ = hidden_states.size()
        bsz, src_len, _ = encoder_hidden_states.size()

        # project to q, k, v
        q = self.q_proj(hidden_states).view(bsz, tgt_len, -1, self.head_dim).transpose(1, 2)    # [bsz, n_heads, tgt, d]
        k = self.k_proj(encoder_hidden_states).view(bsz, src_len, -1, self.head_dim).transpose(1, 2)  # [bsz, n_kv, src, d]
        v = self.v_proj(encoder_hidden_states).view(bsz, src_len, -1, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,
            **kwargs,
        )

        # [bsz, tgt_len, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, -1)
        
        if self.config.use_gated_attention:
            attn_output = attn_output * torch.sigmoid(self.attn_gate(hidden_states))
            
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        # encoder self-attention is non-causal, no sliding window
        self.self_attn.is_causal = False
        # self.self_attn.sliding_window = None

        self.mlp = MLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        attention_mask: Optional[torch.Tensor] = None,  # [bsz, 1, seq_len, seq_len] additive
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,      # [bsz, src_len, hidden]
        encoder_attention_mask: Optional[torch.Tensor] = None,     # [cross-attn mask [bsz, 1, tgt_len, src_len]
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
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
            hidden_states = self.attn_mhc(residual, hidden_states)
        else:
            hidden_states = residual + hidden_states
        
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
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        if self.config.use_mhc:
            hidden_states = self.mlp_mhc(residual, hidden_states)
        else:
            hidden_states = residual + hidden_states

        return hidden_states


class DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = Attention(config=config, layer_idx=layer_idx)

        self.mlp = MLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        if self.config.use_mhc:
            hidden_states = self.attn_mhc(residual, hidden_states)
        else:
            hidden_states = residual + hidden_states
        
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
            
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        if self.config.use_mhc:
            hidden_states = self.mlp_mhc(residual, hidden_states)
        else:
            hidden_states = residual + hidden_states

        return hidden_states
    
    
@auto_docstring
class TransformerPreTrainedModel(PreTrainedModel):
    config: TransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["EncoderLayer", "DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": DecoderLayer,
        "attentions": Attention,
    }
    
    
class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    @staticmethod
    def compute_default_rope_parameters(config, device=None):
        # BC: transformers 5.x removed ROPE_INIT_FUNCTIONS["default"]; provide a vanilla equivalent.
        base = float(getattr(config, "rope_theta", 10000.0))
        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        return inv_freq, 1.0

    def __init__(self, config: TransformerConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type, self.compute_default_rope_parameters)

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
    
@auto_docstring
class TransformerEncoderModel(TransformerPreTrainedModel):
    """
    Encoder-only transformer model with RoPE and non-causal self-attention.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        if config.vocab_size > 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [EncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [bsz, seq_len] (1 for keep, 0 for pad)
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,  # [bsz, src_len, hidden]
        encoder_attention_mask: Optional[torch.Tensor] = None, # [bsz, src_len]
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, seq_len, _ = inputs_embeds.size()

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)

        hidden_states = inputs_embeds

        # shared RoPE embeddings across all layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones(bsz, seq_len, device=inputs_embeds.device, dtype=torch.long)
            
        mask_mapping = {'full_attention': attention_mask}
        if self.has_sliding_layers:
            mask_mapping['sliding_attention'] = attention_mask
            
        if attention_mask.dim() == 2 and self.config._attn_implementation not in ['flash_attention_2', 'flash_attention_3']:
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
                attention_mask=mask_mapping[encoder_layer.attention_type],
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        # we reuse BaseModelOutputWithPast for API uniformity (past_key_values is always None here)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )
    

@auto_docstring
class TransformerDecoderModel(TransformerPreTrainedModel):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.config = config
        
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        if config.vocab_size > 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,  # [bsz, src_len, hidden]
        encoder_attention_mask: Optional[torch.Tensor] = None, # [bsz, src_len]
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        bsz, seq_len, _ = inputs_embeds.size()

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
                
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
              
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class TransformerForCausalLM(TransformerPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = TransformerDecoderModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        

# if __name__ == '__main__':
#     import os
#     if os.path.isfile('.env.local'):
#         from dotenv import load_dotenv
#         load_dotenv('.env.local')
        
#     torch.manual_seed(42)
        
#     inputs = torch.rand((2, 10, 256), device='cuda').bfloat16()
#     attn_mask = torch.ones((2, 10), device='cuda').bfloat16()
#     attn_mask[1, 5:] = 0
    
#     with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        
#         encoder = TransformerEncoderModel(TransformerConfig(
#             vocab_size=0, hidden_size=256, intermediate_size=1024, num_hidden_layers=4, num_attention_heads=16,
#             max_window_layers=0, use_sliding_window=True, sliding_window=2,
#             # attn_implementation='flash_attention_2',
#             attn_implementation='sdpa'
#         )).to('cuda')
        
#         outputs = encoder.forward(inputs_embeds=inputs, attention_mask=attn_mask, output_attentions=False)
        
        # print(f"{outputs = }")
        
        # decoder = TransformerDecoderModel(TransformerConfig(
        #     vocab_size=0, hidden_size=256, intermediate_size=1024, num_hidden_layers=4, num_attention_heads=16,
        #     max_window_layers=0, use_sliding_window=True, sliding_window=2,
        #     # attn_implementation='flash_attention_2'
        #     attn_implementation='sdpa'
        # )).to('cuda')
        
        # outputs = decoder.forward(inputs_embeds=inputs, attention_mask=attn_mask, output_attentions=False)