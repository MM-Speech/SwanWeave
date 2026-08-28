from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from modules.commons.hf._compat import (
    ACT2FN,
    GradientCheckpointingLayer,
    MoeModelOutputWithPast,
    PreTrainedModel,
    TransformersKwargs,
    Unpack,
    auto_docstring,
    check_model_inputs,
)

from modules.commons.hf.transformer_dit_config import TransformerDiTConfig
from modules.commons.hf.transformer import (
    Attention,
    MLP,
    RMSNorm,
    CrossAttention,
    RotaryEmbedding,
    make_padding_mask_4d,
    make_sliding_window_noncausal_mask_4d,
)
from modules.commons.hf.transformer_dit import AdaLayerNormZero, AdaLayerNormZero_Final


class _ExpertMLP(nn.Module):
    def __init__(self, config: TransformerDiTConfig):
        super().__init__()
        intermediate_size = int(getattr(config, "moe_intermediate_size", 0) or config.intermediate_size)
        self.ffn = MLP(config, intermediate_size=intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.ffn(x.unsqueeze(1)).squeeze(1)
        return self.ffn(x)


class _PackedExpertMLP(nn.Module):
    def __init__(self, config: TransformerDiTConfig, num_experts: int):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(getattr(config, "moe_intermediate_size", 0) or config.intermediate_size)
        self.num_experts = int(num_experts)
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))
        self.up_weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))
        self.down_weight = nn.Parameter(torch.empty(self.num_experts, self.intermediate_size, self.hidden_size))

    def extra_repr(self) -> str:
        lines = [
            f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}, num_experts={self.num_experts}"
        ]
        for name, param in self.named_parameters(recurse=False):
            lines.append(
                f"{name}: shape={tuple(param.shape)}, requires_grad={param.requires_grad}"
            )
        return "\n".join(lines)

    def reset_packed_weights(self, std: float) -> None:
        nn.init.normal_(self.gate_weight, mean=0.0, std=std)
        nn.init.normal_(self.up_weight, mean=0.0, std=std)
        nn.init.normal_(self.down_weight, mean=0.0, std=std)

    def forward_packed(self, x: torch.Tensor) -> torch.Tensor:
        gate_hidden = torch.bmm(x, self.gate_weight)
        up_hidden = torch.bmm(x, self.up_weight)
        hidden = self.act_fn(gate_hidden) * up_hidden
        return torch.bmm(hidden, self.down_weight)


class ExpertChoiceMoEFFN(nn.Module):
    """
    Expert-Choice FFN for variable-length wav latent sequences.

    Design goals:
      - valid-token-only routing: padding tokens never participate in ranking or capacity.
      - ratio-based capacity: expert capacity is computed from valid-token count.
      - reverse-linear schedule: more routed capacity at later denoising progress.
      - shared fallback: every valid token always sees >=1 shared dense expert.
      - no extra auxiliary loss.

    Expected extra runtime kwarg (optional):
      - ec_step_ratio: normalized denoising progress in [0, 1], where
        0 = early / high-noise stage, 1 = late / low-noise stage.
        If omitted, a config default is used.
    """

    def __init__(self, config: TransformerDiTConfig, moe_ordinal: int):
        super().__init__()
        self.config = config
        self.dim = int(config.hidden_size)
        self.moe_ordinal = int(moe_ordinal)

        self.num_routed = int(getattr(config, "moe_num_routed", 0))
        self.num_shared = max(1, int(getattr(config, "moe_num_shared", 1)))
        if self.num_routed <= 0:
            raise ValueError("ExpertChoiceMoEFFN requires config.moe_num_routed > 0")

        self.routed_experts = _PackedExpertMLP(config, self.num_routed)
        self.shared_experts = nn.ModuleList([_ExpertMLP(config) for _ in range(self.num_shared)])
        self.gate = nn.Linear(self.dim, self.num_routed, bias=False)

        # Reverse-linear capacity schedule.
        self.capacity_ratio_min = float(getattr(config, "moe_capacity_factor_min", 1.0))
        self.capacity_ratio_max = float(getattr(config, "moe_capacity_factor_max", 2.0))
        if self.capacity_ratio_min <= 0.0 or self.capacity_ratio_max <= 0.0:
            raise ValueError("capacity ratios must be > 0")
        if self.capacity_ratio_max < self.capacity_ratio_min:
            raise ValueError("moe_capacity_factor_max must be >= moe_capacity_factor_min")

        # Early-layer protection.
        self.early_dense_moe_layers = int(getattr(config, "ec_early_dense_moe_layers", 1))
        self.early_capacity_boost_moe_layers = int(getattr(config, "ec_early_capacity_boost_moe_layers", 1))
        self.early_capacity_ratio = float(
            getattr(config, "ec_early_capacity_ratio", self.capacity_ratio_max)
        )

        # Runtime fallback when caller does not pass ec_step_ratio.
        self.default_step_ratio = float(getattr(config, "ec_default_step_ratio", 0.5))
        self.default_step_ratio = max(0.0, min(1.0, self.default_step_ratio))

    def _packed_routed_experts(
        self,
        x_valid: torch.Tensor,
        chosen: torch.Tensor,
        combine: torch.Tensor,
        routed_out: torch.Tensor,
    ) -> torch.Tensor:
        # Pack routed tokens as [E, K, C], run grouped GEMMs, then scatter-add once.
        expert_token_ids = chosen.transpose(0, 1).contiguous()  # [E, K]
        flat_token_ids = expert_token_ids.reshape(-1)
        expert_x = x_valid.index_select(0, flat_token_ids).view(self.num_routed, -1, self.dim)
        expert_y = self.routed_experts.forward_packed(
            expert_x.to(dtype=self.routed_experts.gate_weight.dtype)
        ).to(routed_out.dtype)

        flat_expert_ids = torch.arange(self.num_routed, device=x_valid.device)[:, None]
        flat_expert_ids = flat_expert_ids.expand_as(expert_token_ids).reshape(-1)
        expert_w = combine[flat_token_ids, flat_expert_ids].view(self.num_routed, -1, 1).to(routed_out.dtype)
        routed_out.index_add_(0, flat_token_ids, (expert_y * expert_w).reshape(-1, self.dim))
        return routed_out

    def _normalize_step_ratio(
        self,
        ec_step_ratio: Optional[torch.Tensor | float],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if ec_step_ratio is None:
            return torch.full((batch_size,), self.default_step_ratio, device=device, dtype=dtype)

        if not torch.is_tensor(ec_step_ratio):
            return torch.full((batch_size,), float(ec_step_ratio), device=device, dtype=dtype)

        ratio = ec_step_ratio.to(device=device, dtype=dtype)
        if ratio.dim() == 0:
            ratio = ratio.expand(batch_size)
        elif ratio.dim() == 1:
            if ratio.shape[0] == 1:
                ratio = ratio.expand(batch_size)
            elif ratio.shape[0] != batch_size:
                raise ValueError(
                    f"ec_step_ratio batch mismatch: got {ratio.shape[0]}, expected {batch_size}"
                )
        elif ratio.dim() == 2:
            if ratio.shape[0] != batch_size:
                raise ValueError(
                    f"ec_step_ratio batch mismatch: got {ratio.shape[0]}, expected {batch_size}"
                )
            if ratio.shape[1] == 1:
                ratio = ratio[:, 0]
            else:
                ratio = ratio.mean(dim=1)
        else:
            raise ValueError(
                f"ec_step_ratio must be scalar, [B], or [B, 1]/[B, T]; got {tuple(ratio.shape)}"
            )
        return ratio.clamp_(0.0, 1.0)

    def _capacity_ratio(self, progress: torch.Tensor) -> torch.Tensor:
        # reverse-linear: later denoising stages receive more routed capacity.
        ratio = self.capacity_ratio_min + (
            self.capacity_ratio_max - self.capacity_ratio_min
        ) * progress

        boost_begin = self.early_dense_moe_layers
        boost_end = boost_begin + self.early_capacity_boost_moe_layers
        if boost_begin <= self.moe_ordinal < boost_end:
            ratio = torch.clamp_min(ratio, self.early_capacity_ratio)

        return ratio

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        ec_step_ratio: Optional[torch.Tensor | float] = None,
        expert_capacity: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
            padding_mask: [B, T] bool, True means valid token.
            ec_step_ratio: scalar / [B] / [B,1] / [B,T], normalized progress in [0,1].
        Returns:
            out: [B, T, C]
        """
        batch_size, seq_len, hidden = x.shape
        device = x.device
        dtype = x.dtype

        if padding_mask is None:
            padding_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)
        else:
            padding_mask = padding_mask.to(device=device, dtype=torch.bool)

        x_flat = x.reshape(batch_size * seq_len, hidden)
        mask_flat = padding_mask.reshape(batch_size * seq_len)
        valid_idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)

        if valid_idx.numel() == 0:
            return torch.zeros_like(x)

        batch_ids = valid_idx // seq_len
        x_valid = x_flat.index_select(0, valid_idx)  # [N_valid, C]
        num_valid = x_valid.shape[0]

        # Shared fallback path for every valid token.
        shared_out = torch.zeros_like(x_valid)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x_valid).to(dtype)
        shared_out = shared_out / float(self.num_shared)

        # Routed experts: expert-choice over valid tokens only.
        gate_logits = self.gate(x_valid).float()  # [N_valid, E]
        gate_probs = torch.softmax(gate_logits, dim=-1).to(dtype)

        if expert_capacity is None:
            progress = self._normalize_step_ratio(ec_step_ratio, batch_size, device, gate_logits.dtype)
            # Ratio-based capacity for variable-length sequences:
            # first compute each sample's routed budget from its valid length and denoising progress,
            # then aggregate the per-sample budgets into a single expert capacity for the pooled routing set.
            valid_counts = padding_mask.sum(dim=1).to(gate_logits.dtype)  # [B]
            sample_capacity_ratio = self._capacity_ratio(progress)         # [B]
            sample_expert_budget = torch.ceil(sample_capacity_ratio * valid_counts / float(self.num_routed))
            expert_capacity = int(torch.clamp(
                sample_expert_budget.sum().to(torch.long),
                min=1,
                max=num_valid,
            ).item())
        else:
            expert_capacity = max(1, min(int(expert_capacity), num_valid))

        chosen = torch.topk(gate_logits, k=expert_capacity, dim=0, largest=True, sorted=False).indices
        selected_mask = torch.zeros_like(gate_probs, dtype=torch.bool)
        selected_mask.scatter_(0, chosen, True)

        combine = gate_probs * selected_mask.to(dtype)
        denom = combine.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        combine = (combine / denom).to(dtype)

        routed_out = torch.zeros_like(x_valid)
        routed_out = self._packed_routed_experts(
            x_valid=x_valid,
            chosen=chosen,
            combine=combine,
            routed_out=routed_out,
        )

        out_valid = shared_out + routed_out
        out_flat = torch.zeros_like(x_flat)
        out_flat.index_copy_(0, valid_idx, out_valid)
        return out_flat.view(batch_size, seq_len, hidden)


def _moe_candidate_positions(config: TransformerDiTConfig) -> list[int]:
    if not bool(getattr(config, "use_moe_ffn", False)):
        return []

    sparse_step = max(1, int(getattr(config, "decoder_sparse_step", 2)))
    mlp_only_layers = set(getattr(config, "mlp_only_layers", []) or [])
    positions: list[int] = []
    for layer_idx in range(int(config.num_hidden_layers)):
        if layer_idx in mlp_only_layers:
            continue
        if (layer_idx + 1) % sparse_step == 0:
            positions.append(layer_idx)
    return positions


class EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TransformerDiTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.self_attn.is_causal = False

        moe_positions = _moe_candidate_positions(config)
        moe_ordinal = moe_positions.index(layer_idx) if layer_idx in moe_positions else -1
        early_dense_moe_layers = int(getattr(config, "ec_early_dense_moe_layers", 1))
        use_moe_here = moe_ordinal >= 0 and moe_ordinal >= early_dense_moe_layers

        if use_moe_here:
            self.mlp = ExpertChoiceMoEFFN(config=config, moe_ordinal=moe_ordinal)
        else:
            self.mlp = MLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = AdaLayerNormZero(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

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
                hidden_size=config.hidden_size,
                n_streams=config.mhc_num_streams,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                use_dynamic=config.mhc_use_dynamic,
                rms_norm_eps=RMSNorm,
            )
            self.mlp_mhc = MHCResidual(
                hidden_size=config.hidden_size,
                n_streams=config.mhc_num_streams,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                use_dynamic=config.mhc_use_dynamic,
                rms_norm_eps=RMSNorm,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_step: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        ec_step_ratio: Optional[torch.Tensor | float] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.input_layernorm(hidden_states, time_step)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            cache_position=None,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if self.config.use_mhc:
            hidden_states = self.attn_mhc(residual, gate_msa[:, None, :] * hidden_states)
        else:
            hidden_states = residual + gate_msa[:, None, :] * hidden_states

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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None, :]

        if isinstance(self.mlp, ExpertChoiceMoEFFN):
            hidden_states = self.mlp(
                hidden_states,
                padding_mask=padding_mask,
                ec_step_ratio=ec_step_ratio,
                expert_capacity=kwargs.pop("expert_capacity", None),
            )
        else:
            hidden_states = self.mlp(hidden_states)

        if self.config.use_mhc:
            hidden_states = self.mlp_mhc(residual, gate_mlp[:, None] * hidden_states)
        else:
            hidden_states = residual + gate_mlp[:, None] * hidden_states

        return hidden_states


@auto_docstring
class TransformerEcMoePreTrainedModel(PreTrainedModel):
    config: TransformerDiTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["EncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": EncoderLayer,
        "attentions": Attention,
    }


@auto_docstring
class TransformerEcMoeDiTModel(TransformerEcMoePreTrainedModel):
    """
    DiT encoder with Expert-Choice MoE FFN for variable-length wav latents.

    Forward interface intentionally follows modules/commons/hf/transformer_dit.py
    and modules/commons/hf/transformer_dit_moe.py. The only extra runtime kwarg is
    optional `ec_step_ratio`, passed through **kwargs.

    configuration:
        - use_moe_ffn: whether to enable MoE FFN.
        - decoder_sparse_step: MoE insertion frequency.
        - moe_intermediate_size: expert FFN hidden size.
        - moe_num_routed: number of routed experts.
        - moe_num_shared: number of shared fallback experts (at least 1 is enforced).
        - moe_capacity_factor_min/max: min/max capacity ratio for reverse-linear schedule.
        - mlp_only_layers: layers forced to stay dense.
        - ec_early_dense_moe_layers: earliest candidate MoE layers forced to dense.
        - ec_early_capacity_boost_moe_layers: number of early MoE layers with boosted capacity.
        - ec_early_capacity_ratio: boosted minimum capacity ratio for early MoE layers.
        - ec_default_step_ratio: fallback denoising progress in [0, 1] when runtime does not pass ec_step_ratio.

    single MOE activations ratio (relative to dense) are approximately: 
        (moe_num_shared + (moe_capacity_factor_min + moe_capacity_factor_max) / 2) * moe_intermediate_size / intermediate_size

    """

    def __init__(self, config: TransformerDiTConfig):
        super().__init__(config)
        self.padding_idx = getattr(config, "pad_token_id", None)

        self.layers = nn.ModuleList([EncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = AdaLayerNormZero_Final(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        if self.config.num_cross_attention_layers > 0 and config.use_caption_pool_in_adaln:
            self.cap_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.cap_gate = nn.Linear(config.hidden_size, config.hidden_size)

        self.post_init()

    def _precompute_expert_capacities(
        self,
        padding_mask: torch.Tensor,
        ec_step_ratio: Optional[torch.Tensor | float],
    ) -> dict[int, int]:
        moe_layers = [
            layer for layer in self.layers[: self.config.num_hidden_layers]
            if isinstance(layer.mlp, ExpertChoiceMoEFFN)
        ]
        if len(moe_layers) == 0:
            return {}

        batch_size = padding_mask.shape[0]
        device = padding_mask.device
        progress = moe_layers[0].mlp._normalize_step_ratio(ec_step_ratio, batch_size, device, torch.float32)
        # Ratio-based capacity for variable-length sequences:
        # first compute each sample's routed budget from its valid length and denoising progress,
        # then aggregate the per-sample budgets into a single expert capacity for the pooled routing set.
        valid_counts = padding_mask.sum(dim=1).to(torch.float32)  # [B]
        num_valid = torch.clamp(valid_counts.sum().to(torch.long), min=1)

        capacities = []
        for layer in moe_layers:
            sample_capacity_ratio = layer.mlp._capacity_ratio(progress)  # [B]
            sample_expert_budget = torch.ceil(
                sample_capacity_ratio * valid_counts / float(layer.mlp.num_routed)
            )
            capacities.append(torch.clamp(sample_expert_budget.sum().to(torch.long), min=1, max=num_valid))

        capacity_values = torch.stack(capacities, dim=0).cpu().tolist()
        return {layer.mlp.moe_ordinal: int(cap) for layer, cap in zip(moe_layers, capacity_values)}

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        batch_size, seq_len, _ = inputs_embeds.size()
        time_step = kwargs.pop("time_step")
        ec_step_ratio = kwargs.pop("ec_step_ratio", None)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)

        if (
            self.config.num_cross_attention_layers > 0
            and self.config.use_caption_pool_in_adaln
            and encoder_hidden_states is not None
            and encoder_attention_mask is not None
        ):
            cap_pool = (encoder_hidden_states * encoder_attention_mask[..., None]).sum(dim=1) / encoder_attention_mask[..., None].sum(dim=1).clamp(min=1.0)
            cap_emb = self.cap_proj(cap_pool)
            cap_gate = torch.tanh(self.cap_gate(time_step + cap_emb.to(time_step)))
            time_step = time_step + cap_emb.to(time_step) * cap_gate

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=inputs_embeds.device, dtype=torch.long)

        padding_mask_2d = kwargs.pop("padding_mask", None)
        if padding_mask_2d is None:
            if attention_mask.dim() == 2:
                padding_mask_2d = attention_mask.to(torch.bool)
            else:
                raise ValueError(
                    "EC routing needs a 2D padding mask [B, T]. "
                    "When passing a 4D attention_mask, also pass padding_mask=[B, T]."
                )
        else:
            padding_mask_2d = padding_mask_2d.to(torch.bool)

        expert_capacities = self._precompute_expert_capacities(padding_mask_2d, ec_step_ratio)

        mask_mapping = {"full_attention": attention_mask}
        if self.has_sliding_layers:
            mask_mapping["sliding_attention"] = attention_mask

        if attention_mask.dim() == 2 and self.config._attn_implementation not in ["flash_attention_2", "flash_attention_3"]:
            mask_mapping["full_attention"] = make_padding_mask_4d(
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
                encoder_attention_mask = torch.ones(
                    batch_size,
                    encoder_hidden_states.shape[1],
                    device=encoder_hidden_states.device,
                    dtype=torch.long,
                )

            if encoder_attention_mask.dim() == 2 and self.config._attn_implementation not in ["flash_attention_2", "flash_attention_3"]:
                encoder_attention_mask = make_padding_mask_4d(
                    attention_mask=encoder_attention_mask,
                    dtype=encoder_hidden_states.dtype,
                    q_len=seq_len,
                    k_len=encoder_hidden_states.shape[1],
                )

        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_kwargs = dict(kwargs)
            if isinstance(encoder_layer.mlp, ExpertChoiceMoEFFN):
                layer_kwargs["expert_capacity"] = expert_capacities.get(encoder_layer.mlp.moe_ordinal)
            hidden_states = encoder_layer(
                hidden_states,
                time_step,
                attention_mask=mask_mapping[encoder_layer.attention_type],
                padding_mask=padding_mask_2d,
                ec_step_ratio=ec_step_ratio,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                **layer_kwargs,
            )

        hidden_states = self.norm(hidden_states, time_step)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )
