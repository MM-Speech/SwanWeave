from typing import Optional

import torch
from torch import nn

from modules.commons.hf._compat import (
    GradientCheckpointingLayer,
    MoeModelOutputWithPast,
    PreTrainedModel,
    Unpack,
    TransformersKwargs,
    auto_docstring,
    check_model_inputs,
)

from modules.commons.hf.transformer_dit_config import TransformerDiTConfig
from modules.commons.hf.transformer import Attention, MLP, RMSNorm, CrossAttention, RotaryEmbedding, make_padding_mask_4d, make_sliding_window_noncausal_mask_4d
from modules.commons.hf.transformer_dit import AdaLayerNormZero, AdaLayerNormZero_Final

import math
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F


class _ExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        inter = getattr(config, "moe_intermediate_size", None)
        if inter is None:
            inter = config.intermediate_size
        self.ffn = MLP(config, intermediate_size=inter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  # [N,C]
            return self.ffn(x.unsqueeze(1)).squeeze(1)
        return self.ffn(x)


def _top_p_select(probs: torch.Tensor, p_tok: torch.Tensor) -> torch.Tensor:
    """
    probs: [N, E]
    p_tok: [N] in (0, 1)
    return selected_mask: [N, E] bool
    """
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    prev_cum = cumsum - sorted_probs

    p_tok = p_tok.view(-1, 1)
    keep_sorted = prev_cum < p_tok
    keep_sorted[..., 0] = True

    selected_mask = torch.zeros_like(keep_sorted, dtype=torch.bool)
    selected_mask.scatter_(1, sorted_idx, keep_sorted)
    return selected_mask


class TaskExpertRouter(nn.Module):
    """
    Selects task experts once per sample from caption embeddings.

    The selected expert indices are shared by every MoE layer in the DiT forward.
    """

    def __init__(self, config: TransformerDiTConfig):
        super().__init__()
        self.config = config
        self.num_task_experts = int(getattr(config, "moe_num_task_experts", 0))
        if self.num_task_experts <= 0:
            raise ValueError("moe_num_task_experts must be > 0 when task expert routing is enabled")

        self.p = float(getattr(config, "moe_task_p", getattr(config, "moe_p", 0.7)))
        self.task_queries = nn.Parameter(torch.empty(1, self.num_task_experts, config.hidden_size))
        nn.init.normal_(self.task_queries, mean=0.0, std=float(getattr(config, "initializer_range", 0.02)))

        self.cross_attn = CrossAttention(config=config, layer_idx=0)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logit_proj = nn.Linear(config.hidden_size, 1, bias=False)

        self.last_task_mask = None
        self.last_task_weights = None
        self.last_task_logits = None

    def _safe_caption_mask(
        self,
        caption_hidden_states: torch.Tensor,
        caption_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, src_len, _ = caption_hidden_states.shape
        if caption_attention_mask is None:
            mask = torch.ones(bsz, src_len, device=caption_hidden_states.device, dtype=torch.bool)
        else:
            if caption_attention_mask.dim() != 2:
                raise ValueError(
                    f"task expert routing expects a 2D caption mask [B, S], got {tuple(caption_attention_mask.shape)}"
                )
            mask = caption_attention_mask.to(device=caption_hidden_states.device, dtype=torch.bool)

        if src_len > 0:
            empty_rows = ~mask.any(dim=1)
            if empty_rows.any():
                mask = mask.clone()
                mask[empty_rows, 0] = True
        return mask

    def forward(
        self,
        caption_hidden_states: Optional[torch.Tensor],
        caption_attention_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if caption_hidden_states is None:
            logits = torch.zeros(batch_size, self.num_task_experts, device=device, dtype=torch.float32)
        else:
            caption_mask = self._safe_caption_mask(caption_hidden_states, caption_attention_mask)
            queries = self.task_queries.expand(batch_size, -1, -1).to(
                device=caption_hidden_states.device,
                dtype=caption_hidden_states.dtype,
            )

            if self.config._attn_implementation in ["flash_attention_2", "flash_attention_3"]:
                attn_mask = caption_mask.to(dtype=torch.long)
            else:
                attn_mask = make_padding_mask_4d(
                    attention_mask=caption_mask.to(dtype=torch.long),
                    dtype=caption_hidden_states.dtype,
                    q_len=self.num_task_experts,
                    k_len=caption_hidden_states.shape[1],
                )

            task_ctx, _ = self.cross_attn(
                hidden_states=queries,
                encoder_hidden_states=caption_hidden_states,
                attention_mask=attn_mask,
            )
            task_ctx = self.norm(task_ctx)
            logits = self.logit_proj(task_ctx).squeeze(-1).float()

        probs = F.softmax(logits, dim=-1)
        p_tok = torch.full((batch_size,), self.p, device=logits.device, dtype=torch.float32)
        selected_mask = _top_p_select(probs, p_tok)
        weights = probs * selected_mask.to(probs.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        self.last_task_mask = selected_mask.detach()
        self.last_task_weights = weights.detach()
        self.last_task_logits = logits.detach()
        return selected_mask, weights.to(dtype=dtype), logits


class DynamicTopPMoEFFN(nn.Module):
    """
    Top-P MoE-FFN with:
      - pure Top-P (no max_k)
      - standard auxiliary load-balancing loss
      - router z-loss
      - capacity / overflow drop
      - t-aware compute budget: p(t), null_bias(t), capacity_factor(t)
      - caption-selected task experts shared across all MoE layers

    forward(x, padding_mask, t, task_expert_mask, task_expert_weights) -> (out, aux)
      - x: [B,T,C]
      - padding_mask: [B,T] bool, True=valid
      - t: [B,C] or [B,T,C]
      - task_expert_mask: [B, K] bool
      - task_expert_weights: [B, K]
      - aux: scalar = load_balance_loss_coef * lb_loss + router_z_loss_coef * z_loss
    """
    def __init__(
        self,
        *,
        config,
        dim: int,
        num_routed: int,
        num_task_experts: int,
        num_null: int,
        p: float = 0.7,

        # gumbel
        use_gumbel: bool = False,
        gumbel_tau_start: float = 1.0,
        gumbel_tau_end: float = 0.3,
        gumbel_tau_anneal_steps: int = 200_000,
        expert_dropout: float = 0.0,

        # capacity / overflow
        capacity_factor_min: float = 1.0,
        capacity_factor_max: float = 2.0,
        overflow_drop: bool = True,

        # t budget
        use_t_budget: bool = True,
        p_min: float = 0.4,
        p_max: float = 0.95,
        null_logit_bias_min: float = -2.0,
        null_logit_bias_max: float = 0.0,

        # aux losses
        load_balance_loss_coef: float = 1e-2,
        router_z_loss_coef: float = 1e-3,
    ):
        super().__init__()
        self.config = config
        self.dim = int(dim)

        self.num_routed = int(num_routed)
        self.num_task_experts = int(num_task_experts)
        self.num_null = int(num_null)
        self.p = float(p)

        # gumbel
        self.use_gumbel = bool(use_gumbel)
        self.gumbel_tau_start = float(gumbel_tau_start)
        self.gumbel_tau_end = float(gumbel_tau_end)
        self.gumbel_tau_anneal_steps = int(gumbel_tau_anneal_steps)
        self.expert_dropout = float(expert_dropout)
        self.register_buffer("gumbel_step", torch.zeros(1), persistent=False)

        # experts
        self.routed_experts = nn.ModuleList([_ExpertMLP(config) for _ in range(self.num_routed)])
        self.task_experts = nn.ModuleList([_ExpertMLP(config) for _ in range(self.num_task_experts)])

        # gate routes over (routed + null)
        self.total_gate_experts = self.num_routed + self.num_null
        if self.total_gate_experts <= 0:
            raise ValueError("num_routed + num_null must be > 0")

        self.gate = nn.Linear(self.dim, self.total_gate_experts, bias=False)

        # time-aware injection to gate input
        self.t_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.t_proj.weight)
        nn.init.zeros_(self.t_proj.bias)

        # ===== capacity / overflow =====
        self.capacity_factor_min = float(capacity_factor_min)
        self.capacity_factor_max = float(capacity_factor_max)
        self.overflow_drop = bool(overflow_drop)

        # ===== t-budget =====
        self.use_t_budget = bool(use_t_budget)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.null_logit_bias_min = float(null_logit_bias_min)
        self.null_logit_bias_max = float(null_logit_bias_max)

        if self.use_t_budget:
            self.t_budget_proj = nn.Linear(self.dim, 1)
            nn.init.zeros_(self.t_budget_proj.weight)
            nn.init.zeros_(self.t_budget_proj.bias)

        # ===== aux loss coeffs =====
        self.load_balance_loss_coef = float(load_balance_loss_coef)
        self.router_z_loss_coef = float(router_z_loss_coef)
        self.null_loss_coef = float(getattr(config, "moe_null_loss_coef", load_balance_loss_coef))

        # DeepSeek-style bias balance. This bias is not trained by gradients:
        # it is updated after optimizer steps and only affects expert selection.
        self.use_bias_balance = bool(getattr(config, "moe_use_bias_balance", False))
        self.bias_update_rate = float(getattr(config, "moe_bias_update_rate", 1e-3))
        self.bias_momentum = float(getattr(config, "moe_bias_momentum", 0.9))
        self.bias_clamp = float(getattr(config, "moe_bias_clamp", 3.0))
        self.register_buffer("routing_bias", torch.zeros(self.num_routed), persistent=True)
        self.register_buffer("load_accumulator", torch.zeros(self.num_routed), persistent=False)
        self.register_buffer("load_ema", torch.zeros(self.num_routed), persistent=False)
        self.register_buffer("load_ema_initialized", torch.zeros((), dtype=torch.bool), persistent=False)

        # optional logging hooks
        self.last_lb_loss = None
        self.last_z_loss = None
        self.last_aux_loss = None

    def _current_tau(self) -> float:
        if (not self.use_gumbel) or (self.gumbel_tau_anneal_steps <= 0):
            return float(self.gumbel_tau_start)
        step = float(self.gumbel_step.item())
        ratio = min(1.0, step / float(self.gumbel_tau_anneal_steps))
        tau = self.gumbel_tau_start + (self.gumbel_tau_end - self.gumbel_tau_start) * ratio
        return max(float(tau), 1e-4)

    @staticmethod
    def _top_p_select(probs: torch.Tensor, p_tok: torch.Tensor) -> torch.Tensor:
        return _top_p_select(probs, p_tok)

    def _load_balance_loss(
        self,
        dispatched_sel: torch.Tensor,   # [N, R] bool
        router_probs: torch.Tensor,     # [N, E] float32, clean router probs
    ) -> torch.Tensor:
        """
        Standard aux load balancing loss.

        我这里用的是 routed experts 上的：
          load_i       = dispatched assignments fraction
          importance_i = router probability mass fraction
          loss = R * sum_i(load_i * importance_i)

        这类 loss 的最优值接近 1，不是 0。
        """
        if self.num_routed <= 0:
            return router_probs.new_tensor(0.0)

        load = dispatched_sel.float().sum(dim=0)  # [R]
        total_load = load.sum()
        if total_load <= 0:
            return router_probs.new_tensor(0.0)

        load = load / total_load.clamp_min(1.0)

        importance = router_probs[:, :self.num_routed].sum(dim=0)  # [R]
        total_importance = importance.sum()
        if total_importance <= 0:
            return router_probs.new_tensor(0.0)

        importance = importance / total_importance.clamp_min(1e-9)

        loss = self.num_routed * torch.sum(load * importance)
        return loss

    @torch.no_grad()
    def _accumulate_bias_balance_load(self, routed_sel: torch.Tensor) -> None:
        if (not self.use_bias_balance) or (not self.training) or self.num_routed <= 0:
            return
        load = routed_sel.to(dtype=torch.float32).sum(dim=0)
        self.load_accumulator.add_(load.to(device=self.load_accumulator.device))

    @torch.no_grad()
    def update_moe_bias_balance(self, distributed: bool = True) -> dict:
        if (not self.use_bias_balance) or self.num_routed <= 0:
            return {"updated": False}

        counts = self.load_accumulator.detach().float().clone()
        if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)

        total = counts.sum()
        if total <= 0:
            self.load_accumulator.zero_()
            return {"updated": False, "tokens": 0.0}

        avg = counts.mean().clamp_min(1e-9)
        load_ratio = counts / avg
        if self.bias_momentum > 0.0:
            if bool(self.load_ema_initialized.item()):
                self.load_ema.mul_(self.bias_momentum).add_(
                    load_ratio.to(self.load_ema.device),
                    alpha=1.0 - self.bias_momentum,
                )
            else:
                self.load_ema.copy_(load_ratio.to(self.load_ema.device))
                self.load_ema_initialized.fill_(True)
            balance_signal = self.load_ema
        else:
            balance_signal = load_ratio.to(self.routing_bias.device)

        center = balance_signal.mean()
        delta = torch.zeros_like(self.routing_bias)
        delta = torch.where(balance_signal > center, delta - self.bias_update_rate, delta)
        delta = torch.where(balance_signal < center, delta + self.bias_update_rate, delta)
        self.routing_bias.add_(delta.to(dtype=self.routing_bias.dtype))
        self.routing_bias.clamp_(-self.bias_clamp, self.bias_clamp)
        self.load_accumulator.zero_()

        return {
            "updated": True,
            "tokens": float(total.item()),
            "load_min": float(load_ratio.min().item()),
            "load_max": float(load_ratio.max().item()),
            "bias_min": float(self.routing_bias.min().item()),
            "bias_max": float(self.routing_bias.max().item()),
        }

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        task_expert_mask: Optional[torch.Tensor] = None,
        task_expert_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B,T,C]
        padding_mask: [B,T] bool, True=valid
        t: [B,C] or [B,T,C]
        return: out, aux
        """
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype

        # mask
        if padding_mask is None:
            padding_mask = torch.ones((B, T), device=device, dtype=torch.bool)
        else:
            padding_mask = padding_mask.to(torch.bool)

        x_flat = x.reshape(B * T, C)
        mask_flat = padding_mask.reshape(B * T)

        idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)  # [N]
        if idx.numel() == 0:
            aux_zero = x.new_tensor(0.0)
            self.last_lb_loss = aux_zero.detach()
            self.last_z_loss = aux_zero.detach()
            self.last_aux_loss = aux_zero.detach()
            return x, aux_zero

        x_valid = x_flat.index_select(0, idx)  # [N, C]
        N = x_valid.shape[0]
        batch_ids = torch.div(idx, T, rounding_mode="floor")

        # ===== gate input: x + time =====
        gate_in = x_valid
        budget = None  # [N,1]

        if t is not None:
            if t.dim() == 2:
                if t.shape[0] != B:
                    raise ValueError(f"t batch mismatch: got {t.shape[0]}, expected {B}")
                t_expanded = t.unsqueeze(1).expand(B, T, t.shape[-1])
            elif t.dim() == 3:
                if t.shape[0] != B:
                    raise ValueError(f"t batch mismatch: got {t.shape[0]}, expected {B}")
                if t.shape[1] == 1:
                    t_expanded = t.expand(B, T, t.shape[-1])
                elif t.shape[1] == T:
                    t_expanded = t
                else:
                    raise ValueError(f"t seq_len mismatch: got {t.shape[1]}, expected 1 or {T}")
            else:
                raise ValueError(f"t must be [B,C] or [B,T,C], got shape {tuple(t.shape)}")

            t_flat = t_expanded.reshape(B * T, -1)
            t_valid = t_flat.index_select(0, idx)  # [N,C]

            gate_in = gate_in + self.t_proj(t_valid)

            if self.use_t_budget:
                budget = torch.sigmoid(self.t_budget_proj(t_valid))  # [N,1]
        else:
            if self.use_t_budget:
                budget = x_valid.new_full((N, 1), 0.5)

        # ===== logits =====
        gate_logits = self.gate(gate_in)  # [N, E], E = R + Nnull
        R = self.num_routed

        # ===== null logit bias (t-aware) =====
        if self.num_null > 0:
            if self.use_t_budget and (budget is not None):
                null_bias = self.null_logit_bias_max + (
                    self.null_logit_bias_min - self.null_logit_bias_max
                ) * budget  # [N,1]
                gate_logits[:, self.num_routed:] = (
                    gate_logits[:, self.num_routed:] + null_bias.to(gate_logits.dtype)
                )
            else:
                gate_logits[:, self.num_routed:] = (
                    gate_logits[:, self.num_routed:] + float(self.null_logit_bias_min)
                )

        # ===== expert dropout =====
        if self.expert_dropout > 0.0 and self.training:
            drop_mask = (torch.rand(self.total_gate_experts, device=device) < self.expert_dropout)
            gate_logits = gate_logits.masked_fill(drop_mask.unsqueeze(0), float("-inf"))
            if drop_mask.all():
                keep_idx = torch.randint(0, self.total_gate_experts, (1,), device=device).item()
                gate_logits[:, keep_idx] = 0.0

        # ===== clean router probs for aux / combine weights =====
        clean_router_probs = F.softmax(gate_logits.float(), dim=-1)  # [N, E], float32
        selection_probs = clean_router_probs
        if self.use_bias_balance and R > 0:
            selection_logits = gate_logits.float().clone()
            selection_logits[:, :R] = selection_logits[:, :R] + self.routing_bias.to(selection_logits)
            selection_probs = F.softmax(selection_logits, dim=-1)

        # ===== router z-loss =====
        z = torch.logsumexp(gate_logits.float(), dim=-1)  # [N]
        z_loss = (z * z).mean()  # float32

        # ===== probs for combine weights (optionally gumbel) =====
        if self.use_gumbel and self.training:
            tau = self._current_tau()
            with torch.no_grad():
                self.gumbel_step += 1
            u = torch.rand_like(gate_logits).clamp_(1e-9, 1 - 1e-9)
            g = -torch.log(-torch.log(u))
            logits_for_prob = (gate_logits + g) / tau
            combine_probs = F.softmax(logits_for_prob, dim=-1, dtype=torch.float32).to(dtype)
        else:
            combine_probs = clean_router_probs.to(dtype)

        # ===== per-token p (t-aware) =====
        if self.use_t_budget and (budget is not None):
            p_tok = (
                self.p_min + (self.p_max - self.p_min) * budget.squeeze(-1)
            ).clamp(0.01, 0.999).to(torch.float32)  # [N]
        else:
            p_tok = torch.full((N,), float(self.p), device=device, dtype=torch.float32)

        selected_mask = self._top_p_select(selection_probs, p_tok)  # [N,E]
        
        # ===== 防止全部坍缩到 null =====
        if R > 0:
            has_routed = selected_mask[:, :R].any(dim=-1)          # [N]
            if not has_routed.all():
                best_routed = selection_probs[:, :R].argmax(dim=-1)   # [N]
                no_routed = ~has_routed
                selected_mask[no_routed, best_routed[no_routed]] = True

        # ===== weights: mask probs then renorm =====
        weights_all = combine_probs * selected_mask.to(dtype)
        denom = weights_all.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights_all = weights_all / denom
        weights_routed = weights_all[:, :R] if R > 0 else None

        # ===== task experts selected once from caption =====
        task_out = torch.zeros_like(x_valid)
        if self.num_task_experts > 0:
            if task_expert_mask is None:
                raise ValueError("task_expert_mask is required when task experts are enabled")
            if task_expert_weights is None:
                raise ValueError("task_expert_weights is required when task experts are enabled")
            if task_expert_mask.shape != (B, self.num_task_experts):
                raise ValueError(
                    "task_expert_mask must be [B, num_task_experts], "
                    f"got {tuple(task_expert_mask.shape)} for B={B}, num_task_experts={self.num_task_experts}"
                )
            if task_expert_weights.shape != (B, self.num_task_experts):
                raise ValueError(
                    "task_expert_weights must be [B, num_task_experts], "
                    f"got {tuple(task_expert_weights.shape)} for B={B}, num_task_experts={self.num_task_experts}"
                )

            task_mask_valid = task_expert_mask.to(device=device, dtype=torch.bool).index_select(0, batch_ids)
            task_weights_valid = task_expert_weights.to(device=device, dtype=dtype).index_select(0, batch_ids)
            for e_id, expert in enumerate(self.task_experts):
                token_ids = torch.nonzero(task_mask_valid[:, e_id], as_tuple=False).squeeze(1)
                if token_ids.numel() == 0:
                    continue
                seg_x = x_valid.index_select(0, token_ids)
                seg_w = task_weights_valid[token_ids, e_id].unsqueeze(-1)
                y = expert(seg_x).to(dtype)
                task_out.index_add_(0, token_ids, y * seg_w)

        # ===== routed dispatch with capacity =====
        routed_out = torch.zeros_like(x_valid)
        lb_loss = clean_router_probs.new_tensor(0.0)

        if R > 0:
            routed_sel = selected_mask[:, :R]  # [N, R]
            self._accumulate_bias_balance_load(routed_sel)
            token_ids, expert_ids = torch.nonzero(routed_sel, as_tuple=True)
            M = token_ids.numel()

            dispatched_sel = torch.zeros_like(routed_sel)

            if M > 0:
                w = weights_routed[token_ids, expert_ids].unsqueeze(-1).to(dtype)  # [M,1]

                # sort by expert
                order = torch.argsort(expert_ids)
                token_ids = token_ids[order]
                expert_ids = expert_ids[order]
                w = w[order]

                x_pairs = x_valid.index_select(0, token_ids)  # [M,C]
                counts = torch.bincount(expert_ids, minlength=R)

                # t-aware capacity factor (scalar)
                if self.use_t_budget and (budget is not None):
                    b_mean = float(budget.mean().detach().cpu())
                    cap_factor = self.capacity_factor_min + (
                        self.capacity_factor_max - self.capacity_factor_min
                    ) * b_mean
                else:
                    cap_factor = self.capacity_factor_max

                # pure top-p: capacity should be based on assignments M, not tokens N
                cap = int(math.ceil(cap_factor * float(M) / float(R)))
                cap = max(cap, 1)

                start = 0
                for e_id, cnt in enumerate(counts.tolist()):
                    if cnt == 0:
                        continue
                    end = start + cnt

                    seg_token_ids = token_ids[start:end]
                    seg_w = w[start:end]       # [cnt,1]
                    seg_x = x_pairs[start:end] # [cnt,C]

                    if self.overflow_drop and (cnt > cap):
                        keep = torch.topk(seg_w.squeeze(-1), k=cap, largest=True).indices
                        seg_token_ids = seg_token_ids[keep]
                        seg_w = seg_w[keep]
                        seg_x = seg_x[keep]

                    dispatched_sel[seg_token_ids, e_id] = True

                    y = self.routed_experts[e_id](seg_x).to(dtype)
                    routed_out.index_add_(0, seg_token_ids, y * seg_w)

                    start = end

            lb_loss = self._load_balance_loss(routed_sel, clean_router_probs)  # 用 drop 前的

        out_valid = routed_out + task_out

        # scatter back
        out_flat = torch.zeros_like(x_flat)
        out_flat.index_copy_(0, idx, out_valid)
        out = out_flat.reshape(B, T, C)
        
        null_frac_loss = clean_router_probs[:, R:].sum(dim=-1).mean()  # null 吃掉的平均概率

        aux = (
            self.load_balance_loss_coef * lb_loss
            + self.router_z_loss_coef * z_loss
            + self.null_loss_coef * null_frac_loss
        ).to(dtype)

        self.last_lb_loss = lb_loss.detach()
        self.last_z_loss = z_loss.detach()
        self.last_aux_loss = aux.detach().float()

        return out, aux

class EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TransformerDiTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.self_attn.is_causal = False

        # ===== MoE 决策=====
        use_moe_here = (
            (layer_idx not in config.mlp_only_layers)
            and (getattr(config, "use_moe_ffn", False))
            and ((layer_idx + 1) % config.decoder_sparse_step == 0)
        )

        if use_moe_here:
            num_routed = int(getattr(config, "moe_num_routed", 0))
            num_task_experts = int(getattr(config, "moe_num_task_experts", 0))
            num_null   = int(getattr(config, "moe_num_null", 0))

            self.mlp = DynamicTopPMoEFFN(
                config=config,
                dim=config.hidden_size,
                num_routed=num_routed,
                num_task_experts=num_task_experts,
                num_null=num_null,
                p=float(getattr(config, "moe_p", 0.7)),

                use_gumbel=bool(getattr(config, "moe_use_gumbel", False)),
                gumbel_tau_start=float(getattr(config, "moe_gumbel_tau_start", 1.0)),
                gumbel_tau_end=float(getattr(config, "moe_gumbel_tau_end", 0.3)),
                gumbel_tau_anneal_steps=int(getattr(config, "moe_gumbel_tau_anneal_steps", 200_000)),
                expert_dropout=float(getattr(config, "moe_expert_dropout", 0.0)),

                capacity_factor_min=float(getattr(config, "moe_capacity_factor_min", 1.0)),
                capacity_factor_max=float(getattr(config, "moe_capacity_factor_max", 2.0)),
                overflow_drop=bool(getattr(config, "moe_overflow_drop", True)),

                use_t_budget=bool(getattr(config, "moe_use_t_budget", True)),
                p_min=float(getattr(config, "moe_p_min", 0.4)),
                p_max=float(getattr(config, "moe_p_max", 0.95)),
                null_logit_bias_min=float(getattr(config, "moe_null_logit_bias_min", -2.0)),
                null_logit_bias_max=float(getattr(config, "moe_null_logit_bias_max", 0.0)),

                load_balance_loss_coef=float(getattr(config, "moe_load_balance_loss_coef", 1e-2)),
                router_z_loss_coef=float(getattr(config, "moe_router_z_loss_coef", 1e-3)),
            )
        else:
            self.mlp = MLP(config, intermediate_size=config.intermediate_size)

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
        attention_mask: Optional[torch.Tensor] = None,  # self-attn mask (2D or 4D additive)
        padding_mask: Optional[torch.Tensor] = None,    # MoE mask 用 [B,T] bool
        task_expert_mask: Optional[torch.Tensor] = None,
        task_expert_weights: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
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

        # Feed-forward / MoE-FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None, :]

        if isinstance(self.mlp, DynamicTopPMoEFFN):
            hidden_states, aux = self.mlp(
                hidden_states,
                padding_mask=padding_mask,
                t=time_step,
                task_expert_mask=task_expert_mask,
                task_expert_weights=task_expert_weights,
            )
        else:
            hidden_states, aux = self.mlp(hidden_states), None

        if self.config.use_mhc:
            hidden_states = self.mlp_mhc(residual, gate_mlp[:, None] * hidden_states)
        else:
            hidden_states = residual + gate_mlp[:, None] * hidden_states

        return hidden_states, aux

@auto_docstring
class TransformerMoePreTrainedModel(PreTrainedModel):
    config: TransformerDiTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["EncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": EncoderLayer,
        "attentions": Attention,
    }


@auto_docstring
class TransformerMoeDiTModel(TransformerMoePreTrainedModel):
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

        self.task_expert_router = None
        if getattr(config, "use_moe_ffn", False) and int(getattr(config, "moe_num_task_experts", 0)) > 0:
            self.task_expert_router = TaskExpertRouter(config)

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
    ) -> MoeModelOutputWithPast:
        
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

        padding_mask_2d = kwargs.pop("padding_mask", None)

        if padding_mask_2d is None:
            if attention_mask.dim() == 2:
                padding_mask_2d = attention_mask.to(torch.bool)
            else:
                raise ValueError(
                    "MoE routing needs a 2D padding mask [B, T]. "
                    "When passing a 4D attention_mask, also pass padding_mask=[B, T]."
                )
        else:
            padding_mask_2d = padding_mask_2d.to(torch.bool)

        caption_mask_2d = encoder_attention_mask
        if encoder_hidden_states is not None and caption_mask_2d is None:
            caption_mask_2d = torch.ones(
                bsz,
                encoder_hidden_states.shape[1],
                device=encoder_hidden_states.device,
                dtype=torch.long,
            )

        task_expert_mask = None
        task_expert_weights = None
        if self.task_expert_router is not None:
            task_expert_mask, task_expert_weights, _ = self.task_expert_router(
                encoder_hidden_states,
                caption_mask_2d,
                batch_size=bsz,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )

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
                encoder_attention_mask = torch.ones(bsz, encoder_hidden_states.shape[1], device=encoder_hidden_states.device, dtype=torch.long)
                
            if encoder_attention_mask.dim() == 2 and self.config._attn_implementation not in ['flash_attention_2', 'flash_attention_3']:
                encoder_attention_mask = make_padding_mask_4d(
                    attention_mask=encoder_attention_mask,
                    dtype=encoder_hidden_states.dtype,
                    q_len=seq_len,
                    k_len=encoder_hidden_states.shape[1],
                )

        aux_list = []
        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, aux = encoder_layer(
                hidden_states,
                time_step,
                attention_mask=mask_mapping[encoder_layer.attention_type],
                padding_mask=padding_mask_2d, 
                task_expert_mask=task_expert_mask,
                task_expert_weights=task_expert_weights,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                **kwargs,
            )
            if aux is not None:
                aux_list.append(aux)

        hidden_states = self.norm(hidden_states, time_step)

        router_aux_loss = (
            torch.stack(aux_list, 0).mean()
            if len(aux_list) > 0
            else hidden_states.new_tensor(0.0)
        )

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            router_logits=router_aux_loss,
        )
