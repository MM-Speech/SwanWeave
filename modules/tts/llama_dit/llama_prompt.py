
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from modules.tts.f5_dit.f5_modules import AdaLayerNormZero, AdaLayerNormZero_Final
from utils.nn.seq_utils import sequence_mask
from modules.tts.llama_dit.llama_ca import precompute_freqs_cis, Attention, CrossAttention, FeedForward
from typing import Optional

@dataclass
class ModelArgs:
    encoder_dim: int = 1024
    encoder_n_layers: int = 24
    crossattn_n_layers: int = 24
    encoder_n_heads: int = 16
    encoder_n_kv_heads: int = None
    mlp_extend: float = None
    max_seq_len: int = 16384
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = 2
    use_causal_attn: bool = False
    use_caption_pool_in_adaln: bool = False
    use_caption_pool_gate_in_adaln: bool = False
    use_qk_norm: bool = False
    use_dynamic_cross_gate: bool = False
    use_gated_attention: bool = False
    use_moe_ffn: bool = False
    moe_p: float = 0.7                # Top-P routing threshold
    moe_num_routed: int = 8           # routed experts count
    moe_num_shared: int = 2           # shared experts count (always-on)
    moe_num_null: int = 1             # null experts count (no compute)
    moe_aux_loss_weight: float = 0.01 # training-time weight (you can anneal in loop)
    moe_use_gumbel: bool = False
    moe_gumbel_tau_start: float = 1.0       # 初始温度（训练早期）
    moe_gumbel_tau_end: float = 0.3         # 最终温度（训练后期）
    moe_gumbel_tau_anneal_steps: int = 200_000  # 退火步数
    moe_expert_dropout: float = 0.0   # 每个 step 随机屏蔽部分专家（0~1）
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

import math
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn

class DynamicTopPMoEFFN(nn.Module):
    """
    Top-P MoE-FFN with:
      - pure Top-P (no max_k)
      - standard auxiliary load-balancing loss
      - router z-loss
      - capacity / overflow drop
      - t-aware compute budget: p(t), null_bias(t), capacity_factor(t)

    forward(x, padding_mask, t) -> (out, aux)
      - x: [B,T,C]
      - padding_mask: [B,T] bool, True=valid
      - t: [B,C] or [B,T,C]
      - aux: scalar = load_balance_loss_coef * lb_loss + router_z_loss_coef * z_loss (+ null_frac penalty)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        num_routed: int,
        num_shared: int,
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
        self.dim = int(dim)
        self.num_routed = int(num_routed)
        self.num_shared = int(num_shared)
        self.num_null = int(num_null)
        self.p = float(p)

        self.use_gumbel = bool(use_gumbel)
        self.gumbel_tau_start = float(gumbel_tau_start)
        self.gumbel_tau_end = float(gumbel_tau_end)
        self.gumbel_tau_anneal_steps = int(gumbel_tau_anneal_steps)
        self.expert_dropout = float(expert_dropout)
        self.register_buffer("gumbel_step", torch.zeros(1), persistent=False)

        self.routed_experts = nn.ModuleList(
            [FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier) for _ in range(self.num_routed)]
        )
        self.shared_experts = nn.ModuleList(
            [FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier) for _ in range(self.num_shared)]
        )

        self.total_gate_experts = self.num_routed + self.num_null
        self.gate = nn.Linear(self.dim, self.total_gate_experts, bias=False)

        self.t_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.t_proj.weight)
        nn.init.zeros_(self.t_proj.bias)

        self.capacity_factor_min = float(capacity_factor_min)
        self.capacity_factor_max = float(capacity_factor_max)
        self.overflow_drop = bool(overflow_drop)

        self.use_t_budget = bool(use_t_budget)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.null_logit_bias_min = float(null_logit_bias_min)
        self.null_logit_bias_max = float(null_logit_bias_max)

        if self.use_t_budget:
            self.t_budget_proj = nn.Linear(self.dim, 1)
            nn.init.zeros_(self.t_budget_proj.weight)
            nn.init.zeros_(self.t_budget_proj.bias)

        self.load_balance_loss_coef = float(load_balance_loss_coef)
        self.router_z_loss_coef = float(router_z_loss_coef)

    def _current_tau(self) -> float:
        if (not self.use_gumbel) or (self.gumbel_tau_anneal_steps <= 0):
            return float(self.gumbel_tau_start)
        step = float(self.gumbel_step.item())
        ratio = min(1.0, step / float(self.gumbel_tau_anneal_steps))
        tau = self.gumbel_tau_start + (self.gumbel_tau_end - self.gumbel_tau_start) * ratio
        return max(float(tau), 1e-4)

    @staticmethod
    def _top_p_select(probs: torch.Tensor, p_tok: torch.Tensor) -> torch.Tensor:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        prev_cum = cumsum - sorted_probs

        p_tok = p_tok.view(-1, 1)
        keep_sorted = prev_cum < p_tok
        keep_sorted[..., 0] = True

        selected_mask = torch.zeros_like(keep_sorted, dtype=torch.bool)
        selected_mask.scatter_(1, sorted_idx, keep_sorted)
        return selected_mask

    def _load_balance_loss(
        self,
        dispatched_sel: torch.Tensor,  # [N, R] bool
        router_probs: torch.Tensor,  # [N, E] float32
    ) -> torch.Tensor:
        if self.num_routed <= 0:
            return router_probs.new_tensor(0.0)

        load = dispatched_sel.float().sum(dim=0)  # [R]
        total_load = load.sum()
        if total_load <= 0:
            return router_probs.new_tensor(0.0)
        load = load / total_load.clamp_min(1.0)

        importance = router_probs[:, : self.num_routed].sum(dim=0)  # [R]
        total_importance = importance.sum()
        if total_importance <= 0:
            return router_probs.new_tensor(0.0)
        importance = importance / total_importance.clamp_min(1e-9)

        return self.num_routed * torch.sum(load * importance)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype

        if padding_mask is None:
            padding_mask = torch.ones((B, T), device=device, dtype=torch.bool)
        else:
            padding_mask = padding_mask.to(torch.bool)

        x_flat = x.reshape(B * T, C)
        mask_flat = padding_mask.reshape(B * T)

        idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)  # [N]
        if idx.numel() == 0:
            aux_zero = x.new_tensor(0.0)
            return x, aux_zero

        x_valid = x_flat.index_select(0, idx)  # [N, C]
        N = x_valid.shape[0]

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

        gate_logits = self.gate(gate_in)  # [N, E], E=R+Nnull

        if self.num_null > 0:
            if self.use_t_budget and (budget is not None):
                null_bias = self.null_logit_bias_max + (
                    self.null_logit_bias_min - self.null_logit_bias_max
                ) * budget
                gate_logits[:, self.num_routed :] = gate_logits[:, self.num_routed :] + null_bias.to(gate_logits.dtype)
            else:
                gate_logits[:, self.num_routed :] = gate_logits[:, self.num_routed :] + float(self.null_logit_bias_min)

        if self.expert_dropout > 0.0 and self.training:
            drop_mask = (torch.rand(self.total_gate_experts, device=device) < self.expert_dropout)
            gate_logits = gate_logits.masked_fill(drop_mask.unsqueeze(0), float("-inf"))
            if drop_mask.all():
                keep_idx = torch.randint(0, self.total_gate_experts, (1,), device=device).item()
                gate_logits[:, keep_idx] = 0.0

        router_probs = F.softmax(gate_logits.float(), dim=-1)  # [N, E], float32

        z = torch.logsumexp(gate_logits.float(), dim=-1)  # [N]
        z_loss = (z * z).mean()  # float32

        if self.use_gumbel and self.training:
            tau = self._current_tau()
            with torch.no_grad():
                self.gumbel_step += 1
            u = torch.rand_like(gate_logits).clamp_(1e-9, 1 - 1e-9)
            g = -torch.log(-torch.log(u))
            logits_for_prob = (gate_logits + g) / tau
            combine_probs = F.softmax(logits_for_prob, dim=-1, dtype=torch.float32).to(dtype)
        else:
            combine_probs = router_probs.to(dtype)

        if self.use_t_budget and (budget is not None):
            p_tok = (self.p_min + (self.p_max - self.p_min) * budget.squeeze(-1)).clamp(0.01, 0.999).to(
                torch.float32
            )
        else:
            p_tok = torch.full((N,), float(self.p), device=device, dtype=torch.float32)

        selected_mask = self._top_p_select(router_probs, p_tok)  # [N, E]

        R = self.num_routed
        if R > 0:
            has_routed = selected_mask[:, :R].any(dim=-1)
            if not has_routed.all():
                best_routed = router_probs[:, :R].argmax(dim=-1)
                no_routed = ~has_routed
                selected_mask[no_routed, best_routed[no_routed]] = True

        weights_all = combine_probs * selected_mask.to(dtype)
        denom = weights_all.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights_all = weights_all / denom
        weights_routed = weights_all[:, :R] if R > 0 else None

        shared_out = torch.zeros_like(x_valid)
        if self.num_shared > 0:
            for se in self.shared_experts:
                shared_out = shared_out + se(x_valid).to(shared_out.dtype)
            shared_out = shared_out / float(self.num_shared)

        routed_out = torch.zeros_like(x_valid)
        lb_loss = router_probs.new_tensor(0.0)

        if R > 0:
            routed_sel = selected_mask[:, :R]  # [N, R]
            token_ids, expert_ids = torch.nonzero(routed_sel, as_tuple=True)
            M = token_ids.numel()

            if M > 0:
                w = weights_routed[token_ids, expert_ids].unsqueeze(-1).to(dtype)  # [M,1]

                order = torch.argsort(expert_ids)
                token_ids = token_ids[order]
                expert_ids = expert_ids[order]
                w = w[order]

                x_pairs = x_valid.index_select(0, token_ids)  # [M,C]
                counts = torch.bincount(expert_ids, minlength=R)

                if self.use_t_budget and (budget is not None):
                    b_mean = float(budget.mean().detach().cpu())
                    cap_factor = self.capacity_factor_min + (self.capacity_factor_max - self.capacity_factor_min) * b_mean
                else:
                    cap_factor = self.capacity_factor_max

                cap = int(math.ceil(cap_factor * float(M) / float(R)))
                cap = max(cap, 1)

                start = 0
                for e_id, cnt in enumerate(counts.tolist()):
                    if cnt == 0:
                        continue
                    end = start + cnt

                    seg_token_ids = token_ids[start:end]
                    seg_w = w[start:end]
                    seg_x = x_pairs[start:end]

                    if self.overflow_drop and (cnt > cap):
                        keep = torch.topk(seg_w.squeeze(-1), k=cap, largest=True).indices
                        seg_token_ids = seg_token_ids[keep]
                        seg_w = seg_w[keep]
                        seg_x = seg_x[keep]

                    y = self.routed_experts[e_id](seg_x).to(dtype)
                    routed_out.index_add_(0, seg_token_ids, y * seg_w)

                    start = end

            lb_loss = self._load_balance_loss(routed_sel, router_probs)

        out_valid = routed_out + shared_out

        out_flat = torch.zeros_like(x_flat)
        out_flat.index_copy_(0, idx, out_valid)
        out = out_flat.reshape(B, T, C)

        null_frac_loss = router_probs[:, R:].sum(dim=-1).mean()

        aux = (
            self.load_balance_loss_coef * lb_loss
            + self.router_z_loss_coef * z_loss
            + self.load_balance_loss_coef * null_frac_loss
        ).to(dtype)

        return out, aux

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = int(layer_idx)
        self.encoder_n_heads = args.encoder_n_heads
        self.encoder_dim = args.encoder_dim
        self.head_dim = args.encoder_dim // args.encoder_n_heads

        self.attention = Attention(args)
        self.cross_attention = CrossAttention(
            self.encoder_dim,
            self.encoder_n_heads,
        )

        self.use_moe_ffn = bool(getattr(args, "use_moe_ffn", False))
        decoder_sparse_step = int(getattr(args, "decoder_sparse_step", 1) or 1)
        decoder_sparse_step = max(decoder_sparse_step, 1)

        mlp_only_layers = getattr(args, "mlp_only_layers", ())
        if isinstance(mlp_only_layers, list):
            mlp_only_layers = tuple(mlp_only_layers)

        self.use_moe_ffn_here = (
            self.use_moe_ffn
            and (self.layer_idx not in set(mlp_only_layers))
            and ((self.layer_idx + 1) % decoder_sparse_step == 0)
        )

        if self.use_moe_ffn_here:
            self.feed_forward = DynamicTopPMoEFFN(
                dim=args.encoder_dim,
                hidden_dim=args.encoder_dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                num_routed=args.moe_num_routed,
                num_shared=args.moe_num_shared,
                num_null=args.moe_num_null,
                p=args.moe_p,
                use_gumbel=args.moe_use_gumbel,
                gumbel_tau_start=args.moe_gumbel_tau_start,
                gumbel_tau_end=args.moe_gumbel_tau_end,
                gumbel_tau_anneal_steps=args.moe_gumbel_tau_anneal_steps,
                expert_dropout=args.moe_expert_dropout,
                capacity_factor_min=args.moe_capacity_factor_min,
                capacity_factor_max=args.moe_capacity_factor_max,
                overflow_drop=args.moe_overflow_drop,
                use_t_budget=args.moe_use_t_budget,
                p_min=args.moe_p_min,
                p_max=args.moe_p_max,
                null_logit_bias_min=args.moe_null_logit_bias_min,
                null_logit_bias_max=args.moe_null_logit_bias_max,
                load_balance_loss_coef=getattr(args, "moe_load_balance_loss_coef", 1e-2),
                router_z_loss_coef=getattr(args, "moe_router_z_loss_coef", 1e-3),
            )
        else:
            self.feed_forward = FeedForward(
                dim=args.encoder_dim,
                hidden_dim=args.encoder_dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
            )

        self.attention_norm = AdaLayerNormZero(args.encoder_dim)
        self.cross_attention_norm = nn.LayerNorm(args.encoder_dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(args.encoder_dim, elementwise_affine=False, eps=1e-6)

        if args.use_dynamic_cross_gate:
            self.cross_gating_proj = nn.Linear(self.encoder_dim, self.encoder_dim)
            nn.init.zeros_(self.cross_gating_proj.weight)
            nn.init.constant_(self.cross_gating_proj.bias, -2.0)
        else:
            self.cross_gate = nn.Parameter(torch.zeros(args.encoder_dim))

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            context: Optional[torch.Tensor] = None,
            context_lens: Optional[torch.Tensor] = None,
            use_cache: bool = False,
    ):
        """
        x:    [B, T, C]
        mask: [B, T] bool
        t:    [B, C] time embedding
        """
        dtype = x.dtype

        # === Self-Attn ===
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attention_norm(x, emb=t)
        attn_output = self.attention(norm, start_pos, freqs_cis, mask=mask, use_cache=use_cache)
        h = x + gate_msa.unsqueeze(1) * attn_output

        # === Cross-Attn ===
        if context is not None:
            norm = self.cross_attention_norm(h.float()).to(dtype)
            cross_attn_output = self.cross_attention(
                norm,
                context,
                context_lens.to(torch.long),
                query_mask=mask
            )

            if self.args.use_dynamic_cross_gate:
                cross_gate = torch.sigmoid(self.cross_gating_proj(norm))
                h = h + cross_gate.to(dtype) * cross_attn_output
            else:
                h = h + self.cross_gate.to(dtype) * cross_attn_output

        # === FFN / MoE-FFN ===
        norm = self.ffn_norm(h.float()).to(dtype) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        moe_aux = None
        if self.use_moe_ffn_here:
            ff_output, moe_aux = self.feed_forward(norm, padding_mask=mask, t=t)
        else:
            ff_output = self.feed_forward(norm)

        out = h + gate_mlp.unsqueeze(1) * ff_output

        if self.use_moe_ffn:
            return out, moe_aux
        return out

class LLaMa(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.encoder_n_layers = params.encoder_n_layers

        self.layers = nn.ModuleList()
        for idx in range(params.encoder_n_layers):
            self.layers.append(TransformerBlock(params, layer_idx=idx))

        self.norm = AdaLayerNormZero_Final(params.encoder_dim)
        self.out_proj = nn.Linear(params.encoder_dim, params.encoder_dim)

        self._use_cap_mod = bool(getattr(self.params, "use_caption_pool_in_adaln", False))
        if self._use_cap_mod:
            self.cap_embedder = nn.Linear(params.encoder_dim, params.encoder_dim)
            self.cap_gate = nn.Linear(params.encoder_dim, params.encoder_dim)
            nn.init.zeros_(self.cap_embedder.weight)
            nn.init.zeros_(self.cap_embedder.bias)
            nn.init.zeros_(self.cap_gate.weight)
            nn.init.zeros_(self.cap_gate.bias)

        freqs_cis = precompute_freqs_cis(
            self.params.encoder_dim // self.params.encoder_n_heads,
            self.params.max_seq_len
        )
        self.register_buffer("freqs_cis", torch.view_as_real(freqs_cis), persistent=False)

    def forward(self, x, t, attn_mask,
                context=None, context_lens=None,
                start_pos=0, use_cache=False, do_checkpoint=False):
        """
        x: [B,T,C]
        t: [B,C] time embedding
        attn_mask: [B,T] bool
        context/context_lens: caption / text encoder 输出（可选）
        """
        freqs_cis = torch.view_as_complex(self.freqs_cis.float())[start_pos: start_pos + x.size(1)]

        # caption pooling（如有） -> 可选注入到 t
        if (context is not None) and (context_lens is not None):
            B, Lc, C = context.size()
            device = context.device

            lengths = context_lens.to(device=device).long().view(-1)         # [B]
            idxs = torch.arange(Lc, device=device).unsqueeze(0).expand(B, Lc)
            cap_mask = (idxs < lengths.unsqueeze(1))                         # [B,Lc]

            cap_mask_f = cap_mask.to(context.dtype).unsqueeze(-1)            # [B,Lc,1]
            cap_sum = (context * cap_mask_f).sum(dim=1)                      # [B,C]
            denom = cap_mask_f.sum(dim=1).clamp(min=1.0)                     # [B,1]
            cap_pool = cap_sum / denom                                       # [B,C]

            if self._use_cap_mod:
                cap_emb = self.cap_embedder(cap_pool)
                cap_gate = torch.tanh(self.cap_gate(t + cap_emb.to(t.dtype)))
                t = t + cap_emb.to(t.dtype) * cap_gate
        else:
            cap_pool = None  # 目前 MoE 没显式用到 caption pool，hidden 里已经有信息

        use_moe = bool(getattr(self.params, "use_moe_ffn", False))
        total_moe_aux = None
        moe_aux_count = 0

        for i, layer in enumerate(self.layers):
            # 有 MoE 时，为简单起见仍然关闭 checkpoint（否则需要自定义 ckpt 返回 aux）
            if torch.is_grad_enabled() and do_checkpoint and (not use_moe):
                def create_custom_forward(module):
                    def custom_forward(x_, t_, start_pos_, freqs_cis_, attn_mask_,
                                       context_, context_lens_, use_cache_):
                        return module(x_, t_, start_pos_, freqs_cis_, attn_mask_,
                                      context=context_, context_lens=context_lens_,
                                      use_cache=use_cache_)
                    return custom_forward

                ckpt_kwargs = {"use_reentrant": False}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x, t, start_pos, freqs_cis, attn_mask, context, context_lens, use_cache,
                    **ckpt_kwargs,
                )
            else:
                out = layer(
                    x, t, start_pos, freqs_cis, attn_mask,
                    context=context, context_lens=context_lens,
                    use_cache=use_cache,
                )
                if use_moe:
                    x, moe_aux = out
                    if moe_aux is not None:
                        moe_aux_count += 1
                        total_moe_aux = moe_aux if total_moe_aux is None else (total_moe_aux + moe_aux)
                else:
                    x = out

        x = self.norm(x, t)
        x = self.out_proj(x)

        if use_moe:
            if total_moe_aux is not None:
                denom = float(max(int(moe_aux_count), 1))
                total_moe_aux = total_moe_aux / denom
            else:
                total_moe_aux = x.new_tensor(0.0)
            return x, total_moe_aux

        return x