import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union


def _safe_temperature(temp: float) -> float:
    return max(temp, 1e-6)

def _top_p_sample(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """
    logits: [N, V]
    返回：采样得到的 token 索引 [N]
    """
    temperature = _safe_temperature(temperature)
    probs = F.softmax(logits / temperature, dim=-1)  # [N,V]
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # 保留累计概率 <= p 的集合，并确保至少保留一个元素
    keep = (cumsum <= p) | (torch.arange(probs.size(-1), device=probs.device) == 0)
    # 重新归一化
    masked = sorted_probs * keep
    denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    masked = masked / denom
    sel = torch.multinomial(masked, num_samples=1)             # [N,1]
    return sorted_idx.gather(-1, sel).squeeze(-1)              # [N]

def _inv_alpha(alpha: torch.Tensor, schedule: str) -> torch.Tensor:
    """
    α -> t 的反函数，逐元素。
    """
    if schedule == "linear":
        return 1.0 - alpha
    elif schedule == "cosine":
        z = (2 * alpha - 1).clamp(-1 + 1e-7, 1 - 1e-7)
        return torch.arccos(z) / torch.pi
    elif schedule == "quadratic":
        return 1.0 - torch.sqrt(alpha.clamp_min(0.0))
    else:
        raise ValueError(schedule)

def _build_uniform_alpha_grid(steps: int, schedule: str, alpha_min: float, device: torch.device):
    """
    几何衰减：alpha_{k+1} = r * alpha_k, r = (alpha_min/alpha_0)^(1/steps).
    返回 t_grid, a_grid: [steps+1]
    """
    a0 = 1.0
    r = (alpha_min / a0) ** (1.0 / steps)
    a = [a0 * (r ** k) for k in range(steps + 1)]
    a = torch.tensor(a, device=device)
    t = _inv_alpha(a, schedule)
    return t, a

def _build_uniform_alpha_grid_refine(steps: int, schedule: str, alpha_min: float, device: torch.device):
    """
    为 remask_refine 解码构造 alpha/t 网格。

    关键点：remask_refine 的最后一轮 (k=steps-1) 不再 remask，只输出该轮预测。
    因此我们希望最后一轮使用的 alpha 就是 alpha_min（而不是 alpha_min^((steps-1)/steps)）。

    返回 t_grid, a_grid: [steps+1]，其中 a_grid[steps-1] == alpha_min 且 a_grid[steps] == alpha_min。
    """
    if steps <= 1:
        a = torch.ones(2, device=device)
        t = _inv_alpha(a, schedule)
        return t, a

    denom = steps - 1
    k = torch.arange(steps + 1, device=device, dtype=torch.float32)
    exponents = k / float(denom)
    a = torch.tensor(alpha_min, device=device, dtype=torch.float32) ** exponents
    a = a.clamp(min=float(alpha_min), max=1.0)
    a[-1] = float(alpha_min)

    t = _inv_alpha(a, schedule)
    return t, a


class MaskFlowMatching(nn.Module):
    """
    Mask Flow Matching for discrete tokens with padding and optional conditioning.
    - Backbone transformer: forward(tokens: [B,T], t: [B], cond: Optional[Any], attn_mask: Optional[Bool[B,T]]) -> logits [B,T,V]
    - Focus: mask-flow computations (schedule, masking, CFM weights), padding handling, conditional support.
    """

    def __init__(
        self,
        vocab_size: int,
        mask_id: int,
        backbone: nn.Module = None,
        schedule: str = "cosine",            # 'linear' | 'cosine' | 'quadratic'
        num_steps: int = 12,
        use_cfm_weight: bool = True,         # w(t) = -alpha'(t)/(alpha(t)+eps)
        pad_id: Optional[int] = None,        # enable padding support if not None
        t_eps: float = 1e-3,                 # sample t in [0, 1 - t_eps]
        enforce_one_mask: bool = True,       # ensure at least one masked non-pad per sample at train-time
        eps: float = 1e-8,
        w_clip: Optional[float] = None,      # optional clip for w(t)
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.pad_id = pad_id

        self.backbone = backbone
        self.schedule = schedule
        self.num_steps = num_steps
        self.use_cfm_weight = use_cfm_weight

        self.t_eps = float(t_eps)
        self.enforce_one_mask = enforce_one_mask
        self.eps = eps
        self.w_clip = w_clip

    # ========= Schedule =========
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        # α(0)=1, α(1)=0
        if self.schedule == "linear":
            return 1.0 - t
        elif self.schedule == "cosine":
            return 0.5 * (1.0 + torch.cos(torch.pi * t))
        elif self.schedule == "quadratic":
            return (1.0 - t) ** 2
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def dalpha(self, t: torch.Tensor) -> torch.Tensor:
        if self.schedule == "linear":
            return -torch.ones_like(t)
        elif self.schedule == "cosine":
            return -0.5 * torch.pi * torch.sin(torch.pi * t)
        elif self.schedule == "quadratic":
            return -2.0 * (1.0 - t)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    # ========= Utilities =========
    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # sample t in [0, 1 - t_eps]
        high = 1.0 - max(self.t_eps, 0.0)
        return torch.rand(batch_size, device=device) * high

    def _build_valid_mask(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        valid positions = not padding. If pad_id is None and padding_mask is None, all positions valid.
        Returns bool[B,T]
        """
        if padding_mask is not None:
            # assume padding_mask: True for pad positions
            valid = ~padding_mask.bool()
        elif self.pad_id is not None:
            valid = (x != self.pad_id)
        else:
            valid = torch.ones_like(x, dtype=torch.bool)
        return valid

    def _build_xt(
        self,
        x0: torch.Tensor,
        alpha_t: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        maskable_mask: Optional[torch.Tensor] = None,  # bool[B,T], True means this position CAN be masked
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build x_t by masking only positions that are both valid (non-pad) and maskable.

        Args:
            x0: [B,T]
            alpha_t: [B]
            padding_mask: bool[B,T], True at pad
            maskable_mask: optional bool[B,T]; positions with False will never be masked.

        Returns:
            x_t: [B, T] with mask_id inserted
            mask: [B, T] boolean WHERE newly masked
            valid: [B, T] boolean non-pad positions
        """
        B, T = x0.shape
        device = x0.device
        valid = self._build_valid_mask(x0, padding_mask)  # [B,T]

        if maskable_mask is None:
            maskable = valid
        else:
            maskable = maskable_mask.bool() & valid

        p = alpha_t.view(B, 1).expand(B, T)
        bern = torch.bernoulli(p.clamp(0.0, 1.0)).bool()
        mask = bern & maskable

        if self.enforce_one_mask:
            maskable_counts = maskable.sum(dim=1)  # [B]
            mask_counts = mask.sum(dim=1)          # [B]
            need = (maskable_counts > 0) & (mask_counts == 0)
            if need.any():
                for b in torch.nonzero(need, as_tuple=False).squeeze(-1).tolist():
                    idx_maskable = torch.nonzero(maskable[b], as_tuple=False).squeeze(-1)
                    j = torch.randint(low=0, high=idx_maskable.numel(), size=(1,), device=device).item()
                    mask[b, idx_maskable[j]] = True

        x_t = torch.where(mask, torch.full_like(x0, self.mask_id), x0)
        return x_t, mask, valid

    # def _loss_weight(self, t: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
    #     w = -self.dalpha(t)  # >= 0
    #     if self.use_cfm_weight:
    #         w = w / (alpha_t + self.eps)
    #     if self.w_clip is not None:
    #         w = torch.clamp(w, max=self.w_clip)
    #     return w  # [B]

    def _loss_weight(self, t: torch.Tensor, alpha_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.use_cfm_weight:
            return torch.ones_like(t)

        if alpha_t is None:
            alpha_t = self.alpha(t)

        w = (-self.dalpha(t)) / (alpha_t + self.eps)
        if self.w_clip is not None:
            w = w.clamp(min=-float(self.w_clip), max=float(self.w_clip))
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        return w

    def _attn_mask_from_valid(self, valid: torch.Tensor) -> torch.Tensor:
        """
        Produce an attention mask for transformer if needed.
        We pass 'valid' (True for keep) directly; the backbone must know how to consume it.
        """
        return valid  # [B,T] bool

    def _match_token_ids(self, x: torch.Tensor, token_ids: Optional[Any]) -> torch.Tensor:
        """Return bool[B,T] where x equals any of token_ids."""
        if token_ids is None:
            return torch.zeros_like(x, dtype=torch.bool)

        if isinstance(token_ids, int):
            ids = torch.tensor([token_ids], device=x.device, dtype=x.dtype)
        elif torch.is_tensor(token_ids):
            ids = token_ids.to(device=x.device, dtype=x.dtype).flatten()
        else:
            ids = torch.tensor(list(token_ids), device=x.device, dtype=x.dtype).flatten()

        if ids.numel() == 0:
            return torch.zeros_like(x, dtype=torch.bool)

        return (x.unsqueeze(-1) == ids.view(1, 1, -1)).any(dim=-1)

    def compute_loss(
        self,
        x0: torch.Tensor,                     # [B, T]
        t: Optional[torch.Tensor] = None,     # [B]
        padding_mask: Optional[torch.Tensor] = None,  # [B,T] True at pad
        cond: Optional[object] = None,        # forwarded to transformer
        loss_mask: Optional[torch.Tensor] = None,     # bool[B,T]; True contributes to loss
        maskable_mask: Optional[torch.Tensor] = None, # bool[B,T]; True can be masked
        never_mask_ids: Optional[Any] = None,         # ids never masked (e.g., BOS/EOS)
        ignore_loss_ids: Optional[Any] = None,        # ids ignored in loss
        *,
        self_conditioning: bool = False,
        self_conditioning_prob: float = 0.5,
        self_conditioning_mode: str = "replace_masked",  # replace_masked | replace_maskable
        return_details: bool = False,
        return_logits: bool = False,
        return_x_t: bool = False,
        return_per_sample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        device = x0.device
        B, T = x0.shape

        if t is None:
            t = self._sample_t(B, device)
        else:
            t = t.to(device)

        alpha_t = self.alpha(t)  # [B]
        valid = self._build_valid_mask(x0, padding_mask)  # [B,T]

        # ----- lossable: where CE is allowed to be computed -----
        lossable = valid
        if loss_mask is not None:
            lossable &= loss_mask.to(device=device).bool()
        if ignore_loss_ids is not None:
            lossable &= ~self._match_token_ids(x0, ignore_loss_ids)

        lossable_counts = lossable.sum(dim=1)  # [B]
        if (lossable_counts > 0).sum().item() == 0:
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return {
                "loss": zero,
                "t": t.detach(),
                "alpha_t": alpha_t.detach(),
                "masked_ratio_nonpad": torch.zeros(B, device=device),
                "masked_ratio_lossable": torch.zeros(B, device=device),
                "lossable_tokens": lossable_counts.detach(),
                "maskable_tokens": torch.zeros(B, device=device, dtype=lossable_counts.dtype),
            }

        # ----- maskable: where masks are allowed to be sampled -----
        if maskable_mask is None:
            maskable = lossable.clone()
        else:
            maskable = valid & maskable_mask.to(device=device).bool()

        if never_mask_ids is not None:
            maskable &= ~self._match_token_ids(x0, never_mask_ids)

        # if user accidentally makes (maskable & lossable) empty, auto-fix to keep training alive
        bad = (lossable_counts > 0) & (maskable.sum(dim=1) == 0)
        if bad.any():
            maskable = maskable | (bad.view(B, 1) & lossable)

        # sample x_t (mask only on maskable)
        x_t, mask, valid = self._build_xt(x0, alpha_t, padding_mask, maskable_mask=maskable)

        # ensure at least one masked position contributes to loss (when lossable exists)
        if self.enforce_one_mask:
            pos_loss = mask & lossable
            pos_loss_count = pos_loss.sum(dim=1)
            need = (lossable_counts > 0) & (pos_loss_count == 0)
            if need.any():
                for b in torch.nonzero(need, as_tuple=False).squeeze(-1).tolist():
                    cand = (maskable[b] & lossable[b])
                    idx = torch.nonzero(cand, as_tuple=False).squeeze(-1)
                    if idx.numel() == 0:
                        continue
                    j = idx[torch.randint(low=0, high=idx.numel(), size=(1,), device=device)].item()
                    mask[b, j] = True
                    x_t[b, j] = self.mask_id

        attn_mask = self._attn_mask_from_valid(valid)

        x_in = x_t
        x_sc = None
        self_cond_mask = None

        use_sc = bool(self_conditioning)
        if use_sc:
            p_sc = float(self_conditioning_prob)
            if p_sc <= 0.0:
                use_sc = False
            elif p_sc < 1.0:
                use_sc = bool((torch.rand((), device=device) < p_sc).item())

        if use_sc:
            with torch.no_grad():
                logits_sc = self.backbone(x_t, t, cond=cond, attn_mask=attn_mask)  # [B,T,V]
                x_sc = torch.argmax(logits_sc, dim=-1).to(dtype=x_t.dtype)  # [B,T]

            if self_conditioning_mode == "replace_masked":
                # 只把“本轮被 mask 的位置”用 x_sc 回填（更接近经典 diffusion self-conditioning：pred x0 作为条件，但不动未 mask 的位置）。
                self_cond_mask = mask & valid
            elif self_conditioning_mode == "replace_maskable":
                # 把“所有 maskable 位置（例如 timestamp 位）”都用 x_sc 回填（更像 refine：即便没 mask，也让模型在第二次 forward 时看到“上一轮自己的预测”，从而学会在其基础上纠错/收敛）。
                self_cond_mask = maskable & valid
            else:
                raise ValueError(f"Unknown self_conditioning_mode: {self_conditioning_mode}")

            x_in = torch.where(self_cond_mask, x_sc, x_t)

        logits = self.backbone(x_in, t, cond=cond, attn_mask=attn_mask)  # [B,T,V]

        # CE on masked & lossable
        pos = mask & lossable
        if pos.any():
            idx = pos.view(-1).nonzero(as_tuple=False).squeeze(-1)  # [M]
            logits_masked = logits.view(B * T, self.vocab_size).index_select(0, idx)  # [M,V]
            targets_masked = x0.view(-1).index_select(0, idx)  # [M]
            ce_masked = F.cross_entropy(logits_masked, targets_masked, reduction="none")  # [M]

            ce_sum = torch.zeros(B, device=logits.device)
            sample_ids = torch.arange(B, device=logits.device).unsqueeze(1).expand(B, T)[pos].long()
            ce_sum.scatter_add_(0, sample_ids, ce_masked)
        else:
            ce_sum = torch.zeros(B, device=logits.device)

        w = self._loss_weight(t, alpha_t)  # [B]

        nonpad_counts = valid.sum(dim=1).clamp(min=1)
        lossable_counts_clamped = lossable.sum(dim=1).clamp(min=1)

        if self.use_cfm_weight:
            loss_per_sample = w * ce_sum / lossable_counts_clamped.float()
        else:
            masked_counts = pos.sum(dim=1).clamp(min=1)
            loss_per_sample = w * ce_sum / masked_counts.float()

        loss = loss_per_sample.mean()

        masked_valid = mask & valid
        
        out: Dict[str, torch.Tensor] = {
            "loss": loss,
            "t": t.detach(),
            "alpha_t": alpha_t.detach(),
            "masked_ratio_nonpad": (masked_valid.sum(dim=1).float() / nonpad_counts.float()).detach(),
            "masked_ratio_lossable": (pos.sum(dim=1).float() / lossable_counts_clamped.float()).detach(),
            "lossable_tokens": lossable_counts.detach(),
            "maskable_tokens": maskable.sum(dim=1).detach(),
        }

        if return_details:
            out["valid"] = valid
            out["lossable"] = lossable
            out["maskable"] = maskable
            out["mask"] = mask
            out["pos"] = pos
            out["attn_mask"] = attn_mask
            out["x_in"] = x_in

            out["self_cond_used"] = torch.full((B,), bool(use_sc), device=device, dtype=torch.bool)
            out["self_cond_prob"] = torch.full((B,), float(self_conditioning_prob), device=device, dtype=torch.float32)

            if x_sc is not None:
                out["x_sc"] = x_sc
            if self_cond_mask is not None:
                out["self_cond_mask"] = self_cond_mask

        if return_x_t:
            out["x_t"] = x_t

        if return_logits:
            out["logits"] = logits

        if return_per_sample:
            out["ce_sum"] = ce_sum.detach()
            out["w"] = w.detach()
            out["loss_per_sample"] = loss_per_sample.detach()

        return out

    def backbone_inference(self, x, t_k, cond, attn_mask, guidance_kwargs=None):
        # maybe override for cfg
        return self.backbone(x, t_k, cond=cond, attn_mask=attn_mask)
    
    @torch.no_grad()
    def infer(
        self,
        x_init: torch.Tensor,
        steps: int = None,
        temperature: float = 1.0,
        topk_per_step: int = None,
        token_topk: int = None,
        token_topp: float = None,
        cond=None,
        padding_mask: torch.Tensor = None,
        return_all_steps: bool = False,
        schedule_mode: str = "uniform_alpha",
        alpha_min: float = 1e-3,
        confidence_greedy: float = 0.9,
        use_margin: bool = True,
        guidance_kwargs: dict = None,
        remask_last: bool = True,
        remask_frac: float = 0.05,
        remask_thresh: float = 0.5,
        inference_mode: str = "remask_refine",         # "unmask" | "remask_refine"
        preserve_input_tokens: bool = True,     # remask_refine 时：是否固定 x_init 里非 mask 的 token
        ensure_no_mask: bool = True,            # 修复：保证 valid 位置最终不残留 mask_id
        *,
        return_dict: bool = False,
        return_scores: bool = False,
        return_margin: bool = False,
        return_topk_logits: Optional[int] = None,
        return_meta: bool = False,
    ):
        x = x_init.clone()
        B, T = x.shape
        steps = steps or self.num_steps

        want_dict = bool(return_dict or return_scores or (return_topk_logits is not None) or return_meta)

        if inference_mode in ("unmask", "iterative_unmask"):
            decoder = MaskCFMIterativeUnmaskDecoder()
            return decoder.decode(
                model=self,
                x_init=x_init,
                steps=steps,
                temperature=temperature,
                topk_per_step=topk_per_step,
                token_topk=token_topk,
                token_topp=token_topp,
                cond=cond,
                padding_mask=padding_mask,
                return_all_steps=return_all_steps,
                schedule_mode=schedule_mode,
                alpha_min=alpha_min,
                confidence_greedy=confidence_greedy,
                use_margin=use_margin,
                guidance_kwargs=guidance_kwargs,
                remask_last=remask_last,
                remask_frac=remask_frac,
                remask_thresh=remask_thresh,
                ensure_no_mask=ensure_no_mask,
                preserve_input_tokens=preserve_input_tokens,
                return_dict=want_dict,
                return_scores=return_scores,
                return_margin=return_margin,
                return_topk_logits=return_topk_logits,
                return_meta=return_meta,
            )

        if inference_mode in ("remask_refine", "remask", "refine"):
            decoder = MaskCFMRemaskRefineDecoder()
            return decoder.decode(
                model=self,
                x_init=x_init,
                steps=steps,
                temperature=temperature,
                token_topk=token_topk,
                token_topp=token_topp,
                cond=cond,
                padding_mask=padding_mask,
                return_all_steps=return_all_steps,
                schedule_mode=schedule_mode,
                alpha_min=alpha_min,
                confidence_greedy=confidence_greedy,
                use_margin=use_margin,
                guidance_kwargs=guidance_kwargs,
                preserve_input_tokens=preserve_input_tokens,
                ensure_no_mask=ensure_no_mask,
                return_dict=want_dict,
                return_scores=return_scores,
                return_margin=return_margin,
                return_topk_logits=return_topk_logits,
                return_meta=return_meta,
            )

        raise ValueError(f"Unknown inference_mode: {inference_mode}")


class MaskCFMTimeGrid:
    @staticmethod
    def build(
        model,
        steps: int,
        schedule_mode: str,
        alpha_min: float,
        device: torch.device,
    ):
        if schedule_mode == "uniform_alpha":
            return _build_uniform_alpha_grid(
                steps, schedule=model.schedule, alpha_min=alpha_min, device=device
            )
        if schedule_mode == "linear_t":
            t_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
            a_grid = model.alpha(t_grid)
            return t_grid, a_grid
        raise ValueError(f"Unknown schedule_mode: {schedule_mode}")


class MaskCFMTokenSampler:
    @staticmethod
    def sample_selected_positions(
        *,
        x: torch.Tensor,                 # [B,T]
        fill_mask: torch.Tensor,         # [B,T] bool
        logits_f: torch.Tensor,          # [B,T,V] (mask_id/pad_id already filtered)
        probs_all: torch.Tensor,         # [B,T,V] = softmax(logits_f/temp)
        temperature: float,
        token_topk: Optional[int],
        token_topp: Optional[float],
        confidence_greedy: float,
    ) -> Tuple[torch.Tensor, bool]:
        if not fill_mask.any():
            return x, False

        B, T = x.shape
        V = logits_f.shape[-1]
        device = x.device

        x_new = x.clone()
        idx = fill_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)  # [M]
        logits_sel = logits_f.view(B * T, V).index_select(0, idx)      # [M,V]
        probs_sel = probs_all.view(B * T, V).index_select(0, idx)      # [M,V]
        conf_sel = probs_sel.max(dim=-1).values                        # [M]

        out = torch.empty(idx.numel(), dtype=x.dtype, device=device)

        greedy_idx = (conf_sel >= float(confidence_greedy)).nonzero(as_tuple=False).squeeze(-1)
        if greedy_idx.numel() > 0:
            out[greedy_idx] = logits_sel[greedy_idx].argmax(dim=-1)

        sample_idx = (conf_sel < float(confidence_greedy)).nonzero(as_tuple=False).squeeze(-1)
        if sample_idx.numel() > 0:
            temp = _safe_temperature(temperature)
            if token_topp is not None:
                out[sample_idx] = _top_p_sample(
                    logits_sel[sample_idx], p=float(token_topp), temperature=temp
                )
            elif token_topk is not None:
                k_tok = max(1, min(int(token_topk), V))
                topk_vals, topk_idx = torch.topk(logits_sel[sample_idx], k=k_tok, dim=-1)  # [M,k]
                probs_topk = F.softmax(topk_vals / temp, dim=-1)
                sel = torch.multinomial(probs_topk, num_samples=1).squeeze(-1)  # [M]
                out[sample_idx] = topk_idx.gather(-1, sel.unsqueeze(-1)).squeeze(-1)
            else:
                out[sample_idx] = torch.multinomial(probs_sel[sample_idx], num_samples=1).squeeze(-1)

        x_new.view(B * T)[idx] = out
        return x_new, True


class MaskCFMIterativeUnmaskDecoder:
    @staticmethod
    def _final_t(model, batch_size: int, device: torch.device) -> torch.Tensor:
        t = 1.0 - float(getattr(model, "t_eps", 0.0))
        t = max(0.0, min(1.0, t))
        return torch.full((batch_size,), t, device=device)

    @staticmethod
    def decode_step(
        *,
        model,
        x: torch.Tensor,
        valid: torch.Tensor,
        t_k: torch.Tensor,
        t_next: torch.Tensor,
        alpha_k: torch.Tensor,
        alpha_next: torch.Tensor,
        attn_mask: torch.Tensor,
        cond: Optional[Any],
        temperature: float,
        token_topk: Optional[int],
        token_topp: Optional[float],
        confidence_greedy: float,
        use_margin: bool,
        guidance_kwargs: Optional[dict],
        topk_per_step: Optional[int],
        n_unmask_override: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, bool]:
        B, T = x.shape

        masked = (x == model.mask_id) & valid
        masked_count = masked.sum(dim=1)
        if (masked_count == 0).all():
            return x, False

        if n_unmask_override is not None:
            n_unmask = torch.minimum(masked_count, n_unmask_override)
        elif topk_per_step is not None:
            n_unmask = torch.minimum(masked_count, torch.full_like(masked_count, int(topk_per_step)))
        else:
            nonpad = valid.sum(dim=1)
            target_k = torch.round(alpha_k * nonpad).long()
            target_n = torch.round(alpha_next * nonpad).long()
            n_unmask = (target_k - target_n).clamp_min(1)
            n_unmask = torch.minimum(n_unmask, masked_count)

        Kmax = int(n_unmask.max().item())
        if Kmax == 0:
            return x, False

        logits = model.backbone_inference(x, t_k, cond=cond, attn_mask=attn_mask, guidance_kwargs=guidance_kwargs)

        logits_f = logits.clone()
        neg_inf = torch.finfo(logits_f.dtype).min
        logits_f[..., model.mask_id] = neg_inf
        if getattr(model, "pad_id", None) is not None:
            logits_f[..., model.pad_id] = neg_inf

        temp = _safe_temperature(temperature)
        probs_all = F.softmax(logits_f / temp, dim=-1)

        if use_margin:
            top2_vals, _ = torch.topk(probs_all, k=2, dim=-1)
            score = top2_vals[..., 0] - top2_vals[..., 1]
        else:
            score = probs_all.max(dim=-1).values

        weights = score.masked_fill(~masked, 0.0)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        row_sum = weights.sum(dim=1, keepdim=True)
        all_zero = (row_sum == 0)
        if all_zero.any():
            uniform = masked.float()
            uniform_sum = uniform.sum(dim=1, keepdim=True).clamp_min(1.0)
            weights = torch.where(all_zero, uniform / uniform_sum, weights)

        logw = torch.log(weights + 1e-20)
        U = torch.rand_like(logw).clamp_min(1e-6)
        g = -torch.log(-torch.log(U))
        y = logw + g
        order = torch.argsort(y, dim=1, descending=True)

        top_pos = order[:, :Kmax]
        active = (torch.arange(Kmax, device=x.device).unsqueeze(0) < n_unmask.unsqueeze(1))
        fill_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        fill_mask.scatter_(1, top_pos, active)
        fill_mask &= masked

        x_new, filled = MaskCFMTokenSampler.sample_selected_positions(
            x=x,
            fill_mask=fill_mask,
            logits_f=logits_f,
            probs_all=probs_all,
            temperature=temperature,
            token_topk=token_topk,
            token_topp=token_topp,
            confidence_greedy=confidence_greedy,
        )
        return x_new, filled

    def decode(
        self,
        *,
        model,
        x_init: torch.Tensor,
        steps: int,
        temperature: float,
        topk_per_step: Optional[int],
        token_topk: Optional[int],
        token_topp: Optional[float],
        cond: Optional[Any],
        padding_mask: Optional[torch.Tensor],
        return_all_steps: bool,
        schedule_mode: str,
        alpha_min: float,
        confidence_greedy: float,
        use_margin: bool,
        guidance_kwargs: Optional[dict],
        remask_last: bool,
        remask_frac: float,
        remask_thresh: float,
        ensure_no_mask: bool,
        preserve_input_tokens: bool,
        return_dict: bool = False,
        return_scores: bool = False,
        return_margin: bool = False,
        return_topk_logits: Optional[int] = None,
        return_meta: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        x = x_init.clone()
        B, T = x.shape

        valid = model._build_valid_mask(x, padding_mask)
        attn_mask = model._attn_mask_from_valid(valid)
        traj = [x.clone()] if return_all_steps else None

        if preserve_input_tokens:
            frozen = (x_init != model.mask_id) & valid
        else:
            frozen = torch.zeros_like(valid, dtype=torch.bool)

        t_grid, a_grid = MaskCFMTimeGrid.build(model, steps, schedule_mode, alpha_min, x.device)

        for k in range(steps):
            t_k = t_grid[k].expand(B)
            t_next = t_grid[k + 1].expand(B)
            alpha_k = a_grid[k].expand(B)
            alpha_n = a_grid[k + 1].expand(B)

            x, _ = self.decode_step(
                model=model,
                x=x,
                valid=valid,
                t_k=t_k,
                t_next=t_next,
                alpha_k=alpha_k,
                alpha_next=alpha_n,
                attn_mask=attn_mask,
                cond=cond,
                temperature=temperature,
                token_topk=token_topk,
                token_topp=token_topp,
                confidence_greedy=confidence_greedy,
                use_margin=use_margin,
                guidance_kwargs=guidance_kwargs,
                topk_per_step=topk_per_step,
                n_unmask_override=None,
            )

            if return_all_steps:
                traj.append(x.clone())

            if not ((x == model.mask_id) & valid).any():
                break

        if ensure_no_mask:
            masked = (x == model.mask_id) & valid
            if masked.any():
                t_fill = self._final_t(model, B, x.device)
                masked_count = masked.sum(dim=1)
                x, _ = self.decode_step(
                    model=model,
                    x=x,
                    valid=valid,
                    t_k=t_fill,
                    t_next=t_fill,
                    alpha_k=model.alpha(t_fill),
                    alpha_next=model.alpha(t_fill),
                    attn_mask=attn_mask,
                    cond=cond,
                    temperature=temperature,
                    token_topk=token_topk,
                    token_topp=token_topp,
                    confidence_greedy=confidence_greedy,
                    use_margin=use_margin,
                    guidance_kwargs=guidance_kwargs,
                    topk_per_step=None,
                    n_unmask_override=masked_count,
                )
                if return_all_steps:
                    traj.append(x.clone())

        if remask_last:
            t_final = self._final_t(model, B, x.device)
            logits = model.backbone_inference(x, t_final, cond=cond, attn_mask=attn_mask, guidance_kwargs=guidance_kwargs)

            logits_f = logits.clone()
            neg_inf = torch.finfo(logits_f.dtype).min
            logits_f[..., model.mask_id] = neg_inf
            if getattr(model, "pad_id", None) is not None:
                logits_f[..., model.pad_id] = neg_inf

            probs = F.softmax(logits_f, dim=-1)
            conf = probs.max(dim=-1).values

            filled_pos = (x != model.mask_id) & valid & (~frozen)
            lowconf = filled_pos & (conf < float(remask_thresh))

            nonpad = valid.sum(dim=1)
            Kmax = torch.ceil(float(remask_frac) * nonpad.float()).long()

            score = conf.masked_fill(~lowconf, 2.0)
            order = torch.argsort(score, dim=1, descending=False)

            mask_new = torch.zeros_like(filled_pos)
            for b in range(B):
                k = int(Kmax[b].item())
                if k <= 0:
                    continue
                cand = lowconf[b]
                if not cand.any():
                    continue
                pos = order[b]
                pos = pos[cand[pos]]
                if pos.numel() == 0:
                    continue
                mask_new[b, pos[:k]] = True

            x = torch.where(mask_new, torch.full_like(x, model.mask_id), x)

            masked = (x == model.mask_id) & valid
            if masked.any():
                masked_count = masked.sum(dim=1)
                x, _ = self.decode_step(
                    model=model,
                    x=x,
                    valid=valid,
                    t_k=t_final,
                    t_next=t_final,
                    alpha_k=model.alpha(t_final),
                    alpha_next=model.alpha(t_final),
                    attn_mask=attn_mask,
                    cond=cond,
                    temperature=temperature,
                    token_topk=token_topk,
                    token_topp=token_topp,
                    confidence_greedy=confidence_greedy,
                    use_margin=use_margin,
                    guidance_kwargs=guidance_kwargs,
                    topk_per_step=None,
                    n_unmask_override=masked_count,
                )
                if return_all_steps:
                    traj.append(x.clone())

        if ensure_no_mask:
            masked = (x == model.mask_id) & valid
            if masked.any():
                t_fill = self._final_t(model, B, x.device)
                logits = model.backbone_inference(x, t_fill, cond=cond, attn_mask=attn_mask, guidance_kwargs=guidance_kwargs)

                logits_f = logits.clone()
                neg_inf = torch.finfo(logits_f.dtype).min
                logits_f[..., model.mask_id] = neg_inf
                if getattr(model, "pad_id", None) is not None:
                    logits_f[..., model.pad_id] = neg_inf

                temp = _safe_temperature(temperature)
                probs_all = F.softmax(logits_f / temp, dim=-1)

                x, _ = MaskCFMTokenSampler.sample_selected_positions(
                    x=x,
                    fill_mask=masked,
                    logits_f=logits_f,
                    probs_all=probs_all,
                    temperature=temperature,
                    token_topk=token_topk,
                    token_topp=token_topp,
                    confidence_greedy=confidence_greedy,
                )
                if return_all_steps:
                    traj.append(x.clone())

        out_traj = torch.stack(traj, dim=0) if return_all_steps else None
        if not return_dict:
            return out_traj if return_all_steps else x

        out: Dict[str, Any] = {"tokens": x}
        if out_traj is not None:
            out["traj"] = out_traj

        if return_meta:
            out["valid"] = valid
            out["frozen"] = frozen
            out["t_grid"] = t_grid
            out["a_grid"] = a_grid

        if return_scores or (return_topk_logits is not None):
            t_probe = self._final_t(model, B, x.device)
            logits = model.backbone_inference(x, t_probe, cond=cond, attn_mask=attn_mask, guidance_kwargs=guidance_kwargs)

            logits_f = logits.clone()
            neg_inf = torch.finfo(logits_f.dtype).min
            logits_f[..., model.mask_id] = neg_inf
            if getattr(model, "pad_id", None) is not None:
                logits_f[..., model.pad_id] = neg_inf

            out["t_probe"] = t_probe.detach()

            if return_topk_logits is not None:
                k = int(return_topk_logits)
                k = max(1, k)
                topk_logits, topk_ids = torch.topk(logits_f, k=k, dim=-1)
                out["topk_ids"] = topk_ids
                out["topk_logits"] = topk_logits

            if return_scores:
                temp = _safe_temperature(temperature)
                probs_all = F.softmax(logits_f / temp, dim=-1)

                token_conf = probs_all.gather(-1, x.unsqueeze(-1)).squeeze(-1)
                token_max_prob = probs_all.max(dim=-1).values

                token_conf = token_conf.masked_fill(~valid, 0.0)
                token_max_prob = token_max_prob.masked_fill(~valid, 0.0)

                need_margin = bool(use_margin or return_margin)
                token_margin = None
                if need_margin:
                    top2_vals = torch.topk(probs_all, k=2, dim=-1).values
                    token_margin = (top2_vals[..., 0] - top2_vals[..., 1]).masked_fill(~valid, 0.0)

                out["token_conf"] = token_conf
                out["token_max_prob"] = token_max_prob
                if token_margin is not None:
                    out["token_margin"] = token_margin

                out["score_type"] = "margin" if use_margin else "max_prob"
                out["token_score"] = token_margin if (use_margin and token_margin is not None) else token_max_prob

        return out


class MaskCFMRemaskRefineDecoder:
    @staticmethod
    def _final_t(model, batch_size: int, device: torch.device) -> torch.Tensor:
        t = 1.0 - float(getattr(model, "t_eps", 0.0))
        t = max(0.0, min(1.0, t))
        return torch.full((batch_size,), t, device=device)

    def decode(
        self,
        *,
        model,
        x_init: torch.Tensor,
        steps: int,
        temperature: float,
        token_topk: Optional[int],
        token_topp: Optional[float],
        cond: Optional[Any],
        padding_mask: Optional[torch.Tensor],
        return_all_steps: bool,
        schedule_mode: str,
        alpha_min: float,
        confidence_greedy: float,
        use_margin: bool,
        guidance_kwargs: Optional[dict],
        preserve_input_tokens: bool,
        ensure_no_mask: bool,
        return_dict: bool = False,
        return_scores: bool = False,
        return_margin: bool = False,
        return_topk_logits: Optional[int] = None,
        return_meta: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        x = x_init.clone()
        B, T = x.shape

        valid = model._build_valid_mask(x, padding_mask)
        attn_mask = model._attn_mask_from_valid(valid)
        traj = [x.clone()] if return_all_steps else None

        if preserve_input_tokens:
            frozen = (x_init != model.mask_id) & valid
        else:
            frozen = torch.zeros_like(valid, dtype=torch.bool)

        candidate = valid & ~frozen

        want_extra = bool(return_scores or (return_topk_logits is not None))
        cache_logits_f = None
        cache_probs_all = None
        cache_t = None

        if not candidate.any():
            out_traj = torch.stack(traj, dim=0) if return_all_steps else None
            if not return_dict:
                return out_traj if return_all_steps else x

            out: Dict[str, Any] = {"tokens": x}
            if out_traj is not None:
                out["traj"] = out_traj
            if return_meta:
                out["valid"] = valid
                out["frozen"] = frozen
                out["candidate"] = candidate
            return out

        t_grid, a_grid = MaskCFMTimeGrid.build(model, steps, schedule_mode, alpha_min, x.device)

        for k in range(steps):
            t_k = t_grid[k].expand(B)

            logits = model.backbone_inference(x, t_k, cond=cond, attn_mask=attn_mask, guidance_kwargs=guidance_kwargs)

            logits_f = logits.clone()
            neg_inf = torch.finfo(logits_f.dtype).min
            logits_f[..., model.mask_id] = neg_inf
            if getattr(model, "pad_id", None) is not None:
                logits_f[..., model.pad_id] = neg_inf

            temp = _safe_temperature(temperature)
            probs_all = F.softmax(logits_f / temp, dim=-1)

            if use_margin:
                top2_vals, _ = torch.topk(probs_all, k=2, dim=-1)
                score = top2_vals[..., 0] - top2_vals[..., 1]
            else:
                score = probs_all.max(dim=-1).values

            x_pred, _ = MaskCFMTokenSampler.sample_selected_positions(
                x=x,
                fill_mask=candidate,
                logits_f=logits_f,
                probs_all=probs_all,
                temperature=temperature,
                token_topk=token_topk,
                token_topp=token_topp,
                confidence_greedy=confidence_greedy,
            )

            if want_extra:
                cache_logits_f = logits_f
                cache_probs_all = probs_all
                cache_t = t_k

            if k == steps - 1:
                x = x_pred
                if return_all_steps:
                    traj.append(x.clone())
                break

            alpha_next = a_grid[k + 1].expand(B)

            nonpad = valid.sum(dim=1)                # [B]
            frozen_cnt = frozen.sum(dim=1)           # [B]
            cand_cnt = candidate.sum(dim=1)          # [B]

            n_mask_next = torch.round(alpha_next * nonpad.float()).long().clamp_min(0)
            keep_total = (nonpad - n_mask_next).clamp_min(0)
            keep_cand = (keep_total - frozen_cnt).clamp_min(0)
            keep_cand = torch.minimum(keep_cand, cand_cnt)

            score_cand = score.masked_fill(~candidate, -1e9)
            order = torch.argsort(score_cand, dim=1, descending=True)

            keep_mask = frozen.clone()
            keep_cand_max = int(keep_cand.max().item())
            if keep_cand_max > 0:
                top_pos = order[:, :keep_cand_max]
                active = (torch.arange(keep_cand_max, device=x.device).unsqueeze(0) < keep_cand.unsqueeze(1))
                keep_sel = torch.zeros_like(candidate)
                keep_sel.scatter_(1, top_pos, active)
                keep_mask |= (keep_sel & candidate)

            remask = candidate & ~keep_mask
            x = torch.where(remask, torch.full_like(x_pred, model.mask_id), x_pred)

            if return_all_steps:
                traj.append(x.clone())

        if ensure_no_mask:
            masked = (x == model.mask_id) & valid
            if masked.any():
                t_fill = self._final_t(model, B, x.device)
                logits = model.backbone_inference(x, t_fill, cond=cond, attn_mask=attn_mask, guidance_kwargs=guidance_kwargs)

                logits_f = logits.clone()
                neg_inf = torch.finfo(logits_f.dtype).min
                logits_f[..., model.mask_id] = neg_inf
                if getattr(model, "pad_id", None) is not None:
                    logits_f[..., model.pad_id] = neg_inf

                temp = _safe_temperature(temperature)
                probs_all = F.softmax(logits_f / temp, dim=-1)

                x, _ = MaskCFMTokenSampler.sample_selected_positions(
                    x=x,
                    fill_mask=masked,
                    logits_f=logits_f,
                    probs_all=probs_all,
                    temperature=temperature,
                    token_topk=token_topk,
                    token_topp=token_topp,
                    confidence_greedy=confidence_greedy,
                )
                if return_all_steps:
                    traj.append(x.clone())

                if want_extra:
                    cache_logits_f = logits_f
                    cache_probs_all = probs_all
                    cache_t = t_fill

        out_traj = torch.stack(traj, dim=0) if return_all_steps else None
        if not return_dict:
            return out_traj if return_all_steps else x

        out: Dict[str, Any] = {"tokens": x}
        if out_traj is not None:
            out["traj"] = out_traj

        if return_meta:
            out["valid"] = valid
            out["frozen"] = frozen
            out["candidate"] = candidate
            out["t_grid"] = t_grid
            out["a_grid"] = a_grid

        if want_extra and cache_logits_f is not None and cache_probs_all is not None:
            out["t_last"] = cache_t.detach() if torch.is_tensor(cache_t) else cache_t

            if return_topk_logits is not None:
                k = int(return_topk_logits)
                k = max(1, k)
                topk_logits, topk_ids = torch.topk(cache_logits_f, k=k, dim=-1)
                out["topk_ids"] = topk_ids
                out["topk_logits"] = topk_logits

            if return_scores:
                token_conf = cache_probs_all.gather(-1, x.unsqueeze(-1)).squeeze(-1)
                token_max_prob = cache_probs_all.max(dim=-1).values

                token_conf = token_conf.masked_fill(~valid, 0.0)
                token_max_prob = token_max_prob.masked_fill(~valid, 0.0)

                need_margin = bool(use_margin or return_margin)
                token_margin = None
                if need_margin:
                    top2_vals = torch.topk(cache_probs_all, k=2, dim=-1).values
                    token_margin = (top2_vals[..., 0] - top2_vals[..., 1]).masked_fill(~valid, 0.0)

                out["token_conf"] = token_conf
                out["token_max_prob"] = token_max_prob
                if token_margin is not None:
                    out["token_margin"] = token_margin

                out["score_type"] = "margin" if use_margin else "max_prob"
                out["token_score"] = token_margin if (use_margin and token_margin is not None) else token_max_prob

        return out

