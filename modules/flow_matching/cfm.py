from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn

DenoiseFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
GuidanceFn = Callable[..., torch.Tensor]


@dataclass
class ConditionalCFMOdeintConfig:
    atol: float = 1e-4
    rtol: float = 1e-4
    method: str = "euler"


class ConditionalFlowMatching(nn.Module):
    """
    Reusable CFM / flow-matching utilities, similar in spirit to mask_cfm.py.

    This module is intentionally *data-format agnostic*:
    - x can be any shape: [B, ...]
    - cond is opaque (you capture it in denoise_fn closure, or pass to your own backbone)

    Typical usage:
      - Training: build (t, x_t, u_t) using make_flow_matching_pair(...)
      - Inference: integrate dx/dt = v_theta(x,t,cond) with Euler / AMO / torchdiffeq (optional)
      - Guidance: pass a custom guidance_fn(pred_v, t, **kwargs) into infer()/infer_step()
    """

    def __init__(self, *, time_sampler: Optional[Any] = None, t_eps: float = 0.0):
        super().__init__()
        self.time_sampler = time_sampler
        self.t_eps = float(t_eps)

    def sample_t(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        if self.time_sampler is not None:
            t = self.time_sampler.sample([batch_size], device)
            if not torch.is_tensor(t):
                t = torch.tensor(t, device=device)
            t = t.to(device=device)
        else:
            high = 1.0 - max(self.t_eps, 0.0)
            t = torch.rand(batch_size, device=device) * high

        t = t.clamp(min=0.0, max=1.0 - max(self.t_eps, 0.0))
        return t.to(dtype=dtype)

    @staticmethod
    def _broadcast_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            raise TypeError("t must be a torch.Tensor")
        if t.ndim == 0:
            t = t.view(1)
        if t.ndim != 1:
            raise ValueError(f"t must be a 1D tensor (or scalar), got shape={tuple(t.shape)}")
        if t.shape[0] != x.shape[0]:
            raise ValueError(f"t batch mismatch: t={t.shape[0]} vs x={x.shape[0]}")
        return t.reshape(x.shape[0], *([1] * (x.ndim - 1)))

    def make_flow_matching_pair(
        self, *, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Independent CFM path:
          x_t = t * x1 + (1 - t) * x0
          u_t = x1 - x0
        """
        if x0.shape != x1.shape:
            raise ValueError(f"x0/x1 shape mismatch: {tuple(x0.shape)} vs {tuple(x1.shape)}")

        t_ = self._broadcast_t_like_x(t, x0)
        xt = t_ * x1 + (1.0 - t_) * x0
        ut = x1 - x0
        return xt, ut

    def compute_loss(
        self,
        *,
        x1: torch.Tensor,
        denoise_fn: DenoiseFn,
        x0: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """
        Generic CFM regression objective: || v_theta(x_t, t) - (x1 - x0) ||^2

        loss_mask:
          - Optional mask broadcastable to x1/pred shape.
          - For sequence models you can pass [B,T,1] or [B,T] etc.
        """
        if x0 is None:
            x0 = torch.randn_like(x1)

        if t is None:
            t = self.sample_t(batch_size=int(x1.shape[0]), device=x1.device, dtype=x1.dtype)

        xt, ut = self.make_flow_matching_pair(x0=x0, x1=x1, t=t)
        pred = denoise_fn(xt, t)

        err2 = (pred - ut) ** 2
        if loss_mask is not None:
            mask = loss_mask.to(device=err2.device).to(dtype=err2.dtype)
            err2 = err2 * mask

        if reduction == "mean":
            loss = err2.mean()
        elif reduction == "sum":
            loss = err2.sum()
        elif reduction == "none":
            loss = err2
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        return {"loss": loss, "pred": pred, "target": ut, "t": t.detach(), "x_t": xt.detach()}

    @staticmethod
    def build_time_schedule(
        *,
        steps: int,
        device: torch.device,
        dtype: torch.dtype,
        use_sway: bool = True,
        sway_sampling_coef: float = -1.0,
    ) -> torch.Tensor:
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}")

        t_schedule = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
        if use_sway:
            t_schedule = t_schedule + float(sway_sampling_coef) * (
                torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule
            )
        return t_schedule

    @staticmethod
    def _amo_step(x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor, pred_v: torch.Tensor) -> torch.Tensor:
        c = 3.0
        o = torch.clamp(s + c * (s - t), max=1.0)

        pred_x_o = x_t + (o - t) * pred_v
        a = s / o
        b = torch.sqrt(torch.clamp_min((1 - s) ** 2 - (a * (1 - o)) ** 2, 0.0))

        noises = torch.randn_like(x_t)
        prev_sample = a * pred_x_o + b * noises
        return prev_sample.to(pred_v.dtype)
    
    def infer_step(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        denoise_fn: DenoiseFn,
        sampler: str = "euler",  # "euler" | "amo"
        pred_v: Optional[torch.Tensor] = None,
        guidance_fn: Optional[GuidanceFn] = None,
        guidance_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-step update x(t) -> x(t_next).

        Returns:
            x_next: updated sample
            pred_v: the (optionally guided) vector field used for the update
        """
        if pred_v is None:
            pred_v = denoise_fn(x, t)

        if guidance_fn is not None:
            guidance_kwargs = guidance_kwargs or {}
            pred_v = guidance_fn(pred_v, t, **guidance_kwargs)

        if sampler in ("euler", "ode_euler"):
            dt = t_next - t
            x_next = x + dt * pred_v
            return x_next, pred_v

        if sampler == "amo":
            x_next = self._amo_step(x, t, t_next, pred_v)
            return x_next, pred_v

        raise ValueError(f"Unknown sampler for infer_step: {sampler}")

    def infer(
        self,
        *,
        x_init: torch.Tensor,
        steps: int,
        denoise_fn: DenoiseFn,
        sampler: str = "euler",  # "euler" | "amo" | "torchdiffeq_euler"
        use_sway: bool = True,
        sway_sampling_coef: float = -1.0,
        t_schedule: Optional[torch.Tensor] = None,
        odeint_cfg: ConditionalCFMOdeintConfig = ConditionalCFMOdeintConfig(),
        guidance_fn: Optional[GuidanceFn] = None,
        guidance_kwargs: Optional[Dict[str, Any]] = None,
        return_timesteps: bool = False,
        return_all_steps: bool = False,
        return_v: bool = False,
    ):
        x = x_init

        if t_schedule is None:
            t_schedule = self.build_time_schedule(
                steps=steps,
                device=x.device,
                dtype=x.dtype,
                use_sway=use_sway,
                sway_sampling_coef=sway_sampling_coef,
            )
        else:
            t_schedule = t_schedule.to(device=x.device, dtype=x.dtype)
            if int(t_schedule.numel()) != int(steps) + 1:
                raise ValueError(
                    f"t_schedule must have length steps+1, got {int(t_schedule.numel())} vs {int(steps)+1}"
                )

        if sampler in ("euler", "ode_euler", "amo"):
            traj = [x] if return_all_steps else None
            v_traj = [] if (return_all_steps and return_v) else None

            for i in range(int(t_schedule.numel()) - 1):
                t = t_schedule[i]
                t_next = t_schedule[i + 1]

                x, v = self.infer_step(
                    x=x,
                    t=t,
                    t_next=t_next,
                    denoise_fn=denoise_fn,
                    sampler=sampler,
                    guidance_fn=guidance_fn,
                    guidance_kwargs=guidance_kwargs,
                )

                if return_all_steps:
                    traj.append(x)
                    if return_v:
                        v_traj.append(v)

            if return_all_steps:
                x_out = torch.stack(traj, dim=0)
                if return_v:
                    v_out = torch.stack(v_traj, dim=0)
                    if return_timesteps:
                        return x_out, t_schedule, v_out
                    return x_out, v_out

                if return_timesteps:
                    return x_out, t_schedule
                return x_out

            if return_timesteps:
                return x, t_schedule
            return x

        if sampler in ("torchdiffeq", "torchdiffeq_euler"):
            try:
                import torchdiffeq  # type: ignore
            except Exception as e:
                raise ImportError(
                    "sampler='torchdiffeq_euler' requires torchdiffeq to be installed/available."
                ) from e

            if guidance_fn is not None:
                raise ValueError("torchdiffeq sampler currently does not support guidance_fn; use euler/amo instead")

            def ode_fn(t_scalar: torch.Tensor, x_state: torch.Tensor) -> torch.Tensor:
                return denoise_fn(x_state, t_scalar)

            traj = torchdiffeq.odeint(
                ode_fn,
                x_init,
                t_schedule,
                atol=float(odeint_cfg.atol),
                rtol=float(odeint_cfg.rtol),
                method=str(odeint_cfg.method),
            )

            x_final = traj[-1]
            if return_timesteps and return_all_steps:
                return traj, t_schedule
            if return_timesteps:
                return x_final, t_schedule
            if return_all_steps:
                return traj
            return x_final

        raise ValueError(f"Unknown sampler: {sampler}")
    