import argparse
import filecmp
import multiprocessing
import os
import subprocess
import librosa
import re
from functools import partial
from multiprocessing import Pool, Process
import random
from typing import Optional, Sequence
import traceback
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from attrdictionary import AttrDict
import soundfile as sf

from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.commons.base_task import BaseTask
from utils.commons.import_utils import import_module_bystr
from utils.commons.os_utils import kill_void
from utils.commons.io import print_once
from utils.nn.schedulers import WarmupSchedule, CosineSchedule
from utils.nn.model_utils import unwrap_model
from utils.nn.seq_utils import sequence_mask
from utils.audio.mel import MelNet, MultiResolutionMelLoss, MultiResolutionMultiBandMelLoss
from utils.audio.energy import short_time_energy

from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin, TTSDatasetMixin
from modules.vocoder.commons.stft_loss import MultiResolutionSTFTLoss, MultiResolutionTransientSTFTLoss
from modules.vocoder.hifigan.hifigan import MultiPeriodDiscriminator, MultiScaleDiscriminator
from modules.vocoder.hifigan.mel_utils import mel_spectrogram
from modules.vocoder.univnet.mrd import MultiResolutionDiscriminator
from modules.tts.scriptspeech.build_model_utils import build_vae


def get_model_core(module: torch.nn.Module) -> torch.nn.Module:
    module = unwrap_model(module)
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


def parse_reinit_modules_from_hparams():
    modules = []
    raw = hparams.get('reinit_modules_after_load', [])
    if isinstance(raw, str):
        modules.extend([x for x in re.split(r'[\s,]+', raw.strip()) if x])
    elif isinstance(raw, (list, tuple, set)):
        modules.extend([str(x).strip() for x in raw if str(x).strip()])

    alias_flags = {
        'reinit_encoder_after_load': 'encoder',
        'reinit_bottleneck_after_load': 'bottleneck',
        'reinit_decoder_after_load': 'decoder',
        'reinit_latent_upsample_after_load': 'latent_upsample',
    }
    for flag_name, module_name in alias_flags.items():
        if hparams.get(flag_name, False):
            modules.append(module_name)

    dedup_modules = []
    seen = set()
    for module_name in modules:
        if module_name in seen:
            continue
        seen.add(module_name)
        dedup_modules.append(module_name)
    return dedup_modules


def generator_loss(disc_outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(disc_outputs) == 0:
        raise ValueError("generator_loss received empty discriminator outputs")
    per_discriminator = []
    for disc_idx, disc_output in enumerate(disc_outputs):
        if disc_output.numel() == 0:
            raise ValueError(f"generator_loss received empty logits for discriminator {disc_idx}")
        per_discriminator.append((1.0 - disc_output).pow(2).mean())
    return torch.stack(per_discriminator).mean()


def feature_loss(
    fmap_r: Sequence[Sequence[torch.Tensor]],
    fmap_g: Sequence[Sequence[torch.Tensor]],
) -> torch.Tensor:
    if len(fmap_r) == 0 or len(fmap_g) == 0:
        raise ValueError("feature_loss received empty feature maps")
    if len(fmap_r) != len(fmap_g):
        raise ValueError(
            f"feature_loss discriminator count mismatch: {len(fmap_r)} vs {len(fmap_g)}"
        )

    per_discriminator = []
    for disc_idx, (disc_real_fmaps, disc_fake_fmaps) in enumerate(zip(fmap_r, fmap_g)):
        if len(disc_real_fmaps) == 0 or len(disc_fake_fmaps) == 0:
            raise ValueError(f"feature_loss received empty layers for discriminator {disc_idx}")
        if len(disc_real_fmaps) != len(disc_fake_fmaps):
            raise ValueError(
                f"feature_loss layer count mismatch at discriminator {disc_idx}: "
                f"{len(disc_real_fmaps)} vs {len(disc_fake_fmaps)}"
            )
        per_layer = [
            torch.abs(real_layer - fake_layer).mean()
            for real_layer, fake_layer in zip(disc_real_fmaps, disc_fake_fmaps)
        ]
        per_discriminator.append(torch.stack(per_layer).mean())

    return torch.stack(per_discriminator).mean()


def discriminator_loss(
    disc_real_outputs: Sequence[torch.Tensor],
    disc_generated_outputs: Sequence[torch.Tensor],
):
    if len(disc_real_outputs) == 0 or len(disc_generated_outputs) == 0:
        raise ValueError("discriminator_loss received empty discriminator outputs")
    if len(disc_real_outputs) != len(disc_generated_outputs):
        raise ValueError(
            f"discriminator_loss discriminator count mismatch: "
            f"{len(disc_real_outputs)} vs {len(disc_generated_outputs)}"
        )

    real_losses = []
    fake_losses = []
    for disc_idx, (disc_real, disc_fake) in enumerate(zip(disc_real_outputs, disc_generated_outputs)):
        if disc_real.numel() == 0 or disc_fake.numel() == 0:
            raise ValueError(f"discriminator_loss received empty logits for discriminator {disc_idx}")
        real_losses.append((1.0 - disc_real).pow(2).mean())
        fake_losses.append(disc_fake.pow(2).mean())

    return torch.stack(real_losses).mean(), torch.stack(fake_losses).mean()


def snapshot_named_modules(module: torch.nn.Module, module_names):
    snapshots = {}
    for module_name in module_names:
        if not hasattr(module, module_name):
            raise ValueError(f"model_gen has no submodule named `{module_name}` to snapshot")
        child = getattr(module, module_name)
        snapshots[module_name] = {
            key: value.detach().cpu().clone()
            for key, value in child.state_dict().items()
        }
    return snapshots


def flatten_valid_latent_frames(
    mu: torch.Tensor,
    z_lens: torch.Tensor,
    max_frames: Optional[int] = None,
):
    mu = mu.float().transpose(1, 2)  # [B, T, C]
    if z_lens is None:
        flat = mu.reshape(-1, mu.shape[-1])
    else:
        z_lens = z_lens.to(device=mu.device, dtype=torch.long)
        valid_mask = sequence_mask(z_lens, maxlen=mu.shape[1]).reshape(-1)
        flat = mu.reshape(-1, mu.shape[-1])[valid_mask]

    if max_frames is not None and max_frames > 0 and flat.shape[0] > max_frames:
        idx = torch.randperm(flat.shape[0], device=flat.device)[:max_frames]
        flat = flat[idx]
    return flat


def compute_centered_latent_covariance(
    mu: torch.Tensor,
    z_lens: torch.Tensor,
    max_frames: Optional[int] = None,
):
    with torch.cuda.amp.autocast(enabled=False):
        flat = flatten_valid_latent_frames(mu, z_lens, max_frames=max_frames)
        if flat.shape[0] < 2:
            return {
                "flat": flat,
                "mean_c": None,
                "xc": None,
                "cov": None,
                "diag": None,
                "num_frames": int(flat.shape[0]),
            }

        mean_c = flat.mean(dim=0)
        xc = flat - mean_c
        cov = torch.matmul(xc.transpose(0, 1), xc) / max(flat.shape[0], 1)
        diag = torch.diagonal(cov)
    return {
        "flat": flat,
        "mean_c": mean_c,
        "xc": xc,
        "cov": cov,
        "diag": diag,
        "num_frames": int(flat.shape[0]),
    }


def normalize_covariance_trace(cov: torch.Tensor, eps: float = 1e-4):
    dim = cov.shape[0]
    trace_mean = (torch.trace(cov) / max(dim, 1)).clamp_min(float(eps))
    cov_norm = cov / trace_mean
    return cov_norm, trace_mean


def compute_latent_sphere_reg_losses(
    mu: torch.Tensor,
    z_lens: torch.Tensor,
    max_frames: Optional[int] = None,
    target_var: float = 1.0,
    cov_stats=None,
    eps: float = 1e-6,
):
    with torch.cuda.amp.autocast(enabled=False):
        if cov_stats is None:
            cov_stats = compute_centered_latent_covariance(mu, z_lens, max_frames=max_frames)
        mean_c = cov_stats["mean_c"]
        cov = cov_stats["cov"]
        diag = cov_stats["diag"]
        if cov is None:
            zero = mu.new_zeros(())
            return {"sphere_mean": zero, "sphere_var": zero, "sphere_cov": zero, "num_frames": 0}

        offdiag = cov - torch.diag_embed(diag)

        sphere_mean = mean_c.pow(2).mean()
        sphere_var = (diag - float(target_var)).pow(2).mean()
        if offdiag.numel() == diag.numel():
            sphere_cov = cov.new_zeros(())
        else:
            offdiag_mask = ~torch.eye(cov.shape[0], device=cov.device, dtype=torch.bool)
            sphere_cov = offdiag[offdiag_mask].pow(2).mean()

    return {
        "sphere_mean": sphere_mean,
        "sphere_var": sphere_var,
        "sphere_cov": sphere_cov,
        "num_frames": cov_stats["num_frames"],
    }


def compute_latent_min_std_reg_losses(
    mu: torch.Tensor,
    z_lens: torch.Tensor,
    floor: float = 0.3,
    max_frames: Optional[int] = None,
    cov_stats=None,
    eps: float = 1e-6,
):
    with torch.cuda.amp.autocast(enabled=False):
        if cov_stats is None:
            cov_stats = compute_centered_latent_covariance(mu, z_lens, max_frames=max_frames)
        diag = cov_stats["diag"]
        if diag is None:
            zero = mu.new_zeros(())
            return {
                "min_std_loss": zero,
                "latent_min_std": zero,
                "latent_mean_std": zero,
                "latent_low_std_ratio": zero,
                "num_frames": cov_stats["num_frames"],
            }

        std = torch.sqrt(diag.clamp_min(0.0) + float(eps))
        low_std = F.relu(float(floor) - std)
        min_std_loss = low_std.pow(2).mean()
        latent_min_std = std.min()
        latent_mean_std = std.mean()
        latent_low_std_ratio = (std < float(floor)).float().mean()

    return {
        "min_std_loss": min_std_loss,
        "latent_min_std": latent_min_std,
        "latent_mean_std": latent_mean_std,
        "latent_low_std_ratio": latent_low_std_ratio,
        "num_frames": cov_stats["num_frames"],
    }


def compute_normalized_covariance_eig_metrics(cov: torch.Tensor, eps: float = 1e-4):
    with torch.cuda.amp.autocast(enabled=False):
        cov_norm, trace_mean = normalize_covariance_trace(cov, eps=eps)
        eye = torch.eye(cov_norm.shape[0], device=cov_norm.device, dtype=cov_norm.dtype)
        stabilized_cov = cov_norm + float(eps) * eye
        eigvals = torch.linalg.eigvalsh(stabilized_cov)
        min_eig = eigvals.min()
        max_eig = eigvals.max()
        cond_est = max_eig / min_eig.clamp_min(float(eps))
        logdet_per_dim = torch.log(eigvals.clamp_min(float(eps))).mean()
        eigvals_sum = eigvals.sum().clamp_min(float(eps))
        eig_probs = eigvals / eigvals_sum
        spec_entropy = -(eig_probs * torch.log(eig_probs.clamp_min(float(eps)))).sum()
        erank = torch.exp(spec_entropy)
        erank_ratio = erank / cov.shape[0]

    return {
        "cov_norm": cov_norm,
        "trace_mean": trace_mean,
        "eigvals": eigvals,
        "logdet_per_dim": logdet_per_dim,
        "min_eig": min_eig,
        "max_eig": max_eig,
        "cond_est": cond_est,
        "erank": erank,
        "erank_ratio": erank_ratio,
    }


def compute_latent_logdet_reg_losses(
    mu: torch.Tensor,
    z_lens: torch.Tensor,
    max_frames: Optional[int] = None,
    cov_stats=None,
    eig_metrics=None,
    eps: float = 1e-4,
):
    with torch.cuda.amp.autocast(enabled=False):
        if cov_stats is None:
            cov_stats = compute_centered_latent_covariance(mu, z_lens, max_frames=max_frames)
        cov = cov_stats["cov"]
        if cov is None:
            zero = mu.new_zeros(())
            return {
                "norm_logdet_barrier": zero,
                "norm_logdet_per_dim": zero,
                "norm_min_eig": zero,
                "norm_max_eig": zero,
                "norm_cond_est": zero,
                "num_frames": cov_stats["num_frames"],
            }

        if eig_metrics is None:
            eig_metrics = compute_normalized_covariance_eig_metrics(cov, eps=eps)
        logdet_barrier = -eig_metrics["logdet_per_dim"]

    return {
        "norm_logdet_barrier": logdet_barrier,
        "norm_logdet_per_dim": eig_metrics["logdet_per_dim"],
        "norm_min_eig": eig_metrics["min_eig"],
        "norm_max_eig": eig_metrics["max_eig"],
        "norm_cond_est": eig_metrics["cond_est"],
        "num_frames": cov_stats["num_frames"],
    }


def compute_latent_small_eig_reg_losses(
    mu: torch.Tensor,
    z_lens: torch.Tensor,
    floor: float = 0.1,
    max_frames: Optional[int] = None,
    cov_stats=None,
    eig_metrics=None,
    eps: float = 1e-4,
    active_only_avg: bool = False,
):
    with torch.cuda.amp.autocast(enabled=False):
        if cov_stats is None:
            cov_stats = compute_centered_latent_covariance(mu, z_lens, max_frames=max_frames)
        cov = cov_stats["cov"]
        if cov is None:
            zero = mu.new_zeros(())
            return {
                "small_eig_loss": zero,
                "small_eig_active_ratio": zero,
                "norm_min_eig": zero,
                "norm_max_eig": zero,
                "norm_cond_est": zero,
                "num_frames": cov_stats["num_frames"],
            }

        if eig_metrics is None:
            eig_metrics = compute_normalized_covariance_eig_metrics(cov, eps=eps)
        eigvals = eig_metrics["eigvals"]
        active_mask = eigvals < float(floor)
        small_eig_excess = F.relu(float(floor) - eigvals)
        small_eig_penalty = small_eig_excess.pow(2)
        small_eig_active_ratio = active_mask.float().mean()
        if active_only_avg:
            active_count = active_mask.float().sum()
            small_eig_loss = small_eig_penalty.sum() / active_count.clamp_min(1.0)
        else:
            small_eig_loss = small_eig_penalty.mean()

    return {
        "small_eig_loss": small_eig_loss,
        "small_eig_active_ratio": small_eig_active_ratio,
        "norm_min_eig": eig_metrics["min_eig"],
        "norm_max_eig": eig_metrics["max_eig"],
        "norm_cond_est": eig_metrics["cond_est"],
        "num_frames": cov_stats["num_frames"],
    }


def compute_latent_geometry_monitor_metrics(
    mu: torch.Tensor,
    z_lens: torch.Tensor,
    max_frames: Optional[int] = None,
    cov_stats=None,
    eig_metrics=None,
    eps: float = 1e-4,
):
    with torch.cuda.amp.autocast(enabled=False):
        if cov_stats is None:
            cov_stats = compute_centered_latent_covariance(mu, z_lens, max_frames=max_frames)
        cov = cov_stats["cov"]
        if cov is None:
            zero = mu.new_zeros(())
            return {
                "latent_norm_logdet_per_dim": zero,
                "latent_norm_erank": zero,
                "latent_norm_erank_ratio": zero,
                "latent_norm_cond_est": zero,
                "latent_norm_min_eig": zero,
                "latent_norm_max_eig": zero,
                "num_frames": cov_stats["num_frames"],
            }
        if eig_metrics is None:
            eig_metrics = compute_normalized_covariance_eig_metrics(cov, eps=eps)
    return {
        "latent_norm_logdet_per_dim": eig_metrics["logdet_per_dim"],
        "latent_norm_erank": eig_metrics["erank"],
        "latent_norm_erank_ratio": eig_metrics["erank_ratio"],
        "latent_norm_cond_est": eig_metrics["cond_est"],
        "latent_norm_min_eig": eig_metrics["min_eig"],
        "latent_norm_max_eig": eig_metrics["max_eig"],
        "num_frames": cov_stats["num_frames"],
    }


def compute_latent_tail_reg_losses(
    mu: torch.Tensor,
    z_lens: torch.Tensor,
    elem_threshold: float = 12.0,
    frame_threshold: float = 96.0,
    max_frames: Optional[int] = None,
    flat=None,
    eps: float = 1e-6,
):
    with torch.cuda.amp.autocast(enabled=False):
        if flat is None:
            flat = flatten_valid_latent_frames(mu, z_lens, max_frames=max_frames)

        if flat.shape[0] < 1:
            zero = mu.new_zeros(())
            return {
                "tail_elem": zero,
                "tail_frame": zero,
                "tail_elem_ratio": zero,
                "tail_frame_ratio": zero,
                "num_frames": 0,
            }

        elem_excess = F.relu(flat.abs() - float(elem_threshold))
        tail_elem = elem_excess.pow(2).mean()
        tail_elem_ratio = (elem_excess > 0).float().mean()

        frame_norm = torch.linalg.norm(flat, dim=-1)
        frame_excess = F.relu(frame_norm - float(frame_threshold))
        tail_frame = frame_excess.pow(2).mean()
        tail_frame_ratio = (frame_excess > 0).float().mean()

    return {
        "tail_elem": tail_elem,
        "tail_frame": tail_frame,
        "tail_elem_ratio": tail_elem_ratio,
        "tail_frame_ratio": tail_frame_ratio,
        "num_frames": int(flat.shape[0]),
    }


def compute_latent_temporal_diff_reg_losses(
    mu: torch.Tensor,
    z_lens: Optional[torch.Tensor],
    order: int = 1,
    max_frames: Optional[int] = None,
    eps: float = 1e-6,
    loss_mode: str = "global_ratio",
    hinge_target: float = 0.0,
    sample_gate: Optional[torch.Tensor] = None,
):
    if order not in (1, 2):
        raise ValueError(f"unsupported temporal diff order={order}")
    if loss_mode not in ("global_ratio", "per_sample_ratio", "per_sample_hinge"):
        raise ValueError(f"unsupported temporal diff loss_mode={loss_mode}")

    with torch.cuda.amp.autocast(enabled=False):
        mu_t = mu.float().transpose(1, 2)  # [B, T, C]
        bsz, seq_len, dim = mu_t.shape

        if z_lens is None:
            eff_lens = torch.full((bsz,), seq_len, device=mu_t.device, dtype=torch.long)
        else:
            eff_lens = z_lens.to(device=mu_t.device, dtype=torch.long).clamp(min=0, max=seq_len)

        if max_frames is not None and max_frames > 0 and int(eff_lens.sum().item()) > int(max_frames):
            per_sample_cap = max(int(math.ceil(float(max_frames) / max(bsz, 1))), order + 1)
            per_sample_cap = min(per_sample_cap, seq_len)
            eff_lens = eff_lens.clamp(max=per_sample_cap)

        valid_mask = sequence_mask(eff_lens, maxlen=seq_len)
        num_frames = int(valid_mask.sum().item())
        zero = mu.new_zeros(())
        if num_frames <= order:
            return {
                "temporal_diff_loss": zero,
                "temporal_diff_ratio": zero,
                "temporal_diff_mean_l2": zero,
                "temporal_diff_active_ratio": zero,
                "temporal_diff_hinge_active_ratio": zero,
                "num_frames": num_frames,
                "num_diffs": 0,
            }

        energy_per_sample = (mu_t.pow(2) * valid_mask.unsqueeze(-1)).sum(dim=(1, 2))

        if order == 1:
            diff = mu_t[:, 1:] - mu_t[:, :-1]
            diff_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        else:
            diff = mu_t[:, 2:] - 2.0 * mu_t[:, 1:-1] + mu_t[:, :-2]
            diff_mask = valid_mask[:, 2:] & valid_mask[:, 1:-1] & valid_mask[:, :-2]

        num_diffs = int(diff_mask.sum().item())
        if num_diffs <= 0:
            return {
                "temporal_diff_loss": zero,
                "temporal_diff_ratio": zero,
                "temporal_diff_mean_l2": zero,
                "temporal_diff_active_ratio": zero,
                "temporal_diff_hinge_active_ratio": zero,
                "num_frames": num_frames,
                "num_diffs": 0,
            }

        diff_energy_per_sample = (diff.pow(2) * diff_mask.unsqueeze(-1)).sum(dim=(1, 2))
        diff_count_per_sample = diff_mask.sum(dim=1)
        valid_sample_mask = diff_count_per_sample > 0
        diff_energy = diff_energy_per_sample.sum()
        energy = energy_per_sample.sum()
        temporal_diff_ratio = diff_energy / energy.clamp_min(float(eps))
        temporal_diff_mean_l2 = diff_energy / (diff_mask.sum().to(diff_energy.dtype) * float(dim)).clamp_min(1.0)
        temporal_diff_active_ratio = diff_mask.float().mean()
        temporal_diff_hinge_active_ratio = zero

        if loss_mode == "global_ratio":
            temporal_diff_loss = temporal_diff_ratio
        else:
            per_sample_ratio = diff_energy_per_sample / energy_per_sample.clamp_min(float(eps))
            per_sample_ratio = per_sample_ratio[valid_sample_mask]
            if sample_gate is None:
                gate = None
            else:
                gate = sample_gate.detach().to(device=mu_t.device, dtype=per_sample_ratio.dtype).view(-1)
                if gate.numel() != bsz:
                    raise ValueError(f"sample_gate shape mismatch: expected {bsz}, got {gate.numel()}")
                gate = gate.clamp_min(0.0)[valid_sample_mask]

            if loss_mode == "per_sample_ratio":
                per_sample_loss = per_sample_ratio
            else:
                per_sample_loss = F.relu(per_sample_ratio - float(hinge_target)).pow(2)
                temporal_diff_hinge_active_ratio = (per_sample_ratio > float(hinge_target)).float().mean()

            if gate is not None:
                temporal_diff_loss = (per_sample_loss * gate).mean()
            else:
                temporal_diff_loss = per_sample_loss.mean()
            temporal_diff_ratio = per_sample_ratio.mean()

    return {
        "temporal_diff_loss": temporal_diff_loss,
        "temporal_diff_ratio": temporal_diff_ratio,
        "temporal_diff_mean_l2": temporal_diff_mean_l2,
        "temporal_diff_active_ratio": temporal_diff_active_ratio,
        "temporal_diff_hinge_active_ratio": temporal_diff_hinge_active_ratio,
        "num_frames": num_frames,
        "num_diffs": num_diffs,
    }


def get_cached_target_light_mel(task, wavs: torch.Tensor, light_mel_cache: Optional[dict] = None):
    if light_mel_cache is not None and "target" in light_mel_cache:
        return light_mel_cache["target"]
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            light_mel = task.interp_pair_mel_net(wavs.float()).detach()
    if light_mel_cache is not None:
        light_mel_cache["target"] = light_mel
    return light_mel


def compute_mel_flux_transient_gate(
    light_mel: torch.Tensor,
    alpha: float = 1.0,
    threshold: float = 1.0,
    floor: float = 0.25,
    eps: float = 1e-6,
):
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            light_mel = light_mel.float()
            bsz = int(light_mel.shape[0])
            if light_mel.shape[1] <= 1:
                flux = light_mel.new_zeros((bsz,))
                flux_norm = flux
                gate = light_mel.new_ones((bsz,))
            else:
                flux = F.relu(light_mel[:, 1:] - light_mel[:, :-1]).mean(dim=(1, 2))
                flux_norm = flux / flux.mean().clamp_min(float(eps))
                gate = 1.0 / (1.0 + float(alpha) * F.relu(flux_norm - float(threshold)))
                gate = gate.clamp(min=float(floor), max=1.0)

    return gate.detach(), {
        "monitor/latent_delta_gate_mean": gate.mean().detach(),
        "monitor/latent_delta_gate_min": gate.min().detach(),
        "monitor/latent_delta_gate_flux_mean": flux.mean().detach(),
        "monitor/latent_delta_gate_flux_norm_mean": flux_norm.mean().detach(),
    }


def get_interp_local_group(task) -> Optional[dist.ProcessGroup]:
    if not dist.is_available() or not dist.is_initialized():
        return None
    if task._interp_local_group_initialized:
        return task._interp_local_group

    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if local_world_size is not None:
        local_world_size = int(local_world_size)
    elif getattr(task, 'trainer', None) is not None and hasattr(task.trainer, 'all_gpu_ids'):
        local_world_size = len(task.trainer.all_gpu_ids)
    else:
        local_world_size = torch.cuda.device_count()
    if local_world_size <= 1:
        task._interp_local_group_ranks = [dist.get_rank()]
        task._interp_local_group_initialized = True
        return None

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    node_starts = list(range(0, world_size, local_world_size))
    for node_start in node_starts:
        ranks = list(range(node_start, min(node_start + local_world_size, world_size)))
        group = dist.new_group(ranks=ranks)
        if node_start == rank - local_rank:
            task._interp_local_group = group
            task._interp_local_group_ranks = ranks
    if task._interp_local_group_ranks is None:
        raise RuntimeError(
            f"failed to build interp local group for rank={rank}, "
            f"local_rank={local_rank}, local_world_size={local_world_size}, world_size={world_size}"
        )
    task._interp_local_group_initialized = True
    return task._interp_local_group


def all_gather_local_fixed_tensor(tensor: torch.Tensor, group, tag: str):
    if group is None or not dist.is_available() or not dist.is_initialized():
        return tensor.contiguous()

    tensor = tensor.contiguous()
    group_world_size = dist.get_world_size(group=group)
    if group_world_size <= 1:
        return tensor

    local_shape = torch.tensor(tensor.shape, device=tensor.device, dtype=torch.long)
    gathered_shapes = [torch.empty_like(local_shape) for _ in range(group_world_size)]
    dist.all_gather(gathered_shapes, local_shape, group=group)
    expected_shape = tuple(tensor.shape)
    if any(tuple(shape.tolist()) != expected_shape for shape in gathered_shapes):
        print_once(f"| WARN: skip interp_mid_loss because `{tag}` shape mismatch across local group")
        return None

    gathered = [torch.empty_like(tensor) for _ in range(group_world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=0)


def compute_interp_mid_losses(
    task,
    wavs: torch.Tensor,
    model_outputs: dict,
    light_mel: Optional[torch.Tensor] = None,
    light_mel_cache: Optional[dict] = None,
):
    zero = wavs.new_zeros(())
    lambda_interp_mid = float(hparams.get('lambda_interp_mid', 0.0))
    if lambda_interp_mid <= 0.0:
        return None

    mu = model_outputs['mu'].transpose(1, 2).contiguous()  # [B, T, C]
    local_bs = mu.shape[0]
    if local_bs <= 1 and (not dist.is_available() or not dist.is_initialized()):
        return {
            'interp_mid': zero,
            'monitor/interp_pair_valid_ratio': zero,
            'monitor/interp_pair_escape_ratio': zero,
            'monitor/interp_pair_sim_mean': zero,
        }

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            if light_mel is None:
                light_mel = get_cached_target_light_mel(task, wavs, light_mel_cache)
            else:
                light_mel = light_mel.detach()
            mel_mean = light_mel.mean(dim=1)
            mel_std = light_mel.std(dim=1, unbiased=False)
            pair_key = torch.cat([mel_mean, mel_std], dim=-1)
            pair_key = F.normalize(pair_key, dim=-1, eps=1e-6)

    local_group = get_interp_local_group(task)
    pair_key_all = all_gather_local_fixed_tensor(pair_key.detach(), local_group, "interp_pair_key")
    mu_all = all_gather_local_fixed_tensor(mu.detach(), local_group, "interp_pair_mu")
    light_mel_all = all_gather_local_fixed_tensor(light_mel.detach(), local_group, "interp_pair_mel")
    if pair_key_all is None or mu_all is None or light_mel_all is None:
        return {
            'interp_mid': zero,
            'monitor/interp_pair_valid_ratio': zero,
            'monitor/interp_pair_escape_ratio': zero,
            'monitor/interp_pair_sim_mean': zero,
        }

    if local_group is None or task._interp_local_group_ranks is None:
        local_rank_idx = 0
    else:
        local_rank_idx = task._interp_local_group_ranks.index(dist.get_rank())
    self_offset = local_rank_idx * local_bs
    total_candidates = pair_key_all.shape[0]

    sims = torch.matmul(pair_key.detach(), pair_key_all.transpose(0, 1))
    topk = max(int(hparams.get('interp_pair_topk', 4)), 1)
    temperature = max(float(hparams.get('interp_pair_temperature', 0.2)), 1e-6)
    escape_prob = min(max(float(hparams.get('interp_pair_escape_prob', 0.1)), 0.0), 1.0)

    valid_anchor_indices = []
    partner_indices = []
    chosen_sims = []
    escape_count = 0

    for local_idx in range(local_bs):
        self_idx = self_offset + local_idx
        valid_indices = torch.arange(total_candidates, device=mu.device, dtype=torch.long)
        valid_indices = valid_indices[valid_indices != self_idx]
        if valid_indices.numel() <= 0:
            continue

        if random.random() < escape_prob:
            picked = valid_indices[torch.randint(valid_indices.numel(), size=(1,), device=mu.device)].item()
            escape_count += 1
        else:
            valid_sims = sims[local_idx, valid_indices]
            actual_topk = min(topk, valid_indices.numel())
            top_vals, top_pos = torch.topk(valid_sims, k=actual_topk, largest=True)
            top_indices = valid_indices[top_pos]
            top_weights = torch.softmax(top_vals / temperature, dim=0)
            picked = top_indices[torch.multinomial(top_weights, num_samples=1)].item()

        valid_anchor_indices.append(local_idx)
        partner_indices.append(picked)
        chosen_sims.append(sims[local_idx, picked].detach())

    if len(valid_anchor_indices) <= 0:
        return {
            'interp_mid': zero,
            'monitor/interp_pair_valid_ratio': zero,
            'monitor/interp_pair_escape_ratio': zero,
            'monitor/interp_pair_sim_mean': zero,
        }

    anchor_idx = torch.tensor(valid_anchor_indices, device=mu.device, dtype=torch.long)
    partner_idx = torch.tensor(partner_indices, device=mu.device, dtype=torch.long)
    mu_mid = 0.5 * (mu[anchor_idx] + mu_all[partner_idx])

    model_gen_core = get_model_core(task.model_gen)
    y_mid = model_gen_core.decode(mu_mid)
    with torch.cuda.amp.autocast(enabled=False):
        mel_mid_pred = task.interp_pair_mel_net(y_mid.squeeze(1))
        mel_mid_tgt = 0.5 * (light_mel[anchor_idx] + light_mel_all[partner_idx])    # heuristic, not mel_mid_tgt = log10(0.5 * (10**mel_a + 10**mel_b))
        interp_mid = F.l1_loss(mel_mid_pred, mel_mid_tgt) * lambda_interp_mid

    interp_pair_valid_ratio = mu.new_tensor(len(valid_anchor_indices) / max(local_bs, 1))
    interp_pair_escape_ratio = mu.new_tensor(escape_count / max(len(valid_anchor_indices), 1))
    interp_pair_sim_mean = torch.stack(chosen_sims).mean() if len(chosen_sims) > 0 else zero
    return {
        'interp_mid': interp_mid,
        'monitor/interp_pair_valid_ratio': interp_pair_valid_ratio.detach(),
        'monitor/interp_pair_escape_ratio': interp_pair_escape_ratio.detach(),
        'monitor/interp_pair_sim_mean': interp_pair_sim_mean.detach(),
    }


class WavVAETask(FastDatasetMixin, BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = import_module_bystr(hparams['dataset_cls'])
        self.hparams = hparams
        self.config = AttrDict(hparams)
        self._reinit_module_names = []
        self._fresh_model_gen_state = {}
        self._interp_local_group = None
        self._interp_local_group_initialized = False
        self._interp_local_group_ranks = None

    def build_model(self):
        if hparams.get('model_version', 'v10') == 'v10':
            from modules.vae.wavvae_v10 import build_wavvae
        elif hparams.get('model_version', 'v11') == 'v11':
            from modules.vae.wavvae_v11 import build_wavvae
        elif hparams.get('model_version', 'v12') == 'v12':
            from modules.vae.wavvae_v12 import build_wavvae
        elif hparams.get('model_version', 'v13') == 'v13':
            from modules.vae.wavvae_v13 import build_wavvae
        self.model_gen = build_wavvae(hparams)
        self._reinit_module_names = parse_reinit_modules_from_hparams()
        if self._reinit_module_names:
            model_gen_core = get_model_core(self.model_gen)
            self._fresh_model_gen_state = snapshot_named_modules(model_gen_core, self._reinit_module_names)
            print_once(f"| Will reinit model_gen submodules after load: {self._reinit_module_names}")
        else:
            self._fresh_model_gen_state = {}

        if torch.__version__.split(".")[0] == '2' and hparams.get("torch_compile", False):
            self.model_gen = torch.compile(
                self.model_gen, backend='inductor', mode='default',
                dynamic=False, fullgraph=False,
            )

        self._maybe_freeze_model_gen_modules()

        self.model_disc = torch.nn.ModuleDict()
        self.num_disc = 0
        self.model_disc['mpd'] = MultiPeriodDiscriminator(hparams['mpd'], use_cond=hparams['use_cond_disc'])
        self.num_disc += 1
        if hparams['use_msd']:
            self.model_disc['msd'] = MultiScaleDiscriminator(use_cond=hparams['use_cond_disc'])
            self.num_disc += 1
        if hparams['use_mrd']:
            self.model_disc['mrd'] = MultiResolutionDiscriminator(hparams)
            self.num_disc += 1
        load_ckpt_disc = hparams.get('load_ckpt_disc')
        if load_ckpt_disc:
            load_ckpt(self.model_disc, load_ckpt_disc, 'model_disc', force=True, strict=True)

        self.stft_loss = MultiResolutionSTFTLoss(**hparams.get('stft_loss_params', {}))
        self.transient_stft_loss = None
        if hparams.get('lambda_transient_stft', 0.0) > 0.0:
            transient_params = dict(hparams.get('transient_stft_params', {}))
            transient_params.setdefault('use_log_mag', hparams.get('transient_stft_use_log_mag', True))
            transient_params.setdefault('positive_only', hparams.get('transient_stft_positive_only', True))
            transient_params.setdefault('onset_weight', hparams.get('transient_stft_onset_weight', 0.0))
            transient_params.setdefault('loss_type', hparams.get('transient_stft_loss_type', 'l1'))
            transient_params.setdefault('eps', hparams.get('transient_stft_eps', 1e-7))
            if 'resolutions' not in transient_params:
                default_resolutions = hparams.get('stft_loss_params', {}).get('resolutions')
                if default_resolutions is not None:
                    transient_params['resolutions'] = default_resolutions
            self.transient_stft_loss = MultiResolutionTransientSTFTLoss(**transient_params)
            print_once(f"Use MultiResolutionTransientSTFTLoss: {transient_params}")
        
        if hparams.get('use_mr_mel', False):
            mel_params = hparams.get('mr_mel')
            mel_params = [dict(zip(list(mel_params.keys()), row)) for row in zip(*list(mel_params.values()))]
            mel_params = [{**d, 'audio_sample_rate': hparams['audio_sample_rate']} for d in mel_params]
            if len(hparams.get('mel_band_edges_hz', [])) > 0:
                mel_band_edges_hz = hparams.get('mel_band_edges_hz', [])
                mel_band_weights = hparams.get('mel_band_weights', [])
                assert len(mel_band_edges_hz) == len(mel_band_weights) + 1
                self.mel_loss = MultiResolutionMultiBandMelLoss(mel_params, mel_band_edges_hz, mel_band_weights)
                print(f"Use MultiResolutionMultiBandMelLoss: {mel_params}, {self.mel_loss.band_slices_per_melnet = }")
            else:
                self.mel_loss = MultiResolutionMelLoss(mel_params)
                print(f"Use MultiResolutionMelLoss: {mel_params}")
        else:
            self.mel_loss = MultiResolutionMelLoss([hparams])

        interp_mel_hparams = dict(hparams)
        interp_mel_hparams['audio_num_mel_bins'] = int(hparams.get('interp_pair_mel_bins', 80))
        self.interp_pair_mel_net = MelNet(interp_mel_hparams)
        if hparams.get('lambda_interp_mid', 0.0) > 0.0:
            print_once(
                "| Enable interp_mid_loss: "
                f"mel_bins={interp_mel_hparams['audio_num_mel_bins']}, "
                f"topk={hparams.get('interp_pair_topk', 4)}, "
                f"temperature={hparams.get('interp_pair_temperature', 0.2)}, "
                f"escape_prob={hparams.get('interp_pair_escape_prob', 0.1)}"
            )

        return {'trainable': [self.model_gen, self.model_disc], 'others': []}
    
    def _maybe_freeze_model_gen_modules(self):
        if not hparams.get('freeze_encoder', False):
            return
        model_gen_core = get_model_core(self.model_gen)
        frozen_modules = []
        for module_name in ('encoder', 'bottleneck'):
            if not hasattr(model_gen_core, module_name):
                raise ValueError(f"model_gen has no submodule named `{module_name}` to freeze")
            getattr(model_gen_core, module_name).requires_grad_(False)
            frozen_modules.append(module_name)
        print_once(f"| Freeze model_gen submodules: {frozen_modules}")

    def _get_trainable_model_gen_parameters(self):
        return [p for p in self.model_gen.parameters() if p.requires_grad]

    def _maybe_reinit_model_gen_submodules_after_load(self):
        if not self._reinit_module_names:
            return
        model_gen_core = get_model_core(self.model_gen)
        reinit_done = []
        for module_name in self._reinit_module_names:
            if not hasattr(model_gen_core, module_name):
                raise ValueError(f"model_gen has no submodule named `{module_name}` to reinit")
            fresh_state = self._fresh_model_gen_state.get(module_name)
            if fresh_state is None:
                raise ValueError(f"missing fresh init snapshot for `{module_name}`")
            getattr(model_gen_core, module_name).load_state_dict(fresh_state, strict=True)
            reinit_done.append(module_name)
        print_once(f"| Reinitialized model_gen submodules after ckpt load: {reinit_done}")

    def load_model(self):
        if hparams.get('load_ckpt', '') != '':
            load_ckpt(self.model_gen, hparams['load_ckpt'], 'model_gen', strict=False)
            self._maybe_reinit_model_gen_submodules_after_load()
            if hparams.get('do_load_ckpt_disc', False):
                try:
                    load_ckpt(self.model_disc, hparams['load_ckpt'], 'model_disc', strict=False)
                except:
                    traceback.print_exc()
        elif self._reinit_module_names:
            print_once("| Skip reinit_modules_after_load because load_ckpt is empty")

    def _in_disc_only_warmup(self):
        disc_start_steps = int(hparams.get('disc_start_steps', 0))
        disc_warmup_steps = int(hparams.get('disc_warmup_steps', 0))
        return disc_warmup_steps > 0 and disc_start_steps <= self.global_step < (disc_start_steps + disc_warmup_steps)

    def _sample_rope_position_offset(self, wavs: torch.Tensor) -> torch.Tensor:
        batch_size = int(wavs.shape[0])
        device = wavs.device
        offset = torch.zeros(batch_size, device=device, dtype=torch.long)
        if (not self.training) or (not hparams.get('rope_random_position_offset', False)):
            return offset

        max_sec = float(hparams.get('rope_random_position_offset_max_sec', 0.0) or 0.0)
        if max_sec <= 0.0:
            return offset

        model_gen_core = get_model_core(self.model_gen)
        hop_length = int(getattr(model_gen_core, 'hop_length'))
        max_position_embeddings = int(model_gen_core.encoder.transformer.config.max_position_embeddings)
        seq_len_lat = max(1, (int(wavs.shape[-1]) + hop_length - 1) // hop_length)
        max_offset_from_sec = int(math.floor(max_sec * hparams['audio_sample_rate'] / hop_length))
        max_offset = min(max_offset_from_sec, max(max_position_embeddings - seq_len_lat, 0))
        if max_offset <= 0:
            return offset
        return torch.randint(0, max_offset + 1, (batch_size,), device=device, dtype=torch.long)

    def _build_model_gen_train_kwargs(self, sample, wavs: torch.Tensor):
        wav_lengths = sample.get('wav_lengths')
        if wav_lengths is not None:
            wav_lengths = wav_lengths.to(device=wavs.device, dtype=torch.long)
        pos_offset = torch.zeros([], device=wavs.device, dtype=torch.long)
        if hparams.get('transformer_n_layers', 2) > 0:
            pos_offset = self._sample_rope_position_offset(wavs)
        return {
            'audio_lengths': wav_lengths,
            'pos_offset': pos_offset,
        }, {
            'monitor/rope_pos_offset_mean_frames': pos_offset.float().mean().detach(),
            'monitor/rope_pos_offset_max_frames': pos_offset.max().detach(),
        }

    def build_optimizer(self):
        gen_params = self._get_trainable_model_gen_parameters()
        if len(gen_params) == 0:
            raise ValueError("model_gen has no trainable parameters")
        optimizer_gen = torch.optim.AdamW(gen_params, lr=hparams['lr'],
                                        betas=[hparams['adam_b1'], hparams['adam_b2']])

        optimizer_disc = torch.optim.AdamW(self.model_disc.parameters(),
                                        lr=hparams.get('disc_lr', hparams['lr']),
                                        betas=[hparams['adam_b1'], hparams['adam_b2']])
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return (
            WarmupSchedule(
                optimizer[0], lr=hparams['lr'], warmup_updates=hparams.get('warmup_updates', 0)
            ),
            WarmupSchedule(
                optimizer[1], lr=hparams.get('disc_lr', hparams['lr']), warmup_updates=hparams.get('warmup_updates', 0)
            ),
        )

    def _training_step(self, sample, batch_idx, optimizer_idx):
        if self.trainer.proc_rank_local == 0 and random.random() < 0.0001:
            kill_void()

        sample['wavs'] = sample['wavs'].float()

        if self.trainer.proc_rank == 0 and self.global_step < 20:
            save_dir = f'{hparams["work_dir"]}/sample_batches/step_{self.global_step}'
            os.makedirs(save_dir, exist_ok=True)
            for i in range(len(sample['wavs'])):
                sf.write(f"{save_dir}/{i}.wav", sample['wavs'][i].cpu().numpy(), hparams['audio_sample_rate'], 'PCM_16')

        y = sample['wavs']
        loss_output = {}
        if optimizer_idx == 0:
            model_gen_kwargs, rope_monitor = self._build_model_gen_train_kwargs(sample, y)
            #######################
            #      Generator      #
            #######################
            if self.training and self._in_disc_only_warmup():
                if self.global_step == int(hparams.get('disc_start_steps', 0)):
                    print_once(
                        f"| Enter disc-only warmup for {hparams.get('disc_warmup_steps', 0)} steps "
                        f"(global_step in [{hparams.get('disc_start_steps', 0)}, "
                        f"{hparams.get('disc_start_steps', 0) + hparams.get('disc_warmup_steps', 0)}))"
                    )
                with torch.no_grad():
                    model_outputs = self.model_gen(y, **model_gen_kwargs)
                    self.y_ = model_outputs['recon'].detach()
                return None

            model_outputs = self.model_gen(y, **model_gen_kwargs)
            loss_output.update(rope_monitor)

            y_ = model_outputs['recon']     # [B, 1, T]
            y = y.unsqueeze(1)
            with torch.no_grad():
                y_monitor = torch.nan_to_num(y_.detach().float(), nan=0.0, posinf=1.0e6, neginf=-1.0e6)
                y_abs = y_monitor.abs()
                y_target_monitor = y.detach().float()
                loss_output['monitor/recon_abs_max'] = y_abs.amax()
                loss_output['monitor/recon_abs_mean'] = y_abs.mean()
                loss_output['monitor/recon_rms'] = y_monitor.pow(2).mean().sqrt()
                loss_output['monitor/recon_mean'] = y_monitor.mean()
                loss_output['monitor/recon_out_of_range_ratio'] = (y_abs > 1.0).float().mean()
                loss_output['monitor/recon_nonfinite_ratio'] = (~torch.isfinite(y_.detach())).float().mean()
                loss_output['monitor/target_rms'] = y_target_monitor.pow(2).mean().sqrt()
            # y_mel = mel_spectrogram(y.squeeze(1), hparams).transpose(1, 2)
            # y_hat_mel = mel_spectrogram(y_.squeeze(1), hparams).transpose(1, 2)
            # loss_output['mel'] = F.l1_loss(y_hat_mel, y_mel) * hparams['lambda_mel']

            with torch.cuda.amp.autocast(enabled=False):
                loss_output['mel'] = self.mel_loss(y_.squeeze(1), y.squeeze(1)) * hparams['lambda_mel']

                if hparams['use_ms_stft']:
                    sc, mag = self.stft_loss(y_.squeeze(1), y.squeeze(1))
                    loss_output['sc'] = sc * hparams['lambda_stft'] / 2
                    loss_output['mag'] = mag * hparams['lambda_stft'] / 2
                if self.transient_stft_loss is not None:
                    loss_output['transient_stft'] = self.transient_stft_loss(
                        y_.squeeze(1), y.squeeze(1)
                    ) * hparams['lambda_transient_stft']

            if hparams.get('lambda_energy', 0.0) > 0.0:
                y_energy = short_time_energy(y.squeeze(1), sample_rate=hparams['audio_sample_rate'], use_log_db=True)
                y_hat_energy = short_time_energy(y_.squeeze(1), sample_rate=hparams['audio_sample_rate'], use_log_db=True)
                loss_output['energy'] = F.l1_loss(y_hat_energy, y_energy) * hparams['lambda_energy']

            light_mel_cache = {}
            interp_mid_losses = compute_interp_mid_losses(
                self,
                sample['wavs'],
                model_outputs,
                light_mel_cache=light_mel_cache,
            )
            if interp_mid_losses is not None:
                loss_output.update(interp_mid_losses)

            if self.training and self.global_step >= hparams.get('disc_start_steps', 0):
                _, y_p_hat_g, fmap_f_r, fmap_f_g = unwrap_model(self.model_disc)['mpd'](y, y_, None)
                loss_output['a_p'] = hparams['lambda_adv'] * generator_loss(y_p_hat_g) * hparams.get('lambda_mpd', 1.0) / self.num_disc
                loss_output['a_p_fm'] = feature_loss(fmap_f_r, fmap_f_g) * hparams.get('lambda_fm', 0.0) / self.num_disc
                if hparams['use_msd']:
                    _, y_s_hat_g, fmap_s_r, fmap_s_g = unwrap_model(self.model_disc)['msd'](y, y_, None)
                    loss_output['a_s'] = hparams['lambda_adv'] * generator_loss(y_s_hat_g) * hparams.get('lambda_msd', 1.0) / self.num_disc
                    loss_output['a_s_fm'] = feature_loss(fmap_s_r, fmap_s_g) * hparams.get('lambda_fm', 0.0) / self.num_disc
                if hparams['use_mrd']:
                    mrd_res = unwrap_model(self.model_disc)['mrd'](y_)
                    y_r_hat_g = [x[1] for x in mrd_res]
                    fmap_r_g = [x[0] for x in mrd_res]
                    fmap_r_r = [x[0] for x in unwrap_model(self.model_disc)['mrd'](y)]
                    loss_output['a_r'] = hparams['lambda_adv'] * generator_loss(y_r_hat_g) * hparams.get('lambda_mrd', 1.0) / self.num_disc
                    loss_output['a_r_fm'] = feature_loss(fmap_r_r, fmap_r_g) * hparams.get('lambda_fm', 0.0) / self.num_disc

            kl_start_steps = hparams.get('kl_start_steps', 0)
            if self.global_step >= kl_start_steps:
                if 0 < self.global_step - kl_start_steps < hparams.get('kl_annealing_step', 0):
                    lambda_kl = hparams.get('lambda_kl', 0.001) * (self.global_step - kl_start_steps) / hparams.get('kl_annealing_step', 0)
                else:
                    lambda_kl = hparams.get('lambda_kl', 0.001)
                loss_output['kl_loss'] = model_outputs['kl'] * lambda_kl
                loss_output['monitor/lambda_kl'] = lambda_kl

            latent_cov_stats_cache = {}
            latent_eig_metrics_cache = {}
            latent_temporal_reg_cache = {}

            def get_cached_cov_stats(max_frames):
                cache_key = -1 if max_frames is None else int(max_frames)
                if cache_key not in latent_cov_stats_cache:
                    latent_cov_stats_cache[cache_key] = compute_centered_latent_covariance(
                        model_outputs['mu'],
                        model_outputs.get('z_lens'),
                        max_frames=max_frames,
                    )
                return latent_cov_stats_cache[cache_key]

            def get_cached_eig_metrics(max_frames, eps):
                cache_key = (-1 if max_frames is None else int(max_frames), float(eps))
                if cache_key not in latent_eig_metrics_cache:
                    cov_stats = get_cached_cov_stats(max_frames)
                    cov = cov_stats["cov"]
                    if cov is None:
                        latent_eig_metrics_cache[cache_key] = None
                    else:
                        latent_eig_metrics_cache[cache_key] = compute_normalized_covariance_eig_metrics(cov, eps=eps)
                return latent_eig_metrics_cache[cache_key]

            def get_cached_temporal_reg(
                order,
                max_frames,
                eps,
                loss_mode="global_ratio",
                hinge_target=0.0,
                sample_gate=None,
            ):
                cache_key = (
                    int(order),
                    -1 if max_frames is None else int(max_frames),
                    float(eps),
                    str(loss_mode),
                    float(hinge_target),
                    0 if sample_gate is None else id(sample_gate),
                )
                if cache_key not in latent_temporal_reg_cache:
                    latent_temporal_reg_cache[cache_key] = compute_latent_temporal_diff_reg_losses(
                        model_outputs['mu'],
                        model_outputs.get('z_lens'),
                        order=order,
                        max_frames=max_frames,
                        eps=eps,
                        loss_mode=loss_mode,
                        hinge_target=hinge_target,
                        sample_gate=sample_gate,
                    )
                return latent_temporal_reg_cache[cache_key]

            latent_sphere_reg_weight = hparams.get('latent_sphere_reg_weight', 0.0)
            if latent_sphere_reg_weight > 0.0:
                sphere_max_frames = hparams.get('latent_sphere_reg_max_frames', 8192)
                sphere_losses = compute_latent_sphere_reg_losses(
                    model_outputs['mu'],
                    model_outputs.get('z_lens'),
                    max_frames=sphere_max_frames,
                    target_var=hparams.get('latent_sphere_reg_target_var', 1.0),
                    cov_stats=get_cached_cov_stats(sphere_max_frames),
                )
                mean_w = latent_sphere_reg_weight * hparams.get('latent_sphere_reg_mean_weight', 1.0)
                var_w = latent_sphere_reg_weight * hparams.get('latent_sphere_reg_var_weight', 1.0)
                cov_w = latent_sphere_reg_weight * hparams.get('latent_sphere_reg_cov_weight', 1.0)
                if mean_w > 0:
                    loss_output['sphere_mean'] = sphere_losses['sphere_mean'] * mean_w
                if var_w > 0:
                    loss_output['sphere_var'] = sphere_losses['sphere_var'] * var_w
                if cov_w > 0:
                    loss_output['sphere_cov'] = sphere_losses['sphere_cov'] * cov_w

            latent_logdet_reg_weight = hparams.get('latent_logdet_reg_weight', 0.0)
            if latent_logdet_reg_weight > 0.0:
                logdet_max_frames = hparams.get('latent_logdet_max_frames', 8192)
                logdet_eps = hparams.get('latent_logdet_eps', 1e-4)
                logdet_losses = compute_latent_logdet_reg_losses(
                    model_outputs['mu'],
                    model_outputs.get('z_lens'),
                    max_frames=logdet_max_frames,
                    cov_stats=get_cached_cov_stats(logdet_max_frames),
                    eig_metrics=get_cached_eig_metrics(logdet_max_frames, logdet_eps),
                    eps=logdet_eps,
                )
                loss_output['latent_logdet'] = logdet_losses['norm_logdet_barrier'] * latent_logdet_reg_weight
                loss_output['monitor/latent_norm_logdet_per_dim'] = logdet_losses['norm_logdet_per_dim'].detach()
                loss_output['monitor/latent_norm_min_eig'] = logdet_losses['norm_min_eig'].detach()
                loss_output['monitor/latent_norm_max_eig'] = logdet_losses['norm_max_eig'].detach()
                loss_output['monitor/latent_norm_cond_est'] = logdet_losses['norm_cond_est'].detach()

            latent_small_eig_reg_weight = hparams.get('latent_small_eig_reg_weight', 0.0)
            if latent_small_eig_reg_weight > 0.0:
                small_eig_max_frames = hparams.get('latent_small_eig_max_frames', 8192)
                small_eig_eps = hparams.get('latent_small_eig_eps', 1e-4)
                small_eig_losses = compute_latent_small_eig_reg_losses(
                    model_outputs['mu'],
                    model_outputs.get('z_lens'),
                    floor=hparams.get('latent_small_eig_floor', 0.1),
                    max_frames=small_eig_max_frames,
                    cov_stats=get_cached_cov_stats(small_eig_max_frames),
                    eig_metrics=get_cached_eig_metrics(small_eig_max_frames, small_eig_eps),
                    eps=small_eig_eps,
                    active_only_avg=hparams.get('latent_small_eig_active_only_avg', False),
                )
                loss_output['latent_small_eig'] = small_eig_losses['small_eig_loss'] * latent_small_eig_reg_weight
                loss_output['monitor/latent_small_eig_active_ratio'] = small_eig_losses['small_eig_active_ratio'].detach()
                loss_output['monitor/latent_norm_min_eig'] = small_eig_losses['norm_min_eig'].detach()
                loss_output['monitor/latent_norm_max_eig'] = small_eig_losses['norm_max_eig'].detach()
                loss_output['monitor/latent_norm_cond_est'] = small_eig_losses['norm_cond_est'].detach()

            latent_min_std_reg_weight = hparams.get('latent_min_std_reg_weight', 0.0)
            if latent_min_std_reg_weight > 0.0:
                min_std_max_frames = hparams.get('latent_min_std_max_frames', 8192)
                min_std_losses = compute_latent_min_std_reg_losses(
                    model_outputs['mu'],
                    model_outputs.get('z_lens'),
                    floor=hparams.get('latent_min_std_floor', 0.3),
                    max_frames=min_std_max_frames,
                    cov_stats=get_cached_cov_stats(min_std_max_frames),
                    eps=hparams.get('latent_min_std_eps', 1e-6),
                )
                loss_output['latent_min_std'] = min_std_losses['min_std_loss'] * latent_min_std_reg_weight
                loss_output['monitor/latent_min_std'] = min_std_losses['latent_min_std'].detach()
                loss_output['monitor/latent_mean_std'] = min_std_losses['latent_mean_std'].detach()
                loss_output['monitor/latent_low_std_ratio'] = min_std_losses['latent_low_std_ratio'].detach()

            latent_delta_reg_weight = hparams.get('latent_delta_reg_weight', 0.0)
            if latent_delta_reg_weight > 0.0:
                delta_max_frames = hparams.get('latent_delta_max_frames', 8192)
                delta_loss_mode = hparams.get('latent_delta_loss_mode', 'global_ratio')
                delta_hinge_target = hparams.get('latent_delta_hinge_target', 0.0)
                delta_sample_gate = None
                if hparams.get('latent_delta_use_transient_gate', False):
                    delta_gate_metric = hparams.get('latent_delta_gate_metric', 'mel_flux')
                    if delta_gate_metric != 'mel_flux':
                        raise ValueError(f"unsupported latent_delta_gate_metric={delta_gate_metric}")
                    light_mel = get_cached_target_light_mel(self, sample['wavs'], light_mel_cache)
                    delta_sample_gate, delta_gate_monitors = compute_mel_flux_transient_gate(
                        light_mel,
                        alpha=hparams.get('latent_delta_gate_alpha', 1.0),
                        threshold=hparams.get('latent_delta_gate_threshold', 1.0),
                        floor=hparams.get('latent_delta_gate_floor', 0.25),
                        eps=hparams.get('latent_delta_gate_eps', 1e-6),
                    )
                    loss_output.update(delta_gate_monitors)
                delta_losses = get_cached_temporal_reg(
                    order=1,
                    max_frames=delta_max_frames,
                    eps=hparams.get('latent_delta_eps', 1e-6),
                    loss_mode=delta_loss_mode,
                    hinge_target=delta_hinge_target,
                    sample_gate=delta_sample_gate,
                )
                loss_output['latent_delta'] = delta_losses['temporal_diff_loss'] * latent_delta_reg_weight
                loss_output['monitor/latent_delta_ratio'] = delta_losses['temporal_diff_ratio'].detach()
                loss_output['monitor/latent_delta_mean_l2'] = delta_losses['temporal_diff_mean_l2'].detach()
                loss_output['monitor/latent_delta_active_ratio'] = delta_losses['temporal_diff_active_ratio'].detach()
                loss_output['monitor/latent_delta_hinge_active_ratio'] = delta_losses['temporal_diff_hinge_active_ratio'].detach()

            latent_delta2_reg_weight = hparams.get('latent_delta2_reg_weight', 0.0)
            if latent_delta2_reg_weight > 0.0:
                delta2_max_frames = hparams.get('latent_delta2_max_frames', 8192)
                delta2_loss_mode = hparams.get('latent_delta2_loss_mode', 'global_ratio')
                delta2_losses = get_cached_temporal_reg(
                    order=2,
                    max_frames=delta2_max_frames,
                    eps=hparams.get('latent_delta2_eps', 1e-6),
                    loss_mode=delta2_loss_mode,
                )
                loss_output['latent_delta2'] = delta2_losses['temporal_diff_loss'] * latent_delta2_reg_weight
                loss_output['monitor/latent_delta2_ratio'] = delta2_losses['temporal_diff_ratio'].detach()
                loss_output['monitor/latent_delta2_mean_l2'] = delta2_losses['temporal_diff_mean_l2'].detach()
                loss_output['monitor/latent_delta2_active_ratio'] = delta2_losses['temporal_diff_active_ratio'].detach()

            if hparams.get('latent_geometry_monitor', False) and self.global_step % hparams.get('latent_geometry_monitor_step', 100) == 0:
                geometry_max_frames = hparams.get('latent_geometry_monitor_max_frames', 8192)
                geometry_eps = hparams.get('latent_geometry_monitor_eps', 1e-4)
                geometry_metrics = compute_latent_geometry_monitor_metrics(
                    model_outputs['mu'],
                    model_outputs.get('z_lens'),
                    max_frames=geometry_max_frames,
                    cov_stats=get_cached_cov_stats(geometry_max_frames),
                    eig_metrics=get_cached_eig_metrics(geometry_max_frames, geometry_eps),
                    eps=geometry_eps,
                )
                loss_output['monitor/latent_norm_logdet_per_dim'] = geometry_metrics['latent_norm_logdet_per_dim'].detach()
                loss_output['monitor/latent_norm_erank_ratio'] = geometry_metrics['latent_norm_erank_ratio'].detach()
                loss_output['monitor/latent_norm_cond_est'] = geometry_metrics['latent_norm_cond_est'].detach()
                loss_output['monitor/latent_norm_min_eig'] = geometry_metrics['latent_norm_min_eig'].detach()
                loss_output['monitor/latent_norm_max_eig'] = geometry_metrics['latent_norm_max_eig'].detach()

            latent_tail_reg_weight = hparams.get('lambda_latent_tail', 0.0)
            if latent_tail_reg_weight > 0.0:
                tail_max_frames = hparams.get('latent_tail_max_frames', 8192)
                tail_cov_stats = get_cached_cov_stats(tail_max_frames)
                tail_losses = compute_latent_tail_reg_losses(
                    model_outputs['mu'],
                    model_outputs.get('z_lens'),
                    elem_threshold=hparams.get('latent_tail_elem_threshold', 12.0),
                    frame_threshold=hparams.get('latent_tail_frame_threshold', 96.0),
                    max_frames=tail_max_frames,
                    flat=tail_cov_stats['flat'],
                )
                elem_w = latent_tail_reg_weight * hparams.get('latent_tail_elem_weight', 1.0)
                frame_w = latent_tail_reg_weight * hparams.get('latent_tail_frame_weight', 0.0)
                if elem_w > 0:
                    loss_output['latent_tail_elem'] = tail_losses['tail_elem'] * elem_w
                if frame_w > 0:
                    loss_output['latent_tail_frame'] = tail_losses['tail_frame'] * frame_w
                loss_output['monitor/latent_tail_elem_ratio'] = tail_losses['tail_elem_ratio'].detach()
                loss_output['monitor/latent_tail_frame_ratio'] = tail_losses['tail_frame_ratio'].detach()

            loss_output['monitor/mu'] = model_outputs['mu'].mean().detach()
            loss_output['monitor/logvar'] = model_outputs['logvar'].mean().detach()
            
            self.y_ = y_.detach()
        else:
            #######################
            #    Discriminator    #
            #######################
            if self.global_step >= hparams.get('disc_start_steps', 0):
                if not self.training:
                    return None
                y = y.unsqueeze(1)
                y_ = self.y_
                # MPD
                y_p_hat_r, y_p_hat_g, _, _ = unwrap_model(self.model_disc)['mpd'](y, y_.detach(), None)
                loss_output['r_p'], loss_output['f_p'] = discriminator_loss(y_p_hat_r, y_p_hat_g)
                loss_output['r_p'] = loss_output['r_p'] / self.num_disc
                loss_output['f_p'] = loss_output['f_p'] / self.num_disc
                # MSD
                if hparams['use_msd']:
                    y_s_hat_r, y_s_hat_g, _, _ = unwrap_model(self.model_disc)['msd'](y, y_.detach(), None)
                    loss_output['r_s'], loss_output['f_s'] = discriminator_loss(y_s_hat_r, y_s_hat_g)
                    loss_output['r_s'] = loss_output['r_s'] / self.num_disc
                    loss_output['f_s'] = loss_output['f_s'] / self.num_disc
                # MRD
                if hparams['use_mrd']:
                    y_r_hat_r = [x[1] for x in unwrap_model(self.model_disc)['mrd'](y)]
                    y_r_hat_g = [x[1] for x in unwrap_model(self.model_disc)['mrd'](y_.detach())]
                    loss_output['r_r'], loss_output['f_r'] = discriminator_loss(y_r_hat_r, y_r_hat_g)
                    loss_output['r_r'] = loss_output['r_r'] / self.num_disc
                    loss_output['f_r'] = loss_output['f_r'] / self.num_disc

        total_loss = sum(v for k, v in loss_output.items() if not k.startswith('monitor/'))
        loss_output['bs'] = sample['wavs'].shape[0]
        loss_output['ntokens'] = sample['wavs'].shape[0] * sample['wavs'].shape[1] // hparams['hop_size']

        return total_loss, loss_output

    def on_before_optimization(self, opt_idx):

        grad_norm_dict = super().on_before_optimization(opt_idx)

        if opt_idx == 0:
            gen_params = self._get_trainable_model_gen_parameters()
            if len(gen_params) > 0:
                nn.utils.clip_grad_norm_(gen_params, hparams['generator_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.model_disc.parameters(), hparams["discriminator_grad_norm"])

        return grad_norm_dict

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        infer_steps = self.hparams.get('infer_steps', 12)
        outputs = self._validation_step(sample, batch_idx, infer_steps)
        return outputs

    def _validation_step(self, sample, batch_idx, infer_steps):
        outputs = {}
        if self.trainer.proc_rank == 0:
            pass
        return outputs

    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        infer_steps = hparams['infer_steps']
        return self._validation_step(sample, batch_idx, infer_steps)
    
    
