import os
import math
import json
from typing import List, Tuple, Optional, Dict, Any, Sequence, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import remove_weight_norm

from modules.codec.dac.nn.layers import WNConv1d, WNConvTranspose1d
from modules.commons.hf.transformer import TransformerEncoderModel
from modules.commons.hf.transformer_config import TransformerConfig

from utils.audio.transform import (
    apply_batch_variable_lowpass_torch,
    estimate_effective_bandwidth_torch,
)
from utils.commons.io import print_once, json_dumps
from utils.commons.ckpt_utils import get_all_ckpts
from utils.nn.seq_utils import sequence_mask


def build_wavvae(hparams, infer=False, attn_implementation='flash_attention_2', verbose=True):
    
    config = ModelArgs(
        sample_rate=hparams['audio_sample_rate'],
        kernel_size=hparams.get('kernel_size', 7),
        encoder_dim=hparams.get('encoder_dim', 32),
        encoder_dim_max=hparams.get('encoder_dim_max', 1024),
        encoder_kernel_sizes=hparams.get('encoder_kernel_sizes', (7, 7, 7, 7, 7, 7)),
        encoder_rates=hparams.get('encoder_rates', (2, 3, 5, 4, 4, 4)),
        encoder_dilations=hparams.get('encoder_dilations', ((1, 3, 9), (1, 3, 9), (1, 3, 9), (1, 3, 9), (1, 3, 9), (1, 3, 9))),
        transformer_n_layers=hparams.get('transformer_n_layers', 0),
        transformer_n_head=hparams.get('transformer_n_head', 16),
        transformer_n_kv_heads=hparams.get('transformer_n_kv_heads', 8),
        transformer_use_sliding_window=hparams.get('transformer_use_sliding_window', True),
        transformer_sliding_window_size=hparams.get('transformer_sliding_window_size', 16),
        encoder_smoothing_layers=hparams.get('encoder_smoothing_layers', 0),
        encoder_smoothing_kernel_size=hparams.get('encoder_smoothing_kernel_size', 5),
        encoder_smoothing_expansion=hparams.get('encoder_smoothing_expansion', 2),
        latent_dim=hparams.get('latent_dim', 64),
        decoder_dim=hparams.get('decoder_dim', 1920),
        decoder_rates=hparams.get('decoder_rates', (5, 4, 4, 4, 3, 2)),
        attn_implementation=attn_implementation,
        gradient_checkpointing=hparams.get('gradient_checkpointing', False),
        encoder_use_antialiasing=hparams.get('encoder_use_antialiasing', True),
        encoder_antialias_taps_base=hparams.get('encoder_antialias_taps_base', 11),
        encoder_antialias_taps_scale=hparams.get('encoder_antialias_taps_scale', 4),
        encoder_antialias_cutoff_scale=hparams.get('encoder_antialias_cutoff_scale', 0.97),
        encoder_resample_mode=hparams.get('encoder_resample_mode', 'phase0'),
        decoder_use_antialias_upsample=hparams.get('decoder_use_antialias_upsample', True),
        decoder_antialias_taps_base=hparams.get('decoder_antialias_taps_base', 11),
        decoder_antialias_taps_scale=hparams.get('decoder_antialias_taps_scale', 4),
        decoder_antialias_cutoff_scale=hparams.get('decoder_antialias_cutoff_scale', 0.97),
        decoder_resample_mode=hparams.get('decoder_resample_mode', 'linear'),
        decoder_use_aa_activation=hparams.get('decoder_use_aa_activation', True),
        decoder_aa_activation_oversample_factor=hparams.get('decoder_aa_activation_oversample_factor', 2),
        decoder_aa_activation_taps=hparams.get('decoder_aa_activation_taps', 13),
        decoder_aa_activation_cutoff_scale=hparams.get('decoder_aa_activation_cutoff_scale', 0.97),
        decoder_use_highpass_prior=hparams.get('decoder_use_highpass_prior', True),
        decoder_highpass_prior_dim=hparams.get('decoder_highpass_prior_dim', 256),
        decoder_highpass_prior_kernel_size=hparams.get('decoder_highpass_prior_kernel_size', 7),
        decoder_highpass_prior_gain_init=hparams.get('decoder_highpass_prior_gain_init', 0.1),
        decoder_highpass_prior_gain_max=hparams.get('decoder_highpass_prior_gain_max', 0.5),
        decoder_use_adaa_snakebeta=hparams.get('decoder_use_adaa_snakebeta', False),
    )
    
    if infer:
        config.latent_mean = hparams.get('vae_latent_mean', None)
        config.latent_std = hparams.get('vae_latent_std', None)
        config.latent_norm_mode = hparams.get('vae_latent_norm_mode', hparams.get('latent_norm_mode', config.latent_norm_mode))
        config.latent_stats_path = hparams.get('vae_latent_stats_path', hparams.get('latent_stats_path', None))
        config.bandwidth_candidates_hz = tuple(
            float(x) for x in hparams.get('vae_bandwidth_candidates_hz', config.bandwidth_candidates_hz)
        )
        config.bandwidth_transition_hz = float(
            hparams.get('vae_bandwidth_transition_hz', config.bandwidth_transition_hz)
        )
        print_once(
            f"| load VAE with mean={config.latent_mean}, std={config.latent_std}, latent_norm_mode={config.latent_norm_mode}, latent_stats_path={config.latent_stats_path}"
        )
        model = InferenceWrapper(config)
    else:
        model = WavVAE(config)

    if verbose:
        # window_info = compute_wavvae_window_size(config)
        window_info = compute_wavvae_e2e_window_size(config)
        print_once(f"WavVAE window info: {json_dumps(window_info)}")
        
    return model


@dataclass
class ModelArgs:
    # audio
    sample_rate: int = 24000
    audio_channels: int = 1
    
    kernel_size: int = 7
    
    # encoder
    encoder_dim: int = 16
    encoder_dim_max: int = 1024
    encoder_kernel_sizes: Tuple[int] = (7, 7, 7, 7, 7, 7)
    encoder_rates: Tuple[int] = (2, 3, 5, 4, 4, 4)
    encoder_dilations: Tuple[Tuple[int, ...], ...] = ((1, 3, 9), (1, 3, 9), (1, 3, 9), (1, 3, 9), (1, 3, 9), (1, 3, 9))
    encoder_use_antialiasing: bool = True
    encoder_antialias_taps_base: int = 11
    encoder_antialias_taps_scale: int = 4
    encoder_antialias_cutoff_scale: float = 0.97
    encoder_resample_mode: str = 'phase0'
    
    # transformer
    transformer_n_layers: int = 0
    transformer_n_head: int = 16
    transformer_n_kv_heads: int = 8
    transformer_use_sliding_window: bool = True
    transformer_sliding_window_size: int = 16
    encoder_smoothing_layers: int = 0
    encoder_smoothing_kernel_size: int = 5
    encoder_smoothing_expansion: int = 2
    attn_implementation: str = 'flash_attention_2'

    # bottleneck
    latent_dim: int = 48
    latent_mean: float = None
    latent_std: float = None
    latent_stats_path: Optional[str] = None
    latent_norm_mode: str = "per_channel"   # global | per_channel | pca
    remove_weight_norm_infer: bool = False
    bandwidth_candidates_hz: Tuple[float, ...] = (4000.0, 5500.0, 8000.0, 11025.0, 12000.0, 16000.0, 20000.0, 22050.0)
    bandwidth_transition_hz: float = 1000.0
    
    # decoder
    decoder_dim: int = 2048
    decoder_rates: Tuple[int] = (5, 4, 4, 4, 3, 2)
    decoder_use_antialias_upsample: bool = True
    decoder_antialias_taps_base: int = 11
    decoder_antialias_taps_scale: int = 4
    decoder_antialias_cutoff_scale: float = 0.97
    decoder_resample_mode: str = 'linear'
    decoder_use_aa_activation: bool = True
    decoder_aa_activation_oversample_factor: int = 2
    decoder_aa_activation_taps: int = 13
    decoder_aa_activation_cutoff_scale: float = 0.97
    decoder_use_highpass_prior: bool = True
    decoder_highpass_prior_dim: int = 256
    decoder_highpass_prior_kernel_size: int = 7
    decoder_highpass_prior_gain_init: float = 0.1
    decoder_highpass_prior_gain_max: float = 0.5
    decoder_use_adaa_snakebeta: bool = False

    gradient_checkpointing: bool = False


@torch.jit.script
def snake_plain(x, alpha, beta):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


@torch.jit.script
def snake_logscale(x, alpha, beta):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    alpha = torch.exp(alpha)
    beta = torch.exp(beta)
    x = x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


@torch.jit.script
def adaa_snakebeta_plain(x, alpha, beta):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    beta = beta + 1e-9
    x_prev = torch.cat([torch.zeros_like(x[..., :1]), x[..., :-1]], dim=-1)
    x_sum = x + x_prev
    x_diff = x - x_prev
    sinc = torch.sinc((alpha * x_diff) / 3.141592653589793)
    x = 0.5 / beta + 0.5 * x_sum - torch.cos(alpha * x_sum) * sinc / (2.0 * beta)
    return x.reshape(shape)


@torch.jit.script
def adaa_snakebeta_logscale(x, alpha, beta):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    alpha = torch.exp(alpha)
    beta = torch.exp(beta) + 1e-9
    x_prev = torch.cat([torch.zeros_like(x[..., :1]), x[..., :-1]], dim=-1)
    x_sum = x + x_prev
    x_diff = x - x_prev
    sinc = torch.sinc((alpha * x_diff) / 3.141592653589793)
    x = 0.5 / beta + 0.5 * x_sum - torch.cos(alpha * x_sum) * sinc / (2.0 * beta)
    return x.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels, logscale=True, init=1.0):
        super().__init__()
        self.logscale = logscale

        if logscale:
            self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
            self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        else:
            self.alpha = nn.Parameter(torch.full((1, channels, 1), init))
            self.beta = nn.Parameter(torch.full((1, channels, 1), init))

    def forward(self, x):
        if torch.is_grad_enabled():
            orig_dtype = x.dtype
            x_work = x.float()
            alpha = self.alpha.float()
            beta = self.beta.float()

            if self.logscale:
                y = snake_logscale(x_work, alpha, beta)
            else:
                y = snake_plain(x_work, alpha, beta)

            return y.to(orig_dtype)

        alpha = self.alpha.to(dtype=x.dtype)
        beta = self.beta.to(dtype=x.dtype)

        if self.logscale:
            return snake_logscale(x, alpha, beta)
        else:
            return snake_plain(x, alpha, beta)


class ADAASnakeBeta1d(nn.Module):
    def __init__(self, channels, logscale=True, init=1.0):
        super().__init__()
        self.logscale = logscale

        if logscale:
            self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
            self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        else:
            self.alpha = nn.Parameter(torch.full((1, channels, 1), init))
            self.beta = nn.Parameter(torch.full((1, channels, 1), init))

    def forward(self, x):
        if torch.is_grad_enabled():
            orig_dtype = x.dtype
            if self.logscale:
                y = adaa_snakebeta_logscale(x.float(), self.alpha.float(), self.beta.float())
            else:
                y = adaa_snakebeta_plain(x.float(), self.alpha.float(), self.beta.float())
            return y.to(orig_dtype)

        alpha = self.alpha.to(dtype=x.dtype)
        beta = self.beta.to(dtype=x.dtype)
        if self.logscale:
            return adaa_snakebeta_logscale(x, alpha, beta)
        return adaa_snakebeta_plain(x, alpha, beta)


def _ensure_odd(x: int) -> int:
    x = int(x)
    return x if x % 2 == 1 else x + 1


def _suggest_fir_taps(stride: int, base_taps: int = 9, taps_scale: int = 4) -> int:
    return _ensure_odd(max(int(base_taps), int(taps_scale) * int(stride) + 1))


def _design_lowpass_fir(num_taps: int, cutoff: float, window: str = 'hann') -> torch.Tensor:
    num_taps = _ensure_odd(num_taps)
    cutoff = float(max(1e-4, min(0.999, cutoff)))

    n = torch.arange(num_taps, dtype=torch.float32) - (num_taps - 1) / 2
    h = cutoff * torch.sinc(cutoff * n)

    if window == 'hann':
        w = torch.hann_window(num_taps, periodic=False, dtype=torch.float32)
    else:
        w = torch.ones(num_taps, dtype=torch.float32)

    h = h * w
    h = h / h.sum().clamp_min(1e-12)
    return h


class FixedLowpass1d(nn.Module):
    def __init__(self, channels: int, num_taps: int, cutoff: float, window: str = 'hann'):
        super().__init__()
        kernel = _design_lowpass_fir(num_taps=num_taps, cutoff=cutoff, window=window)
        kernel = kernel.view(1, 1, -1).repeat(channels, 1, 1)
        self.register_buffer('kernel', kernel)
        self.channels = channels
        self.pad = kernel.shape[-1] // 2

    def forward(self, x):
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        kernel = self.kernel.to(dtype=x.dtype, device=x.device)
        return F.conv1d(x, kernel, padding=self.pad, groups=self.channels)


class AASnake1d(nn.Module):
    def __init__(
        self,
        channels: int,
        logscale: bool = True,
        init: float = 1.0,
        oversample_factor: int = 2,
        filt_taps: int = 13,
        cutoff_scale: float = 0.95,
        resample_mode: str = 'linear',
    ):
        super().__init__()
        self.channels = channels
        self.oversample_factor = int(max(1, oversample_factor))
        self.resample_mode = resample_mode
        self.snake = Snake1d(channels, logscale=logscale, init=init)
        cutoff = min(0.999, max(1e-4, float(cutoff_scale) / self.oversample_factor))
        self.lowpass = FixedLowpass1d(
            channels=channels,
            num_taps=_ensure_odd(filt_taps),
            cutoff=cutoff,
            window='hann',
        )

    def forward(self, x):
        if self.oversample_factor == 1:
            return self.snake(x)

        target_len = x.shape[-1]
        x_up = F.interpolate(
            x,
            scale_factor=self.oversample_factor,
            mode=self.resample_mode,
            align_corners=False if self.resample_mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None,
        )
        y_up = self.snake(x_up)
        y_up = self.lowpass(y_up)
        y = F.interpolate(
            y_up,
            size=target_len,
            mode=self.resample_mode,
            align_corners=False if self.resample_mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None,
        )
        return y


def _make_decoder_activation(
    channels: int,
    use_aa_activation: bool,
    use_adaa_snakebeta: bool = False,
    aa_oversample_factor: int = 2,
    aa_filt_taps: int = 13,
    aa_cutoff_scale: float = 0.95,
    aa_resample_mode: str = 'linear',
):
    if use_aa_activation:
        if use_adaa_snakebeta:
            return ADAASnakeBeta1d(channels)
        return AASnake1d(
            channels,
            oversample_factor=aa_oversample_factor,
            filt_taps=aa_filt_taps,
            cutoff_scale=aa_cutoff_scale,
            resample_mode=aa_resample_mode,
        )
    return Snake1d(channels)


class AntiAliasedDownsample1d(nn.Module):
    def __init__(
        self,
        channels: int,
        stride: int,
        taps: int,
        cutoff_scale: float = 0.95,
        mode: str = 'linear',
    ):
        super().__init__()
        self.stride = int(stride)
        self.mode = str(mode).lower()
        cutoff = min(0.999, max(1e-4, float(cutoff_scale) / self.stride))
        self.lowpass = FixedLowpass1d(channels, num_taps=taps, cutoff=cutoff, window='hann')

    def forward(self, x):
        in_len = x.shape[-1]
        if in_len % self.stride != 0:
            raise RuntimeError(
                f"AntiAliasedDownsample1d requires input length divisible by stride: "
                f"input_len={in_len}, stride={self.stride}."
            )
        x = self.lowpass(x)
        if self.mode in {"phase0", "decimate"}:
            return x[..., ::self.stride]
        return F.interpolate(
            x,
            size=in_len // self.stride,
            mode=self.mode,
            align_corners=False if self.mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None,
        )


class AntiAliasedUpsample1d(nn.Module):
    def __init__(
        self,
        channels: int,
        stride: int,
        taps: int,
        cutoff_scale: float = 0.95,
        mode: str = 'linear',
    ):
        super().__init__()
        self.stride = int(stride)
        self.mode = mode
        cutoff = min(0.999, max(1e-4, float(cutoff_scale) / self.stride))
        self.taps = _ensure_odd(int(taps))
        self.cutoff = cutoff
        self.lowpass = FixedLowpass1d(channels, num_taps=taps, cutoff=cutoff, window='hann')

    @staticmethod
    def zero_interlace_with_stride(x, stride: int):
        stride = int(stride)
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        out = x.new_zeros(x.shape[:-1] + (x.shape[-1] * stride,))
        out[..., ::stride] = x
        return out

    def zero_interlace(self, x):
        return self.zero_interlace_with_stride(x, self.stride)

    def low_high(self, x):
        zero = self.zero_interlace(x) * self.stride
        low = self.lowpass(zero)
        if low.shape[-1] != zero.shape[-1]:
            raise RuntimeError(
                f"AntiAliasedUpsample1d length mismatch after lowpass: got {low.shape[-1]}, expected {zero.shape[-1]}"
            )
        high = zero - low
        return low, high

    def forward(self, x):
        low, _ = self.low_high(x)
        return low


def _init_depthwise_identity_wn_conv(conv: nn.Module):
    with torch.no_grad():
        weight = getattr(conv, "weight_v", None)
        if weight is None:
            weight = getattr(conv, "weight", None)
        if weight is not None:
            weight.zero_()
            center = weight.shape[-1] // 2
            weight[:, 0, center] = 1.0
        weight_g = getattr(conv, "weight_g", None)
        if weight_g is not None:
            weight_g.fill_(1.0)
        if getattr(conv, "bias", None) is not None:
            conv.bias.zero_()


def _init_pointwise_partial_identity_wn_conv(conv: nn.Module):
    with torch.no_grad():
        weight = getattr(conv, "weight_v", None)
        if weight is None:
            weight = getattr(conv, "weight", None)
        if weight is not None:
            weight.zero_()
            out_ch, in_ch = weight.shape[0], weight.shape[1]
            center = weight.shape[-1] // 2
            for out_idx in range(out_ch):
                in_idx = out_idx if out_idx < in_ch else 0
                weight[out_idx, in_idx, center] = 1.0
        weight_g = getattr(conv, "weight_g", None)
        if weight_g is not None:
            weight_g.zero_()
            n_identity = min(weight_g.numel(), weight.shape[0], weight.shape[1]) if weight is not None else 0
            weight_g.view(-1)[:n_identity].fill_(1.0)
        if getattr(conv, "bias", None) is not None:
            conv.bias.zero_()


def _bounded_sigmoid_raw_from_value(value: float, max_value: float) -> float:
    value = float(value)
    max_value = float(max_value)
    if max_value <= 0.0:
        raise ValueError(f"decoder_highpass_prior_gain_max must be positive, got {max_value}")
    if value < 0.0 or value > max_value:
        raise ValueError(
            f"decoder_highpass_prior_gain_init must be in [0, gain_max], "
            f"got init={value}, gain_max={max_value}"
        )
    p = min(max(value / max_value, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class AntiAliasedDownsampleConv1d(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int,
        kernel_size: int,
        taps_base: int,
        taps_scale: int,
        cutoff_scale: float,
        mode: str = 'linear',
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"AntiAliasedDownsampleConv1d requires odd kernel_size, got {kernel_size}")
        taps = _suggest_fir_taps(stride, taps_base, taps_scale)
        self.resample = AntiAliasedDownsample1d(
            in_dim,
            stride=stride,
            taps=taps,
            cutoff_scale=cutoff_scale,
            mode=mode,
        )
        self.proj = WNConv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        in_len = x.shape[-1]
        expected_len = in_len // self.resample.stride
        x = self.resample(x)
        x = self.proj(x)
        if x.shape[-1] != expected_len:
            raise RuntimeError(
                f"AntiAliasedDownsampleConv1d length mismatch: got {x.shape[-1]}, expected {expected_len}"
            )
        return x


class AntiAliasedUpsampleConv1d(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int,
        kernel_size: int,
        taps_base: int,
        taps_scale: int,
        cutoff_scale: float,
        mode: str = 'linear',
        use_highpass_prior: bool = True,
        prior_source_dim: Optional[int] = None,
        prior_stride: Optional[int] = None,
        highpass_prior_kernel_size: int = 7,
        highpass_prior_gain_init: float = 0.1,
        highpass_prior_gain_max: float = 0.5,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"AntiAliasedUpsampleConv1d requires odd kernel_size, got {kernel_size}")
        if use_highpass_prior and highpass_prior_kernel_size % 2 == 0:
            raise ValueError(
                f"decoder_highpass_prior_kernel_size must be odd, got {highpass_prior_kernel_size}"
            )
        if use_highpass_prior:
            _bounded_sigmoid_raw_from_value(highpass_prior_gain_init, highpass_prior_gain_max)
        taps = _suggest_fir_taps(stride, taps_base, taps_scale)
        self.resample = AntiAliasedUpsample1d(
            in_dim,
            stride=stride,
            taps=taps,
            cutoff_scale=cutoff_scale,
            mode=mode,
        )
        self.use_highpass_prior = bool(use_highpass_prior)
        self.prior_source_dim = int(prior_source_dim) if prior_source_dim is not None else int(in_dim)
        self.prior_stride = int(prior_stride) if prior_stride is not None else int(stride)
        if self.use_highpass_prior:
            if self.prior_stride <= 0:
                raise ValueError(f"prior_stride must be positive, got {self.prior_stride}")
            self.prior_lowpass = FixedLowpass1d(
                channels=self.prior_source_dim,
                num_taps=self.resample.taps,
                cutoff=self.resample.cutoff,
                window='hann',
            )
            self.prior_depthwise = WNConv1d(
                self.prior_source_dim,
                self.prior_source_dim,
                kernel_size=highpass_prior_kernel_size,
                stride=1,
                padding=highpass_prior_kernel_size // 2,
                groups=self.prior_source_dim,
            )
            self.prior_pointwise = WNConv1d(
                self.prior_source_dim,
                in_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            _init_depthwise_identity_wn_conv(self.prior_depthwise)
            _init_pointwise_partial_identity_wn_conv(self.prior_pointwise)
            self.prior_gain_max = float(highpass_prior_gain_max)
            self.prior_gain_raw = nn.Parameter(torch.tensor(
                _bounded_sigmoid_raw_from_value(highpass_prior_gain_init, highpass_prior_gain_max),
                dtype=torch.float32,
            ))
        else:
            self.prior_lowpass = None
            self.prior_depthwise = None
            self.prior_pointwise = None
            self.prior_gain_max = float(highpass_prior_gain_max)
            self.prior_gain_raw = None
        self.proj = WNConv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    @property
    def highpass_prior_gain(self) -> Optional[torch.Tensor]:
        if self.prior_gain_raw is None:
            return None
        return self.prior_gain_max * torch.sigmoid(self.prior_gain_raw)

    def forward(self, x, x0: Optional[torch.Tensor] = None):
        in_len = x.shape[-1]
        expected_len = in_len * self.resample.stride
        x = self.resample(x)
        if self.use_highpass_prior:
            if x0 is None:
                raise ValueError("AntiAliasedUpsampleConv1d high-pass prior requires x0.")
            if x0.shape[1] != self.prior_source_dim:
                raise ValueError(f"Expected x0 channels={self.prior_source_dim}, got {x0.shape[1]}")
            x0_up = AntiAliasedUpsample1d.zero_interlace_with_stride(x0, self.prior_stride) * self.prior_stride
            if x0_up.shape[-1] != expected_len:
                raise RuntimeError(
                    f"AntiAliasedUpsampleConv1d x0 prior length mismatch: "
                    f"got {x0_up.shape[-1]}, expected {expected_len}, prior_stride={self.prior_stride}."
                )
            x0_low = self.prior_lowpass(x0_up)
            if x0_low.shape[-1] != expected_len:
                raise RuntimeError(
                    f"AntiAliasedUpsampleConv1d x0 prior lowpass length mismatch: "
                    f"got {x0_low.shape[-1]}, expected {expected_len}."
                )
            high = x0_up - x0_low
            prior = self.prior_depthwise(high)
            prior = self.prior_pointwise(prior)
            if prior.shape[-1] != expected_len:
                raise RuntimeError(
                    f"AntiAliasedUpsampleConv1d prior length mismatch: got {prior.shape[-1]}, expected {expected_len}"
                )
            gain = self.highpass_prior_gain.to(dtype=x.dtype, device=x.device)
            x = x + gain * prior
        x = self.proj(x)
        if x.shape[-1] != expected_len:
            raise RuntimeError(
                f"AntiAliasedUpsampleConv1d length mismatch: got {x.shape[-1]}, expected {expected_len}"
            )
        return x


class ResidualUnit(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        dilation: int = 1,
        kernel_size: int = 7,
        use_aa_activation: bool = True,
        use_adaa_snakebeta: bool = False,
        aa_oversample_factor: int = 2,
        aa_filt_taps: int = 13,
        aa_cutoff_scale: float = 0.95,
        aa_resample_mode: str = 'linear',
    ):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        act1 = _make_decoder_activation(
            dim,
            use_aa_activation=use_aa_activation,
            use_adaa_snakebeta=use_adaa_snakebeta,
            aa_oversample_factor=aa_oversample_factor,
            aa_filt_taps=aa_filt_taps,
            aa_cutoff_scale=aa_cutoff_scale,
            aa_resample_mode=aa_resample_mode,
        )
        act2 = _make_decoder_activation(
            dim,
            use_aa_activation=use_aa_activation,
            use_adaa_snakebeta=use_adaa_snakebeta,
            aa_oversample_factor=aa_oversample_factor,
            aa_filt_taps=aa_filt_taps,
            aa_cutoff_scale=aa_cutoff_scale,
            aa_resample_mode=aa_resample_mode,
        )
        self.block = nn.Sequential(
            act1,
            WNConv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, padding=pad),
            act2,
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        if y.shape[-1] != x.shape[-1]:
            raise RuntimeError(
                f"ResidualUnit length mismatch: input_len={x.shape[-1]}, block_len={y.shape[-1]}. "
                "Check kernel_size/dilation padding; residual blocks must preserve time length."
            )
        return x + y
    

class EncoderResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, kernel_size: int = 7):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.block = nn.Sequential(
            nn.SiLU(),
            WNConv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, padding=pad),
            nn.SiLU(),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        if y.shape[-1] != x.shape[-1]:
            raise RuntimeError(
                f"EncoderResidualUnit length mismatch: input_len={x.shape[-1]}, block_len={y.shape[-1]}. "
                "Check kernel_size/dilation padding; residual blocks must preserve time length."
            )
        return x + y


class _IdentityModule(nn.Module):
    def forward(self, x):
        return x


def _zero_init_conv(conv: nn.Module):
    if hasattr(conv, "weight_g") and conv.weight_g is not None:
        nn.init.zeros_(conv.weight_g)
    elif hasattr(conv, "weight") and conv.weight is not None:
        nn.init.zeros_(conv.weight)
    if hasattr(conv, "bias") and conv.bias is not None:
        nn.init.zeros_(conv.bias)


class EncoderSmoothingBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, expansion: int = 2):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"encoder_smoothing_kernel_size must be odd, got {kernel_size}")
        pad = kernel_size // 2
        hidden_dim = dim * max(1, int(expansion))
        self.dwconv = WNConv1d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)
        self.in_act = nn.SiLU()
        self.mid_act = nn.SiLU()
        self.out_act = nn.SiLU()
        self.pw1 = WNConv1d(dim, hidden_dim, kernel_size=1)
        self.pw2 = WNConv1d(hidden_dim, dim, kernel_size=1)
        _zero_init_conv(self.pw2)

    def forward(self, x):
        y = self.in_act(x)
        y = self.dwconv(y)
        y = self.mid_act(y)
        y = self.pw1(y)
        y = self.out_act(y)
        y = self.pw2(y)
        return x + y
    

class EncoderBlock(nn.Module):
    def __init__(self, config: ModelArgs, in_dim: int = 16, out_dim: int = 16, stride: int = 1, kernel_size: int = 7, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 3, 9]
        if config.encoder_use_antialiasing and stride > 1:
            downsample = AntiAliasedDownsampleConv1d(
                in_dim,
                out_dim,
                stride=stride,
                kernel_size=config.kernel_size,
                taps_base=config.encoder_antialias_taps_base,
                taps_scale=config.encoder_antialias_taps_scale,
                cutoff_scale=config.encoder_antialias_cutoff_scale,
                mode=config.encoder_resample_mode,
            )
        else:
            downsample = WNConv1d(
                in_dim,
                out_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        self.block = nn.Sequential(
            EncoderResidualUnit(in_dim, dilation=dilations[0], kernel_size=kernel_size),
            EncoderResidualUnit(in_dim, dilation=dilations[1], kernel_size=kernel_size),
            EncoderResidualUnit(in_dim, dilation=dilations[2], kernel_size=kernel_size),
            nn.SiLU(),
            downsample,
        )

    def forward(self, x):
        return self.block(x)
    

class Encoder(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
    ):
        super().__init__()
        self.config = config
        
        d_in = config.audio_channels
        d_model = config.encoder_dim
        strides = config.encoder_rates
        self.hop_size = int(np.prod(strides))
        
        # Create first convolution
        self.block = [WNConv1d(d_in, d_model, kernel_size=config.kernel_size, padding=config.kernel_size // 2)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for i, stride in enumerate(strides):
            out_dim = min(d_model * 2, config.encoder_dim_max)
            self.block += [EncoderBlock(
                config, d_model, out_dim, stride=stride, 
                kernel_size=config.encoder_kernel_sizes[i],
                dilations=config.encoder_dilations[i],
            )]
            d_model = out_dim

        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

        if config.transformer_n_layers > 0:
            self.transformer = TransformerEncoderModel(TransformerConfig(
                hidden_size=d_model, intermediate_size=d_model * 4, 
                num_hidden_layers=config.transformer_n_layers,
                num_attention_heads=config.transformer_n_head, 
                num_key_value_heads=config.transformer_n_kv_heads,
                use_sliding_window=config.transformer_use_sliding_window, 
                sliding_window=config.transformer_sliding_window_size,
                max_window_layers=0,
                attn_implementation=config.attn_implementation,
                use_cache=False,
            ))
            if config.gradient_checkpointing:
                try:
                    self.transformer.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False, "determinism_check": "none"}
                    )
                except TypeError:
                    self.transformer.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
        if config.encoder_smoothing_layers > 0:
            self.smoothing_tail = nn.Sequential(*[
                EncoderSmoothingBlock(
                    d_model,
                    kernel_size=config.encoder_smoothing_kernel_size,
                    expansion=config.encoder_smoothing_expansion,
                )
                for _ in range(config.encoder_smoothing_layers)
            ])
        else:
            self.smoothing_tail = nn.Identity()

    def _build_position_ids(
        self,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        pos_offset: Union[int, torch.Tensor] = 0,
    ) -> torch.Tensor:
        base = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        offset = torch.as_tensor(pos_offset, device=device, dtype=torch.long).reshape(-1)
        if offset.numel() == 1:
            offset = offset.expand(batch_size)
        elif offset.numel() != batch_size:
            raise ValueError(
                f"pos_offset must be scalar or have batch size {batch_size}, got shape {tuple(offset.shape)}"
            )
        return base + offset[:, None]

    def forward(self, x, x_mask=None, pos_offset=0):
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            try:
                x = torch.utils.checkpoint.checkpoint(
                    self.block, x, use_reentrant=False, determinism_check="none"
                )
            except TypeError:
                x = torch.utils.checkpoint.checkpoint(self.block, x, use_reentrant=False)
        else:
            x = self.block(x)
        x_lat_mask = None
        if x_mask is not None:
            x_mask = x_mask[:, ::self.hop_size]
            x_lat_mask = x_mask[:, None, :].to(dtype=x.dtype)

        if self.config.transformer_n_layers > 0:
            x = self.transformer(
                inputs_embeds=x.transpose(1, 2), 
                attention_mask=x_mask,
                position_ids=self._build_position_ids(
                    seq_len=x.shape[2],
                    batch_size=x.shape[0],
                    device=x.device,
                    pos_offset=pos_offset,
                ),
            ).last_hidden_state.transpose(1, 2) + x
            
        if x_lat_mask is not None:
            x = x * x_lat_mask
        x = self.smoothing_tail(x)
        if x_lat_mask is not None:
            x = x * x_lat_mask

        return x
    

class VAEBottleneck(nn.Module):
    def __init__(
        self, 
        input_dim=1024, 
        z_dim=32, 
    ):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.mu = WNConv1d(input_dim, z_dim, kernel_size=1)
        self.lv = WNConv1d(input_dim, z_dim, kernel_size=1)

    def encode(self, x):
        # x: [B, C_in, T_enc]
        mu = self.mu(x)
        logvar = self.lv(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        orig_dtype = mu.dtype
        with torch.cuda.amp.autocast(enabled=False):
            mu_32 = mu.float()
            logvar_32 = logvar.float()
            std = torch.exp(0.5 * logvar_32)
            eps = torch.randn_like(std)
            z = mu_32 + std * eps   # [B, z_dim, T_code]
        return z.to(orig_dtype)

    def forward(self, x, kl_weight=1.0):
        mu, logvar = self.encode(x)
        logvar = logvar.clamp(min=-30, max=20)
        z = self.reparameterize(mu, logvar)

        # KL loss
        with torch.cuda.amp.autocast(enabled=False):
            mu_32 = mu.float()
            logvar_32 = logvar.float()
            kl = -0.5 * (1 + logvar_32 - mu_32.pow(2) - logvar_32.exp())
            kl = kl.mean()

        return z, mu, logvar, kl * kl_weight
    

class DecoderBlock(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        kernel_size: int = 7,
        prior_source_dim: Optional[int] = None,
        prior_stride: Optional[int] = None,
    ):
        super().__init__()
        self.pre_act = _make_decoder_activation(
            input_dim,
            use_aa_activation=config.decoder_use_aa_activation,
            use_adaa_snakebeta=config.decoder_use_adaa_snakebeta,
            aa_oversample_factor=config.decoder_aa_activation_oversample_factor,
            aa_filt_taps=config.decoder_aa_activation_taps,
            aa_cutoff_scale=config.decoder_aa_activation_cutoff_scale,
            aa_resample_mode=config.decoder_resample_mode,
        )

        if config.decoder_use_antialias_upsample and stride > 1:
            self.upsample = AntiAliasedUpsampleConv1d(
                input_dim,
                output_dim,
                stride=stride,
                kernel_size=config.kernel_size,
                taps_base=config.decoder_antialias_taps_base,
                taps_scale=config.decoder_antialias_taps_scale,
                cutoff_scale=config.decoder_antialias_cutoff_scale,
                mode=config.decoder_resample_mode,
                use_highpass_prior=config.decoder_use_highpass_prior,
                prior_source_dim=prior_source_dim,
                prior_stride=prior_stride,
                highpass_prior_kernel_size=config.decoder_highpass_prior_kernel_size,
                highpass_prior_gain_init=config.decoder_highpass_prior_gain_init,
                highpass_prior_gain_max=config.decoder_highpass_prior_gain_max,
            )
        else:
            self.upsample = WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            )

        self.res_blocks = nn.Sequential(
            ResidualUnit(
                output_dim,
                dilation=1,
                kernel_size=kernel_size,
                use_aa_activation=config.decoder_use_aa_activation,
                use_adaa_snakebeta=config.decoder_use_adaa_snakebeta,
                aa_oversample_factor=config.decoder_aa_activation_oversample_factor,
                aa_filt_taps=config.decoder_aa_activation_taps,
                aa_cutoff_scale=config.decoder_aa_activation_cutoff_scale,
                aa_resample_mode=config.decoder_resample_mode,
            ),
            ResidualUnit(
                output_dim,
                dilation=3,
                kernel_size=kernel_size,
                use_aa_activation=config.decoder_use_aa_activation,
                use_adaa_snakebeta=config.decoder_use_adaa_snakebeta,
                aa_oversample_factor=config.decoder_aa_activation_oversample_factor,
                aa_filt_taps=config.decoder_aa_activation_taps,
                aa_cutoff_scale=config.decoder_aa_activation_cutoff_scale,
                aa_resample_mode=config.decoder_resample_mode,
            ),
            ResidualUnit(
                output_dim,
                dilation=9,
                kernel_size=kernel_size,
                use_aa_activation=config.decoder_use_aa_activation,
                use_adaa_snakebeta=config.decoder_use_adaa_snakebeta,
                aa_oversample_factor=config.decoder_aa_activation_oversample_factor,
                aa_filt_taps=config.decoder_aa_activation_taps,
                aa_cutoff_scale=config.decoder_aa_activation_cutoff_scale,
                aa_resample_mode=config.decoder_resample_mode,
            ),
        )

    def forward(self, x, x0: Optional[torch.Tensor] = None):
        x = self.pre_act(x)
        if isinstance(self.upsample, AntiAliasedUpsampleConv1d):
            x = self.upsample(x, x0=x0)
        else:
            x = self.upsample(x)
        x = self.res_blocks(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
    ):
        super().__init__()
        self.config = config
        
        channels = config.decoder_dim
        rates = config.decoder_rates
        d_out = config.audio_channels

        self.conv_pre = WNConv1d(
            config.latent_dim,
            channels,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
        )
        prior_dim = int(config.decoder_highpass_prior_dim)
        if prior_dim <= 0:
            raise ValueError(f"decoder_highpass_prior_dim must be positive, got {prior_dim}")
        self.prior_pre = WNConv1d(
            config.latent_dim,
            prior_dim,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
        ) if config.decoder_use_highpass_prior else None

        # Add upsampling + MRF blocks
        blocks = []
        cumulative_upps = 1
        output_dim = channels
        for i, stride in enumerate(rates):
            input_dim = channels // 2 ** ((i + 1) // 2)
            output_dim = channels // 2 ** (i // 2 + 1)
            cumulative_upps *= int(stride)
            blocks.append(DecoderBlock(
                config,
                input_dim,
                output_dim,
                stride,
                config.kernel_size,
                prior_source_dim=prior_dim,
                prior_stride=cumulative_upps,
            ))
        self.blocks = nn.ModuleList(blocks)

        # Add final conv layer
        self.final_act = _make_decoder_activation(
            output_dim,
            use_aa_activation=config.decoder_use_aa_activation,
            use_adaa_snakebeta=config.decoder_use_adaa_snakebeta,
            aa_oversample_factor=config.decoder_aa_activation_oversample_factor,
            aa_filt_taps=config.decoder_aa_activation_taps,
            aa_cutoff_scale=config.decoder_aa_activation_cutoff_scale,
            aa_resample_mode=config.decoder_resample_mode,
        )

        self.conv_post = WNConv1d(output_dim, d_out, kernel_size=config.kernel_size, padding=config.kernel_size // 2)

    def forward(self, x):
        x0 = self.prior_pre(x) if self.prior_pre is not None else None
        x = self.conv_pre(x)
        for block in self.blocks:
            x = block(x, x0=x0)
        x = self.final_act(x)
        x = self.conv_post(x)
        return x
    

class WavVAE(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
    ):
        super().__init__()
        self.config = config

        self.hop_length = self.frame_length = int(np.prod(config.encoder_rates))
        self.sample_rate = config.sample_rate
        
        self.encoder = Encoder(config)
        self.bottleneck = VAEBottleneck(self.encoder.enc_dim, config.latent_dim)
        self.decoder = Decoder(config)
        
    def _ckpt(self, fn, *args):
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            try:
                return torch.utils.checkpoint.checkpoint(
                    fn, *args, use_reentrant=False, determinism_check="none"
                )
            except TypeError:
                return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _decode_channel_first(self, latent: torch.Tensor, use_checkpoint: bool = True):
        if use_checkpoint:
            return self._ckpt(self.decoder, latent)
        return self.decoder(latent)
    
    def preprocess(self, audio_data, audio_lengths=None):
        if audio_data.ndim == 2:
            audio_data = audio_data.unsqueeze(1)    # [B, C, T]
        if audio_lengths is None:
            audio_lengths = torch.full((audio_data.shape[0],), audio_data.shape[-1], device=audio_data.device, dtype=torch.long)
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.frame_length) * self.frame_length - length
        if right_pad > 0:
            audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data, audio_lengths

    def encode(
        self,
        audio_data: torch.Tensor,
        audio_lengths: torch.Tensor = None,
        pos_offset: Union[int, torch.Tensor] = 0,
    ):
        audio_data, audio_lengths = self.preprocess(audio_data, audio_lengths)

        z_enc = self.encoder(
            audio_data, 
            sequence_mask(audio_lengths, maxlen=audio_data.shape[-1]),
            pos_offset=pos_offset
        )                # [B, C, T_enc]
        z_lens = (audio_lengths + self.hop_length - 1) // self.hop_length
        z, mu, logvar, kl = self.bottleneck(z_enc)

        ret = {
            "z": z, 
            "mu": mu, 
            "logvar": logvar, 
            "kl": kl,
            "z_lens": z_lens
        }

        return ret

    def encode_latent(
        self,
        audio_data,
        audio_lengths=None,
        pos_offset: Union[int, torch.Tensor] = 0,
    ):
        audio_data, audio_lengths = self.preprocess(audio_data, audio_lengths)
        z_enc = self.encoder(
            audio_data,
            sequence_mask(audio_lengths, maxlen=audio_data.shape[-1]),
            pos_offset=pos_offset,
        )                # [B, C, T_enc]
        z_lens = (audio_lengths + self.hop_length - 1) // self.hop_length
        mu, logvar = self.bottleneck.encode(z_enc)
        z = self.bottleneck.reparameterize(mu, logvar)
        return z.permute(0, 2, 1)   # [B, T, C]
        
    def decode(self, latent):
        # latent [B, T, C]
        latent = latent.transpose(1, 2)
        return self._decode_channel_first(latent, use_checkpoint=True)     # [B, 1, T]

    def forward(
        self,
        audio_data: torch.Tensor,
        audio_lengths=None,
        pos_offset: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        enc = self.encode(audio_data, audio_lengths, pos_offset=pos_offset)
        enc['recon'] = self._decode_channel_first(enc['z'], use_checkpoint=True)
        return enc


class InferenceWrapper(WavVAE):
    def __init__(self, config):
        super().__init__(config)
        self.context_info = compute_wavvae_window_size(config)
        self.decoder_context_info = compute_wavvae_decoder_window_size(config)
        self.latent_stats: Optional[Dict[str, Any]] = None
        self._latent_stats_warning_cache = set()
        self._weight_norm_removed = False
        self._load_latent_stats(config.latent_stats_path)
        if bool(getattr(config, "remove_weight_norm_infer", False)):
            self.apply_remove_weight_norm()
        # self.eval()

    def _warn_once(self, key: str, message: str):
        if key in self._latent_stats_warning_cache:
            return
        self._latent_stats_warning_cache.add(key)
        print(f"| WARN: {message}")

    @staticmethod
    def _remove_weight_norm_from_module(module: nn.Module) -> bool:
        if not (hasattr(module, "weight_g") and hasattr(module, "weight_v")):
            return False
        try:
            remove_weight_norm(module)
            return True
        except (ValueError, AttributeError):
            return False

    def apply_remove_weight_norm(self):
        if self._weight_norm_removed:
            return self
        removed_count = 0
        for module in self.modules():
            removed_count += int(self._remove_weight_norm_from_module(module))
        self._weight_norm_removed = True
        print_once(f"| VAE removed weight norm for infer ({removed_count} modules)")
        return self

    @staticmethod
    def _fmt_mem_mib(num_bytes: Optional[int]) -> str:
        if num_bytes is None:
            return "-"
        return f"{float(num_bytes) / (1024.0 ** 2):.1f}MiB"

    def _maybe_take_cuda_mem_snapshot(
        self,
        device: torch.device,
        tag: str,
        profile: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.synchronize(dev_idx)
        snap = {
            "tag": tag,
            "allocated_bytes": int(torch.cuda.memory_allocated(dev_idx)),
            "reserved_bytes": int(torch.cuda.memory_reserved(dev_idx)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(dev_idx)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(dev_idx)),
        }
        if profile is not None:
            profile.append(snap)
        return snap

    def _print_cuda_mem_profile(self, profile: List[Dict[str, Any]], device: torch.device) -> None:
        if device.type != "cuda" or len(profile) == 0:
            return
        print("| VAE encode memory profile")
        for snap in profile:
            print(
                "| "
                f"{snap['tag']}: "
                f"alloc={self._fmt_mem_mib(snap['allocated_bytes'])}, "
                f"reserved={self._fmt_mem_mib(snap['reserved_bytes'])}, "
                f"max_alloc={self._fmt_mem_mib(snap['max_allocated_bytes'])}, "
                f"max_reserved={self._fmt_mem_mib(snap['max_reserved_bytes'])}"
            )

    def _load_latent_stats(self, path: Optional[str]):
        self.latent_stats = None
        if not path:
            return
        try:
            with np.load(path, allow_pickle=False) as data:
                stats: Dict[str, Any] = {}
                for key in data.files:
                    value = data[key]
                    if isinstance(value, np.ndarray) and value.shape == ():
                        stats[key] = value.item()
                    else:
                        stats[key] = torch.from_numpy(np.array(value))

            latent_dim = int(stats.get("latent_dim"))
            if latent_dim != self.config.latent_dim:
                raise ValueError(
                    f"latent_dim mismatch in latent stats: stats={latent_dim}, model={self.config.latent_dim}"
                )

            self.latent_stats = stats
            self.config.latent_stats_path = path
            print_once(f"| VAE loaded latent stats from {path}")
        except Exception as e:
            self._warn_once("latent_stats_load", f"failed to load latent stats from {path}: {e}")
            self.latent_stats = None

    def _mode_key_prefix(self, key: str) -> str:
        if key not in {"z", "mu"}:
            raise ValueError(f"unsupported latent key: {key}")
        return key

    def _stats_tensor(self, name: str, ref: torch.Tensor) -> Optional[torch.Tensor]:
        if self.latent_stats is None:
            return None
        value = self.latent_stats.get(name)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device=ref.device, dtype=torch.float32)
        return torch.tensor(value, device=ref.device, dtype=torch.float32)

    def _has_required_stats(self, key: str, mode: str) -> bool:
        if self.latent_stats is None:
            return False
        prefix = self._mode_key_prefix(key)
        required = {
            "global": [f"{prefix}_global_mean", f"{prefix}_global_std"],
            "per_channel": [f"{prefix}_mean_c", f"{prefix}_std_c"],
            "pca": [f"{prefix}_mean_c", f"{prefix}_pca_whiten_mat", f"{prefix}_pca_unwhiten_mat"],
        }.get(mode, [])
        if not required:
            return False
        for name in required:
            if name not in self.latent_stats:
                return False

        if mode in {"per_channel", "pca"}:
            mean_c = self.latent_stats[f"{prefix}_mean_c"]
            if not isinstance(mean_c, torch.Tensor) or mean_c.shape != (self.config.latent_dim,):
                return False
        if mode == "per_channel":
            std_c = self.latent_stats[f"{prefix}_std_c"]
            if not isinstance(std_c, torch.Tensor) or std_c.shape != (self.config.latent_dim,):
                return False
        if mode == "pca":
            expected_shape = (self.config.latent_dim, self.config.latent_dim)
            whiten = self.latent_stats[f"{prefix}_pca_whiten_mat"]
            unwhiten = self.latent_stats[f"{prefix}_pca_unwhiten_mat"]
            if not isinstance(whiten, torch.Tensor) or whiten.shape != expected_shape:
                return False
            if not isinstance(unwhiten, torch.Tensor) or unwhiten.shape != expected_shape:
                return False
        return True

    def _get_effective_norm_mode(self, requested_mode: Optional[str] = None, key: str = "z") -> str:
        mode = requested_mode if requested_mode is not None else getattr(self.config, "latent_norm_mode", "per_channel")
        mode = "none" if mode is None else str(mode).lower()
        valid_modes = {"none", "global", "per_channel", "pca"}
        if mode not in valid_modes:
            self._warn_once(
                f"latent_norm_mode_invalid_{mode}",
                f"unsupported latent_norm_mode={mode}, fallback to scalar global stats or none.",
            )
            mode = "none"

        if mode == "none":
            return "none"

        if mode != "none" and self._has_required_stats(key, mode):
            return mode

        if mode != "none" and self.latent_stats is not None and not self._has_required_stats(key, mode):
            self._warn_once(
                f"latent_stats_missing_{key}_{mode}",
                f"latent stats missing fields for key={key}, mode={mode}; fallback to global stats or none.",
            )

        if self._has_required_stats(key, "global"):
            return "global"

        if key == "z" and self.config.latent_mean is not None and self.config.latent_std is not None:
            return "global"

        return "none"

    def _normalize_latent(self, latent: torch.Tensor, key: str = "z", mode: Optional[str] = None) -> torch.Tensor:
        eff_mode = self._get_effective_norm_mode(mode, key=key)
        if eff_mode == "none":
            return latent

        prefix = self._mode_key_prefix(key)
        orig_dtype = latent.dtype
        work = latent.float()

        if eff_mode == "global":
            if self._has_required_stats(key, "global"):
                mean = self._stats_tensor(f"{prefix}_global_mean", work)
                std = self._stats_tensor(f"{prefix}_global_std", work)
            else:
                mean = work.new_tensor(float(self.config.latent_mean))
                std = work.new_tensor(float(self.config.latent_std))
            work = (work - mean) / (std + 1e-12)
        elif eff_mode == "per_channel":
            mean = self._stats_tensor(f"{prefix}_mean_c", work).view(*([1] * (work.ndim - 1)), -1)
            std = self._stats_tensor(f"{prefix}_std_c", work).view(*([1] * (work.ndim - 1)), -1)
            work = (work - mean) / (std + 1e-12)
        elif eff_mode == "pca":
            mean = self._stats_tensor(f"{prefix}_mean_c", work).view(*([1] * (work.ndim - 1)), -1)
            whiten = self._stats_tensor(f"{prefix}_pca_whiten_mat", work)
            work = torch.matmul(work - mean, whiten)

        return work.to(dtype=orig_dtype)

    def _denormalize_latent(self, latent: torch.Tensor, key: str = "z", mode: Optional[str] = None) -> torch.Tensor:
        eff_mode = self._get_effective_norm_mode(mode, key=key)
        if eff_mode == "none":
            return latent

        prefix = self._mode_key_prefix(key)
        orig_dtype = latent.dtype
        work = latent.float()

        if eff_mode == "global":
            if self._has_required_stats(key, "global"):
                mean = self._stats_tensor(f"{prefix}_global_mean", work)
                std = self._stats_tensor(f"{prefix}_global_std", work)
            else:
                mean = work.new_tensor(float(self.config.latent_mean))
                std = work.new_tensor(float(self.config.latent_std))
            work = work * (std + 1e-12) + mean
        elif eff_mode == "per_channel":
            mean = self._stats_tensor(f"{prefix}_mean_c", work).view(*([1] * (work.ndim - 1)), -1)
            std = self._stats_tensor(f"{prefix}_std_c", work).view(*([1] * (work.ndim - 1)), -1)
            work = work * (std + 1e-12) + mean
        elif eff_mode == "pca":
            mean = self._stats_tensor(f"{prefix}_mean_c", work).view(*([1] * (work.ndim - 1)), -1)
            unwhiten = self._stats_tensor(f"{prefix}_pca_unwhiten_mat", work)
            work = torch.matmul(work, unwhiten) + mean

        return work.to(dtype=orig_dtype)

    def normalize_latent(self, latent: torch.Tensor, key: str = "z", mode: Optional[str] = None) -> torch.Tensor:
        return self._normalize_latent(latent, key=key, mode=mode)

    def denormalize_latent(self, latent: torch.Tensor, key: str = "z", mode: Optional[str] = None) -> torch.Tensor:
        return self._denormalize_latent(latent, key=key, mode=mode)

    # ---------- helpers ----------
    def _sec_to_samples(self, sec: float) -> int:
        return max(1, int(round(sec * self.sample_rate)))

    def _align_up(self, x: int, multiple: int) -> int:
        return ((x + multiple - 1) // multiple) * multiple

    def _get_total_radius_samples(self) -> int:
        if "total_radius_samples" in self.context_info:
            return int(self.context_info["total_radius_samples"])
        tp = self.context_info.get("transformer_part", {})
        if tp.get("mode") == "sliding_window" and "total_radius_samples" in tp:
            return int(tp["total_radius_samples"])
        return int(self.context_info["conv_radius_samples"])

    def _latent_len_from_samples(self, n_samples: int) -> int:
        # Stable rounding: ceil(n_samples / hop)
        return max(1, (n_samples + self.hop_length - 1) // self.hop_length)

    def _get_total_radius_seconds(self) -> float:
        if "total_radius_seconds" in self.context_info:
            return float(self.context_info["total_radius_seconds"])
        tp = self.context_info.get("transformer_part", {})
        if tp.get("mode") == "sliding_window" and "total_radius_seconds" in tp:
            return float(tp["total_radius_seconds"])
        return float(self.context_info["conv_radius_seconds"])

    def _encoder_chunking_supported(self) -> bool:
        return bool(self.context_info.get("chunking_supported", True))

    def _decoder_chunking_supported(self) -> bool:
        return bool(self.decoder_context_info.get("chunking_supported", True))

    def _validate_encode_chunking_supported(self, chunk_mode: str) -> None:
        if chunk_mode != "full" and not self._encoder_chunking_supported():
            raise ValueError("chunked encode is unsupported for the current encoder window configuration.")

    def _validate_decode_chunking_supported(self, chunk_mode: str) -> None:
        if chunk_mode != "full" and not self._decoder_chunking_supported():
            raise ValueError("chunked decode is unsupported for the current decoder window configuration.")
    
    def _auto_chunk_sec(
        self,
        safety_sec: float = 0.5,
        min_sec: float = 2.0,
        max_sec: float = 20.0,
    ) -> float:
        r_sec = self._get_total_radius_seconds()
        # 2R + safety, then clamp
        chunk_sec = 2.0 * r_sec + safety_sec
        chunk_sec = max(min_sec, min(max_sec, chunk_sec))
        return float(chunk_sec)

    def _get_decoder_total_radius_seconds(self) -> float:
        if "overlap_seconds_conservative" in self.decoder_context_info:
            return float(self.decoder_context_info["overlap_seconds_conservative"])
        if "radius_seconds_equivalent" in self.decoder_context_info:
            return float(self.decoder_context_info["radius_seconds_equivalent"])
        left = float(self.decoder_context_info.get("left_radius_seconds_max", 0.0))
        right = float(self.decoder_context_info.get("right_radius_seconds_max", 0.0))
        return max(left, right)

    def _get_decoder_overlap_frames_conservative(self) -> int:
        if "latent_overlap_frames_conservative" in self.decoder_context_info:
            return int(self.decoder_context_info["latent_overlap_frames_conservative"])
        left = float(self.decoder_context_info.get("latent_left_radius_frames_max", 0.0))
        right = float(self.decoder_context_info.get("latent_right_radius_frames_max", 0.0))
        if left > 0.0 or right > 0.0:
            return int(math.ceil(max(left, right)))
        radius = float(self.decoder_context_info.get("latent_radius_frames_equivalent", 0.0))
        return max(1, int(math.ceil(radius)))

    def _auto_decode_chunk_sec(
        self,
        safety_sec: float = 0.5,
        min_sec: float = 2.0,
        max_sec: float = 20.0,
    ) -> float:
        r_sec = self._get_decoder_total_radius_seconds()
        chunk_sec = 2.0 * r_sec + safety_sec
        chunk_sec = max(min_sec, min(max_sec, chunk_sec))
        return float(chunk_sec)

    def _resolve_chunk_mode(
        self,
        chunk_sec: Optional[Union[float, str]],
    ) -> Tuple[str, Optional[float]]:
        if chunk_sec is None:
            return "full", None
        if isinstance(chunk_sec, str):
            chunk_sec_norm = chunk_sec.strip().lower()
            if chunk_sec_norm == "auto":
                return "auto_chunk", self._auto_chunk_sec()
            raise ValueError(f"unsupported chunk_sec string: {chunk_sec!r}; expected None, 'auto', or a positive number")
        if isinstance(chunk_sec, bool):
            raise ValueError(f"unsupported chunk_sec type: {type(chunk_sec).__name__}")
        try:
            chunk_sec_value = float(chunk_sec)
        except (TypeError, ValueError):
            raise ValueError(f"unsupported chunk_sec value: {chunk_sec!r}") from None
        if not math.isfinite(chunk_sec_value) or chunk_sec_value <= 0.0:
            raise ValueError(f"chunk_sec must be a positive finite number, got {chunk_sec!r}")
        return "fixed_chunk", chunk_sec_value

    def _resolve_decode_chunk_mode(
        self,
        chunk_sec: Optional[Union[float, str]],
    ) -> Tuple[str, Optional[float]]:
        if isinstance(chunk_sec, str) and chunk_sec.strip().lower() == "auto":
            return "auto_chunk", self._auto_decode_chunk_sec()
        return self._resolve_chunk_mode(chunk_sec)

    def _encode_latent_full(
        self,
        audio: torch.Tensor,
        audio_lengths_t: torch.Tensor,
        normalize: bool,
        deterministic: bool,
        return_mu_logvar: bool,
        return_kl: bool,
        estimate_bandwidth: bool,
        bandwidth_candidates_hz: Optional[Sequence[float]],
        bandwidth_return_details: bool,
        return_dict: bool,
        profile_memory: bool,
        memory_profile: Optional[List[Dict[str, Any]]],
        chunk_mode: str = "full",
    ):
        B, _, T_in_max = audio.shape
        device = audio.device
        hop = int(self.hop_length)
        T_in_aligned = self._align_up(int(T_in_max), hop)
        target_T_full = int(T_in_aligned // hop)
        z_lengths_t = ((audio_lengths_t + hop - 1) // hop).clamp(min=0, max=target_T_full)

        audio_full = audio
        if T_in_aligned > T_in_max:
            audio_full = nn.functional.pad(audio_full, (0, T_in_aligned - T_in_max))

        enc_mask = sequence_mask(audio_lengths_t, maxlen=audio_full.shape[-1])
        z_enc = self.encoder(audio_full, enc_mask)
        self._maybe_take_cuda_mem_snapshot(device, "after_full_encoder_forward", memory_profile)
        mu_full, logvar_full = self.bottleneck.encode(z_enc)
        z_full = mu_full if deterministic else self.bottleneck.reparameterize(mu_full, logvar_full)
        self._maybe_take_cuda_mem_snapshot(device, "after_full_reparameterize", memory_profile)

        z_pad = z_full.transpose(1, 2)
        if z_pad.shape[1] != target_T_full:
            raise RuntimeError(
                f"Full encode latent length mismatch: got {z_pad.shape[1]}, expected {target_T_full}. "
                f"T_in_aligned={T_in_aligned}, hop={hop}."
            )

        if normalize:
            z_pad = self._normalize_latent(z_pad, key="mu" if deterministic else "z")

        t_lat = torch.arange(target_T_full, device=device)[None, :]
        z_valid_mask = (t_lat < z_lengths_t[:, None]).to(z_pad.dtype)[:, :, None]
        z_pad = z_pad * z_valid_mask
        self._maybe_take_cuda_mem_snapshot(device, "final", memory_profile)

        if profile_memory and memory_profile is not None:
            self._print_cuda_mem_profile(memory_profile, device)

        if not return_dict:
            return z_pad, z_lengths_t

        ret: Dict[str, Any] = {
            "z": z_pad,
            "z_lengths": z_lengths_t,
            "chunk_mode": chunk_mode,
            "deterministic": deterministic,
        }

        mu_pad = None
        lv_pad = None
        need_mu = deterministic or return_mu_logvar or return_kl
        if need_mu:
            mu_pad = mu_full.transpose(1, 2)
            if mu_pad.shape[1] != target_T_full:
                raise RuntimeError(
                    f"Full encode mu length mismatch: got {mu_pad.shape[1]}, expected {target_T_full}."
                )
            mu_pad = mu_pad * z_valid_mask

        if return_mu_logvar or return_kl:
            lv_pad = logvar_full.transpose(1, 2)
            if lv_pad.shape[1] != target_T_full:
                raise RuntimeError(
                    f"Full encode logvar length mismatch: got {lv_pad.shape[1]}, expected {target_T_full}."
                )
            lv_pad = lv_pad * z_valid_mask

        if deterministic:
            ret["mu"] = mu_pad
        if return_mu_logvar:
            ret["mu"] = mu_pad
            ret["logvar"] = lv_pad

        if return_kl:
            with torch.cuda.amp.autocast(enabled=False):
                mu32 = mu_pad.float()
                lv32 = lv_pad.float().clamp(min=-30, max=20)
                kl = -0.5 * (1 + lv32 - mu32.pow(2) - lv32.exp())
                frame_mask = z_valid_mask.float()
                denom = (frame_mask.sum(dim=1).squeeze(-1) * mu32.shape[-1]).clamp_min(1.0)
                kl = (kl * frame_mask).sum(dim=(1, 2)) / denom
            ret["kl"] = kl

        if estimate_bandwidth:
            candidate_cutoffs_hz = self.config.bandwidth_candidates_hz if bandwidth_candidates_hz is None else bandwidth_candidates_hz
            bandwidth_out = estimate_effective_bandwidth_torch(
                audio.squeeze(1),
                sample_rate=self.sample_rate,
                wav_lens=audio_lengths_t,
                candidate_cutoffs_hz=candidate_cutoffs_hz,
                return_details=bandwidth_return_details,
            )
            if bandwidth_return_details:
                ret["bandwidth_hz"] = bandwidth_out["bandwidth_hz"]
                ret["bandwidth_confidence"] = bandwidth_out["confidence"]
                ret["bandwidth_selected_tail_ratio"] = bandwidth_out["selected_tail_ratio"]
                ret["bandwidth_selected_drop_db"] = bandwidth_out["selected_drop_db"]
            else:
                ret["bandwidth_hz"] = bandwidth_out

        if profile_memory and memory_profile is not None:
            ret["memory_profile"] = memory_profile

        return ret

    def _decode_full(
        self,
        latent: torch.Tensor,
        latent_lengths_valid: torch.Tensor,
        normalize: bool,
        bandwidth_hz: Optional[Union[float, Sequence[float], torch.Tensor]],
        apply_bandwidth_lowpass: bool,
        bandwidth_transition_hz: float,
        return_dict: bool,
        chunk_mode: str = "full",
    ):
        device = latent.device
        T = latent.shape[1]

        if normalize:
            latent = self._denormalize_latent(latent, key="z")

        t_lat = torch.arange(T, device=device)[None, :]
        latent_valid_mask = (t_lat < latent_lengths_valid[:, None]).to(latent.dtype)[:, :, None]
        latent = latent * latent_valid_mask

        target_S_full = int(T * self.hop_length)
        out_lengths = latent_lengths_valid * self.hop_length
        audio = super(InferenceWrapper, self).decode(latent)
        if audio.shape[-1] != target_S_full:
            raise RuntimeError(
                f"Full decode audio length mismatch: got {audio.shape[-1]}, expected {target_S_full}. "
                f"latent_T={T}, hop={self.hop_length}."
            )

        t = torch.arange(target_S_full, device=device)[None, None, :]
        mask = (t < out_lengths[:, None, None]).to(audio.dtype)
        audio = audio * mask
        if apply_bandwidth_lowpass and bandwidth_hz is not None:
            audio = apply_batch_variable_lowpass_torch(
                audio,
                sample_rate=self.sample_rate,
                cutoff_hz=bandwidth_hz,
                wav_lens=out_lengths,
                transition_width_hz=bandwidth_transition_hz,
            )

        if not return_dict:
            return audio, out_lengths

        return {
            "audio": audio,
            "audio_lengths": out_lengths,
            "chunk_mode": chunk_mode,
        }

    # ---------- encode ----------
    def encode_latent(
        self,
        audio: torch.Tensor,                        # [B,T] or [B,1,T]
        audio_lengths: Optional[torch.Tensor] = None,
        chunk_sec: Optional[Union[float, str]] = None,  # None => full, "auto" => heuristic, number => explicit chunk sec
        max_batch_size: int = 32,                   # encoder/bottleneck chunk batch size
        normalize: bool = True,
        deterministic: bool = False,
        return_mu_logvar: bool = False,
        return_kl: bool = False,
        estimate_bandwidth: bool = False,
        bandwidth_candidates_hz: Optional[Sequence[float]] = None,
        bandwidth_return_details: bool = False,
        return_dict: bool = False,
        profile_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
          z:         [B, T_max, C_z]
          z_lengths: [B]
          optional mu/logvar: same shape as z
          optional kl: [B], mean KL on valid latent frames
        """
        if not return_dict:
            return_mu_logvar = False
            return_kl = False
            estimate_bandwidth = False
            
        # ---- 0) normalize input ----
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)  # [B,1,T]
        assert audio.ndim == 3 and audio.shape[1] == 1, "audio must be [B,T] or [B,1,T]"

        B, _, T_in_max = audio.shape
        device = audio.device
        memory_profile: Optional[List[Dict[str, Any]]] = [] if profile_memory else None
        if profile_memory and device.type == "cuda" and torch.cuda.is_available():
            dev_idx = device.index if device.index is not None else torch.cuda.current_device()
            torch.cuda.synchronize(dev_idx)
            torch.cuda.reset_peak_memory_stats(dev_idx)
        self._maybe_take_cuda_mem_snapshot(device, "start", memory_profile)

        hop = int(self.hop_length)         # encoder hop in samples
        frame = int(self.frame_length)     # == hop for your WavVAE
        assert hop == frame

        chunk_mode, resolved_chunk_sec = self._resolve_chunk_mode(chunk_sec)
        self._validate_encode_chunking_supported(chunk_mode)

        # Output shape depends only on input T after hop alignment.
        T_in_aligned = self._align_up(int(T_in_max), hop)     # samples
        target_T_full = int(T_in_aligned // hop)              # latent frames

        # Keep length math on tensors to avoid graph breaks.
        if audio_lengths is None:
            audio_lengths_t = torch.full((B,), int(T_in_max), device=device, dtype=torch.long)
        elif isinstance(audio_lengths, torch.Tensor):
            audio_lengths_t = audio_lengths.to(device=device, dtype=torch.long)
        else:
            audio_lengths_t = torch.tensor(audio_lengths, device=device, dtype=torch.long)

        audio_lengths_t = audio_lengths_t.clamp(min=1, max=int(T_in_max))
        z_lengths_t = ((audio_lengths_t + hop - 1) // hop).clamp(min=0, max=target_T_full)  # [B]

        if chunk_mode == "full":
            return self._encode_latent_full(
                audio=audio,
                audio_lengths_t=audio_lengths_t,
                normalize=normalize,
                deterministic=deterministic,
                return_mu_logvar=return_mu_logvar,
                return_kl=return_kl,
                estimate_bandwidth=estimate_bandwidth,
                bandwidth_candidates_hz=bandwidth_candidates_hz,
                bandwidth_return_details=bandwidth_return_details,
                return_dict=return_dict,
                profile_memory=profile_memory,
                memory_profile=memory_profile,
                chunk_mode=chunk_mode,
            )

        # ---- 1) chunk size (seconds -> samples -> latent_len) ----
        chunk_samples_target = self._sec_to_samples(resolved_chunk_sec)
        # Align to hop to keep encoder output T_enc stable.
        chunk_samples = self._align_up(chunk_samples_target, hop)
        chunk_latent_len = chunk_samples // hop

        if T_in_aligned <= chunk_samples:
            return self._encode_latent_full(
                audio=audio,
                audio_lengths_t=audio_lengths_t,
                normalize=normalize,
                deterministic=deterministic,
                return_mu_logvar=return_mu_logvar,
                return_kl=return_kl,
                estimate_bandwidth=estimate_bandwidth,
                bandwidth_candidates_hz=bandwidth_candidates_hz,
                bandwidth_return_details=bandwidth_return_details,
                return_dict=return_dict,
                profile_memory=profile_memory,
                memory_profile=memory_profile,
                chunk_mode="full",
            )

        # context margin from receptive field
        radius_samples_total = self._get_total_radius_samples()
        margin_frames = max(0, int(math.ceil(float(radius_samples_total) / float(hop))))   # one-sided margin in latent frames
        margin_samples = margin_frames * hop

        min_chunk_latent = 2 * margin_frames + 1
        if chunk_latent_len < min_chunk_latent:
            raise ValueError(
                f"chunk_sec={chunk_sec} is too small: chunk_latent_len={chunk_latent_len}, "
                f"but it must be >= {min_chunk_latent} (=2*margin_frames+1). "
                f"Increase chunk_sec or reduce transformer window/layers."
            )

        step_samples = chunk_samples - 2 * margin_samples
        if step_samples <= 0:
            raise ValueError(
                f"step_samples<=0: chunk_samples={chunk_samples}, margin_samples={margin_samples}. "
                f"Increase chunk_sec or reduce context radius."
            )

        # Use an exact final chunk start instead of padding the last chunk and trimming later.
        # This keeps stitched latent length exactly equal to target_T_full.
        last_start = T_in_aligned - chunk_samples
        if last_start <= 0:
            chunk_starts_samples = [0]
        else:
            chunk_starts_samples = [0]
            while chunk_starts_samples[-1] < last_start:
                next_start = min(chunk_starts_samples[-1] + step_samples, last_start)
                if next_start <= chunk_starts_samples[-1]:
                    raise RuntimeError(
                        f"Invalid chunk start schedule: {next_start=} <= {chunk_starts_samples[-1]=}."
                    )
                chunk_starts_samples.append(next_start)

        # ---- 2) cut all chunks across batch, then cat into one big batch ----
        all_chunks: List[torch.Tensor] = []
        chunk_valid_lengths: List[int] = []
        chunk_start_frames: List[int] = []
        sample_chunk_start: List[int] = [0 for _ in range(B)]
        sample_chunk_count: List[int] = [0 for _ in range(B)]
        audio_lengths_list = [int(x) for x in audio_lengths_t.detach().cpu().tolist()]

        global_k = 0
        for b in range(B):
            # Chunk partition uses the input tensor length only to keep shape fixed.
            len_b = int(T_in_max)

            wav_b = audio[b:b+1, :, :len_b]  # [1,1,len_b]

            # pad to hop multiple (same as training preprocess idea)
            pad_right = self._align_up(len_b, frame) - len_b
            if pad_right > 0:
                wav_b = nn.functional.pad(wav_b, (0, pad_right))
            T_pad = wav_b.shape[-1]
            if T_pad != T_in_aligned:
                raise RuntimeError(f"Chunk input alignment mismatch: sample_T_pad={T_pad}, batch_T_aligned={T_in_aligned}.")

            sample_chunk_start[b] = global_k

            for start in chunk_starts_samples:
                end = start + chunk_samples
                chunk = wav_b[:, :, start:end]
                if chunk.shape[-1] != chunk_samples:
                    raise RuntimeError(
                        f"Chunk length mismatch: got {chunk.shape[-1]}, expected {chunk_samples}, "
                        f"{start=}, {end=}, {T_pad=}."
                    )
                valid_len = max(0, min(chunk_samples, audio_lengths_list[b] - start))

                all_chunks.append(chunk)                 # [1,1,chunk_samples]
                chunk_valid_lengths.append(valid_len)    # valid real-audio samples inside this chunk
                chunk_start_frames.append(start // hop)
                global_k += 1

            sample_chunk_count[b] = global_k - sample_chunk_start[b]

        if global_k == 0:
            raise RuntimeError("No chunks produced (check input lengths).")

        self._maybe_take_cuda_mem_snapshot(device, "after_chunk_build", memory_profile)
        chunk_wavs = torch.cat(all_chunks, dim=0)  # [N,1,chunk_samples]
        N_total = chunk_wavs.shape[0]
        chunk_lengths = torch.tensor(chunk_valid_lengths, device=device, dtype=torch.long)  # [N]
        self._maybe_take_cuda_mem_snapshot(device, "after_chunk_cat", memory_profile)

        # ---- 3) encoder + bottleneck in (almost) one shot, only split if needed ----
        # reuse the same arange across splits (avoid repeated allocations)
        t = torch.arange(chunk_samples, device=device)[None, :]  # [1,T]

        def run_range(i: int, j: int):
            wav_batch = chunk_wavs[i:j]            # [B_c,1,chunk_samples]
            len_batch = chunk_lengths[i:j]         # [B_c]

            # IMPORTANT: mask shape must be [B_c, T] (not [B_c,1,T])
            enc_mask = (t < len_batch[:, None])    # [B_c,T], bool

            z_enc = self.encoder(wav_batch, enc_mask)          # [B_c,C_enc,T_enc]
            mu_b, logvar_b = self.bottleneck.encode(z_enc)     # [B_c,C_z,T_enc]
            # logvar_b = logvar_b.clamp(min=-30, max=20)
            return mu_b, logvar_b

        mu_list, logvar_list = [], []
        if N_total <= max_batch_size:
            mu_b, logvar_b = run_range(0, N_total)
            mu_list.append(mu_b)
            logvar_list.append(logvar_b)
        else:
            for i in range(0, N_total, max_batch_size):
                j = min(N_total, i + max_batch_size)
                mu_b, logvar_b = run_range(i, j)
                mu_list.append(mu_b)
                logvar_list.append(logvar_b)

        self._maybe_take_cuda_mem_snapshot(device, "after_encoder_forward", memory_profile)
        mu_chunks = torch.cat(mu_list, dim=0)         # [N,C_z,T_enc]
        logvar_chunks = torch.cat(logvar_list, dim=0) # [N,C_z,T_enc]
        self._maybe_take_cuda_mem_snapshot(device, "after_mu_logvar_cat", memory_profile)
        z_chunks = mu_chunks if deterministic else self.bottleneck.reparameterize(mu_chunks, logvar_chunks)
        self._maybe_take_cuda_mem_snapshot(device, "after_reparameterize", memory_profile)

        _, C_z, T_enc = z_chunks.shape
        if T_enc != chunk_latent_len:
            raise RuntimeError(
                f"T_enc={T_enc} != chunk_latent_len={chunk_latent_len}. "
                f"chunk_samples={chunk_samples}, hop={hop}. Check alignment logic."
            )

        # ---- 4) stitch back per sample, trimming overlap margins ----
        z_per_sample: List[torch.Tensor] = []
        mu_per_sample: List[torch.Tensor] = []
        logvar_per_sample: List[torch.Tensor] = []
        kl_per_sample: List[torch.Tensor] = []
        need_mu = (deterministic and return_dict) or return_mu_logvar or return_kl
        need_logvar = return_mu_logvar or return_kl

        frame_ids = torch.arange(target_T_full, device=device)  # used by KL/mask

        for b in range(B):
            start_k = sample_chunk_start[b]
            ncb = sample_chunk_count[b]

            if ncb <= 0:
                z_per_sample.append(torch.zeros(target_T_full, C_z, device=device))
                if need_mu:
                    mu_per_sample.append(torch.zeros(target_T_full, C_z, device=device))
                if need_logvar:
                    logvar_per_sample.append(torch.zeros(target_T_full, C_z, device=device))
                if return_kl:
                    kl_per_sample.append(torch.tensor(0.0, device=device))
                continue

            kept_z, kept_mu, kept_logvar = [], [], []
            starts_f = chunk_start_frames[start_k:start_k + ncb]
            if len(starts_f) != ncb:
                raise RuntimeError(f"Chunk start bookkeeping mismatch: got {len(starts_f)}, expected {ncb}.")
            boundaries_f = [0]
            for ci in range(ncb - 1):
                prev_end_f = starts_f[ci] + T_enc
                next_start_f = starts_f[ci + 1]
                if next_start_f > prev_end_f:
                    raise RuntimeError(
                        f"Chunk schedule has a latent gap for sample {b}: "
                        f"prev_end={prev_end_f}, next_start={next_start_f}."
                    )
                boundary_f = (prev_end_f + next_start_f) // 2
                if boundary_f <= boundaries_f[-1]:
                    raise RuntimeError(
                        f"Non-monotonic stitch boundary for sample {b}: "
                        f"{boundary_f=} after {boundaries_f[-1]=}."
                    )
                boundaries_f.append(boundary_f)
            boundaries_f.append(target_T_full)

            for ci in range(ncb):
                k = start_k + ci
                z_k = z_chunks[k]
                mu_k = mu_chunks[k]
                lv_k = logvar_chunks[k]

                s_f = boundaries_f[ci] - starts_f[ci]
                e_f = boundaries_f[ci + 1] - starts_f[ci]

                if s_f < 0 or e_f > T_enc or s_f >= e_f:
                    raise RuntimeError(
                        f"Invalid chunk stitch slice for sample {b}, chunk {ci}: "
                        f"{s_f=}, {e_f=}, {T_enc=}, start_f={starts_f[ci]}, "
                        f"boundary=({boundaries_f[ci]}, {boundaries_f[ci + 1]})."
                    )

                kept_z.append(z_k[:, s_f:e_f])
                if need_mu:
                    kept_mu.append(mu_k[:, s_f:e_f])
                if need_logvar:
                    kept_logvar.append(lv_k[:, s_f:e_f])

            z_cat = torch.cat(kept_z, dim=-1)  # [C_z,T]
            if z_cat.shape[-1] != target_T_full:
                raise RuntimeError(
                    f"Chunk stitch latent length mismatch for sample {b}: "
                    f"got {z_cat.shape[-1]}, expected {target_T_full}."
                )

            z_per_sample.append(z_cat.transpose(0, 1))  # [target_T_full,C_z]

            if need_mu:
                mu_cat = torch.cat(kept_mu, dim=-1)
                if mu_cat.shape[-1] != target_T_full:
                    raise RuntimeError(
                        f"Chunk stitch mu length mismatch for sample {b}: "
                        f"got {mu_cat.shape[-1]}, expected {target_T_full}."
                    )

                mu_per_sample.append(mu_cat.transpose(0, 1))

            if need_logvar:
                lv_cat = torch.cat(kept_logvar, dim=-1)
                if lv_cat.shape[-1] != target_T_full:
                    raise RuntimeError(
                        f"Chunk stitch logvar length mismatch for sample {b}: "
                        f"got {lv_cat.shape[-1]}, expected {target_T_full}."
                    )
                logvar_per_sample.append(lv_cat.transpose(0, 1))

                if return_kl:
                    valid_T_b = z_lengths_t[b]  # tensor scalar
                    valid_mask = (frame_ids < valid_T_b).to(mu_cat.dtype)  # [T]
                    denom = valid_mask.sum().clamp(min=1.0)

                    with torch.cuda.amp.autocast(enabled=False):
                        mu32 = mu_cat.float()
                        lv32 = lv_cat.float().clamp(min=-30, max=20)
                        kl = -0.5 * (1 + lv32 - mu32.pow(2) - lv32.exp())  # [C_z,T]
                        kl = (kl * valid_mask[None, :]).sum() / (mu32.shape[0] * denom)
                    kl_per_sample.append(kl)

        self._maybe_take_cuda_mem_snapshot(device, "after_stitch", memory_profile)
        # ---- 5) stack to batch (fixed length) ----
        z_pad = torch.stack(z_per_sample, dim=0)  # [B,target_T_full,C_z]
        self._maybe_take_cuda_mem_snapshot(device, "after_stack", memory_profile)

        if normalize:
            z_pad = self._normalize_latent(z_pad, key="mu" if deterministic else "z")
            self._maybe_take_cuda_mem_snapshot(device, "after_normalize", memory_profile)

        t_lat = torch.arange(target_T_full, device=device)[None, :]
        z_valid_mask = (t_lat < z_lengths_t[:, None]).to(z_pad.dtype)[:, :, None]  # [B,T,1]
        z_pad = z_pad * z_valid_mask
        self._maybe_take_cuda_mem_snapshot(device, "final", memory_profile)

        if profile_memory and memory_profile is not None:
            self._print_cuda_mem_profile(memory_profile, device)

        if not return_dict:
            return z_pad, z_lengths_t

        ret: Dict[str, Any] = {
            "z": z_pad,
            "z_lengths": z_lengths_t,
            "chunk_mode": chunk_mode,
            "chunk_sec": z_pad.new_tensor(float(chunk_samples) / self.sample_rate),
            "margin_sec": z_pad.new_tensor(float(margin_samples) / self.sample_rate),
            "deterministic": deterministic,
        }

        mu_pad = None
        lv_pad = None
        if need_mu:
            mu_pad = torch.stack(mu_per_sample, dim=0) if len(mu_per_sample) else torch.zeros(B, target_T_full, C_z, device=device)
            mu_pad = mu_pad * z_valid_mask
        if need_logvar:
            lv_pad = torch.stack(logvar_per_sample, dim=0) if len(logvar_per_sample) else torch.zeros(B, target_T_full, C_z, device=device)
            lv_pad = lv_pad * z_valid_mask

        if deterministic:
            ret["mu"] = mu_pad
        if return_mu_logvar:
            ret["mu"] = mu_pad
            ret["logvar"] = lv_pad

        if return_kl:
            ret["kl"] = torch.stack(kl_per_sample, dim=0) if len(kl_per_sample) else torch.zeros(B, device=device)

        if estimate_bandwidth:
            candidate_cutoffs_hz = self.config.bandwidth_candidates_hz if bandwidth_candidates_hz is None else bandwidth_candidates_hz
            bandwidth_out = estimate_effective_bandwidth_torch(
                audio.squeeze(1),
                sample_rate=self.sample_rate,
                wav_lens=audio_lengths_t,
                candidate_cutoffs_hz=candidate_cutoffs_hz,
                return_details=bandwidth_return_details,
            )
            if bandwidth_return_details:
                ret["bandwidth_hz"] = bandwidth_out["bandwidth_hz"]
                ret["bandwidth_confidence"] = bandwidth_out["confidence"]
                ret["bandwidth_selected_tail_ratio"] = bandwidth_out["selected_tail_ratio"]
                ret["bandwidth_selected_drop_db"] = bandwidth_out["selected_drop_db"]
            else:
                ret["bandwidth_hz"] = bandwidth_out

        if profile_memory and memory_profile is not None:
            ret["memory_profile"] = memory_profile

        return ret

    # ---------- decode ----------
    def _sec_to_latent_frames(self, sec: float) -> int:
        # Latent frame rate: one frame per hop_length samples.
        n_samples = self._sec_to_samples(sec)
        return max(1, (n_samples + self.hop_length - 1) // self.hop_length)

    def decode(
        self,
        latent: torch.Tensor,                       # [B,T,C]
        latent_lengths: Optional[torch.Tensor] = None,
        max_batch_size: int = 32,
        normalize: bool = True,
        chunk_sec: Optional[Union[float, str]] = None,  # None => full, "auto" => heuristic, number => explicit chunk sec
        overlap_sec: Optional[float] = None,        # None => use conservative automatic overlap
        bandwidth_hz: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        apply_bandwidth_lowpass: bool = False,
        bandwidth_transition_hz: Optional[float] = None,
        return_dict: bool = False,                  # False => audio only; True => include lengths
    ):
        if latent.ndim != 3:
            raise ValueError("latent must be [B,T,C]")

        device = latent.device
        B, T, C = latent.shape
        if C != self.config.latent_dim:
            raise ValueError(f"latent dim mismatch: got C={C}, expect {self.config.latent_dim}")

        if latent_lengths is None:
            latent_lengths_valid = torch.full((B,), T, device=device, dtype=torch.long)
        else:
            latent_lengths_valid = latent_lengths.to(device).clamp(min=0, max=T)

        # Output length is fixed by latent length * hop.
        target_S_full = int(T * self.hop_length)
        out_lengths = latent_lengths_valid * self.hop_length
        if bandwidth_transition_hz is None:
            bandwidth_transition_hz = float(self.config.bandwidth_transition_hz)

        chunk_mode, resolved_chunk_sec = self._resolve_decode_chunk_mode(chunk_sec)
        self._validate_decode_chunking_supported(chunk_mode)
        if chunk_mode == "full":
            return self._decode_full(
                latent=latent,
                latent_lengths_valid=latent_lengths_valid,
                normalize=normalize,
                bandwidth_hz=bandwidth_hz,
                apply_bandwidth_lowpass=apply_bandwidth_lowpass,
                bandwidth_transition_hz=bandwidth_transition_hz,
                return_dict=return_dict,
                chunk_mode=chunk_mode,
            )

        # ---- chunked overlap decode ----
        chunk_T = self._sec_to_latent_frames(resolved_chunk_sec)
        if T <= chunk_T:
            return self._decode_full(
                latent=latent,
                latent_lengths_valid=latent_lengths_valid,
                normalize=normalize,
                bandwidth_hz=bandwidth_hz,
                apply_bandwidth_lowpass=apply_bandwidth_lowpass,
                bandwidth_transition_hz=bandwidth_transition_hz,
                return_dict=return_dict,
                chunk_mode="full",
            )

        if normalize:
            latent = self._denormalize_latent(latent, key="z")

        t_lat = torch.arange(T, device=device)[None, :]
        latent_valid_mask = (t_lat < latent_lengths_valid[:, None]).to(latent.dtype)[:, :, None]
        latent = latent * latent_valid_mask

        if overlap_sec is None:
            overlap_T = self._get_decoder_overlap_frames_conservative()
        else:
            overlap_T = self._sec_to_latent_frames(overlap_sec)
        overlap_sec = float(overlap_T * self.hop_length / self.sample_rate)

        # There must be a valid center region after trimming overlaps.
        if chunk_T < 2 * overlap_T + 1:
            raise ValueError(f"chunk_sec too small: chunk_T={chunk_T} < 2*overlap_T+1={2*overlap_T+1}")

        step_T = chunk_T - 2 * overlap_T
        if step_T <= 0:
            raise ValueError("step_T <= 0, increase chunk_sec or decrease overlap_sec")

        last_start_T = T - chunk_T
        if last_start_T <= 0:
            chunk_starts_T = [0]
        else:
            chunk_starts_T = [0]
            while chunk_starts_T[-1] < last_start_T:
                next_start = min(chunk_starts_T[-1] + step_T, last_start_T)
                if next_start <= chunk_starts_T[-1]:
                    raise RuntimeError(
                        f"Invalid decode chunk start schedule: {next_start=} <= {chunk_starts_T[-1]=}."
                    )
                chunk_starts_T.append(next_start)

        # 1) collect all chunks into one large batch
        all_z = []
        chunk_map = [[] for _ in range(B)]  # per-sample chunk indices
        decode_chunk_start_frames: List[int] = []
        global_k = 0

        active_bs = torch.nonzero(latent_lengths_valid > 0, as_tuple=True)[0].tolist()
        for b in active_bs:
            # Partition by input latent T to keep output shape fixed.
            for start in chunk_starts_T:
                end = start + chunk_T
                z = latent[b:b+1, start:end, :]
                if z.shape[1] != chunk_T:
                    raise RuntimeError(
                        f"Decode chunk latent length mismatch: got {z.shape[1]}, expected {chunk_T}, "
                        f"{start=}, {end=}, T={T}."
                    )
                all_z.append(z)
                chunk_map[b].append(global_k)
                decode_chunk_start_frames.append(start)
                global_k += 1

        if global_k == 0:
            audio = torch.zeros(B, 1, target_S_full, device=device)
            return {"audio": audio, "audio_lengths": out_lengths, "chunk_mode": chunk_mode} if return_dict else (audio, out_lengths)

        z_chunks = torch.cat(all_z, dim=0)  # [N,chunk_T,C]

        # 2) decode chunks by mini-batch
        wav_chunks = []
        for i in range(0, z_chunks.shape[0], max_batch_size):
            z_b = z_chunks[i:i+max_batch_size]   # [B_c,chunk_T,C]
            wav_b = super(InferenceWrapper, self).decode(z_b)  # [B_c,1,chunk_T*hop] in the common case
            expected_chunk_S = int(chunk_T * self.hop_length)
            if wav_b.shape[-1] != expected_chunk_S:
                raise RuntimeError(
                    f"Chunk decode audio length mismatch: got {wav_b.shape[-1]}, expected {expected_chunk_S}."
                )
            wav_chunks.append(wav_b)
        wav_chunks = torch.cat(wav_chunks, dim=0)  # [N,1,Tc]

        # 3) Stitch only the valid center region of each chunk.
        # Internal chunk boundaries are chosen inside the region where both adjacent chunks have
        # full decoder context. This avoids mixing boundary-contaminated samples into the result.
        audio = torch.zeros(B, 1, target_S_full, device=device, dtype=wav_chunks.dtype)

        for b in range(B):
            idxs = chunk_map[b]
            if len(idxs) == 0:
                continue

            starts_f = [decode_chunk_start_frames[k] for k in idxs]
            boundaries_f = [0]
            for ci in range(len(idxs) - 1):
                prev_start_f = starts_f[ci]
                next_start_f = starts_f[ci + 1]
                prev_valid_end_f = prev_start_f + chunk_T - overlap_T
                next_valid_start_f = next_start_f + overlap_T
                if next_valid_start_f > prev_valid_end_f:
                    raise RuntimeError(
                        f"Decode chunk valid centers do not overlap for sample {b}: "
                        f"prev_valid_end={prev_valid_end_f}, next_valid_start={next_valid_start_f}, "
                        f"chunk_T={chunk_T}, overlap_T={overlap_T}."
                    )
                boundary_f = (prev_valid_end_f + next_valid_start_f) // 2
                if boundary_f <= boundaries_f[-1]:
                    raise RuntimeError(
                        f"Non-monotonic decode stitch boundary for sample {b}: "
                        f"{boundary_f=} after {boundaries_f[-1]=}."
                    )
                boundaries_f.append(boundary_f)
            boundaries_f.append(T)

            for ci, k in enumerate(idxs):
                w = wav_chunks[k, 0]
                start_f = decode_chunk_start_frames[k]
                s_f = boundaries_f[ci] - start_f
                e_f = boundaries_f[ci + 1] - start_f
                if s_f < 0 or e_f > chunk_T or s_f >= e_f:
                    raise RuntimeError(
                        f"Invalid decode chunk stitch slice for sample {b}, chunk {ci}: "
                        f"{s_f=}, {e_f=}, {chunk_T=}, start_f={start_f}, "
                        f"boundary=({boundaries_f[ci]}, {boundaries_f[ci + 1]})."
                    )

                src_start_S = int(s_f * self.hop_length)
                src_end_S = int(e_f * self.hop_length)
                dst_start_S = int(boundaries_f[ci] * self.hop_length)
                dst_end_S = int(boundaries_f[ci + 1] * self.hop_length)
                if src_start_S < 0 or src_end_S > w.numel() or dst_start_S < 0 or dst_end_S > target_S_full:
                    raise RuntimeError(
                        f"Invalid decode chunk placement for sample {b}, chunk {ci}: "
                        f"src=({src_start_S}, {src_end_S})/{w.numel()}, "
                        f"dst=({dst_start_S}, {dst_end_S})/{target_S_full}."
                    )
                audio[b, 0, dst_start_S:dst_end_S] = w[src_start_S:src_end_S]

        t = torch.arange(target_S_full, device=device)[None, None, :]
        mask = (t < out_lengths[:, None, None]).to(audio.dtype)
        audio = audio * mask
        if apply_bandwidth_lowpass and bandwidth_hz is not None:
            audio = apply_batch_variable_lowpass_torch(
                audio,
                sample_rate=self.sample_rate,
                cutoff_hz=bandwidth_hz,
                wav_lens=out_lengths,
                transition_width_hz=bandwidth_transition_hz,
            )

        return {
            "audio": audio,
            "audio_lengths": out_lengths,
            "chunk_mode": chunk_mode,
            "chunk_sec": torch.tensor(resolved_chunk_sec, device=device),
            "overlap_sec": torch.tensor(overlap_sec, device=device),
        } if return_dict else (audio, out_lengths)


def compute_wavvae_window_size(config, seq_len_samples=None):
    k0 = int(config.kernel_size)
    sr = int(config.sample_rate)
    strides = [int(x) for x in config.encoder_rates]
    enc_ks = getattr(config, "encoder_kernel_sizes", [k0] * len(strides))
    enc_ks = [int(x) for x in enc_ks]
    enc_ds = getattr(config, "encoder_dilations", [(1, 3, 9)] * len(strides))
    enc_ds = [tuple(int(x) for x in stage) if isinstance(stage, (list, tuple)) else (int(stage),) for stage in enc_ds]
    if len(enc_ks) != len(strides) or len(enc_ds) != len(strides):
        raise ValueError("encoder kernel/dilation config length mismatch")
    aa = bool(getattr(config, "encoder_use_antialiasing", False))
    aa_taps = [_suggest_fir_taps(s, getattr(config, "encoder_antialias_taps_base", 9), getattr(config, "encoder_antialias_taps_scale", 4)) for s in strides]
    encoder_resample_mode = _check_resample_mode(str(getattr(config, "encoder_resample_mode", "phase0")))
    if aa and k0 % 2 == 0:
        raise ValueError(f"encoder AA projection requires odd kernel_size, got {k0}")

    interval = (0, 0)
    for s, k, ds, taps in reversed(list(zip(strides, enc_ks, enc_ds, aa_taps))):
        if aa and s > 1:
            interval = _backprop_same_conv_interval(interval, kernel_size=k0)
            if encoder_resample_mode in {"phase0", "decimate"}:
                interval = _backprop_decimate_interval(interval, stride=s)
            else:
                interval = _backprop_interpolate_down_ratio_interval(interval, stride=s, mode=encoder_resample_mode)
            interval = _backprop_same_conv_interval(interval, kernel_size=taps)
        else:
            interval = _backprop_strided_conv1d_interval(
                interval,
                stride=s,
                kernel_size=2 * s,
                padding=math.ceil(s / 2),
            )
        for d in reversed(ds):
            interval = _backprop_same_conv_interval(interval, kernel_size=k, dilation=d)
    interval = _backprop_same_conv_interval(interval, kernel_size=k0)

    hop = int(np.prod(strides))
    conv_info = _interval_to_radius_info(interval, sr)
    smooth_layers = max(int(getattr(config, "encoder_smoothing_layers", 0)), 0)
    smooth_kernel = int(getattr(config, "encoder_smoothing_kernel_size", 5))
    smooth_r = smooth_layers * max((smooth_kernel - 1) // 2, 0)
    L = int(getattr(config, "transformer_n_layers", 0))
    use_sw = bool(getattr(config, "transformer_use_sliding_window", False))
    W = getattr(config, "transformer_sliding_window_size", None)
    attn_r = 0
    chunking_supported = (L <= 0) or use_sw
    tp = {"chunking_supported": bool(chunking_supported)}
    if L <= 0:
        tp.update({"mode": "none", "attn_layers": 0, "attn_total_radius_frames": 0})
    elif not use_sw:
        tp.update({"mode": "global", "attn_layers": L, "description": "global attention is non-local; chunked encode is disabled."})
        if seq_len_samples is not None:
            tp["sequence_receptive_field_samples"] = int(seq_len_samples)
    else:
        per_layer = max(int(W) - 1, 0)
        attn_r = L * per_layer
        tp.update({"mode": "sliding_window", "attn_layers": L, "attn_window_param": int(W), "attn_radius_frames_per_layer": per_layer, "attn_total_radius_frames": attn_r, "attn_extra_radius_samples": attn_r * hop})

    total_interval = (interval[0] - (smooth_r + attn_r) * hop, interval[1] + (smooth_r + attn_r) * hop)
    total_info = _interval_to_radius_info(total_interval, sr)
    return {
        "conv_receptive_field_samples": conv_info["receptive_field_samples"],
        "conv_left_radius_samples": conv_info["left_radius_samples"],
        "conv_right_radius_samples": conv_info["right_radius_samples"],
        "conv_radius_samples": conv_info["radius_samples_conservative"],
        "conv_radius_seconds": conv_info["radius_seconds_conservative"],
        "hop_size": int(hop),
        "encoder_kernel_sizes": enc_ks,
        "encoder_dilations": [list(int(x) for x in stage) for stage in enc_ds],
        "encoder_antialias_taps": aa_taps if aa else None,
        "encoder_resample_mode": encoder_resample_mode,
        "smoothing_part": {"layers": smooth_layers, "kernel_size": smooth_kernel, "radius_frames_per_layer": max((smooth_kernel - 1) // 2, 0), "total_radius_frames": smooth_r, "extra_radius_samples": smooth_r * hop, "extra_radius_seconds": float(smooth_r * hop / sr)},
        "total_receptive_field_samples": total_info["receptive_field_samples"],
        "total_left_radius_samples": total_info["left_radius_samples"],
        "total_right_radius_samples": total_info["right_radius_samples"],
        "total_radius_samples": total_info["radius_samples_conservative"],
        "total_radius_seconds": total_info["radius_seconds_conservative"],
        "transformer_part": tp,
        "chunking_supported": bool(chunking_supported),
    }


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError(f"b must be positive, got {b}")
    return -((-a) // b)


def _backprop_conv1d_interval(interval: Tuple[int, int], kernel_size: int, padding: int, dilation: int = 1) -> Tuple[int, int]:
    left, right = interval
    eff = (int(kernel_size) - 1) * int(dilation)
    return left - int(padding), right - int(padding) + eff


def _backprop_strided_conv1d_interval(interval: Tuple[int, int], stride: int, kernel_size: int, padding: int, dilation: int = 1) -> Tuple[int, int]:
    left, right = interval
    eff = (int(kernel_size) - 1) * int(dilation)
    return left * int(stride) - int(padding), right * int(stride) - int(padding) + eff


def _backprop_same_conv_interval(interval: Tuple[int, int], kernel_size: int, dilation: int = 1) -> Tuple[int, int]:
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd for same-padding conv receptive field, got {kernel_size}")
    radius = ((kernel_size - 1) * dilation) // 2
    return interval[0] - radius, interval[1] + radius


def _backprop_decimate_interval(interval: Tuple[int, int], stride: int) -> Tuple[int, int]:
    return interval[0] * int(stride), interval[1] * int(stride)


def _check_resample_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in {"linear", "nearest", "nearest-exact", "phase0", "decimate"}:
        raise ValueError(f"unsupported 1D resample mode for window analysis: {mode}")
    return mode


def _backprop_interpolate_scale_up_interval(interval: Tuple[int, int], stride: int, mode: str = "linear") -> Tuple[int, int]:
    mode = _check_resample_mode(mode)
    if mode in {"phase0", "decimate"}:
        raise ValueError(f"{mode} is not an interpolate upsample mode")
    s = int(stride)
    x0 = (interval[0] + 0.5) / s - 0.5
    x1 = (interval[1] + 0.5) / s - 0.5
    return math.floor(x0), math.ceil(x1)


def _backprop_interpolate_down_ratio_interval(interval: Tuple[int, int], stride: int, mode: str = "linear") -> Tuple[int, int]:
    mode = _check_resample_mode(mode)
    if mode in {"phase0", "decimate"}:
        raise ValueError(f"{mode} is not an interpolate downsample mode")
    s = int(stride)
    x0 = (interval[0] + 0.5) * s - 0.5
    x1 = (interval[1] + 0.5) * s - 0.5
    return math.floor(x0), math.ceil(x1)


def _backprop_zero_interlace_lowpass_interval(interval: Tuple[int, int], stride: int, taps: int) -> Tuple[int, int]:
    s = int(stride)
    if s <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    radius = _ensure_odd(int(taps)) // 2
    return _ceil_div(int(interval[0]) - radius, s), (int(interval[1]) + radius) // s


def _backprop_adaa_snakebeta_interval(interval: Tuple[int, int]) -> Tuple[int, int]:
    return int(interval[0]) - 1, int(interval[1])


def _union_intervals(*intervals: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    valid = [iv for iv in intervals if iv is not None]
    if not valid:
        return None
    return min(int(iv[0]) for iv in valid), max(int(iv[1]) for iv in valid)


def _backprop_aa_snake_interval(interval: Tuple[int, int], oversample_factor: int, filt_taps: int, mode: str = "linear") -> Tuple[int, int]:
    s = max(1, int(oversample_factor))
    if s == 1:
        return interval
    interval = _backprop_interpolate_down_ratio_interval(interval, s, mode=mode)
    interval = _backprop_same_conv_interval(interval, kernel_size=_ensure_odd(filt_taps))
    interval = _backprop_interpolate_scale_up_interval(interval, s, mode=mode)
    return interval


def _interval_to_radius_info(interval: Tuple[int, int], sample_rate: int) -> Dict[str, Any]:
    left, right = int(interval[0]), int(interval[1])
    left_r, right_r = max(0, -left), max(0, right)
    rf = right - left + 1
    cons = max(left_r, right_r)
    return {
        "interval": [left, right],
        "receptive_field_samples": int(rf),
        "left_radius_samples": int(left_r),
        "right_radius_samples": int(right_r),
        "radius_samples_conservative": int(cons),
        "radius_seconds_conservative": float(cons / int(sample_rate)),
        "radius_samples_equivalent": float((rf - 1) / 2.0),
        "radius_seconds_equivalent": float((rf - 1) / (2.0 * int(sample_rate))),
    }


def _backprop_conv_transpose_interval(interval: Tuple[int, int], stride: int, kernel_size: int, padding: int) -> Tuple[int, int]:
    left, right = interval
    in_left = _ceil_div(left + padding - (kernel_size - 1), stride)
    in_right = (right + padding) // stride
    return in_left, in_right


def _resolve_wavvae_decoder_window_config(config):
    base_k = int(config.kernel_size)
    rates = [int(x) for x in config.decoder_rates]
    ks = getattr(config, "decoder_kernel_sizes", [base_k] * len(rates))
    ks = [int(x) for x in ks]
    ds = getattr(config, "decoder_dilations", [(1, 3, 9)] * len(rates))
    ds = [tuple(int(x) for x in stage) if isinstance(stage, (list, tuple)) else (int(stage),) for stage in ds]
    if len(ks) != len(rates) or len(ds) != len(rates):
        raise ValueError("decoder kernel/dilation config length mismatch")
    if bool(getattr(config, "decoder_use_antialias_upsample", False)) and base_k % 2 == 0:
        raise ValueError(f"decoder AA projection requires odd kernel_size, got {base_k}")
    highpass_prior = bool(getattr(config, "decoder_use_highpass_prior", True))
    highpass_prior_dim = int(getattr(config, "decoder_highpass_prior_dim", 256))
    if highpass_prior and highpass_prior_dim <= 0:
        raise ValueError(f"decoder_highpass_prior_dim must be positive, got {highpass_prior_dim}")
    highpass_prior_kernel_size = int(getattr(config, "decoder_highpass_prior_kernel_size", 7))
    if highpass_prior and highpass_prior_kernel_size % 2 == 0:
        raise ValueError(f"decoder_highpass_prior_kernel_size must be odd, got {highpass_prior_kernel_size}")
    highpass_prior_gain_init = float(getattr(config, "decoder_highpass_prior_gain_init", 0.1))
    highpass_prior_gain_max = float(getattr(config, "decoder_highpass_prior_gain_max", 0.5))
    if highpass_prior:
        _bounded_sigmoid_raw_from_value(highpass_prior_gain_init, highpass_prior_gain_max)
    cumulative_upps = []
    cumulative = 1
    for rate in rates:
        cumulative *= int(rate)
        cumulative_upps.append(int(cumulative))
    return {
        "sample_rate": int(config.sample_rate),
        "encoder_hop_size": int(np.prod(config.encoder_rates)),
        "decoder_rates": rates,
        "decoder_cumulative_upps": cumulative_upps,
        "decoder_upsample_factor": int(np.prod(rates)),
        "decoder_input_kernel_size": int(getattr(config, "decoder_input_kernel_size", base_k)),
        "decoder_output_kernel_size": int(getattr(config, "decoder_output_kernel_size", base_k)),
        "decoder_kernel_sizes": ks,
        "decoder_dilations": ds,
        "aa_proj_kernel_size": int(base_k),
        "aa_upsample": bool(getattr(config, "decoder_use_antialias_upsample", False)),
        "aa_upsample_taps": [_suggest_fir_taps(s, getattr(config, "decoder_antialias_taps_base", 9), getattr(config, "decoder_antialias_taps_scale", 4)) for s in rates],
        "aa_activation": bool(getattr(config, "decoder_use_aa_activation", False)),
        "adaa_snakebeta": bool(getattr(config, "decoder_use_adaa_snakebeta", False)),
        "aa_activation_oversample_factor": int(getattr(config, "decoder_aa_activation_oversample_factor", 2)),
        "aa_activation_taps": _ensure_odd(int(getattr(config, "decoder_aa_activation_taps", 13))),
        "resample_mode": str(getattr(config, "decoder_resample_mode", "linear")),
        "highpass_prior": highpass_prior,
        "highpass_prior_dim": highpass_prior_dim,
        "highpass_prior_kernel_size": highpass_prior_kernel_size,
        "highpass_prior_gain_init": highpass_prior_gain_init,
        "highpass_prior_gain_max": highpass_prior_gain_max,
    }


def compute_wavvae_decoder_window_size(config):
    cfg = _resolve_wavvae_decoder_window_config(config)
    sr, up = int(cfg["sample_rate"]), int(cfg["decoder_upsample_factor"])
    def act(iv):
        if not cfg["aa_activation"]:
            return iv
        if cfg["adaa_snakebeta"]:
            return _backprop_adaa_snakebeta_interval(iv)
        return _backprop_aa_snake_interval(iv, cfg["aa_activation_oversample_factor"], cfg["aa_activation_taps"], cfg["resample_mode"])
    def res(iv, k, d):
        iv = act(iv)
        iv = _backprop_same_conv_interval(iv, kernel_size=k, dilation=d)
        return act(iv)
    max_lf = max_rf = max_ls = max_rs = max_total = 0
    max_phase = 0
    reversed_blocks = list(zip(
        cfg["decoder_rates"],
        cfg["decoder_cumulative_upps"],
        cfg["decoder_kernel_sizes"],
        cfg["decoder_dilations"],
        cfg["aa_upsample_taps"],
    ))
    for phase in range(up):
        iv = (phase, phase)
        x0_prior_iv = None
        iv = _backprop_same_conv_interval(iv, kernel_size=cfg["decoder_output_kernel_size"])
        iv = act(iv)
        for s, cumulative_upps, k, ds, taps in reversed(reversed_blocks):
            for d in reversed(ds):
                iv = res(iv, k, d)
            if cfg["aa_upsample"] and s > 1:
                iv = _backprop_same_conv_interval(iv, kernel_size=cfg["aa_proj_kernel_size"])
                main_iv = _backprop_zero_interlace_lowpass_interval(iv, stride=s, taps=taps)
                if cfg["highpass_prior"]:
                    prior_iv = _backprop_same_conv_interval(iv, kernel_size=cfg["highpass_prior_kernel_size"])
                    # highpass = x0_up - lowpass(x0_up). The lowpass branch is a conservative superset
                    # of the direct zero-interlaced branch for receptive-field accounting.
                    prior_iv = _backprop_zero_interlace_lowpass_interval(
                        prior_iv,
                        stride=cumulative_upps,
                        taps=taps,
                    )
                    x0_prior_iv = _union_intervals(x0_prior_iv, prior_iv)
                iv = main_iv
            else:
                iv = _backprop_conv_transpose_interval(iv, stride=s, kernel_size=2 * s, padding=math.ceil(s / 2))
            iv = act(iv)
        iv = _union_intervals(iv, x0_prior_iv)
        iv = _backprop_same_conv_interval(iv, kernel_size=cfg["decoder_input_kernel_size"])
        l, r = int(iv[0]), int(iv[1])
        lf, rf = max(0, -l), max(0, r)
        ls, rs = float(phase - l * up), float(r * up - phase)
        total = ls + rs + 1.0
        max_lf, max_rf = max(max_lf, lf), max(max_rf, rf)
        max_ls, max_rs = max(max_ls, ls), max(max_rs, rs)
        if total > max_total:
            max_total, max_phase = total, phase
    overlap_f = int(max(max_lf, max_rf))
    return {
        "latent_receptive_field_frames_max": int(max_lf + max_rf + 1),
        "latent_radius_frames_equivalent": float((max_lf + max_rf) / 2.0),
        "upsample_factor": up,
        "input_kernel_size": int(cfg["decoder_input_kernel_size"]),
        "output_kernel_size": int(cfg["decoder_output_kernel_size"]),
        "aa_proj_kernel_size": int(cfg["aa_proj_kernel_size"]),
        "decoder_rates": [int(x) for x in cfg["decoder_rates"]],
        "decoder_cumulative_upps": [int(x) for x in cfg["decoder_cumulative_upps"]],
        "decoder_kernel_sizes": [int(x) for x in cfg["decoder_kernel_sizes"]],
        "decoder_dilations": [list(int(x) for x in stage) for stage in cfg["decoder_dilations"]],
        "decoder_antialias_taps": cfg["aa_upsample_taps"] if cfg["aa_upsample"] else None,
        "decoder_upsample_mode": "zero_interlace_lowpass" if cfg["aa_upsample"] else "conv_transpose",
        "decoder_use_highpass_prior": bool(cfg["highpass_prior"] and cfg["aa_upsample"]),
        "decoder_highpass_prior_source": "latent_prior_pre" if cfg["highpass_prior"] and cfg["aa_upsample"] else None,
        "decoder_highpass_prior_dim": int(cfg["highpass_prior_dim"]) if cfg["highpass_prior"] and cfg["aa_upsample"] else None,
        "decoder_highpass_prior_kernel_size": int(cfg["highpass_prior_kernel_size"]) if cfg["highpass_prior"] and cfg["aa_upsample"] else None,
        "decoder_highpass_prior_gain_init": float(cfg["highpass_prior_gain_init"]) if cfg["highpass_prior"] and cfg["aa_upsample"] else None,
        "decoder_highpass_prior_gain_max": float(cfg["highpass_prior_gain_max"]) if cfg["highpass_prior"] and cfg["aa_upsample"] else None,
        "decoder_use_adaa_snakebeta": bool(cfg["aa_activation"] and cfg["adaa_snakebeta"]),
        "latent_left_radius_frames_max": int(max_lf),
        "latent_right_radius_frames_max": int(max_rf),
        "latent_overlap_frames_conservative": int(overlap_f),
        "overlap_samples_conservative": float(overlap_f * up),
        "overlap_seconds_conservative": float(overlap_f * up / sr),
        "max_phase_with_decoder_window": int(max_phase),
        "left_radius_samples_max": float(max_ls),
        "right_radius_samples_max": float(max_rs),
        "left_radius_seconds_max": float(max_ls / sr),
        "right_radius_seconds_max": float(max_rs / sr),
        "total_receptive_field_samples_max": float(max_total),
        "radius_samples_equivalent": float((max_total - 1.0) / 2.0),
        "radius_seconds_equivalent": float((max_total - 1.0) / (2.0 * sr)),
        "chunking_supported": True,
    }


def compute_wavvae_e2e_window_size(config, seq_len_samples=None):
    enc = compute_wavvae_window_size(config, seq_len_samples=seq_len_samples)
    dec = compute_wavvae_decoder_window_size(config)
    sr = int(config.sample_rate)
    enc_hop, dec_up = int(enc["hop_size"]), int(dec["upsample_factor"])
    left = float(dec["left_radius_samples_max"]) + float(enc["total_left_radius_samples"])
    right = float(dec["right_radius_samples_max"]) + float(enc["total_right_radius_samples"])
    total = left + right + 1.0
    return {
        "encoder_window": enc,
        "decoder_part": dec,
        "e2e_part": {
            "max_phase_with_e2e_window": int(dec["max_phase_with_decoder_window"]),
            "left_radius_samples_max": float(left),
            "right_radius_samples_max": float(right),
            "left_radius_seconds_max": float(left / sr),
            "right_radius_seconds_max": float(right / sr),
            "total_receptive_field_samples_max": float(total),
            "radius_samples_equivalent": float((total - 1.0) / 2.0),
            "radius_seconds_equivalent": float((total - 1.0) / (2.0 * sr)),
            "encoder_left_radius_samples_used": float(enc["total_left_radius_samples"]),
            "encoder_right_radius_samples_used": float(enc["total_right_radius_samples"]),
            "decoder_left_radius_samples_used": float(dec["left_radius_samples_max"]),
            "decoder_right_radius_samples_used": float(dec["right_radius_samples_max"]),
            "decoder_overlap_frames_conservative_used": int(dec["latent_overlap_frames_conservative"]),
        },
        "encoder_hop_size": enc_hop,
        "decoder_upsample_factor": dec_up,
        "hop_matches_decoder": bool(enc_hop == dec_up),
        "chunking_supported": bool(enc.get("chunking_supported", True) and dec.get("chunking_supported", True)),
    }


if __name__ == '__main__':
    config = ModelArgs(
        sample_rate=48000,
        kernel_size=7,
        encoder_dim=128,
        encoder_kernel_sizes=[7, 7, 7, 7, 5, 5],
        encoder_rates=[2, 3, 5, 4, 4, 4],
        encoder_dilations=[[1, 3, 9], [1, 3, 9], [1, 3, 9], [1, 3, 9], [1, 3, 7], [1, 3, 5]],
        transformer_n_layers=0,
        encoder_smoothing_layers=0,
        latent_dim=64,
        decoder_dim=1920,
        decoder_rates=[5, 4, 4, 4, 3, 2]
    )
    window_info = compute_wavvae_e2e_window_size(config)
    print_once(f"WavVAE window info: {json_dumps(window_info)}")
