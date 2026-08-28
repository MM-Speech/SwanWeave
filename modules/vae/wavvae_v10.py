import os
import math
import json
import time
from typing import List, Tuple, Optional, Dict, Any, Sequence, Union
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext

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


class _TimingProfiler:
    def __init__(self, enabled: bool, device: Optional[torch.device]):
        self.enabled = bool(enabled)
        self.device = device
        self._device_index = None
        if self.enabled and device is not None and device.type == "cuda" and torch.cuda.is_available():
            self._device_index = device.index if device.index is not None else torch.cuda.current_device()
        self._stats: Dict[str, Dict[str, Any]] = {}

    def _sync(self):
        if self._device_index is not None:
            torch.cuda.synchronize(self._device_index)

    def start(self) -> Optional[float]:
        if not self.enabled:
            return None
        self._sync()
        return time.perf_counter()

    def stop(self, name: str, start_time: Optional[float]) -> float:
        if not self.enabled or start_time is None:
            return 0.0
        self._sync()
        elapsed = float(time.perf_counter() - start_time)
        stat = self._stats.get(name)
        if stat is None:
            stat = {"name": name, "total_sec": 0.0, "count": 0}
            self._stats[name] = stat
        stat["total_sec"] += elapsed
        stat["count"] += 1
        return elapsed

    @contextmanager
    def section(self, name: str):
        start_time = self.start()
        try:
            yield
        finally:
            self.stop(name, start_time)

    def summary(self) -> List[Dict[str, Any]]:
        out = []
        for stat in self._stats.values():
            total_sec = float(stat["total_sec"])
            count = int(stat["count"])
            out.append(
                {
                    "name": stat["name"],
                    "total_sec": total_sec,
                    "count": count,
                    "avg_sec": total_sec / max(count, 1),
                }
            )
        return out


class _CudaEventProfiler:
    def __init__(self, enabled: bool, device: Optional[torch.device]):
        self.enabled = bool(enabled)
        self.device = device
        self._device_index = None
        if self.enabled and device is not None and device.type == "cuda" and torch.cuda.is_available():
            self._device_index = device.index if device.index is not None else torch.cuda.current_device()
        else:
            self.enabled = False
        self._events: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(self._device_index)
        start.record(stream)
        try:
            yield
        finally:
            end.record(stream)
            self._events.append((name, start, end))

    def summary(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        torch.cuda.synchronize(self._device_index)
        stats: Dict[str, Dict[str, Any]] = {}
        for name, start, end in self._events:
            elapsed = float(start.elapsed_time(end)) / 1000.0
            stat = stats.get(name)
            if stat is None:
                stat = {"name": name, "total_sec": 0.0, "count": 0}
                stats[name] = stat
            stat["total_sec"] += elapsed
            stat["count"] += 1
        out = []
        for stat in stats.values():
            total_sec = float(stat["total_sec"])
            count = int(stat["count"])
            out.append(
                {
                    "name": stat["name"],
                    "total_sec": total_sec,
                    "count": count,
                    "avg_sec": total_sec / max(count, 1),
                }
            )
        return out

def build_wavvae(hparams, infer=False, attn_implementation='flash_attention_2', verbose=True):
    
    config = ModelArgs(
        sample_rate=hparams['audio_sample_rate'],
        kernel_size=hparams.get('kernel_size', 7),
        encoder_dim=hparams.get('encoder_dim', 32),
        encoder_kernel_sizes=hparams.get('encoder_kernel_sizes', (7, 7, 7, 7, 7, 7)),
        encoder_rates=hparams.get('encoder_rates', (2, 3, 5, 4, 4, 4)),
        encoder_dilations=hparams.get('encoder_dilations', ((1, 3, 9), (1, 3, 9), (1, 3, 9), (1, 3, 9), (1, 3, 9), (1, 3, 9))),
        transformer_n_layers=hparams.get('transformer_n_layers', 2),
        transformer_n_head=hparams.get('transformer_n_head', 16),
        transformer_n_kv_heads=hparams.get('transformer_n_kv_heads', 8),
        transformer_use_sliding_window=hparams.get('transformer_use_sliding_window', True),
        transformer_sliding_window_size=hparams.get('transformer_sliding_window_size', 16),
        encoder_smoothing_layers=hparams.get('encoder_smoothing_layers', 1),
        encoder_smoothing_kernel_size=hparams.get('encoder_smoothing_kernel_size', 5),
        encoder_smoothing_expansion=hparams.get('encoder_smoothing_expansion', 2),
        latent_dim=hparams.get('latent_dim', 64),
        decoder_dim=hparams.get('decoder_dim', 1920),
        decoder_rates=hparams.get('decoder_rates', (5, 4, 4, 4, 3, 2)),
        attn_implementation=attn_implementation,
        gradient_checkpointing=hparams.get('gradient_checkpointing', False),
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
    encoder_kernel_sizes: Tuple[int] = (7, 7, 7, 7, 7, 7)
    encoder_rates: Tuple[int] = (2, 3, 5, 4, 4, 4)
    encoder_dilations: Tuple[Tuple[int, ...], ...] = ((1, 3, 9), (1, 3, 9), (1, 3, 9), (1, 3, 9), (1, 3, 9), (1, 3, 9))
    
    # transformer
    transformer_n_layers: int = 8
    transformer_n_head: int = 16
    transformer_n_kv_heads: int = 8
    transformer_use_sliding_window: bool = True
    transformer_sliding_window_size: int = 16
    encoder_smoothing_layers: int = 0
    encoder_smoothing_kernel_size: int = 5
    encoder_smoothing_expansion: int = 2
    attn_implementation: str = 'flash_attention_2'

    # bottleneck
    latent_dim: int = 32
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

        # no-grad / infer 路径缓存
        self._infer_cache_key = None
        self._infer_cache_alpha = None
        self._infer_cache_beta = None

    def _get_infer_cached_alpha_beta(self, x: torch.Tensor):
        device_index = -1 if x.device.index is None else int(x.device.index)
        cache_key = (
            x.device.type,
            device_index,
            str(x.dtype),
            bool(self.logscale),
            int(self.alpha._version),
            int(self.beta._version),
        )

        if (
            self._infer_cache_key != cache_key
            or self._infer_cache_alpha is None
            or self._infer_cache_beta is None
        ):
            if self.logscale:
                alpha = torch.exp(self.alpha.detach()).to(device=x.device, dtype=x.dtype)
                beta = torch.exp(self.beta.detach()).to(device=x.device, dtype=x.dtype)
            else:
                alpha = self.alpha.detach().to(device=x.device, dtype=x.dtype)
                beta = self.beta.detach().to(device=x.device, dtype=x.dtype)

            self._infer_cache_key = cache_key
            self._infer_cache_alpha = alpha
            self._infer_cache_beta = beta

        return self._infer_cache_alpha, self._infer_cache_beta

    def forward(self, x):
        # 有梯度：保持原来的 fp32 稳定路径
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

        # 无梯度：使用缓存好的 alpha/beta
        alpha, beta = self._get_infer_cached_alpha_beta(x)

        # 这里 alpha/beta 已经是 exp 后的值，所以统一走 plain
        return snake_plain(x, alpha, beta)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, kernel_size: int = 7):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y

    def _forward_impl(self, x, timing_recorder=None, prefix: Optional[str] = None):
        if timing_recorder is None or prefix is None:
            return self.forward(x)
        residual = x
        with timing_recorder.section(f"{prefix}.snake0"):
            y = self.block[0](x)
        with timing_recorder.section(f"{prefix}.conv_dilated"):
            y = self.block[1](y)
        with timing_recorder.section(f"{prefix}.snake1"):
            y = self.block[2](y)
        with timing_recorder.section(f"{prefix}.conv1x1"):
            y = self.block[3](y)
        pad = (residual.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            residual = residual[..., pad:-pad]
        return residual + y
    

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
        self.in_act = Snake1d(dim)
        self.mid_act = Snake1d(dim)
        self.out_act = Snake1d(hidden_dim)
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

    def _forward_impl(self, x, timing_recorder=None, prefix: Optional[str] = None):
        if timing_recorder is None or prefix is None:
            return self.forward(x)
        with timing_recorder.section(f"{prefix}.in_act"):
            y = self.in_act(x)
        with timing_recorder.section(f"{prefix}.dwconv"):
            y = self.dwconv(y)
        with timing_recorder.section(f"{prefix}.mid_act"):
            y = self.mid_act(y)
        with timing_recorder.section(f"{prefix}.pw1"):
            y = self.pw1(y)
        with timing_recorder.section(f"{prefix}.out_act"):
            y = self.out_act(y)
        with timing_recorder.section(f"{prefix}.pw2"):
            y = self.pw2(y)
        return x + y
    

class EncoderBlock(nn.Module):
    def __init__(self, in_dim: int = 16, out_dim: int = 16, stride: int = 1, kernel_size: int = 7, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 3, 9]
        self.block = nn.Sequential(
            ResidualUnit(in_dim, dilation=dilations[0], kernel_size=kernel_size),
            ResidualUnit(in_dim, dilation=dilations[1], kernel_size=kernel_size),
            ResidualUnit(in_dim, dilation=dilations[2], kernel_size=kernel_size),
            Snake1d(in_dim),
            WNConv1d(
                in_dim,
                out_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)

    def _forward_impl(self, x, timing_recorder=None, prefix: Optional[str] = None):
        if timing_recorder is None or prefix is None:
            return self.forward(x)
        x = self.block[0]._forward_impl(x, timing_recorder=timing_recorder, prefix=f"{prefix}.res0")
        x = self.block[1]._forward_impl(x, timing_recorder=timing_recorder, prefix=f"{prefix}.res1")
        x = self.block[2]._forward_impl(x, timing_recorder=timing_recorder, prefix=f"{prefix}.res2")
        with timing_recorder.section(f"{prefix}.silu"):
            x = self.block[3](x)
        with timing_recorder.section(f"{prefix}.downsample_conv"):
            x = self.block[4](x)
        return x
    

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
        # d_model = d_model // 4
        # for i, stride in enumerate(strides):
        #     d_model *= 2
        #     self.block += [EncoderBlock(
        #         d_model, stride=stride, kernel_size=config.kernel_size,
        #         dilations=config.encoder_dilations[i],
        #     )]

        # Increase width every two downsampling stages instead of every stage.
        for i, stride in enumerate(strides):
            out_dim = config.encoder_dim * (2 ** (i // 2 + 1))
            self.block += [EncoderBlock(
                d_model, out_dim, stride=stride, 
                kernel_size=config.encoder_kernel_sizes[i],
                dilations=config.encoder_dilations[i],
            )]
            d_model = out_dim

        # Wrap black into nn.Sequential
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

    def _forward_impl(
        self,
        x,
        x_mask=None,
        pos_offset=0,
        timing_recorder: Optional[_TimingProfiler] = None,
        layer_timing_recorder=None,
        timing_detail: str = "block",
    ):
        if timing_recorder is not None:
            for idx, layer in enumerate(self.block):
                if timing_detail == "layer" and layer_timing_recorder is not None:
                    if isinstance(layer, EncoderBlock):
                        x = layer._forward_impl(
                            x,
                            timing_recorder=layer_timing_recorder,
                            prefix=f"encoder.block.{idx}",
                        )
                    else:
                        with layer_timing_recorder.section(f"encoder.block.{idx}.conv"):
                            x = layer(x)
                else:
                    with timing_recorder.section(f"encoder.block.{idx}"):
                        x = layer(x)
        elif self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
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
            if timing_recorder is not None:
                with timing_recorder.section("encoder.transformer"):
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
            else:
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

        if timing_recorder is not None and isinstance(self.smoothing_tail, nn.Sequential):
            for idx, layer in enumerate(self.smoothing_tail):
                if timing_detail == "layer" and layer_timing_recorder is not None and isinstance(layer, EncoderSmoothingBlock):
                    x = layer._forward_impl(
                        x,
                        timing_recorder=layer_timing_recorder,
                        prefix=f"encoder.smoothing.{idx}",
                    )
                else:
                    with timing_recorder.section(f"encoder.smoothing.{idx}"):
                        x = layer(x)
        else:
            x = self.smoothing_tail(x)

        if x_lat_mask is not None:
            x = x * x_lat_mask

        return x

    def forward(self, x, x_mask=None, pos_offset=0):
        return self._forward_impl(x, x_mask=x_mask, pos_offset=pos_offset, timing_recorder=None)
    

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

    def encode_with_timing(self, x, timing_recorder=None, prefix: str = "bottleneck"):
        if timing_recorder is None:
            return self.encode(x)
        with timing_recorder.section(f"{prefix}.mu"):
            mu = self.mu(x)
        with timing_recorder.section(f"{prefix}.lv"):
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
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, kernel_size: int = 7):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1, kernel_size=kernel_size),
            ResidualUnit(output_dim, dilation=3, kernel_size=kernel_size),
            ResidualUnit(output_dim, dilation=9, kernel_size=kernel_size),
        )

    def forward(self, x):
        return self.block(x)


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

        # Add first conv layer
        layers = [WNConv1d(config.latent_dim, channels, kernel_size=config.kernel_size, padding=config.kernel_size // 2)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2 ** ((i + 1) // 2)
            output_dim = channels // 2 ** (i // 2 + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride, config.kernel_size)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=config.kernel_size, padding=config.kernel_size // 2),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

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
        )                # [B, 1024, T_enc]
        z_lens = audio_lengths // self.hop_length
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
        )                # [B, 1024, T_enc]
        z_lens = audio_lengths // self.hop_length
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

    @staticmethod
    def _ordered_timing_summary(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        total_items = [item for item in summary if item["name"] == "encode.total"]
        stage_items = [item for item in summary if item["name"].startswith("full.") or item["name"].startswith("chunk.")]
        encoder_items = [item for item in summary if item["name"].startswith("encoder.")]
        bottleneck_items = [item for item in summary if item["name"].startswith("bottleneck.")]
        other_items = [
            item for item in summary
            if item["name"] != "encode.total"
            and not item["name"].startswith("full.")
            and not item["name"].startswith("chunk.")
            and not item["name"].startswith("encoder.")
            and not item["name"].startswith("bottleneck.")
        ]
        return total_items + stage_items + encoder_items + bottleneck_items + other_items

    @staticmethod
    def _timing_type_name(name: str) -> Optional[str]:
        if name.startswith("encoder.block.") or name.startswith("encoder.smoothing.") or name.startswith("bottleneck."):
            if name.endswith(".snake0") or name.endswith(".snake1") or name.endswith(".post_snake"):
                return "snake"
            if name.endswith(".conv_dilated"):
                return "conv_dilated"
            if name.endswith(".conv1x1"):
                return "conv1x1"
            if name.endswith(".downsample_conv"):
                return "downsample_conv"
            if name.endswith(".in_act") or name.endswith(".mid_act") or name.endswith(".out_act"):
                return "smoothing_act"
            if name.endswith(".dwconv"):
                return "smoothing_dwconv"
            if name.endswith(".pw1"):
                return "smoothing_pw1"
            if name.endswith(".pw2"):
                return "smoothing_pw2"
            if name == "encoder.block.0.conv":
                return "input_conv"
            if name == "bottleneck.mu":
                return "bottleneck_mu"
            if name == "bottleneck.lv":
                return "bottleneck_lv"
        if name == "encoder.transformer":
            return "transformer"
        return None

    def _build_timing_type_summary(self, summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for item in summary:
            type_name = self._timing_type_name(item["name"])
            if type_name is None:
                continue
            stat = stats.get(type_name)
            if stat is None:
                stat = {"name": type_name, "total_sec": 0.0, "count": 0}
                stats[type_name] = stat
            stat["total_sec"] += float(item["total_sec"])
            stat["count"] += int(item["count"])

        out = []
        for stat in stats.values():
            total_sec = float(stat["total_sec"])
            count = int(stat["count"])
            out.append(
                {
                    "name": stat["name"],
                    "total_sec": total_sec,
                    "count": count,
                    "avg_sec": total_sec / max(count, 1),
                }
            )
        out.sort(key=lambda x: x["total_sec"], reverse=True)
        return out

    def _print_timing_profile(
        self,
        meta: Dict[str, Any],
        summary: List[Dict[str, Any]],
        type_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if len(summary) == 0:
            return
        requested = meta.get("requested_chunk_mode")
        executed = meta.get("executed_path")
        resolved_chunk_sec = meta.get("resolved_chunk_sec")
        resolved_chunk_str = "-" if resolved_chunk_sec is None else f"{float(resolved_chunk_sec):.3f}"
        print("| VAE encode timing profile")
        print(
            "| "
            f"requested_chunk_mode={requested}, "
            f"executed_path={executed}, "
            f"resolved_chunk_sec={resolved_chunk_str}, "
            f"max_batch_size={meta.get('max_batch_size')}, "
            f"deterministic={meta.get('deterministic')}, "
            f"normalize={meta.get('normalize')}, "
            f"estimate_bandwidth={meta.get('estimate_bandwidth')}, "
            f"profile_timing_detail={meta.get('profile_timing_detail')}, "
            f"device={meta.get('device')}"
        )
        if executed == "chunk":
            print(
                "| "
                f"chunk_samples={meta.get('chunk_samples')}, "
                f"chunk_latent_len={meta.get('chunk_latent_len')}, "
                f"margin_frames={meta.get('margin_frames')}, "
                f"num_chunks={meta.get('num_chunks')}, "
                f"num_micro_batches={meta.get('num_micro_batches')}"
            )
        for item in summary:
            print(
                "| "
                f"{item['name']}: "
                f"total={item['total_sec']:.4f}s, "
                f"count={item['count']}, "
                f"avg={item['avg_sec']:.4f}s"
            )
        if type_summary:
            print("| VAE encode timing profile by type")
            for item in type_summary:
                print(
                    "| "
                    f"{item['name']}: "
                    f"total={item['total_sec']:.4f}s, "
                    f"count={item['count']}, "
                    f"avg={item['avg_sec']:.4f}s"
                )

    def _finalize_encode_timing_profile(
        self,
        result: Any,
        return_dict: bool,
        timing_profiler: Optional[_TimingProfiler],
        layer_timing_profiler: Optional[Union[_TimingProfiler, _CudaEventProfiler]],
        timing_meta: Dict[str, Any],
    ):
        if timing_profiler is None or not timing_profiler.enabled:
            return result
        summary = timing_profiler.summary()
        if layer_timing_profiler is not None and getattr(layer_timing_profiler, "enabled", False):
            summary = summary + layer_timing_profiler.summary()
        summary = self._ordered_timing_summary(summary)
        type_summary = self._build_timing_type_summary(summary)
        self._print_timing_profile(timing_meta, summary, type_summary=type_summary)
        if return_dict:
            result["timing_profile_meta"] = timing_meta
            result["timing_profile_summary"] = summary
            result["timing_profile_type_summary"] = type_summary
        return result

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
        timing_profiler: Optional[_TimingProfiler] = None,
        layer_timing_profiler: Optional[Union[_TimingProfiler, _CudaEventProfiler]] = None,
        timing_detail: str = "block",
        chunk_mode: str = "full",
    ):
        B, _, T_in_max = audio.shape
        device = audio.device
        hop = int(self.hop_length)
        with timing_profiler.section("full.input_align") if timing_profiler is not None else nullcontext():
            T_in_aligned = self._align_up(int(T_in_max), hop)
            target_T_full = int(T_in_aligned // hop)
            z_lengths_t = ((audio_lengths_t + hop - 1) // hop).clamp(min=0, max=target_T_full)

            audio_full = audio
            if T_in_aligned > T_in_max:
                audio_full = nn.functional.pad(audio_full, (0, T_in_aligned - T_in_max))

            enc_mask = sequence_mask(audio_lengths_t, maxlen=audio_full.shape[-1])
        if timing_profiler is not None:
            with timing_profiler.section("full.encoder_total"):
                z_enc = self.encoder._forward_impl(
                    audio_full,
                    enc_mask,
                    timing_recorder=timing_profiler,
                    layer_timing_recorder=layer_timing_profiler,
                    timing_detail=timing_detail,
                )
        else:
            z_enc = self.encoder(audio_full, enc_mask)
        self._maybe_take_cuda_mem_snapshot(device, "after_full_encoder_forward", memory_profile)
        with timing_profiler.section("full.bottleneck_encode") if timing_profiler is not None else nullcontext():
            if timing_detail == "layer" and layer_timing_profiler is not None:
                mu_full, logvar_full = self.bottleneck.encode_with_timing(
                    z_enc,
                    timing_recorder=layer_timing_profiler,
                    prefix="bottleneck",
                )
            else:
                mu_full, logvar_full = self.bottleneck.encode(z_enc)
        with timing_profiler.section("full.reparameterize") if timing_profiler is not None else nullcontext():
            z_full = mu_full if deterministic else self.bottleneck.reparameterize(mu_full, logvar_full)
        self._maybe_take_cuda_mem_snapshot(device, "after_full_reparameterize", memory_profile)

        with timing_profiler.section("full.latent_align") if timing_profiler is not None else nullcontext():
            z_pad = z_full.transpose(1, 2)
            if z_pad.shape[1] > target_T_full:
                z_pad = z_pad[:, :target_T_full, :]
            elif z_pad.shape[1] < target_T_full:
                z_pad = nn.functional.pad(z_pad, (0, 0, 0, target_T_full - z_pad.shape[1]))

        if normalize:
            with timing_profiler.section("full.normalize") if timing_profiler is not None else nullcontext():
                z_pad = self._normalize_latent(z_pad, key="mu" if deterministic else "z")

        with timing_profiler.section("full.output_mask") if timing_profiler is not None else nullcontext():
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
            if mu_pad.shape[1] > target_T_full:
                mu_pad = mu_pad[:, :target_T_full, :]
            elif mu_pad.shape[1] < target_T_full:
                pad_frames = target_T_full - mu_pad.shape[1]
                mu_pad = nn.functional.pad(mu_pad, (0, 0, 0, pad_frames))
            mu_pad = mu_pad * z_valid_mask

        if return_mu_logvar or return_kl:
            lv_pad = logvar_full.transpose(1, 2)
            if lv_pad.shape[1] > target_T_full:
                lv_pad = lv_pad[:, :target_T_full, :]
            elif lv_pad.shape[1] < target_T_full:
                pad_frames = target_T_full - lv_pad.shape[1]
                lv_pad = nn.functional.pad(lv_pad, (0, 0, 0, pad_frames))
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
            with timing_profiler.section("full.bandwidth_estimate") if timing_profiler is not None else nullcontext():
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
            if audio.shape[-1] > target_S_full:
                audio = audio[:, :, :target_S_full]
            else:
                audio = nn.functional.pad(audio, (0, target_S_full - audio.shape[-1]))

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
        profile_timing: bool = False,
        profile_timing_detail: str = "layer",
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
        profile_timing_detail = str(profile_timing_detail).lower()
        if profile_timing_detail not in {"block", "layer"}:
            raise ValueError(f"unsupported profile_timing_detail={profile_timing_detail!r}, expected 'block' or 'layer'")
        timing_profiler: Optional[_TimingProfiler] = _TimingProfiler(True, device) if profile_timing else None
        if profile_timing and profile_timing_detail == "layer":
            if device.type == "cuda" and torch.cuda.is_available():
                layer_timing_profiler: Optional[Union[_TimingProfiler, _CudaEventProfiler]] = _CudaEventProfiler(True, device)
            else:
                layer_timing_profiler = _TimingProfiler(True, device)
        else:
            layer_timing_profiler = None
        total_start = timing_profiler.start() if timing_profiler is not None else None
        if profile_memory and device.type == "cuda" and torch.cuda.is_available():
            dev_idx = device.index if device.index is not None else torch.cuda.current_device()
            torch.cuda.synchronize(dev_idx)
            torch.cuda.reset_peak_memory_stats(dev_idx)
        self._maybe_take_cuda_mem_snapshot(device, "start", memory_profile)

        hop = int(self.hop_length)         # encoder hop in samples
        frame = int(self.frame_length)     # == hop for your WavVAE
        assert hop == frame

        requested_chunk_mode, resolved_chunk_sec = self._resolve_chunk_mode(chunk_sec)
        timing_meta: Dict[str, Any] = {
            "requested_chunk_mode": requested_chunk_mode,
            "executed_path": None,
            "resolved_chunk_sec": resolved_chunk_sec,
            "max_batch_size": int(max_batch_size),
            "deterministic": bool(deterministic),
            "normalize": bool(normalize),
            "estimate_bandwidth": bool(estimate_bandwidth),
            "profile_timing_detail": profile_timing_detail,
            "device": str(device),
        }

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

        if requested_chunk_mode == "full":
            result = self._encode_latent_full(
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
                timing_profiler=timing_profiler,
                layer_timing_profiler=layer_timing_profiler,
                timing_detail=profile_timing_detail,
                chunk_mode=requested_chunk_mode,
            )
            timing_meta["executed_path"] = "full"
            if timing_profiler is not None:
                timing_profiler.stop("encode.total", total_start)
            return self._finalize_encode_timing_profile(result, return_dict, timing_profiler, layer_timing_profiler, timing_meta)

        # ---- 1) chunk size (seconds -> samples -> latent_len) ----
        with timing_profiler.section("chunk.setup") if profile_timing else nullcontext():
            chunk_samples_target = self._sec_to_samples(resolved_chunk_sec)
            # Align to hop to keep encoder output T_enc stable.
            chunk_samples = self._align_up(chunk_samples_target, hop)
            chunk_latent_len = chunk_samples // hop

            # context margin from receptive field
            radius_samples_total = self._get_total_radius_samples()
            margin_frames = max(0, radius_samples_total // hop)   # one-sided margin in latent frames
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

        if T_in_aligned <= chunk_samples:
            result = self._encode_latent_full(
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
                timing_profiler=timing_profiler,
                layer_timing_profiler=layer_timing_profiler,
                timing_detail=profile_timing_detail,
                chunk_mode="full",
            )
            timing_meta["executed_path"] = "full"
            if timing_profiler is not None:
                timing_profiler.stop("encode.total", total_start)
            return self._finalize_encode_timing_profile(result, return_dict, timing_profiler, layer_timing_profiler, timing_meta)

        # ---- 2) cut all chunks across batch, then cat into one big batch ----
        all_chunks: List[torch.Tensor] = []
        chunk_valid_lengths: List[int] = []
        sample_chunk_start: List[int] = [0 for _ in range(B)]
        sample_chunk_count: List[int] = [0 for _ in range(B)]

        global_k = 0
        with timing_profiler.section("chunk.build") if profile_timing else nullcontext():
            for b in range(B):
                # Chunk partition uses the input tensor length only to keep shape fixed.
                len_b = int(T_in_max)

                wav_b = audio[b:b+1, :, :len_b]  # [1,1,len_b]

                # pad to hop multiple (same as training preprocess idea)
                pad_right = self._align_up(len_b, frame) - len_b
                if pad_right > 0:
                    wav_b = nn.functional.pad(wav_b, (0, pad_right))
                T_pad = wav_b.shape[-1]

                sample_chunk_start[b] = global_k

                start = 0
                while start < T_pad:
                    end = min(start + chunk_samples, T_pad)
                    chunk = wav_b[:, :, start:end]
                    L_raw = end - start

                    if L_raw < chunk_samples:
                        chunk = nn.functional.pad(chunk, (0, chunk_samples - L_raw))

                    all_chunks.append(chunk)                 # [1,1,chunk_samples]
                    chunk_valid_lengths.append(L_raw)        # raw valid samples inside this chunk
                    global_k += 1

                    if end == T_pad:
                        break
                    start += step_samples

                sample_chunk_count[b] = global_k - sample_chunk_start[b]

        if global_k == 0:
            raise RuntimeError("No chunks produced (check input lengths).")

        self._maybe_take_cuda_mem_snapshot(device, "after_chunk_build", memory_profile)
        with timing_profiler.section("chunk.cat") if profile_timing else nullcontext():
            chunk_wavs = torch.cat(all_chunks, dim=0)  # [N,1,chunk_samples]
            N_total = chunk_wavs.shape[0]
            chunk_lengths = torch.tensor(chunk_valid_lengths, device=device, dtype=torch.long)  # [N]
        self._maybe_take_cuda_mem_snapshot(device, "after_chunk_cat", memory_profile)
        timing_meta["executed_path"] = "chunk"
        timing_meta["chunk_samples"] = int(chunk_samples)
        timing_meta["chunk_latent_len"] = int(chunk_latent_len)
        timing_meta["margin_frames"] = int(margin_frames)
        timing_meta["num_chunks"] = int(N_total)
        timing_meta["num_micro_batches"] = int((N_total + max_batch_size - 1) // max_batch_size)

        # ---- 3) encoder + bottleneck in (almost) one shot, only split if needed ----
        # reuse the same arange across splits (avoid repeated allocations)
        t = torch.arange(chunk_samples, device=device)[None, :]  # [1,T]

        def run_range(i: int, j: int):
            wav_batch = chunk_wavs[i:j]            # [B_c,1,chunk_samples]
            len_batch = chunk_lengths[i:j]         # [B_c]

            # IMPORTANT: mask shape must be [B_c, T] (not [B_c,1,T])
            enc_mask = (t < len_batch[:, None])    # [B_c,T], bool

            if profile_timing:
                with timing_profiler.section("chunk.encoder_total"):
                    z_enc = self.encoder._forward_impl(
                        wav_batch,
                        enc_mask,
                        timing_recorder=timing_profiler,
                        layer_timing_recorder=layer_timing_profiler,
                        timing_detail=profile_timing_detail,
                    )
                with timing_profiler.section("chunk.bottleneck_encode"):
                    if profile_timing_detail == "layer" and layer_timing_profiler is not None:
                        mu_b, logvar_b = self.bottleneck.encode_with_timing(
                            z_enc,
                            timing_recorder=layer_timing_profiler,
                            prefix="bottleneck",
                        )
                    else:
                        mu_b, logvar_b = self.bottleneck.encode(z_enc)     # [B_c,C_z,T_enc]
            else:
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
        with timing_profiler.section("chunk.mu_logvar_cat") if profile_timing else nullcontext():
            mu_chunks = torch.cat(mu_list, dim=0)         # [N,C_z,T_enc]
            logvar_chunks = torch.cat(logvar_list, dim=0) # [N,C_z,T_enc]
        self._maybe_take_cuda_mem_snapshot(device, "after_mu_logvar_cat", memory_profile)
        with timing_profiler.section("chunk.reparameterize") if profile_timing else nullcontext():
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

        with timing_profiler.section("chunk.stitch") if profile_timing else nullcontext():
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
                for ci in range(ncb):
                    k = start_k + ci
                    z_k = z_chunks[k]
                    mu_k = mu_chunks[k]
                    lv_k = logvar_chunks[k]

                    if ncb == 1:
                        s_f, e_f = 0, T_enc
                    elif ci == 0:
                        s_f, e_f = 0, T_enc - margin_frames
                    elif ci == ncb - 1:
                        s_f, e_f = margin_frames, T_enc
                    else:
                        s_f, e_f = margin_frames, T_enc - margin_frames

                    if s_f >= e_f:
                        raise RuntimeError(f"Empty crop for sample {b}, chunk {ci}: {s_f=} {e_f=} {margin_frames=}")

                    kept_z.append(z_k[:, s_f:e_f])
                    if need_mu:
                        kept_mu.append(mu_k[:, s_f:e_f])
                    if need_logvar:
                        kept_logvar.append(lv_k[:, s_f:e_f])

                # z_cat must be built before shape fix-up.
                z_cat = torch.cat(kept_z, dim=-1)  # [C_z,T]

                if z_cat.shape[-1] > target_T_full:
                    z_cat = z_cat[:, :target_T_full]
                elif z_cat.shape[-1] < target_T_full:
                    z_cat = nn.functional.pad(z_cat, (0, target_T_full - z_cat.shape[-1]))

                z_per_sample.append(z_cat.transpose(0, 1))  # [target_T_full,C_z]

                if need_mu:
                    mu_cat = torch.cat(kept_mu, dim=-1)
                    if mu_cat.shape[-1] > target_T_full:
                        mu_cat = mu_cat[:, :target_T_full]
                    elif mu_cat.shape[-1] < target_T_full:
                        mu_cat = nn.functional.pad(mu_cat, (0, target_T_full - mu_cat.shape[-1]))

                    mu_per_sample.append(mu_cat.transpose(0, 1))

                if need_logvar:
                    lv_cat = torch.cat(kept_logvar, dim=-1)
                    if lv_cat.shape[-1] > target_T_full:
                        lv_cat = lv_cat[:, :target_T_full]
                    elif lv_cat.shape[-1] < target_T_full:
                        lv_cat = nn.functional.pad(lv_cat, (0, target_T_full - lv_cat.shape[-1]))
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
        with timing_profiler.section("chunk.stack") if profile_timing else nullcontext():
            z_pad = torch.stack(z_per_sample, dim=0)  # [B,target_T_full,C_z]
        self._maybe_take_cuda_mem_snapshot(device, "after_stack", memory_profile)

        if normalize:
            with timing_profiler.section("chunk.normalize") if profile_timing else nullcontext():
                z_pad = self._normalize_latent(z_pad, key="mu" if deterministic else "z")
            self._maybe_take_cuda_mem_snapshot(device, "after_normalize", memory_profile)

        with timing_profiler.section("chunk.output_mask") if profile_timing else nullcontext():
            t_lat = torch.arange(target_T_full, device=device)[None, :]
            z_valid_mask = (t_lat < z_lengths_t[:, None]).to(z_pad.dtype)[:, :, None]  # [B,T,1]
            z_pad = z_pad * z_valid_mask
        self._maybe_take_cuda_mem_snapshot(device, "final", memory_profile)

        if profile_memory and memory_profile is not None:
            self._print_cuda_mem_profile(memory_profile, device)

        if not return_dict:
            result = (z_pad, z_lengths_t)
            if timing_profiler is not None:
                timing_profiler.stop("encode.total", total_start)
            return self._finalize_encode_timing_profile(result, return_dict, timing_profiler, layer_timing_profiler, timing_meta)

        ret: Dict[str, Any] = {
            "z": z_pad,
            "z_lengths": z_lengths_t,
            "chunk_mode": requested_chunk_mode,
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
            with timing_profiler.section("chunk.bandwidth_estimate") if profile_timing else nullcontext():
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

        if timing_profiler is not None:
            timing_profiler.stop("encode.total", total_start)
        return self._finalize_encode_timing_profile(ret, return_dict, timing_profiler, layer_timing_profiler, timing_meta)

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

        # 1) collect all chunks into one large batch
        all_z = []
        chunk_map = [[] for _ in range(B)]  # per-sample chunk indices
        global_k = 0

        active_bs = torch.nonzero(latent_lengths_valid > 0, as_tuple=True)[0].tolist()
        for b in active_bs:
            # Partition by input latent T to keep output shape fixed.
            Tb = int(T)
            start = 0
            while start < Tb:
                end = min(start + chunk_T, Tb)
                z = latent[b:b+1, start:end, :]
                if z.shape[1] < chunk_T:
                    z = torch.nn.functional.pad(z, (0, 0, 0, chunk_T - z.shape[1]))
                all_z.append(z)
                chunk_map[b].append(global_k)
                global_k += 1
                if end == Tb:
                    break
                start += step_T

        if global_k == 0:
            audio = torch.zeros(B, 1, target_S_full, device=device)
            return {"audio": audio, "audio_lengths": out_lengths, "chunk_mode": chunk_mode} if return_dict else (audio, out_lengths)

        z_chunks = torch.cat(all_z, dim=0)  # [N,chunk_T,C]

        # 2) decode chunks by mini-batch
        wav_chunks = []
        for i in range(0, z_chunks.shape[0], max_batch_size):
            z_b = z_chunks[i:i+max_batch_size]   # [B_c,chunk_T,C]
            wav_b = super(InferenceWrapper, self).decode(z_b)  # [B_c,1,chunk_T*hop] in the common case
            wav_chunks.append(wav_b)
        wav_chunks = torch.cat(wav_chunks, dim=0)  # [N,1,Tc]

        # 3) trim overlap and stitch back per sample
        # Convert overlap_T to samples for trimming.
        overlap_S = int(overlap_T * self.hop_length)

        audio = torch.zeros(B, 1, target_S_full, device=device)

        for b in range(B):
            idxs = chunk_map[b]
            if len(idxs) == 0:
                continue

            segs = []
            num_chunks_b = len(idxs)
            for ci, k in enumerate(idxs):
                w = wav_chunks[k, 0]
                if num_chunks_b == 1:
                    s, e = 0, w.numel()
                elif ci == 0:
                    s, e = 0, max(0, w.numel() - overlap_S)
                elif ci == len(idxs) - 1:
                    s, e = min(overlap_S, w.numel()), w.numel()
                else:
                    s, e = min(overlap_S, w.numel()), max(0, w.numel() - overlap_S)
                segs.append(w[s:e])

            w_cat = torch.cat(segs, dim=0)
            if w_cat.numel() > target_S_full:
                w_cat = w_cat[:target_S_full]
            audio[b, 0, :w_cat.numel()] = w_cat

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
    """
    Compute the waveform receptive field for one latent frame.

    Required config fields:
      - sample_rate
      - kernel_size
      - encoder_rates
      - transformer_n_layers
      - transformer_use_sliding_window
      - transformer_sliding_window_size

    If seq_len_samples is provided and transformer uses global attention,
    the reported total receptive field will be based on the full sequence length.
    """
    # 1) Convolutional encoder receptive field.
    k0 = int(config.kernel_size)
    strides = [int(x) for x in config.encoder_rates]
    encoder_kernel_sizes = getattr(config, "encoder_kernel_sizes", None)
    if encoder_kernel_sizes is None:
        encoder_kernel_sizes = [k0] * len(strides)
    else:
        encoder_kernel_sizes = [int(x) for x in encoder_kernel_sizes]
        if len(encoder_kernel_sizes) != len(strides):
            raise ValueError(
                f"encoder_kernel_sizes length mismatch: expected {len(strides)}, got {len(encoder_kernel_sizes)}"
            )

    encoder_dilations = getattr(config, "encoder_dilations", None)
    if encoder_dilations is None:
        encoder_dilations = [(1, 3, 9)] * len(strides)
    else:
        encoder_dilations = [
            tuple(int(x) for x in stage) if isinstance(stage, (list, tuple)) else (int(stage),)
            for stage in encoder_dilations
        ]
        if len(encoder_dilations) != len(strides):
            raise ValueError(
                f"encoder_dilations length mismatch: expected {len(strides)}, got {len(encoder_dilations)}"
            )

    sr = int(config.sample_rate)

    R = k0
    jump = 1
    for s, block_kernel_size, block_dilation_list in zip(strides, encoder_kernel_sizes, encoder_dilations):
        residual_extra = sum((int(block_kernel_size) - 1) * int(d) for d in block_dilation_list)
        downsample_extra = 2 * int(s) - 1
        R += jump * (residual_extra + downsample_extra)
        jump *= s

    hop_size = jump

    conv_radius_samples = (R - 1) / 2.0
    conv_radius_seconds = conv_radius_samples / sr

    result = {
        "conv_receptive_field_samples": int(R),
        "conv_radius_samples": float(conv_radius_samples),
        "conv_radius_seconds": float(conv_radius_seconds),
        "hop_size": int(hop_size),
        "encoder_kernel_sizes": [int(x) for x in encoder_kernel_sizes],
        "encoder_dilations": [list(int(x) for x in stage) for stage in encoder_dilations],
    }

    smooth_layers = max(int(getattr(config, "encoder_smoothing_layers", 0)), 0)
    smooth_kernel = int(getattr(config, "encoder_smoothing_kernel_size", 5))
    smooth_radius_frames_per_layer = max((smooth_kernel - 1) // 2, 0)
    smooth_total_radius_frames = smooth_layers * smooth_radius_frames_per_layer
    smooth_extra_radius_samples = smooth_total_radius_frames * hop_size
    result["smoothing_part"] = {
        "layers": int(smooth_layers),
        "kernel_size": int(smooth_kernel),
        "radius_frames_per_layer": int(smooth_radius_frames_per_layer),
        "total_radius_frames": int(smooth_total_radius_frames),
        "extra_radius_samples": int(smooth_extra_radius_samples),
        "extra_radius_seconds": float(smooth_extra_radius_samples / sr),
    }

    total_radius = conv_radius_samples + smooth_extra_radius_samples
    total_R = int(round(2 * total_radius + 1))
    result["total_receptive_field_samples"] = int(total_R)
    result["total_radius_samples"] = float(total_radius)
    result["total_radius_seconds"] = float(total_radius / sr)

    # 2) Transformer contribution.
    L = config.transformer_n_layers
    use_sw = getattr(config, "transformer_use_sliding_window", False)
    W = getattr(config, "transformer_sliding_window_size", None)

    transformer_info = {}

    if int(L) <= 0:
        transformer_info["mode"] = "none"
        transformer_info["description"] = (
            "No transformer layers: encoder receptive field is determined by conv + smoothing only."
        )
        transformer_info["attn_layers"] = 0
        transformer_info["attn_window_param"] = int(W) if W is not None else None
        transformer_info["attn_radius_frames_per_layer"] = 0
        transformer_info["attn_total_radius_frames"] = 0
        transformer_info["attn_extra_radius_samples"] = 0
        transformer_info["attn_extra_total_samples"] = 0
        transformer_info["total_receptive_field_samples"] = int(total_R)
        transformer_info["total_radius_samples"] = float(total_radius)
        transformer_info["total_radius_seconds"] = float(total_radius / sr)
    elif not use_sw:
        transformer_info["mode"] = "global"
        transformer_info["description"] = (
            "Global self-attention: each latent frame can attend to the full encoder sequence. "
            "Use seq_len_samples to report sequence-specific numbers."
        )
        if seq_len_samples is not None:
            total_R = seq_len_samples
            total_radius = (total_R - 1) / 2.0
            transformer_info.update({
                "total_receptive_field_samples": int(total_R),
                "total_radius_samples": float(total_radius),
                "total_radius_seconds": float(total_radius / sr),
            })
    else:
        transformer_info["mode"] = "sliding_window"
        transformer_info["attn_layers"] = int(L)

        if W is None:
            transformer_info["warning"] = (
                "transformer_use_sliding_window=True but transformer_sliding_window_size is missing"
            )
        else:
            radius_frames_per_layer = max(W - 1, 0)
            radius_frames = L * radius_frames_per_layer

            extra_radius_samples = radius_frames * hop_size
            extra_total_samples = 2 * extra_radius_samples

            total_R = R + extra_total_samples + 2 * smooth_extra_radius_samples
            total_radius = (total_R - 1) / 2.0

            transformer_info.update({
                "attn_window_param": int(W),
                "attn_radius_frames_per_layer": int(radius_frames_per_layer),
                "attn_total_radius_frames": int(radius_frames),
                "attn_extra_radius_samples": int(extra_radius_samples),
                "attn_extra_total_samples": int(extra_total_samples),
                "total_receptive_field_samples": int(total_R),
                "total_radius_samples": float(total_radius),
                "total_radius_seconds": float(total_radius / sr),
            })
            result["total_receptive_field_samples"] = int(total_R)
            result["total_radius_samples"] = float(total_radius)
            result["total_radius_seconds"] = float(total_radius / sr)

    result["transformer_part"] = transformer_info
    return result


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError(f"b must be positive, got {b}")
    return -((-a) // b)


def _backprop_same_conv_interval(
    interval: Tuple[int, int],
    kernel_size: int,
    dilation: int = 1,
) -> Tuple[int, int]:
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd for same-padding conv receptive field, got {kernel_size}")
    radius = ((kernel_size - 1) * dilation) // 2
    return interval[0] - radius, interval[1] + radius


def _backprop_conv_transpose_interval(
    interval: Tuple[int, int],
    stride: int,
    kernel_size: int,
    padding: int,
) -> Tuple[int, int]:
    left, right = interval
    in_left = _ceil_div(left + padding - (kernel_size - 1), stride)
    in_right = (right + padding) // stride
    return in_left, in_right


def _resolve_wavvae_decoder_window_config(config):
    sr = int(config.sample_rate)
    base_kernel_size = int(config.kernel_size)
    encoder_hop_size = int(np.prod(config.encoder_rates))
    decoder_rates = [int(x) for x in config.decoder_rates]
    decoder_upsample_factor = int(np.prod(decoder_rates))
    decoder_input_kernel_size = int(getattr(config, "decoder_input_kernel_size", base_kernel_size))
    decoder_output_kernel_size = int(getattr(config, "decoder_output_kernel_size", base_kernel_size))
    decoder_kernel_sizes = getattr(config, "decoder_kernel_sizes", None)
    if decoder_kernel_sizes is None:
        decoder_kernel_sizes = [base_kernel_size] * len(decoder_rates)
    else:
        decoder_kernel_sizes = [int(x) for x in decoder_kernel_sizes]
        if len(decoder_kernel_sizes) != len(decoder_rates):
            raise ValueError(
                f"decoder_kernel_sizes length mismatch: expected {len(decoder_rates)}, got {len(decoder_kernel_sizes)}"
            )

    decoder_dilations = getattr(config, "decoder_dilations", None)
    if decoder_dilations is None:
        decoder_dilations = [(1, 3, 9)] * len(decoder_rates)
    else:
        decoder_dilations = [
            tuple(int(x) for x in stage) if isinstance(stage, (list, tuple)) else (int(stage),)
            for stage in decoder_dilations
        ]
        if len(decoder_dilations) != len(decoder_rates):
            raise ValueError(
                f"decoder_dilations length mismatch: expected {len(decoder_rates)}, got {len(decoder_dilations)}"
            )

    if decoder_input_kernel_size % 2 == 0:
        raise ValueError(f"decoder_input_kernel_size must be odd, got {decoder_input_kernel_size}")
    if decoder_output_kernel_size % 2 == 0:
        raise ValueError(f"decoder_output_kernel_size must be odd, got {decoder_output_kernel_size}")
    for kernel_size in decoder_kernel_sizes:
        if kernel_size % 2 == 0:
            raise ValueError(f"decoder_kernel_sizes must be odd, got {decoder_kernel_sizes}")

    return {
        "sample_rate": int(sr),
        "encoder_hop_size": int(encoder_hop_size),
        "decoder_rates": [int(x) for x in decoder_rates],
        "decoder_upsample_factor": int(decoder_upsample_factor),
        "decoder_input_kernel_size": int(decoder_input_kernel_size),
        "decoder_output_kernel_size": int(decoder_output_kernel_size),
        "decoder_kernel_sizes": [int(x) for x in decoder_kernel_sizes],
        "decoder_dilations": [tuple(int(x) for x in stage) for stage in decoder_dilations],
    }


def compute_wavvae_decoder_window_size(config):
    """
    Compute the decoder-side aggregation window:
    one reconstructed waveform sample -> latent neighborhood.
    """
    decoder_cfg = _resolve_wavvae_decoder_window_config(config)
    sr = int(decoder_cfg["sample_rate"])
    decoder_rates = decoder_cfg["decoder_rates"]
    decoder_upsample_factor = int(decoder_cfg["decoder_upsample_factor"])
    decoder_input_kernel_size = int(decoder_cfg["decoder_input_kernel_size"])
    decoder_output_kernel_size = int(decoder_cfg["decoder_output_kernel_size"])
    decoder_kernel_sizes = decoder_cfg["decoder_kernel_sizes"]
    decoder_dilations = decoder_cfg["decoder_dilations"]

    max_latent_receptive_field_frames = 0
    max_latent_left_radius_frames = 0
    max_latent_right_radius_frames = 0
    max_decoder_left_radius_samples = 0.0
    max_decoder_right_radius_samples = 0.0
    max_decoder_total_receptive_field_samples = 0.0
    max_decoder_phase = 0

    for phase in range(decoder_upsample_factor):
        interval = (phase, phase)
        interval = _backprop_same_conv_interval(interval, kernel_size=decoder_output_kernel_size)

        for stride, block_kernel_size, block_dilation_list in zip(
            reversed(decoder_rates),
            reversed(decoder_kernel_sizes),
            reversed(decoder_dilations),
        ):
            for dilation in reversed(block_dilation_list):
                interval = _backprop_same_conv_interval(
                    interval,
                    kernel_size=int(block_kernel_size),
                    dilation=dilation,
                )
            interval = _backprop_conv_transpose_interval(
                interval,
                stride=int(stride),
                kernel_size=2 * int(stride),
                padding=math.ceil(int(stride) / 2),
            )

        interval = _backprop_same_conv_interval(interval, kernel_size=decoder_input_kernel_size)
        latent_left_idx, latent_right_idx = interval
        latent_receptive_field_frames = latent_right_idx - latent_left_idx + 1
        latent_left_radius_frames = max(0, -latent_left_idx)
        latent_right_radius_frames = max(0, latent_right_idx)
        max_latent_receptive_field_frames = max(
            max_latent_receptive_field_frames,
            latent_receptive_field_frames,
        )
        max_latent_left_radius_frames = max(max_latent_left_radius_frames, latent_left_radius_frames)
        max_latent_right_radius_frames = max(max_latent_right_radius_frames, latent_right_radius_frames)

        decoder_left_radius_samples = float(phase - latent_left_idx * decoder_upsample_factor)
        decoder_right_radius_samples = float(latent_right_idx * decoder_upsample_factor - phase)
        decoder_total_receptive_field_samples = decoder_left_radius_samples + decoder_right_radius_samples + 1.0

        if decoder_total_receptive_field_samples > max_decoder_total_receptive_field_samples:
            max_decoder_total_receptive_field_samples = decoder_total_receptive_field_samples
            max_decoder_phase = int(phase)
        max_decoder_left_radius_samples = max(max_decoder_left_radius_samples, decoder_left_radius_samples)
        max_decoder_right_radius_samples = max(max_decoder_right_radius_samples, decoder_right_radius_samples)

    latent_overlap_frames_conservative = int(max(max_latent_left_radius_frames, max_latent_right_radius_frames))
    overlap_samples_conservative = float(latent_overlap_frames_conservative * decoder_upsample_factor)
    overlap_seconds_conservative = float(overlap_samples_conservative / sr)

    return {
        "latent_receptive_field_frames_max": int(max_latent_receptive_field_frames),
        "latent_radius_frames_equivalent": float((max_latent_receptive_field_frames - 1) / 2.0),
        "upsample_factor": int(decoder_upsample_factor),
        "input_kernel_size": int(decoder_input_kernel_size),
        "output_kernel_size": int(decoder_output_kernel_size),
        "decoder_kernel_sizes": [int(x) for x in decoder_kernel_sizes],
        "decoder_dilations": [list(int(x) for x in stage) for stage in decoder_dilations],
        "latent_left_radius_frames_max": int(max_latent_left_radius_frames),
        "latent_right_radius_frames_max": int(max_latent_right_radius_frames),
        "latent_overlap_frames_conservative": int(latent_overlap_frames_conservative),
        "overlap_samples_conservative": float(overlap_samples_conservative),
        "overlap_seconds_conservative": float(overlap_seconds_conservative),
        "max_phase_with_decoder_window": int(max_decoder_phase),
        "left_radius_samples_max": float(max_decoder_left_radius_samples),
        "right_radius_samples_max": float(max_decoder_right_radius_samples),
        "left_radius_seconds_max": float(max_decoder_left_radius_samples / sr),
        "right_radius_seconds_max": float(max_decoder_right_radius_samples / sr),
        "total_receptive_field_samples_max": float(max_decoder_total_receptive_field_samples),
        "radius_samples_equivalent": float((max_decoder_total_receptive_field_samples - 1.0) / 2.0),
        "radius_seconds_equivalent": float((max_decoder_total_receptive_field_samples - 1.0) / (2.0 * sr)),
    }


def compute_wavvae_e2e_window_size(config, seq_len_samples=None):
    """
    Compute the end-to-end waveform receptive field of the full VAE:
    one reconstructed waveform sample -> latent neighborhood -> input waveform neighborhood.
    """
    encoder_window = compute_wavvae_window_size(config, seq_len_samples=seq_len_samples)
    decoder_part = compute_wavvae_decoder_window_size(config)

    transformer_part = encoder_window.get("transformer_part", {})
    if (
        transformer_part.get("mode") == "global"
        and seq_len_samples is not None
        and "total_radius_samples" in transformer_part
    ):
        encoder_total_radius_samples = float(transformer_part["total_radius_samples"])
    else:
        encoder_total_radius_samples = float(
            encoder_window.get("total_radius_samples", encoder_window["conv_radius_samples"])
        )

    sr = int(config.sample_rate)
    encoder_hop_size = int(encoder_window["hop_size"])
    decoder_upsample_factor = int(decoder_part["upsample_factor"])
    max_e2e_phase = int(decoder_part["max_phase_with_decoder_window"])
    max_e2e_left_radius_samples = float(decoder_part["left_radius_samples_max"]) + encoder_total_radius_samples
    max_e2e_right_radius_samples = float(decoder_part["right_radius_samples_max"]) + encoder_total_radius_samples
    max_e2e_total_receptive_field_samples = (
        float(decoder_part["total_receptive_field_samples_max"]) + 2.0 * encoder_total_radius_samples
    )

    e2e_part = {
        "max_phase_with_e2e_window": int(max_e2e_phase),
        "left_radius_samples_max": float(max_e2e_left_radius_samples),
        "right_radius_samples_max": float(max_e2e_right_radius_samples),
        "left_radius_seconds_max": float(max_e2e_left_radius_samples / sr),
        "right_radius_seconds_max": float(max_e2e_right_radius_samples / sr),
        "total_receptive_field_samples_max": float(max_e2e_total_receptive_field_samples),
        "radius_samples_equivalent": float((max_e2e_total_receptive_field_samples - 1.0) / 2.0),
        "radius_seconds_equivalent": float((max_e2e_total_receptive_field_samples - 1.0) / (2.0 * sr)),
        "encoder_total_radius_samples_used": float(encoder_total_radius_samples),
        "decoder_left_radius_samples_used": float(decoder_part["left_radius_samples_max"]),
        "decoder_right_radius_samples_used": float(decoder_part["right_radius_samples_max"]),
        "decoder_overlap_frames_conservative_used": int(decoder_part["latent_overlap_frames_conservative"]),
    }

    result = {
        "encoder_window": encoder_window,
        "decoder_part": decoder_part,
        "e2e_part": e2e_part,
        "encoder_hop_size": int(encoder_hop_size),
        "decoder_upsample_factor": int(decoder_upsample_factor),
        "hop_matches_decoder": bool(encoder_hop_size == decoder_upsample_factor),
    }
    if encoder_hop_size != decoder_upsample_factor:
        result["warning"] = (
            "encoder hop size and decoder upsample factor do not match; "
            "e2e window still uses decoder output spacing for phase analysis."
        )
    return result


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
