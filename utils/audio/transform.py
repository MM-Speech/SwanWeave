import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Sequence, Optional, Union, Dict, Any

def apply_gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    """
    以 dB 应用线性增益（不改变 dtype）。
    """
    return x * (10.0 ** (gain_db / 20.0))

def true_peak_dbfs(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    """
    近似 True Peak（dBTP）：按通道上采样 resample_poly 检测插值峰值。
    oversample=4~8 通常够用。
    """
    from scipy.signal import resample_poly
    if x.size == 0:
        return -np.inf
    if x.ndim == 1:
        y = resample_poly(x, up=oversample, down=1, axis=0)
        m = float(np.max(np.abs(y)))
    else:
        m = 0.0
        for ch in range(x.shape[1]):
            y = resample_poly(x[:, ch], up=oversample, down=1, axis=0)
            m = max(m, float(np.max(np.abs(y))))
    return -np.inf if m == 0.0 else 20.0 * np.log10(m)

def normalize_lufs(
    x: np.ndarray,
    sr: int,
    target_lufs: float = -16.0,
    tp_limit_db: Optional[float] = -1.0,
    oversample: int = 4,
    meter=None,
) -> np.ndarray:
    """
    LUFS/EBU R128 归一化（不启用压缩，仅线性增益 + True Peak 安全限制）。
    - target_lufs: 立体声常用 -16 LUFS；单声道 -19 LUFS。
    - tp_limit_db: True Peak 上限（dBTP），默认 -1 dBTP；设为 None 则不限制。
    策略：计算达到目标 LUFS 所需增益；若会超过 True Peak 上限，则以 TP 为准。
    """
    import pyloudnorm as pyln
    if meter is None:
        meter = pyln.Meter(sr)  # EBU R128, K-weighting + gating
    loud = meter.integrated_loudness(x)
    if not np.isfinite(loud):
        # 全静音或极短片段时，LUFS 不稳定：原样返回
        return x.copy()
    # 到达目标 LUFS 所需增益
    gain_lufs_db = target_lufs - loud
    if tp_limit_db is None:
        return apply_gain_db(x, gain_lufs_db)
    # 基于原信号的 True Peak 计算“最大允许增益”
    orig_tp_db = true_peak_dbfs(x, sr, oversample=oversample)
    max_allowed_gain_db = tp_limit_db - orig_tp_db  # 正值表示还能抬这么多
    # 实际增益 = 受限于 TP 的较小值（更保守）
    gain_db = min(gain_lufs_db, max_allowed_gain_db)
    y = apply_gain_db(x, gain_db)
    return y

def batch_resample(
    wavs: List[np.ndarray],
    sample_rates: Union[List[int], int],
    tgt_sr: int,
    resamplers=None,
    batch_size=16,
    batch_duration=300,
    use_batch_by_size=True,
    device='cpu'
):
    import torch, torchaudio, math
    from utils.commons.dataset_utils import collate_xd, batch_by_size
    if sample_rates == tgt_sr:
        return wavs
    if resamplers is None:
        resamplers = {}
    if isinstance(sample_rates, int) or isinstance(sample_rates, float):
        if len(wavs) > batch_size:
            if use_batch_by_size:
                wav_lengths = [len(wav) for wav in wavs]
                ordered_idxs = np.argsort(wav_lengths)
                batches = batch_by_size(
                    ordered_idxs, lambda idx: wav_lengths[idx],
                    max_tokens=batch_duration * sample_rates,
                    max_sentences=batch_size,
                )
                wavs_ = [None] * len(wavs)
                for batch in batches:
                    res = batch_resample([wavs[idx] for idx in batch], sample_rates, tgt_sr, resamplers, 
                                         batch_size, batch_duration, use_batch_by_size, device)
                    for idx, wav in zip(batch, res):
                        wavs_[idx] = wav
            else:
                n_batch = math.ceil(len(wavs) / batch_size)
                wavs_ = []
                for batch_i in range(n_batch):
                    res = batch_resample(wavs[batch_i * batch_size: (batch_i + 1) * batch_size], 
                                        sample_rates, tgt_sr, resamplers, batch_size, device)
                    wavs_.extend(res)
            wavs = wavs_
        else:
            wav_lengths = torch.LongTensor([len(w) for w in wavs])
            wavs = collate_xd([torch.from_numpy(w).float() for w in wavs]).to(device)
            if sample_rates != tgt_sr:
                if sample_rates not in resamplers:
                    resamplers[sample_rates] = torchaudio.transforms.Resample(orig_freq=sample_rates, new_freq=tgt_sr).to(device)
                with torch.no_grad(), torch.autocast(device_type=wavs.device.type, enabled=False):
                    wavs = resamplers[sample_rates](wavs.float())
                wav_lengths = (wav_lengths / sample_rates * tgt_sr).long()
            wavs = [wavs[i, :wav_lengths[i]].cpu().numpy() for i in range(len(wav_lengths))]
    elif isinstance(sample_rates, list):
        sr2wav_idx = {}
        for wav_idx, sr in enumerate(sample_rates):
            if sr in sr2wav_idx:
                sr2wav_idx[sr].append(wav_idx)
            else:
                sr2wav_idx[sr] = [wav_idx]
        wavs_ = [None] * len(wavs)
        for sr in sr2wav_idx:
            wav_idxs = sr2wav_idx[sr]
            res = batch_resample([wavs[idx] for idx in wav_idxs], sr, tgt_sr, resamplers, 
                                 batch_size, batch_duration, use_batch_by_size, device)
            for idx, wav in zip(wav_idxs, res):
                wavs_[idx] = wav
        wavs = wavs_

    return wavs

def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    # soundfile can return shape (T, C); some pipelines use (C, T)
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        return np.mean(audio, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported audio ndim={audio.ndim}")

def float_range_normalize(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    # If decoded audio is int-like scaled or out-of-range, normalize conservatively.
    if peak > 1.0:
        audio = audio / peak
    audio = np.clip(audio, -1.0, 1.0)
    return audio

def float_range_normalize_torch(
        wavs,
        wav_lens=None,
    ):
        wavs = wavs.to(dtype=torch.float32)

        if wavs.numel() == 0:
            return wavs
        
        is_batched = len(wavs.shape) == 2
        if not is_batched:
            wavs = wavs.unsqueeze(0)

        if wav_lens is None:
            peak = wavs.abs().amax(dim=-1, keepdim=True)  # [B, 1]
        else:
            T = int(wavs.shape[-1])
            wav_lens = wav_lens.to(device=wavs.device, dtype=torch.long).clamp(min=0, max=T)

            idx = torch.arange(T, device=wavs.device).unsqueeze(0)  # [1, T]
            mask = idx < wav_lens.unsqueeze(1)                      # [B, T]
            abs_valid = wavs.abs().masked_fill(~mask, 0.0)
            peak = abs_valid.amax(dim=-1, keepdim=True)            # [B, 1]

        scale = torch.where(peak > 1.0, peak, torch.ones_like(peak))
        wavs = wavs / scale
        wavs = wavs.clamp(-1.0, 1.0)

        if not is_batched:
            wavs = wavs.squeeze(0)

        return wavs


def _ensure_batched_wavs_torch(
    wavs: torch.Tensor,
    wav_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if wavs.ndim == 1:
        wavs = wavs.unsqueeze(0)
        squeeze_batch = True
    elif wavs.ndim == 2:
        squeeze_batch = False
    elif wavs.ndim == 3 and wavs.shape[1] == 1:
        wavs = wavs[:, 0, :]
        squeeze_batch = False
    else:
        raise ValueError(f"Expected wavs to be [T], [B,T], or [B,1,T], got {tuple(wavs.shape)}")

    B, T = wavs.shape
    if wav_lens is None:
        wav_lens = torch.full((B,), T, device=wavs.device, dtype=torch.long)
    else:
        wav_lens = torch.as_tensor(wav_lens, device=wavs.device, dtype=torch.long).reshape(-1)
        if wav_lens.numel() != B:
            raise ValueError(f"wav_lens size mismatch: got {wav_lens.numel()}, expected {B}")
        wav_lens = wav_lens.clamp(min=0, max=T)
    return wavs, wav_lens, squeeze_batch


def _default_bandwidth_candidates(sample_rate: int) -> Tuple[float, ...]:
    nyquist = float(sample_rate) / 2.0
    candidates = [4000.0, 5500.0, 8000.0, 11025.0, 12000.0, 16000.0, 22050.0, 24000.0]
    return tuple(c for c in candidates if c < nyquist - 1.0)


def _default_bandwidth_stft_params(sample_rate: int) -> Tuple[int, int, int]:
    if sample_rate >= 48000:
        n_fft = 2048
    elif sample_rate >= 32000:
        n_fft = 1024
    else:
        n_fft = 512
    hop_length = max(64, n_fft // 4)
    win_length = n_fft
    return int(n_fft), int(hop_length), int(win_length)


def compute_batch_average_power_spectrum_torch(
    wavs: torch.Tensor,
    sample_rate: int,
    wav_lens: Optional[torch.Tensor] = None,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    smooth_bins: int = 5,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    wavs, wav_lens, _ = _ensure_batched_wavs_torch(wavs, wav_lens=wav_lens)
    wavs = wavs.to(dtype=torch.float32)
    device = wavs.device
    B, T = wavs.shape

    if n_fft is None or hop_length is None or win_length is None:
        n_fft_d, hop_d, win_d = _default_bandwidth_stft_params(sample_rate)
        n_fft = n_fft_d if n_fft is None else n_fft
        hop_length = hop_d if hop_length is None else hop_length
        win_length = win_d if win_length is None else win_length

    n_fft = int(n_fft)
    hop_length = int(hop_length)
    win_length = int(win_length)
    if win_length > n_fft:
        raise ValueError(f"win_length ({win_length}) must be <= n_fft ({n_fft})")

    if T < win_length:
        wavs = F.pad(wavs, (0, win_length - T))
        T = wavs.shape[1]

    window = torch.hann_window(win_length, device=device, dtype=wavs.dtype)
    spec = torch.stft(
        wavs,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        pad_mode="constant",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    power = spec.abs().pow(2.0)

    effective_lens = torch.maximum(wav_lens, torch.full_like(wav_lens, win_length))
    frame_counts = ((effective_lens - win_length) // hop_length + 1).clamp(min=1, max=power.shape[-1])
    frame_ids = torch.arange(power.shape[-1], device=device)[None, :]
    frame_mask = (frame_ids < frame_counts[:, None]).to(power.dtype)[:, None, :]
    avg_power = (power * frame_mask).sum(dim=-1) / frame_mask.sum(dim=-1).clamp_min(1.0)
    avg_power = avg_power.clamp_min(float(eps))

    smooth_bins = max(1, int(smooth_bins))
    if smooth_bins > 1:
        if smooth_bins % 2 == 0:
            smooth_bins += 1
        avg_power = F.avg_pool1d(
            avg_power.unsqueeze(1),
            kernel_size=smooth_bins,
            stride=1,
            padding=smooth_bins // 2,
        ).squeeze(1).clamp_min(float(eps))

    freqs_hz = torch.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate)).to(device=device, dtype=avg_power.dtype)
    return avg_power, freqs_hz


def estimate_effective_bandwidth_torch(
    wavs: torch.Tensor,
    sample_rate: int,
    wav_lens: Optional[torch.Tensor] = None,
    candidate_cutoffs_hz: Optional[Sequence[float]] = None,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    smooth_bins: int = 5,
    edge_band_width_hz: float = 1000.0,
    tail_margin_hz: float = 500.0,
    tail_ratio_threshold: float = 0.02,
    drop_db_threshold: float = 12.0,
    eps: float = 1e-8,
    return_details: bool = False,
) -> Union[torch.Tensor, Dict[str, Any]]:
    wavs, wav_lens, squeeze_batch = _ensure_batched_wavs_torch(wavs, wav_lens=wav_lens)
    device = wavs.device
    dtype = torch.float32
    nyquist = float(sample_rate) / 2.0
    if candidate_cutoffs_hz is None:
        candidate_cutoffs_hz = _default_bandwidth_candidates(sample_rate)

    candidates = torch.as_tensor(candidate_cutoffs_hz, device=device, dtype=dtype).reshape(-1)
    candidates = candidates[(candidates > 0.0) & (candidates < nyquist - 1.0)]
    if candidates.numel() == 0:
        bandwidth_hz = torch.full((wavs.shape[0],), nyquist, device=device, dtype=dtype)
        if squeeze_batch:
            bandwidth_hz = bandwidth_hz[:1]
        return bandwidth_hz[0] if squeeze_batch and not return_details else (
            {"bandwidth_hz": bandwidth_hz} if return_details else bandwidth_hz
        )

    avg_power, freqs_hz = compute_batch_average_power_spectrum_torch(
        wavs,
        sample_rate=sample_rate,
        wav_lens=wav_lens,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        smooth_bins=smooth_bins,
        eps=eps,
    )
    avg_power = avg_power.to(dtype)
    freqs_hz = freqs_hz.to(dtype)
    total_energy = avg_power.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    log_power_db = 10.0 * torch.log10(avg_power.clamp_min(float(eps)))

    cand = candidates[None, :, None]
    freq = freqs_hz[None, None, :]
    edge_band_width_hz = max(float(edge_band_width_hz), 1.0)
    tail_margin_hz = max(float(tail_margin_hz), 0.0)

    left_lo = (cand - edge_band_width_hz).clamp_min(0.0)
    left_hi = cand
    right_lo = cand + tail_margin_hz
    right_hi = (cand + tail_margin_hz + edge_band_width_hz).clamp_max(nyquist)
    tail_lo = cand + tail_margin_hz

    left_mask = (freq >= left_lo) & (freq < left_hi)
    right_mask = (freq >= right_lo) & (freq < right_hi)
    tail_mask = freq >= tail_lo

    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(x.dtype)
        num = (x[:, None, :] * weight).sum(dim=-1)
        den = weight.sum(dim=-1).clamp_min(1.0)
        return num / den

    left_db = masked_mean(log_power_db, left_mask)
    right_db = masked_mean(log_power_db, right_mask)
    drop_db = left_db - right_db
    tail_ratio = ((avg_power[:, None, :] * tail_mask.to(avg_power.dtype)).sum(dim=-1) / total_energy).clamp_min(0.0)

    accepted = (tail_ratio <= float(tail_ratio_threshold)) & (drop_db >= float(drop_db_threshold))
    accepted_float = accepted.to(dtype)
    has_match = accepted.any(dim=-1)
    first_idx = accepted_float.argmax(dim=-1)

    bandwidth_hz = torch.full((wavs.shape[0],), nyquist, device=device, dtype=dtype)
    if has_match.any():
        bandwidth_hz[has_match] = candidates[first_idx[has_match]]

    selected_tail_ratio = torch.where(
        has_match,
        tail_ratio.gather(1, first_idx[:, None]).squeeze(1),
        torch.zeros_like(bandwidth_hz),
    )
    selected_drop_db = torch.where(
        has_match,
        drop_db.gather(1, first_idx[:, None]).squeeze(1),
        torch.zeros_like(bandwidth_hz),
    )
    tail_margin = (float(tail_ratio_threshold) - selected_tail_ratio) / max(float(tail_ratio_threshold), 1e-6)
    drop_margin = (selected_drop_db - float(drop_db_threshold)) / max(float(drop_db_threshold), 1e-6)
    confidence = torch.where(has_match, torch.minimum(tail_margin, drop_margin), torch.zeros_like(bandwidth_hz))

    if squeeze_batch:
        bandwidth_hz = bandwidth_hz[:1]
        confidence = confidence[:1]
        selected_tail_ratio = selected_tail_ratio[:1]
        selected_drop_db = selected_drop_db[:1]
        has_match = has_match[:1]

    if not return_details:
        return bandwidth_hz[0] if squeeze_batch else bandwidth_hz

    details: Dict[str, Any] = {
        "bandwidth_hz": bandwidth_hz[0] if squeeze_batch else bandwidth_hz,
        "confidence": confidence[0] if squeeze_batch else confidence,
        "selected_tail_ratio": selected_tail_ratio[0] if squeeze_batch else selected_tail_ratio,
        "selected_drop_db": selected_drop_db[0] if squeeze_batch else selected_drop_db,
        "has_match": has_match[0] if squeeze_batch else has_match,
        "candidate_cutoffs_hz": candidates,
        "tail_ratio_by_candidate": tail_ratio[0] if squeeze_batch else tail_ratio,
        "drop_db_by_candidate": drop_db[0] if squeeze_batch else drop_db,
    }
    return details


def _apply_variable_lowpass_same_length_torch(
    wavs: torch.Tensor,
    sample_rate: int,
    cutoff_hz: torch.Tensor,
    transition_width_hz: float = 1000.0,
) -> torch.Tensor:
    if wavs.numel() == 0:
        return wavs
    wavs = wavs.to(dtype=torch.float32)
    cutoff_hz = torch.as_tensor(cutoff_hz, device=wavs.device, dtype=wavs.dtype).reshape(-1)
    B, T = wavs.shape
    if cutoff_hz.numel() != B:
        raise ValueError(f"cutoff_hz size mismatch: got {cutoff_hz.numel()}, expected {B}")
    if T <= 1:
        return wavs

    nyquist = float(sample_rate) / 2.0
    transition_width_hz = max(float(transition_width_hz), 1.0)
    keep_fullband = cutoff_hz >= (nyquist - 0.5 * transition_width_hz)
    if bool(torch.all(keep_fullband)):
        return wavs

    spec = torch.fft.rfft(wavs, n=T, dim=-1)
    freqs_hz = torch.fft.rfftfreq(T, d=1.0 / float(sample_rate)).to(device=wavs.device, dtype=wavs.dtype)[None, :]
    low = (cutoff_hz - 0.5 * transition_width_hz).clamp(min=0.0, max=nyquist)[:, None]
    high = (cutoff_hz + 0.5 * transition_width_hz).clamp(min=0.0, max=nyquist)[:, None]
    high = torch.maximum(high, low + 1e-6)

    slope = ((freqs_hz - low) / (high - low)).clamp(0.0, 1.0)
    smooth_mask = 0.5 * (1.0 + torch.cos(math.pi * slope))
    smooth_mask = torch.where(freqs_hz <= low, torch.ones_like(smooth_mask), smooth_mask)
    smooth_mask = torch.where(freqs_hz >= high, torch.zeros_like(smooth_mask), smooth_mask)
    smooth_mask = torch.where(keep_fullband[:, None], torch.ones_like(smooth_mask), smooth_mask)

    filtered = torch.fft.irfft(spec * smooth_mask.to(spec.dtype), n=T, dim=-1)
    return filtered.to(dtype=wavs.dtype)


def apply_batch_variable_lowpass_torch(
    wavs: torch.Tensor,
    sample_rate: int,
    cutoff_hz: Union[float, Sequence[float], torch.Tensor],
    wav_lens: Optional[torch.Tensor] = None,
    transition_width_hz: float = 1000.0,
) -> torch.Tensor:
    wavs_2d, wav_lens, squeeze_batch = _ensure_batched_wavs_torch(wavs, wav_lens=wav_lens)
    orig_dtype = wavs_2d.dtype
    device = wavs_2d.device
    B, T = wavs_2d.shape
    cutoff_hz = torch.as_tensor(cutoff_hz, device=device, dtype=torch.float32).reshape(-1)
    if cutoff_hz.numel() == 1:
        cutoff_hz = cutoff_hz.expand(B)
    if cutoff_hz.numel() != B:
        raise ValueError(f"cutoff_hz size mismatch: got {cutoff_hz.numel()}, expected {B}")

    out = torch.zeros_like(wavs_2d, dtype=torch.float32)
    unique_lens = torch.unique(wav_lens)
    for length in unique_lens.tolist():
        length = int(length)
        idx = torch.nonzero(wav_lens == length, as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        if length <= 0:
            continue
        filtered = _apply_variable_lowpass_same_length_torch(
            wavs_2d[idx, :length],
            sample_rate=sample_rate,
            cutoff_hz=cutoff_hz[idx],
            transition_width_hz=transition_width_hz,
        )
        out[idx, :length] = filtered

    out = out.to(dtype=orig_dtype)
    if wavs.ndim == 3 and wavs.shape[1] == 1:
        out = out.unsqueeze(1)
    elif squeeze_batch:
        out = out.squeeze(0)
    return out
