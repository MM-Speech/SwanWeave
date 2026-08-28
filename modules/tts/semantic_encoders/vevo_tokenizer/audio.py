from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import torch
import torchaudio


PathLike = Union[str, Path]


@dataclass(frozen=True)
class PreparedAudio:
    waveform_24k: torch.Tensor
    waveform_16k: torch.Tensor
    source_sample_rate: int


def _to_mono_tensor(waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(waveform, np.ndarray):
        tensor = torch.from_numpy(waveform)
    else:
        tensor = waveform.detach().cpu()

    tensor = tensor.float()

    if tensor.ndim == 1:
        return tensor

    if tensor.ndim != 2:
        raise ValueError("waveform must be 1D or 2D")

    if tensor.shape[0] == 1 or tensor.shape[1] == 1:
        return tensor.reshape(-1)

    if tensor.shape[0] <= 8 and tensor.shape[1] > tensor.shape[0]:
        return tensor.mean(dim=0)

    if tensor.shape[1] <= 8 and tensor.shape[0] > tensor.shape[1]:
        return tensor.mean(dim=1)

    raise ValueError(
        "ambiguous 2D waveform shape; expected [channels, time] or [time, channels]"
    )


def _resample_if_needed(
    waveform: torch.Tensor,
    source_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    if source_sample_rate == target_sample_rate:
        return waveform

    return torchaudio.functional.resample(
        waveform.unsqueeze(0),
        source_sample_rate,
        target_sample_rate,
    ).squeeze(0)


def prepare_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    device: torch.device,
    load_sample_rate: int = 24000,
    token_sample_rate: int = 16000,
) -> PreparedAudio:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    mono = _to_mono_tensor(waveform)
    waveform_24k = _resample_if_needed(mono, sample_rate, load_sample_rate)
    waveform_16k = _resample_if_needed(waveform_24k, load_sample_rate, token_sample_rate)

    return PreparedAudio(
        waveform_24k=waveform_24k.unsqueeze(0).to(device),
        waveform_16k=waveform_16k.unsqueeze(0).to(device),
        source_sample_rate=sample_rate,
    )


def load_audio_path(
    audio_path: PathLike,
    device: torch.device,
    load_sample_rate: int = 24000,
    token_sample_rate: int = 16000,
) -> PreparedAudio:
    audio_path = Path(audio_path)
    source_waveform, source_sample_rate = librosa.load(
        str(audio_path),
        sr=None,
        mono=True,
    )
    source_tensor = torch.from_numpy(source_waveform.astype(np.float32, copy=False))
    waveform_24k = _resample_if_needed(
        source_tensor,
        source_sample_rate,
        load_sample_rate,
    )
    waveform_16k = _resample_if_needed(
        waveform_24k,
        load_sample_rate,
        token_sample_rate,
    )

    return PreparedAudio(
        waveform_24k=waveform_24k.unsqueeze(0).to(device),
        waveform_16k=waveform_16k.unsqueeze(0).to(device),
        source_sample_rate=int(source_sample_rate),
    )
