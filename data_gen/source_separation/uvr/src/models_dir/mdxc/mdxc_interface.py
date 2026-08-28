from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import os

import audioread
import librosa
from ml_collections import ConfigDict
import numpy as np
from numpy.typing import NDArray
import torch
import yaml

from . import spec_utils
from .tfc_tdf_v3 import TFC_TDF_net

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = "cpu"


@dataclass
class PreparedMix:
    source_idx: int
    org_mix: np.ndarray
    padded_mix: torch.Tensor
    pad_size: int
    chunk_size: int
    hop_size: int
    sr_pitched: int
    sum_buffer: torch.Tensor
    weight_buffer: torch.Tensor


def load_mdxc_models_data(model_path: str = "mdxc/modelparams/model_data.json") -> dict:
    models_data = json.load(open(model_path))
    return models_data


def get_model_hash_from_path(model_path: str = "./mdxc/weights/MDX23C-8KFFT-InstVoc_HQ/MDX23C-8KFFT-InstVoc_HQ.ckpt") -> str:
    try:
        with open(model_path, "rb") as f:
            f.seek(-10000 * 1024, 2)
            model_hash = hashlib.md5(f.read()).hexdigest()
    except Exception:
        model_hash = hashlib.md5(open(model_path, "rb").read()).hexdigest()

    return model_hash


def load_mdxc_model_data(models_data, model_hash, model_path="./mdxc/modelparams") -> ConfigDict:
    model_data_src = models_data[model_hash]
    model_path = os.path.join(model_path, "mdx_c_configs", model_data_src["config_yaml"])
    model_data = yaml.load(open(model_path), Loader=yaml.FullLoader)
    return ConfigDict(model_data)


def load_modle(model_path: str, model_data: ConfigDict, device: str = "cuda") -> torch.nn.Module:
    model = TFC_TDF_net(model_data, device=device)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model


def rerun_mp3(audio_file: NDArray, sample_rate: int = 44100):
    with audioread.audio_open(audio_file) as f:
        track_length = int(f.duration)

    return librosa.load(audio_file, duration=track_length, mono=False, sr=sample_rate)[0]


def prepare_mix(mix):
    audio_path = mix

    if not isinstance(mix, np.ndarray):
        mix, _ = librosa.load(mix, mono=False, sr=44100)

    if isinstance(audio_path, str) and not np.any(mix) and audio_path.endswith(".mp3"):
        mix = rerun_mp3(audio_path)

    if mix.ndim == 1:
        mix = np.asfortranarray([mix, mix])

    return np.asarray(mix, dtype=np.float32)


def pitch_fix(source, sr_pitched, org_mix, semitone_shift) -> np.ndarray:
    source = spec_utils.change_pitch_semitones(source, sr_pitched, semitone_shift=semitone_shift)[0]
    source = spec_utils.match_array_shapes(source, org_mix)
    return source


def seconds_to_chunk_size_samples(seconds: float, *, sample_rate: int, hop_length: int) -> int:
    if seconds <= 0:
        raise ValueError(f"chunk_size_sec must be > 0, got {seconds!r}.")
    raw_samples = float(seconds) * float(sample_rate)
    aligned = int(round(raw_samples / float(hop_length))) * int(hop_length)
    return max(int(hop_length), aligned)


def _get_num_target_instruments(model: torch.nn.Module) -> int:
    try:
        return int(model.num_target_instruments)
    except Exception:
        return int(model.module.num_target_instruments)


def _resolve_chunk_size(prams: dict, model_data: ConfigDict, chunk_size_samples: int | None) -> int:
    if chunk_size_samples is not None:
        return int(chunk_size_samples)

    if prams["is_mdx_c_seg_def"]:
        mdx_segment_size = int(model_data.inference.dim_t)
    else:
        mdx_segment_size = int(prams["segment_size"])

    return int(model_data.audio.hop_length) * (mdx_segment_size - 1)


def _prepare_single_mix(
    mix: np.ndarray,
    *,
    source_idx: int,
    prams: dict,
    chunk_size: int,
    overlap: int,
    num_target_instruments: int,
) -> PreparedMix:
    org_mix = np.asarray(mix, dtype=np.float32)
    sr_pitched = 441000
    semitone_shift = prams["semitone_shift"]
    if semitone_shift != 0:
        mix, sr_pitched = spec_utils.change_pitch_semitones(org_mix, 44100, semitone_shift=-semitone_shift)
    else:
        mix = org_mix

    mix_tensor = torch.tensor(mix, dtype=torch.float32)
    hop_size = chunk_size // overlap
    mix_shape = int(mix_tensor.shape[1])
    pad_size = hop_size - (mix_shape - chunk_size) % hop_size
    padded_mix = torch.cat(
        [
            torch.zeros((2, chunk_size - hop_size), dtype=torch.float32),
            mix_tensor,
            torch.zeros((2, pad_size + chunk_size - hop_size), dtype=torch.float32),
        ],
        dim=1,
    )

    if num_target_instruments > 1:
        sum_buffer = torch.zeros((num_target_instruments, *padded_mix.shape), dtype=torch.float32)
    else:
        sum_buffer = torch.zeros_like(padded_mix)
    weight_buffer = torch.zeros((padded_mix.shape[1],), dtype=torch.float32)

    return PreparedMix(
        source_idx=source_idx,
        org_mix=org_mix,
        padded_mix=padded_mix,
        pad_size=int(pad_size),
        chunk_size=int(chunk_size),
        hop_size=int(hop_size),
        sr_pitched=int(sr_pitched),
        sum_buffer=sum_buffer,
        weight_buffer=weight_buffer,
    )


def _rebuild_stems(
    prepared: PreparedMix,
    *,
    num_target_instruments: int,
    instruments: list[str],
    semitone_shift: int,
) -> dict | np.ndarray:
    trim_start = prepared.chunk_size - prepared.hop_size
    trim_end = prepared.pad_size + prepared.chunk_size - prepared.hop_size
    stop = -trim_end if trim_end > 0 else None
    trimmed = prepared.sum_buffer[..., trim_start:stop]
    weights = prepared.weight_buffer[trim_start:stop].clamp_min(1.0)
    view_shape = (1,) * (trimmed.ndim - 1) + (weights.shape[0],)
    estimated_sources = trimmed / weights.reshape(view_shape)

    if num_target_instruments > 1:
        result = {}
        for key, value in zip(instruments, estimated_sources.numpy()):
            result[key] = pitch_fix(value, prepared.sr_pitched, prepared.org_mix, semitone_shift) if semitone_shift != 0 else value
        return result

    est_source = estimated_sources.numpy()
    return pitch_fix(est_source, prepared.sr_pitched, prepared.org_mix, semitone_shift) if semitone_shift != 0 else est_source


def demix_batch(
    mixes: list[np.ndarray],
    prams: dict,
    model: torch.nn.Module,
    model_data: ConfigDict,
    device: str = "cpu",
    *,
    batch_size: int = 1,
    precision: str = "fp32",
    chunk_size_samples: int | None = None,
) -> list[dict | np.ndarray]:
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}.")
    if len(mixes) == 0:
        return []

    num_target_instruments = _get_num_target_instruments(model)
    overlap = int(prams["overlap_mdx23"])
    semitone_shift = int(prams["semitone_shift"])
    chunk_size = _resolve_chunk_size(prams, model_data, chunk_size_samples)
    instruments = list(model_data.training.instruments) if num_target_instruments > 1 else []

    prepared_mixes = [
        _prepare_single_mix(
            prepare_mix(mix),
            source_idx=source_idx,
            prams=prams,
            chunk_size=chunk_size,
            overlap=overlap,
            num_target_instruments=num_target_instruments,
        )
        for source_idx, mix in enumerate(mixes)
    ]

    chunk_tasks: list[tuple[int, int]] = []
    for prepared in prepared_mixes:
        max_start = prepared.padded_mix.shape[1] - chunk_size
        for start in range(0, max_start + 1, prepared.hop_size):
            chunk_tasks.append((prepared.source_idx, start))

    autocast_enabled = precision == "fp16" and torch.device(device).type == "cuda"

    with torch.no_grad():
        for offset in range(0, len(chunk_tasks), int(batch_size)):
            batch_tasks = chunk_tasks[offset : offset + int(batch_size)]
            batch = torch.stack(
                [
                    prepared_mixes[source_idx].padded_mix[:, start : start + chunk_size]
                    for source_idx, start in batch_tasks
                ],
                dim=0,
            )

            with (torch.autocast(device_type="cuda", dtype=torch.float16) if autocast_enabled else nullcontext()):
                pred = model(batch.to(device))

            pred_cpu = pred.detach().to(torch.float32).cpu()
            for row_idx, (source_idx, start) in enumerate(batch_tasks):
                prepared = prepared_mixes[source_idx]
                prepared.sum_buffer[..., start : start + chunk_size] += pred_cpu[row_idx]
                prepared.weight_buffer[start : start + chunk_size] += 1.0

    return [
        _rebuild_stems(
            prepared,
            num_target_instruments=num_target_instruments,
            instruments=instruments,
            semitone_shift=semitone_shift,
        )
        for prepared in prepared_mixes
    ]


def demix(
    mix: np.ndarray,
    prams: dict,
    model: torch.nn.Module,
    model_data: ConfigDict,
    device: str = "cpu",
    *,
    batch_size: int | None = None,
    precision: str = "fp32",
    chunk_size_samples: int | None = None,
) -> dict | np.ndarray:
    effective_batch_size = int(prams["batch_size"] if batch_size is None else batch_size)
    return demix_batch(
        [mix],
        prams,
        model,
        model_data,
        device,
        batch_size=effective_batch_size,
        precision=precision,
        chunk_size_samples=chunk_size_samples,
    )[0]


def rename_stems(stems: dict) -> dict:
    if not isinstance(stems, dict):
        return stems
    return {k.lower(): v for k, v in stems.items()}
