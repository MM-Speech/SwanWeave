from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np
import torch

from data_gen.source_separation.common import (
    AudioInput,
    ensure_channel_first_tensor,
    load_audio_input,
    match_num_channels,
    normalize_device,
    resample_channels,
    to_uvr_stereo_mix,
    trim_or_pad,
    write_named_outputs,
)

BatchMemoryAudioInput = Mapping[str, object]
UVRAudioInput = Union[AudioInput, BatchMemoryAudioInput]


@dataclass
class UVRModelHandle:
    model_name: str
    device: torch.device
    engine: object
    sample_rate: int = 44100
    precision: str = "fp32"


@dataclass
class NormalizedAudioItem:
    index: int
    input_sr: int
    target_channels: int
    target_length: int
    input_path: Optional[Path]
    mix: np.ndarray
    output_name: str


def _normalize_precision(precision: str, device: torch.device) -> str:
    normalized = str(precision or "fp32").strip().lower()
    if normalized not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported UVR precision: {precision!r}. Expected 'fp32' or 'fp16'.")
    if normalized == "fp16" and device.type != "cuda":
        warnings.warn(
            f"UVR requested precision=fp16 on device {device}, but fp16 inference is only enabled on CUDA. Falling back to fp32.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "fp32"
    return normalized


def _is_batch_audio_input(audio: object) -> bool:
    if isinstance(audio, (str, Path, np.ndarray, torch.Tensor)):
        return False
    if isinstance(audio, Mapping):
        return False
    return isinstance(audio, Sequence)


def _load_audio_item(
    item: UVRAudioInput,
    *,
    index: int,
    sample_rate: Optional[int],
    model_sample_rate: int,
    require_memory_mapping: bool = False,
) -> NormalizedAudioItem:
    item_audio = item
    item_sample_rate = sample_rate
    input_path = None

    if isinstance(item, Mapping):
        if "audio" not in item or "sample_rate" not in item:
            raise ValueError("批量内存音频输入必须显式传入 {'audio': ..., 'sample_rate': int}。")
        item_audio = item["audio"]
        item_sample_rate = int(item["sample_rate"])
        maybe_path = item.get("path")
        if maybe_path is not None:
            input_path = Path(maybe_path).expanduser()
    elif _is_batch_audio_input(item):
        raise TypeError(f"Nested batch input is not supported: {type(item)!r}")
    elif isinstance(item, (str, Path)):
        input_path = Path(item).expanduser()
    elif require_memory_mapping and isinstance(item, (np.ndarray, torch.Tensor)):
        raise ValueError("批量内存音频输入必须显式传入 {'audio': ..., 'sample_rate': int}。")

    wav, input_sr, loaded_path = load_audio_input(item_audio, sample_rate=item_sample_rate)
    if loaded_path is not None:
        input_path = loaded_path

    wav_resampled = resample_channels(wav, input_sr, model_sample_rate)
    mix = to_uvr_stereo_mix(wav_resampled)

    return NormalizedAudioItem(
        index=index,
        input_sr=input_sr,
        target_channels=int(wav.shape[0]),
        target_length=int(wav.shape[-1]),
        input_path=input_path,
        mix=mix,
        output_name="",
    )


def _dedupe_output_names(names: list[str], indices: list[int]) -> list[str]:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1

    deduped = []
    for name, index in zip(names, indices):
        if counts[name] == 1:
            deduped.append(name)
        else:
            deduped.append(f"{name}_{index:04d}")
    return deduped


def _normalize_audio_inputs(
    audio: Union[UVRAudioInput, Sequence[UVRAudioInput]],
    *,
    sample_rate: Optional[int],
    model_sample_rate: int,
) -> tuple[list[NormalizedAudioItem], bool]:
    is_batch = _is_batch_audio_input(audio)
    raw_items = list(audio) if is_batch else [audio]

    normalized = [
        _load_audio_item(
            item,
            index=index,
            sample_rate=sample_rate,
            model_sample_rate=model_sample_rate,
            require_memory_mapping=is_batch,
        )
        for index, item in enumerate(raw_items)
    ]

    output_names = []
    for item in normalized:
        if item.input_path is not None:
            output_names.append(item.input_path.stem or f"item_{item.index:04d}")
        else:
            output_names.append(f"item_{item.index:04d}")

    deduped_names = _dedupe_output_names(output_names, [item.index for item in normalized])
    for item, output_name in zip(normalized, deduped_names):
        item.output_name = output_name

    return normalized, is_batch


def _resolve_item_output_paths(
    items: list[NormalizedAudioItem],
    *,
    output_path: Optional[Union[str, Path]],
    is_batch: bool,
) -> list[Optional[Path]]:
    if output_path is None:
        return [None] * len(items)

    resolved = Path(output_path).expanduser()
    if not is_batch:
        return [resolved]

    if resolved.suffix:
        raise ValueError("批量 UVR 输出时，output_path 必须是目录路径，不能是带后缀的文件路径。")

    return [resolved / f"{item.output_name}.wav" for item in items]


def build_uvr_model(
    device: Union[str, torch.device] = "cuda",
    model_name: str = "MDX23C-8KFFT-InstVoc_HQ",
    model_root=None,
    allow_legacy_fallback: bool = True,
    precision: str = "fp32",
) -> UVRModelHandle:
    from data_gen.source_separation.uvr.src.mdxc import MDXC

    normalized_device = normalize_device(device)
    normalized_precision = _normalize_precision(precision, normalized_device)
    engine = MDXC(
        name=model_name,
        other_metadata={},
        device=normalized_device,
        model_root=model_root,
        allow_legacy_fallback=allow_legacy_fallback,
        precision=normalized_precision,
    )
    return UVRModelHandle(
        model_name=model_name,
        device=normalized_device,
        engine=engine,
        sample_rate=44100,
        precision=normalized_precision,
    )


def _to_channel_first_numpy(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio[None, :]
    return audio


def run_uvr_model(
    audio: Union[UVRAudioInput, Sequence[UVRAudioInput]],
    uvr_model: UVRModelHandle,
    sample_rate: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    batch_size: int = 1,
    chunk_size_sec: Optional[float] = None,
):
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}.")
    if chunk_size_sec is not None and float(chunk_size_sec) <= 0:
        raise ValueError(f"chunk_size_sec must be > 0 when provided, got {chunk_size_sec!r}.")

    items, is_batch = _normalize_audio_inputs(
        audio,
        sample_rate=sample_rate,
        model_sample_rate=uvr_model.sample_rate,
    )
    item_output_paths = _resolve_item_output_paths(items, output_path=output_path, is_batch=is_batch)

    predictions = uvr_model.engine.predict_batch(
        [item.mix for item in items],
        sampling_rate=uvr_model.sample_rate,
        batch_size=batch_size,
        chunk_size_sec=None if chunk_size_sec is None else float(chunk_size_sec),
    )

    results = []
    for item, item_output_path, prediction in zip(items, item_output_paths, predictions):
        separated = prediction["separated"]
        vocals = _to_channel_first_numpy(np.asarray(separated.get("vocals"), dtype=np.float32))

        instrumental = separated.get("instrumental")
        if instrumental is None:
            instrumental = item.mix - match_num_channels(vocals, item.mix.shape[0])
        instrumental = _to_channel_first_numpy(np.asarray(instrumental, dtype=np.float32))

        if item.input_sr != uvr_model.sample_rate:
            vocals = resample_channels(vocals, uvr_model.sample_rate, item.input_sr)
            instrumental = resample_channels(instrumental, uvr_model.sample_rate, item.input_sr)

        vocals = trim_or_pad(match_num_channels(vocals, item.target_channels), item.target_length)
        instrumental = trim_or_pad(match_num_channels(instrumental, item.target_channels), item.target_length)

        outputs = {
            "vocals": ensure_channel_first_tensor(vocals),
            "instrumental": ensure_channel_first_tensor(instrumental),
        }
        write_named_outputs(
            outputs,
            sample_rate=item.input_sr,
            output_path=item_output_path,
            input_path=item.input_path,
        )
        results.append({"sr": item.input_sr, "outputs": outputs})

    return results if is_batch else results[0]


if __name__ == '__main__':
    uvr_model = build_uvr_model(
        precision='fp16'
    )

    import time
    time_start = time.time()
    results = run_uvr_model(
        audio=['user/prompts/remaji.wav'] * 10,
        uvr_model=uvr_model,
        output_path=None,
        batch_size=32,
    )
    time_end = time.time()
    print(f"Time cost: {time_end - time_start}")

    # print(results)
    import soundfile as sf
    tensor = results[0]['outputs']['vocals']
    sf.write('user/prompts/remaji_vocal.wav', tensor.detach().cpu().numpy().T, 44100)


