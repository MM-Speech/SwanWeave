from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch

AudioInput = Union[str, Path, np.ndarray, torch.Tensor]

_LEGACY_WARNINGS: set[tuple[str, str]] = set()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = REPO_ROOT / "pretrained_models" / "source_separation"


def clearvoice_legacy_checkpoint_dirs(model_name: str) -> tuple[Path, ...]:
    return (
        REPO_ROOT / "data_gen" / "source_separation" / "clearvoice" / "checkpoints" / model_name,
        REPO_ROOT / "checkpoints" / model_name,
        REPO_ROOT / "pretrained" / model_name,
    )


def normalize_device(device: Optional[Union[str, torch.device]] = "cuda") -> torch.device:
    if isinstance(device, torch.device):
        requested = device
    else:
        requested = torch.device(device or "cuda")

    if requested.type == "cuda":
        if torch.cuda.is_available():
            if requested.index is None:
                return torch.device("cuda")
            if requested.index < torch.cuda.device_count():
                return requested
        warnings.warn(
            f"Requested CUDA device {requested}, but CUDA is unavailable. Falling back automatically.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif requested.type == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        warnings.warn(
            "Requested MPS device, but MPS is unavailable. Falling back automatically.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif requested.type == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_tool_root(tool: str, model_root: Optional[Union[str, Path]] = None) -> Path:
    if model_root is not None:
        return Path(model_root).expanduser().resolve()

    tool_env = {
        "uvr": "UVR_MODEL_ROOT",
        "clearvoice": "CLEARVOICE_MODEL_ROOT",
    }.get(tool)
    if tool_env and os.environ.get(tool_env):
        return Path(os.environ[tool_env]).expanduser().resolve()

    if os.environ.get("SOURCE_SEPARATION_MODEL_ROOT"):
        return Path(os.environ["SOURCE_SEPARATION_MODEL_ROOT"]).expanduser().resolve() / tool

    return (DEFAULT_MODEL_ROOT / tool).resolve()


def warn_legacy_path(tool: str, path: Path) -> None:
    key = (tool, str(path))
    if key in _LEGACY_WARNINGS:
        return
    _LEGACY_WARNINGS.add(key)
    warnings.warn(
        f"{tool} 正在使用旧目录 `{path}` 加载权重。请迁移到 `{DEFAULT_MODEL_ROOT / tool}`。",
        DeprecationWarning,
        stacklevel=2,
    )


def resolve_existing_path(
    *,
    tool: str,
    description: str,
    preferred: Path,
    legacy_paths: Iterable[Path] = (),
    allow_legacy_fallback: bool = True,
) -> Path:
    if preferred.exists():
        return preferred

    if allow_legacy_fallback:
        for legacy in legacy_paths:
            if legacy.exists():
                warn_legacy_path(tool, legacy)
                return legacy

    candidate_lines = [f"- preferred: {preferred}"]
    for legacy in legacy_paths:
        candidate_lines.append(f"- legacy: {legacy}")
    raise FileNotFoundError(
        f"{tool} 缺少 {description}。\n"
        f"Expected one of:\n" + "\n".join(candidate_lines)
    )


def resolve_clearvoice_checkpoint_dir(
    model_name: str,
    model_root: Optional[Union[str, Path]] = None,
    *,
    allow_legacy_fallback: bool = True,
) -> Path:
    tool_root = resolve_tool_root("clearvoice", model_root)
    preferred = tool_root / model_name
    legacy_paths = clearvoice_legacy_checkpoint_dirs(model_name)
    return resolve_existing_path(
        tool="clearvoice",
        description=f"checkpoint directory for {model_name}",
        preferred=preferred,
        legacy_paths=legacy_paths,
        allow_legacy_fallback=allow_legacy_fallback,
    )


def resolve_clearvoice_aux_weight(
    relative_path: Union[str, Path],
    model_root: Optional[Union[str, Path]] = None,
    *,
    allow_legacy_fallback: bool = True,
) -> Path:
    tool_root = resolve_tool_root("clearvoice", model_root)
    preferred = tool_root / "aux" / relative_path
    legacy_paths = [
        REPO_ROOT
        / "data_gen"
        / "source_separation"
        / "clearvoice"
        / "models"
        / "av_mossformer2_tse"
        / "faceDetector"
        / "s3fd"
        / Path(relative_path).name
    ]
    return resolve_existing_path(
        tool="clearvoice",
        description=f"auxiliary weight `{relative_path}`",
        preferred=preferred,
        legacy_paths=legacy_paths,
        allow_legacy_fallback=allow_legacy_fallback,
    )


def resolve_uvr_weight_dir(
    arch: str,
    model_name: str,
    model_root: Optional[Union[str, Path]] = None,
    *,
    allow_legacy_fallback: bool = True,
) -> Path:
    tool_root = resolve_tool_root("uvr", model_root)
    preferred = tool_root / arch / "weights" / model_name
    legacy_paths = [
        REPO_ROOT
        / "data_gen"
        / "source_separation"
        / "uvr"
        / "src"
        / "models_dir"
        / arch
        / "weights"
        / model_name
    ]
    return resolve_existing_path(
        tool="uvr",
        description=f"weight directory for {arch}/{model_name}",
        preferred=preferred,
        legacy_paths=legacy_paths,
        allow_legacy_fallback=allow_legacy_fallback,
    )


def load_audio_input(audio: AudioInput, sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int, Optional[Path]]:
    if isinstance(audio, (str, Path)):
        audio_path = Path(audio).expanduser()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio input not found: {audio_path}")
        wav, sr = librosa.load(str(audio_path), sr=None, mono=False)
        if wav.ndim == 1:
            wav = wav[None, :]
        return wav.astype(np.float32), int(sr), audio_path

    if sample_rate is None:
        raise ValueError("内存音频输入必须显式提供 sample_rate。")

    if isinstance(audio, torch.Tensor):
        wav = audio.detach().cpu().float().numpy()
    elif isinstance(audio, np.ndarray):
        wav = audio.astype(np.float32, copy=False)
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)!r}")

    if wav.ndim == 1:
        wav = wav[None, :]
    elif wav.ndim == 2:
        if wav.shape[0] > wav.shape[1] and wav.shape[1] <= 8:
            wav = wav.T
    else:
        raise ValueError(f"Audio must be 1D or 2D, got shape={wav.shape}.")

    return wav.astype(np.float32), int(sample_rate), None


def resample_channels(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav.astype(np.float32, copy=False)
    return np.stack(
        [librosa.resample(channel, orig_sr=orig_sr, target_sr=target_sr) for channel in wav],
        axis=0,
    ).astype(np.float32)


def to_uvr_stereo_mix(wav: np.ndarray) -> np.ndarray:
    if wav.shape[0] == 1:
        return np.repeat(wav, 2, axis=0).astype(np.float32)
    if wav.shape[0] >= 2:
        return wav[:2].astype(np.float32)
    raise ValueError(f"Unexpected waveform shape: {wav.shape}")


def ensure_channel_first_tensor(audio: np.ndarray) -> torch.Tensor:
    if audio.ndim == 1:
        audio = audio[None, :]
    return torch.from_numpy(np.ascontiguousarray(audio)).float()


def match_num_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.shape[0] == channels:
        return audio
    if audio.shape[0] == 1 and channels == 2:
        return np.repeat(audio, 2, axis=0)
    if audio.shape[0] == 2 and channels == 1:
        return audio[:1]
    raise ValueError(f"Cannot match channel count from {audio.shape[0]} to {channels}.")


def trim_or_pad(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[-1] == length:
        return audio
    if audio.shape[-1] > length:
        return audio[..., :length]
    return np.pad(audio, ((0, 0), (0, length - audio.shape[-1])))


def derive_output_base(output_path: Union[str, Path], input_path: Optional[Path]) -> Tuple[Path, str]:
    output = Path(output_path)
    if output.suffix:
        return output.with_suffix(""), output.suffix.lstrip(".")
    base_name = input_path.stem if input_path is not None else "output"
    return output / base_name, "wav"


def write_named_outputs(
    outputs: Dict[str, torch.Tensor],
    *,
    sample_rate: int,
    output_path: Optional[Union[str, Path]],
    input_path: Optional[Path],
) -> None:
    if output_path is None:
        return
    base_path, suffix = derive_output_base(output_path, input_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for name, tensor in outputs.items():
        target = base_path.parent / f"{base_path.name}_{name}.{suffix}"
        sf.write(target, tensor.detach().cpu().numpy().T, sample_rate)
