from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import soundfile as sf


DEFAULT_TARGET_SR = 24000
DEFAULT_TARGET_LUFS = -23.0
DEFAULT_GAP_MS = 200
DEFAULT_FADE_MS = 10
DEFAULT_TP_LIMIT_DB = -1.0
DEFAULT_UVR_MODEL_NAME = "MDX23C-8KFFT-InstVoc_HQ"
DEFAULT_DEVICE = "cuda"
DEFAULT_CACHE_SUBDIR = "_cache/uvr"

VIDEO_SUFFIXES = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".flv",
    ".m4v",
}

_SPACE_RE = re.compile(r"\s+")
_SAFE_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]+')
_TQDM_WARNING_SHOWN = False


class _PlainProgress:
    def __init__(self, total: int):
        self.total = int(total)
        self.n = 0

    def set_description_str(self, desc: str, refresh: bool = True) -> None:
        del desc, refresh

    def set_postfix(self, ordered_dict: Optional[Mapping[str, Any]] = None, refresh: bool = True, **kwargs: Any) -> None:
        del ordered_dict, refresh, kwargs

    def write(self, message: str) -> None:
        print(message)

    def update(self, n: int = 1) -> None:
        self.n += int(n)

    def close(self) -> None:
        return None


def _load_uvr_api():
    try:
        from data_gen.source_separation.uvr.uvr_api import build_uvr_model, run_uvr_model
    except ModuleNotFoundError as exc:
        raise RuntimeError("UVR 依赖不可用，请在远程推理环境中运行该 helper。") from exc
    return build_uvr_model, run_uvr_model


def _load_normalize_lufs():
    try:
        from utils.audio.transform import normalize_lufs
    except ModuleNotFoundError as exc:
        raise RuntimeError("LUFS 归一依赖不可用，请在远程推理环境中运行该 helper。") from exc
    return normalize_lufs


def _create_progress_tracker(total: int, *, enabled: bool):
    global _TQDM_WARNING_SHOWN

    if not enabled:
        return _PlainProgress(total)

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        if not _TQDM_WARNING_SHOWN:
            print("[WARN] tqdm 未安装，已回退为普通日志输出。")
            _TQDM_WARNING_SHOWN = True
        return _PlainProgress(total)

    return tqdm(
        total=total,
        desc="Cases",
        unit="case",
        dynamic_ncols=True,
        leave=True,
        file=sys.stderr,
    )


def _truncate_text(text: str, max_length: int = 32) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _update_progress_state(progress: Any, *, case_id: str, summary: Mapping[str, Sequence[Any]]) -> None:
    progress.set_postfix(
        {
            "case": _truncate_text(case_id, max_length=24),
            "ok": len(summary["successes"]),
            "fail": len(summary["failures"]),
            "skip": len(summary["skipped"]),
        },
        refresh=False,
    )


def _progress_write(progress: Any, message: str) -> None:
    progress.write(message)


@dataclass(frozen=True)
class SegmentSpec:
    source: Path
    start_raw: Union[str, float, int]
    end_raw: Union[str, float, int]
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    segments: Tuple[SegmentSpec, ...]


@dataclass
class SourceVocalStemInfo:
    source_path: Path
    sample_rate: int
    vocals: np.ndarray  # [C, T]
    duration_sec: float
    cache_key: str
    cache_hit: bool
    cache_source: str  # "memory" | "disk" | "miss"
    vocal_cache_path: Path


def parse_timestamp(value: Union[str, float, int]) -> float:
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 0:
            raise ValueError(f"timestamp must be non-negative, got {value!r}")
        return seconds

    text = str(value).strip()
    if not text:
        raise ValueError("timestamp cannot be empty")

    if ":" not in text:
        try:
            seconds = float(text)
        except ValueError as exc:
            raise ValueError(f"invalid timestamp: {value!r}") from exc
        if seconds < 0:
            raise ValueError(f"timestamp must be non-negative, got {value!r}")
        return seconds

    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"invalid timestamp: {value!r}")

    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            total = minutes * 60.0 + seconds
        else:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            total = hours * 3600.0 + minutes * 60.0 + seconds
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {value!r}") from exc

    if total < 0:
        raise ValueError(f"timestamp must be non-negative, got {value!r}")
    return total


def prepare_vocal_segments(
    case_spec: Mapping[str, Any],
    output_dir: Union[str, Path],
    *,
    target_sr: int = DEFAULT_TARGET_SR,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    gap_ms: int = DEFAULT_GAP_MS,
    fade_ms: int = DEFAULT_FADE_MS,
    uvr_model_name: str = DEFAULT_UVR_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    cache_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    preparer = _VocalSegmentPreparer(
        target_sr=target_sr,
        target_lufs=target_lufs,
        gap_ms=gap_ms,
        fade_ms=fade_ms,
        uvr_model_name=uvr_model_name,
        device=device,
        cache_dir=cache_dir,
    )
    return preparer.prepare_case(case_spec, output_dir=output_dir, overwrite=overwrite)


def prepare_vocal_segments_from_manifest(
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    target_sr: Optional[int] = None,
    target_lufs: Optional[float] = None,
    gap_ms: Optional[int] = None,
    fade_ms: Optional[int] = None,
    uvr_model_name: Optional[str] = None,
    device: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    skip_existing: bool = False,
    show_progress: bool = True,
) -> Dict[str, Any]:
    manifest = _load_yaml_manifest(manifest_path)
    defaults = manifest.get("defaults", {}) or {}
    cases = manifest.get("cases", []) or []
    if not isinstance(cases, list) or len(cases) == 0:
        raise ValueError("manifest must contain a non-empty 'cases' list")

    effective_target_sr = int(_coalesce(target_sr, defaults.get("target_sr"), DEFAULT_TARGET_SR))
    effective_target_lufs = float(_coalesce(target_lufs, defaults.get("target_lufs"), DEFAULT_TARGET_LUFS))
    effective_gap_ms = int(_coalesce(gap_ms, defaults.get("gap_ms"), DEFAULT_GAP_MS))
    effective_fade_ms = int(_coalesce(fade_ms, defaults.get("fade_ms"), DEFAULT_FADE_MS))
    effective_device = str(_coalesce(device, defaults.get("device"), DEFAULT_DEVICE))
    effective_model_name = str(_coalesce(uvr_model_name, defaults.get("uvr_model_name"), DEFAULT_UVR_MODEL_NAME))

    preparer = _VocalSegmentPreparer(
        target_sr=effective_target_sr,
        target_lufs=effective_target_lufs,
        gap_ms=effective_gap_ms,
        fade_ms=effective_fade_ms,
        uvr_model_name=effective_model_name,
        device=effective_device,
        cache_dir=cache_dir,
    )

    output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "output_dir": str(output_dir_path),
        "successes": [],
        "failures": [],
        "skipped": [],
    }

    progress = _create_progress_tracker(len(cases), enabled=show_progress)
    try:
        for index, raw_case in enumerate(cases):
            fallback_case_id = f"case_{index:04d}"
            case_id = str((raw_case or {}).get("case_id") or fallback_case_id)
            case_file_name = _safe_case_filename(case_id)
            out_wav = output_dir_path / f"{case_file_name}.wav"
            progress.set_description_str(f"Cases {index + 1}/{len(cases)}", refresh=False)
            _update_progress_state(progress, case_id=case_id, summary=summary)

            if skip_existing and out_wav.exists():
                summary["skipped"].append({"case_id": case_id, "output_path": str(out_wav)})
                _update_progress_state(progress, case_id=case_id, summary=summary)
                _progress_write(progress, f"[SKIP] {case_id}: output already exists at {out_wav}")
                progress.update(1)
                continue

            try:
                meta = preparer.prepare_case(raw_case, output_dir=output_dir_path, overwrite=overwrite)
            except Exception as exc:
                sidecar_path = output_dir_path / f"{case_file_name}.json"
                failure_payload = _build_failure_payload(
                    case_id=case_id,
                    case_spec=raw_case,
                    output_path=out_wav,
                    sidecar_path=sidecar_path,
                    error=str(exc),
                    params=preparer.params_dict(),
                )
                _write_json(sidecar_path, failure_payload)
                summary["failures"].append(
                    {"case_id": case_id, "error": str(exc), "sidecar_path": str(sidecar_path)}
                )
                _update_progress_state(progress, case_id=case_id, summary=summary)
                _progress_write(progress, f"[FAIL] {case_id}: {exc}")
                progress.update(1)
                continue

            summary["successes"].append(
                {"case_id": case_id, "output_path": meta["output_path"], "sidecar_path": meta["sidecar_path"]}
            )
            _update_progress_state(progress, case_id=case_id, summary=summary)
            _progress_write(progress, f"[OK] {case_id}: {meta['output_path']}")
            progress.update(1)
    finally:
        progress.close()

    print(
        "[SUMMARY] "
        f"success={len(summary['successes'])} "
        f"failure={len(summary['failures'])} "
        f"skipped={len(summary['skipped'])}"
    )
    return summary


class _VocalSegmentPreparer:
    def __init__(
        self,
        *,
        target_sr: int,
        target_lufs: float,
        gap_ms: int,
        fade_ms: int,
        uvr_model_name: str,
        device: str,
        cache_dir: Optional[Union[str, Path]],
    ):
        self.target_sr = int(target_sr)
        self.target_lufs = float(target_lufs)
        self.gap_ms = int(gap_ms)
        self.fade_ms = int(fade_ms)
        self.uvr_model_name = str(uvr_model_name)
        self.device = str(device)
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        self._uvr_model = None
        self._source_cache: Dict[Path, SourceVocalStemInfo] = {}

    def params_dict(self) -> Dict[str, Any]:
        return {
            "target_sr": self.target_sr,
            "target_lufs": self.target_lufs,
            "gap_ms": self.gap_ms,
            "fade_ms": self.fade_ms,
            "tp_limit_db": DEFAULT_TP_LIMIT_DB,
            "uvr_model_name": self.uvr_model_name,
            "device": self.device,
            "cache_dir": str(self.cache_dir) if self.cache_dir is not None else None,
        }

    def prepare_case(
        self,
        case_spec: Mapping[str, Any],
        *,
        output_dir: Union[str, Path],
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        normalized_case = _normalize_case_spec(case_spec)
        output_dir_path = Path(output_dir).expanduser().resolve()
        output_dir_path.mkdir(parents=True, exist_ok=True)

        case_file_name = _safe_case_filename(normalized_case.case_id)
        output_path = output_dir_path / f"{case_file_name}.wav"
        sidecar_path = output_dir_path / f"{case_file_name}.json"

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {output_path}")

        segment_metas: List[Dict[str, Any]] = []
        rendered_segments: List[np.ndarray] = []

        for index, segment in enumerate(normalized_case.segments):
            source_info = self._load_or_separate_source(segment.source, output_dir=output_dir_path)
            clip, effective_end_sec, end_was_clamped = self._extract_segment(
                vocals=source_info.vocals,
                sample_rate=source_info.sample_rate,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                source_path=segment.source,
            )
            rendered_segments.append(clip)
            segment_metas.append(
                {
                    "index": index,
                    "source": str(segment.source),
                    "start": segment.start_raw,
                    "end": segment.end_raw,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "effective_end_sec": effective_end_sec,
                    "end_was_clamped": end_was_clamped,
                    "source_duration_sec": source_info.duration_sec,
                    "cache_key": source_info.cache_key,
                    "cache_hit": source_info.cache_hit,
                    "cache_source": source_info.cache_source,
                    "vocal_cache_path": str(source_info.vocal_cache_path),
                    "rendered_duration_sec": float(len(clip) / self.target_sr),
                }
            )

        if len(rendered_segments) == 0:
            raise ValueError("no rendered segments were produced")

        merged = self._merge_segments(rendered_segments)
        merged = self._normalize_output_audio(merged)
        merged = _float_range_normalize(merged)

        _write_wav(output_path, merged, self.target_sr)

        metadata = {
            "status": "success",
            "case_id": normalized_case.case_id,
            "output_path": str(output_path),
            "sidecar_path": str(sidecar_path),
            "parameters": self.params_dict(),
            "segments": segment_metas,
            "audio": {
                "sample_rate": self.target_sr,
                "channels": 1,
                "num_samples": int(merged.shape[0]),
                "duration_sec": float(merged.shape[0] / self.target_sr),
                "peak": float(np.max(np.abs(merged))) if merged.size > 0 else 0.0,
            },
            "error": None,
        }
        _write_json(sidecar_path, metadata)
        return metadata

    def _get_uvr_model(self):
        if self._uvr_model is None:
            build_uvr_model, _ = _load_uvr_api()
            self._uvr_model = build_uvr_model(device=self.device, model_name=self.uvr_model_name)
        return self._uvr_model

    def _resolve_cache_root(self, output_dir: Path) -> Path:
        if self.cache_dir is not None:
            cache_root = self.cache_dir
        else:
            cache_root = output_dir / DEFAULT_CACHE_SUBDIR
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root

    def _load_or_separate_source(self, source_path: Path, *, output_dir: Path) -> SourceVocalStemInfo:
        source_path = source_path.expanduser().resolve()
        if source_path in self._source_cache:
            cached = self._source_cache[source_path]
            cached.cache_hit = True
            cached.cache_source = "memory"
            return cached

        if not source_path.exists():
            raise FileNotFoundError(f"source file not found: {source_path}")

        cache_root = self._resolve_cache_root(output_dir)
        stat = source_path.stat()
        key_raw = f"{source_path}|{stat.st_size}|{stat.st_mtime_ns}|{self.uvr_model_name}"
        cache_key = hashlib.sha1(key_raw.encode("utf-8")).hexdigest()[:16]
        cache_dir = cache_root / cache_key
        vocal_cache_path = cache_dir / "vocals.wav"
        cache_meta_path = cache_dir / "meta.json"

        if vocal_cache_path.exists():
            vocals, sample_rate = _load_audio_file(vocal_cache_path)
            info = SourceVocalStemInfo(
                source_path=source_path,
                sample_rate=sample_rate,
                vocals=vocals,
                duration_sec=float(vocals.shape[-1] / sample_rate),
                cache_key=cache_key,
                cache_hit=True,
                cache_source="disk",
                vocal_cache_path=vocal_cache_path,
            )
            self._source_cache[source_path] = info
            return info

        wav, sample_rate = self._decode_source_audio(source_path)
        vocals = self._separate_source_audio(wav, sample_rate)

        cache_dir.mkdir(parents=True, exist_ok=True)
        _write_multichannel_wav(vocal_cache_path, vocals, sample_rate)
        _write_json(
            cache_meta_path,
            {
                "source_path": str(source_path),
                "sample_rate": sample_rate,
                "uvr_model_name": self.uvr_model_name,
                "cache_key": cache_key,
                "duration_sec": float(vocals.shape[-1] / sample_rate),
            },
        )

        info = SourceVocalStemInfo(
            source_path=source_path,
            sample_rate=sample_rate,
            vocals=vocals,
            duration_sec=float(vocals.shape[-1] / sample_rate),
            cache_key=cache_key,
            cache_hit=False,
            cache_source="miss",
            vocal_cache_path=vocal_cache_path,
        )
        self._source_cache[source_path] = info
        return info

    def _decode_source_audio(self, source_path: Path) -> Tuple[np.ndarray, int]:
        if source_path.suffix.lower() in VIDEO_SUFFIXES:
            return self._decode_media_file(source_path)
        return _load_audio_file(source_path)

    def _decode_media_file(self, source_path: Path) -> Tuple[np.ndarray, int]:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError("ffmpeg is required to decode video inputs but was not found in PATH")

        command = [
            ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "2",
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ffmpeg decode failed for {source_path}: {stderr or f'code={result.returncode}'}")
        if not result.stdout:
            raise RuntimeError(f"ffmpeg decode returned empty audio: {source_path}")

        wav, sample_rate = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=True)
        wav = np.asarray(wav, dtype=np.float32).T
        return _ensure_uvr_channels(wav), int(sample_rate)

    def _separate_source_audio(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        _, run_uvr_model = _load_uvr_api()
        result = run_uvr_model(wav, self._get_uvr_model(), sample_rate=sample_rate, output_path=None)
        vocals = result["outputs"]["vocals"].detach().cpu().numpy().astype(np.float32, copy=False)
        return _ensure_uvr_channels(vocals)

    def _extract_segment(
        self,
        *,
        vocals: np.ndarray,
        sample_rate: int,
        start_sec: float,
        end_sec: float,
        source_path: Path,
    ) -> Tuple[np.ndarray, float, bool]:
        duration_sec = vocals.shape[-1] / float(sample_rate)
        if start_sec < 0:
            raise ValueError(f"segment start must be non-negative for {source_path}")
        if end_sec <= start_sec:
            raise ValueError(f"segment end must be greater than start for {source_path}")
        if start_sec >= duration_sec:
            raise ValueError(
                f"segment start {start_sec:.3f}s exceeds source duration {duration_sec:.3f}s for {source_path}"
            )

        effective_end_sec = min(end_sec, duration_sec)
        end_was_clamped = effective_end_sec < end_sec

        start_idx = int(round(start_sec * sample_rate))
        end_idx = min(int(round(effective_end_sec * sample_rate)), vocals.shape[-1])
        if end_idx <= start_idx:
            raise ValueError(f"segment became empty after rounding for {source_path}")

        clip = np.asarray(vocals[:, start_idx:end_idx], dtype=np.float32)
        clip = _apply_linear_fade(clip, sample_rate=sample_rate, fade_ms=self.fade_ms)
        clip = _to_mono(clip)
        if sample_rate != self.target_sr:
            clip = librosa.resample(clip, orig_sr=sample_rate, target_sr=self.target_sr)
        return np.asarray(clip, dtype=np.float32), float(effective_end_sec), bool(end_was_clamped)

    def _merge_segments(self, segments: Sequence[np.ndarray]) -> np.ndarray:
        if len(segments) == 1:
            return np.asarray(segments[0], dtype=np.float32)

        gap_samples = int(round(self.gap_ms * self.target_sr / 1000.0))
        gap = np.zeros((gap_samples,), dtype=np.float32)
        pieces: List[np.ndarray] = []
        for index, segment in enumerate(segments):
            if index > 0 and gap_samples > 0:
                pieces.append(gap)
            pieces.append(np.asarray(segment, dtype=np.float32))
        return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)

    def _normalize_output_audio(self, audio: np.ndarray) -> np.ndarray:
        try:
            normalize_lufs = _load_normalize_lufs()
            normalized = normalize_lufs(
                audio,
                self.target_sr,
                target_lufs=self.target_lufs,
                tp_limit_db=DEFAULT_TP_LIMIT_DB,
            )
        except RuntimeError:
            raise
        except ModuleNotFoundError as exc:
            raise RuntimeError("LUFS 归一依赖缺失，请检查远程环境中的 pyloudnorm/scipy。") from exc
        return np.asarray(normalized, dtype=np.float32)


def _normalize_case_spec(case_spec: Mapping[str, Any]) -> CaseSpec:
    if not isinstance(case_spec, Mapping):
        raise TypeError(f"case_spec must be a mapping, got {type(case_spec)!r}")

    case_id = str(case_spec.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("case_spec.case_id is required")

    raw_segments = case_spec.get("segments")
    if not isinstance(raw_segments, list) or len(raw_segments) == 0:
        raise ValueError(f"case {case_id} must contain a non-empty segments list")

    normalized_segments: List[SegmentSpec] = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            raise TypeError(f"case {case_id} segment #{index} must be a mapping")
        source = str(raw_segment.get("source") or "").strip()
        if not source:
            raise ValueError(f"case {case_id} segment #{index} is missing source")
        if "start" not in raw_segment or "end" not in raw_segment:
            raise ValueError(f"case {case_id} segment #{index} must contain start and end")

        start_raw = raw_segment["start"]
        end_raw = raw_segment["end"]
        start_sec = parse_timestamp(start_raw)
        end_sec = parse_timestamp(end_raw)

        normalized_segments.append(
            SegmentSpec(
                source=Path(source).expanduser().resolve(),
                start_raw=start_raw,
                end_raw=end_raw,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )

    return CaseSpec(case_id=case_id, segments=tuple(normalized_segments))


def _load_yaml_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to load YAML manifests") from exc

    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be a mapping")
    return manifest


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _safe_case_filename(case_id: str) -> str:
    normalized = _SPACE_RE.sub("_", case_id.strip())
    normalized = normalized.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    normalized = _SAFE_NAME_RE.sub("_", normalized).strip("._")
    return normalized or "vocal_segments"


def _ensure_uvr_channels(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 1:
        wav = wav[None, :]
    elif wav.ndim == 2:
        if wav.shape[0] > wav.shape[1] and wav.shape[1] <= 8:
            wav = wav.T
    else:
        raise ValueError(f"unsupported audio shape: {wav.shape}")

    if wav.shape[0] > 2:
        wav = wav[:2]
    return np.asarray(wav, dtype=np.float32, order="C")


def _load_audio_file(path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    audio_path = Path(path).expanduser().resolve()
    try:
        wav, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        wav = np.asarray(wav, dtype=np.float32).T
        return _ensure_uvr_channels(wav), int(sample_rate)
    except Exception:
        wav, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
        if wav.ndim == 1:
            wav = wav[None, :]
        return _ensure_uvr_channels(wav), int(sample_rate)


def _apply_linear_fade(audio: np.ndarray, *, sample_rate: int, fade_ms: int) -> np.ndarray:
    if fade_ms <= 0:
        return np.array(audio, dtype=np.float32, copy=True)
    fade_samples = int(round(fade_ms * sample_rate / 1000.0))
    if fade_samples <= 0:
        return np.array(audio, dtype=np.float32, copy=True)

    audio = np.array(audio, dtype=np.float32, copy=True)
    length = audio.shape[-1]
    if length == 0:
        return audio

    fade_samples = min(fade_samples, max(length // 2, 1))
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    audio[..., :fade_samples] *= ramp
    audio[..., -fade_samples:] *= ramp[::-1]
    return audio


def _to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=0).astype(np.float32, copy=False)


def _float_range_normalize(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


def _write_wav(path: Union[str, Path], audio: np.ndarray, sample_rate: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.stem}.tmp{target.suffix}")
    sf.write(str(tmp), np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")
    os.replace(tmp, target)


def _write_multichannel_wav(path: Union[str, Path], audio: np.ndarray, sample_rate: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.stem}.tmp{target.suffix}")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        wav_to_write = audio.T
    else:
        wav_to_write = audio
    sf.write(str(tmp), wav_to_write, sample_rate, subtype="PCM_16")
    os.replace(tmp, target)


def _write_json(path: Union[str, Path], payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, target)


def _build_failure_payload(
    *,
    case_id: str,
    case_spec: Any,
    output_path: Path,
    sidecar_path: Path,
    error: str,
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "status": "failed",
        "case_id": case_id,
        "output_path": str(output_path),
        "sidecar_path": str(sidecar_path),
        "parameters": dict(params),
        "case_spec": case_spec,
        "segments": None,
        "audio": None,
        "error": error,
    }


def _build_single_case_spec(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.input:
        raise ValueError("--input is required when --manifest is not provided")
    if not args.case_id:
        raise ValueError("--case-id is required when --manifest is not provided")
    if not args.segment:
        raise ValueError("at least one --segment START END is required when --manifest is not provided")

    segments = []
    for start_raw, end_raw in args.segment:
        segments.append({"source": args.input, "start": start_raw, "end": end_raw})
    return {"case_id": args.case_id, "segments": segments}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare vocal segments for downstream demos or inference.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--manifest", type=str, help="YAML manifest path for batch mode.")
    mode_group.add_argument("--input", type=str, help="Single source mp4/wav path.")

    parser.add_argument("--segment", nargs=2, action="append", metavar=("START", "END"))
    parser.add_argument("--case-id", type=str, help="Case id for single-source mode.")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--target-sr", type=int, default=None)
    parser.add_argument("--target-lufs", type=float, default=None)
    parser.add_argument("--gap-ms", type=int, default=None)
    parser.add_argument("--fade-ms", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--uvr-model-name", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar in manifest mode.")

    args = parser.parse_args(argv)

    if args.manifest:
        summary = prepare_vocal_segments_from_manifest(
            args.manifest,
            args.output_dir,
            target_sr=args.target_sr,
            target_lufs=args.target_lufs,
            gap_ms=args.gap_ms,
            fade_ms=args.fade_ms,
            uvr_model_name=args.uvr_model_name,
            device=args.device,
            cache_dir=args.cache_dir,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
            show_progress=not args.no_progress,
        )
        return 1 if summary["failures"] else 0

    try:
        case_spec = _build_single_case_spec(args)
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{_safe_case_filename(str(args.case_id))}.wav"
        if args.skip_existing and output_path.exists():
            print(f"[SKIP] {args.case_id}: output already exists at {output_path}")
            return 0

        meta = prepare_vocal_segments(
            case_spec,
            output_dir,
            target_sr=int(_coalesce(args.target_sr, DEFAULT_TARGET_SR)),
            target_lufs=float(_coalesce(args.target_lufs, DEFAULT_TARGET_LUFS)),
            gap_ms=int(_coalesce(args.gap_ms, DEFAULT_GAP_MS)),
            fade_ms=int(_coalesce(args.fade_ms, DEFAULT_FADE_MS)),
            uvr_model_name=str(_coalesce(args.uvr_model_name, DEFAULT_UVR_MODEL_NAME)),
            device=str(_coalesce(args.device, DEFAULT_DEVICE)),
            cache_dir=args.cache_dir,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1

    print(f"[OK] {meta['case_id']}: {meta['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
