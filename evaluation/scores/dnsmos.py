from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from evaluation.evalBasis import EvalBasis

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
DEFAULT_BATCH_SIZE = 64


def _available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


def _resolve_onnx_threads(value: Optional[int | str] = None) -> int:
    raw_value = os.getenv("DNSMOS_ONNX_THREADS", "1") if value is None else str(value)
    if raw_value.lower() == "auto":
        workers = max(1, int(os.getenv("DNSMOS_NUM_WORKERS", "1")))
        return max(1, _available_cpu_count() // workers)
    return max(1, int(raw_value))


def _make_session(model_path: str | Path, onnx_threads: Optional[int | str] = None) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = _resolve_onnx_threads(onnx_threads)
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = int(os.getenv("DNSMOS_ORT_LOG_LEVEL", "3"))
    return ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return np.ascontiguousarray(audio)
    if audio.ndim == 2:
        # EvalBasis 约定时间维在 axis=0；这里兼容少量 channel-first 输入。
        if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)
        return np.ascontiguousarray(audio, dtype=np.float32)
    raise ValueError(f"DNSMOS expects 1D or 2D audio, got shape={audio.shape}")


def _repeat_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) == 0:
        raise ValueError("DNSMOS received empty audio.")
    if len(audio) >= length:
        return audio
    repeats = int(np.ceil(length / len(audio)))
    return np.tile(audio, repeats)[:length]


def _build_segments(audio: np.ndarray, sampling_rate: int) -> np.ndarray:
    len_samples = int(INPUT_LENGTH * sampling_rate)
    audio = _repeat_to_length(audio, len_samples)
    num_hops = max(1, int(np.floor((len(audio) - len_samples) / sampling_rate)) + 1)
    starts = np.arange(num_hops, dtype=np.int64) * sampling_rate
    return np.stack([audio[start : start + len_samples] for start in starts], axis=0).astype(np.float32, copy=False)


class DNSMOS(EvalBasis):
    def __init__(
        self,
        *,
        ovrl_only: bool = True,
        onnx_threads: Optional[int | str] = None,
        batch_size: Optional[int] = None,
        primary_model_path: str = "pretrained_models/DNSMOS/sig_bak_ovr.onnx",
        p808_model_path: str = "pretrained_models/DNSMOS/model_v8.onnx",
    ):
        super(DNSMOS, self).__init__(name="DNSMOS")
        self.intrusive = False
        self.score_rate = SAMPLING_RATE
        self.ovrl_only = ovrl_only
        self.primary_model_path = primary_model_path
        self.p808_model_path = p808_model_path
        self.compute_score = ComputeScore(
            self.primary_model_path,
            self.p808_model_path,
            enable_p808=not ovrl_only,
            onnx_threads=onnx_threads,
            batch_size=batch_size,
        )

    def _scoring(self, audios, rate):
        return self.compute_score.cal_mos(audios[0], rate, include_p808=not self.ovrl_only)["OVRL"]

    def score_16k(self, audio: np.ndarray, *, return_all: bool = False):
        """Fast path for callers that already have mono 16 kHz audio in memory."""
        result = self.compute_score.cal_mos_16k(audio, include_p808=not self.ovrl_only)
        return result if return_all else result["OVRL"]


class ComputeScore:
    def __init__(
        self,
        primary_model_path,
        p808_model_path=None,
        *,
        enable_p808: bool = False,
        onnx_threads: Optional[int | str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.batch_size = int(batch_size or os.getenv("DNSMOS_BATCH_SIZE", DEFAULT_BATCH_SIZE))
        self.onnx_sess = _make_session(primary_model_path, onnx_threads=onnx_threads)
        self.primary_input_name = self.onnx_sess.get_inputs()[0].name
        self.primary_batch_supported: Optional[bool] = None

        self.enable_p808 = bool(enable_p808)
        self.p808_onnx_sess = None
        self.p808_input_name = None
        self.p808_batch_supported: Optional[bool] = None
        if self.enable_p808:
            if p808_model_path is None:
                raise ValueError("enable_p808=True requires p808_model_path.")
            self.p808_onnx_sess = _make_session(p808_model_path, onnx_threads=onnx_threads)
            self.p808_input_name = self.p808_onnx_sess.get_inputs()[0].name

    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        import librosa

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=frame_size + 1,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr):
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return p_sig(sig), p_bak(bak), p_ovr(ovr)

    def _run_session(self, session, input_name: str, features: np.ndarray, batch_flag_name: str) -> np.ndarray:
        features = np.ascontiguousarray(features, dtype=np.float32)
        if features.shape[0] == 1:
            return np.asarray(session.run(None, {input_name: features})[0])

        batch_supported = getattr(self, batch_flag_name)
        if batch_supported is None:
            try:
                session.run(None, {input_name: features[:2]})
                batch_supported = True
            except Exception:
                batch_supported = False
            setattr(self, batch_flag_name, batch_supported)

        if not batch_supported:
            outputs = [np.asarray(session.run(None, {input_name: features[i : i + 1]})[0]) for i in range(features.shape[0])]
            return np.concatenate(outputs, axis=0)

        outputs = []
        for start in range(0, features.shape[0], self.batch_size):
            outputs.append(np.asarray(session.run(None, {input_name: features[start : start + self.batch_size]})[0]))
        return np.concatenate(outputs, axis=0)

    def _run_primary(self, segments: np.ndarray) -> np.ndarray:
        raw = self._run_session(self.onnx_sess, self.primary_input_name, segments, "primary_batch_supported")
        if raw.ndim == 1:
            return raw[None, :]
        if raw.ndim == 3 and raw.shape[1] == 1:
            return raw[:, 0, :]
        return raw

    def _run_p808(self, segments: np.ndarray) -> np.ndarray:
        if self.p808_onnx_sess is None or self.p808_input_name is None:
            raise RuntimeError("P808 session is not initialized.")
        p808_features = np.stack(
            [self.audio_melspec(audio=segment[:-160]) for segment in segments],
            axis=0,
        ).astype(np.float32, copy=False)
        raw = self._run_session(self.p808_onnx_sess, self.p808_input_name, p808_features, "p808_batch_supported")
        return raw.reshape(raw.shape[0], -1)[:, 0]

    def cal_mos(self, audio, sampling_rate, *, include_p808: Optional[bool] = None):
        if int(sampling_rate) != SAMPLING_RATE:
            raise ValueError(
                f"DNSMOS model expects {SAMPLING_RATE} Hz audio, got {sampling_rate}. "
                "Use DNSMOS.scoring(...) for automatic resampling or DNSMOS.score_16k(...) for the fast path."
            )
        return self.cal_mos_16k(audio, include_p808=include_p808)

    def cal_mos_16k(self, audio, *, include_p808: Optional[bool] = None):
        include_p808 = self.enable_p808 if include_p808 is None else bool(include_p808)
        if include_p808 and self.p808_onnx_sess is None:
            raise RuntimeError("This DNSMOS ComputeScore was created without P808 enabled.")

        audio = _to_mono_float32(audio)
        segments = _build_segments(audio, SAMPLING_RATE)
        mos_raw = self._run_primary(segments)

        mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
            mos_raw[:, 0],
            mos_raw[:, 1],
            mos_raw[:, 2],
        )
        results = {
            "OVRL": float(np.mean(mos_ovr)),
            "SIG": float(np.mean(mos_sig)),
            "BAK": float(np.mean(mos_bak)),
        }
        if include_p808:
            results["P808_MOS"] = float(np.mean(self._run_p808(segments)))
        return results
