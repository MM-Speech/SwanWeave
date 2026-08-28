from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from data_gen.source_separation.clearvoice import ClearVoice
from data_gen.source_separation.clearvoice.dataloader.dataloader import audio_norm
from data_gen.source_separation.clearvoice.utils.decode import decode_one_audio
from data_gen.source_separation.common import (
    AudioInput,
    ensure_channel_first_tensor,
    load_audio_input,
    resample_channels,
    trim_or_pad,
    write_named_outputs,
)

ENHANCEMENT_MODELS = {"FRCRN_SE_16K", "MossFormer2_SE_48K", "MossFormerGAN_SE_16K"}
SEPARATION_MODELS = {"MossFormer2_SS_16K"}
NORMALIZED_INPUT_MODELS = {"FRCRN_SE_16K", "MossFormer2_SS_16K"}


@dataclass
class ClearVoiceModelHandle:
    task: str
    model_name: str
    runtime: ClearVoice
    speech_model: object
    device: torch.device
    sample_rate: int


def _build_clearvoice_model(
    *,
    task: str,
    model_name: str,
    device: str | torch.device = "cuda",
    model_root: Optional[str | Path] = None,
    allow_legacy_fallback: bool = True,
) -> ClearVoiceModelHandle:
    runtime = ClearVoice(
        task=task,
        model_names=[model_name],
        device=device,
        model_root=model_root,
        allow_legacy_fallback=allow_legacy_fallback,
        allow_download=False,
    )
    speech_model = runtime.models[0]
    return ClearVoiceModelHandle(
        task=task,
        model_name=speech_model.name,
        runtime=runtime,
        speech_model=speech_model,
        device=speech_model.device,
        sample_rate=int(speech_model.args.sampling_rate),
    )


def _coerce_handle(
    model: ClearVoiceModelHandle | ClearVoice,
    *,
    expected_task: Optional[str] = None,
) -> ClearVoiceModelHandle:
    if isinstance(model, ClearVoiceModelHandle):
        handle = model
    elif isinstance(model, ClearVoice):
        if len(model.models) != 1:
            raise ValueError("ClearVoice 运行接口只支持单模型实例。")
        speech_model = model.models[0]
        handle = ClearVoiceModelHandle(
            task=getattr(speech_model.args, "task", expected_task or ""),
            model_name=speech_model.name,
            runtime=model,
            speech_model=speech_model,
            device=speech_model.device,
            sample_rate=int(speech_model.args.sampling_rate),
        )
    else:
        raise TypeError(f"Unsupported clearvoice model handle: {type(model)!r}")

    if expected_task and handle.task != expected_task:
        raise ValueError(f"Expected clearvoice task `{expected_task}`, got `{handle.task}`.")
    return handle


def _normalize_channel_input(channel_audio: np.ndarray, model_name: str) -> tuple[np.ndarray, float]:
    processed = channel_audio.astype(np.float32, copy=False)
    scalar = 1.0
    if model_name in NORMALIZED_INPUT_MODELS:
        processed, scalar = audio_norm(processed)
    return processed.astype(np.float32, copy=False), float(scalar)


def _decode_channel(
    handle: ClearVoiceModelHandle,
    channel_audio: np.ndarray,
) -> np.ndarray | list[np.ndarray]:
    with torch.no_grad():
        return decode_one_audio(
            handle.speech_model.model,
            handle.device,
            channel_audio[None, :],
            handle.speech_model.args,
        )


def _postprocess_single_output(
    audio: np.ndarray,
    *,
    scalar: float,
    input_length: int,
) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float32)
    output = trim_or_pad(output[None, :], input_length)[0]
    return output * scalar


def _finalize_outputs(
    outputs: Dict[str, np.ndarray],
    *,
    input_sr: int,
    model_sr: int,
    input_length: int,
) -> Dict[str, torch.Tensor]:
    finalized = {}
    for name, audio in outputs.items():
        if model_sr != input_sr:
            audio = resample_channels(audio, model_sr, input_sr)
        audio = trim_or_pad(audio, input_length)
        finalized[name] = ensure_channel_first_tensor(audio)
    return finalized


def build_clearvoice_enhancer(
    device: str | torch.device = "cuda",
    model_name: str = "MossFormer2_SE_48K",
    model_root: Optional[str | Path] = None,
    allow_legacy_fallback: bool = True,
) -> ClearVoiceModelHandle:
    if model_name not in ENHANCEMENT_MODELS:
        raise ValueError(f"Unsupported clearvoice enhancement model: {model_name}")
    return _build_clearvoice_model(
        task="speech_enhancement",
        model_name=model_name,
        device=device,
        model_root=model_root,
        allow_legacy_fallback=allow_legacy_fallback,
    )


def run_clearvoice_enhancement(
    audio: AudioInput,
    model: ClearVoiceModelHandle | ClearVoice,
    sample_rate: Optional[int] = None,
    output_path: Optional[str | Path] = None,
    include_residual: bool = True,
) -> Dict[str, Dict[str, torch.Tensor] | int]:
    handle = _coerce_handle(model, expected_task="speech_enhancement")
    wav, input_sr, input_path = load_audio_input(audio, sample_rate)
    model_wav = resample_channels(wav, input_sr, handle.sample_rate)

    enhanced_channels = []
    for channel_audio in model_wav:
        processed, scalar = _normalize_channel_input(channel_audio, handle.model_name)
        decoded = _decode_channel(handle, processed)
        enhanced_channels.append(
            _postprocess_single_output(
                decoded,
                scalar=scalar,
                input_length=processed.shape[0],
            )
        )

    finalized_outputs = _finalize_outputs(
        {"enhanced": np.stack(enhanced_channels, axis=0)},
        input_sr=input_sr,
        model_sr=handle.sample_rate,
        input_length=wav.shape[1],
    )

    if include_residual:
        finalized_outputs["residual"] = ensure_channel_first_tensor(
            wav - finalized_outputs["enhanced"].detach().cpu().numpy()
        )

    write_named_outputs(
        finalized_outputs,
        sample_rate=input_sr,
        output_path=output_path,
        input_path=input_path,
    )
    return {"sr": input_sr, "outputs": finalized_outputs}


def build_clearvoice_separator(
    device: str | torch.device = "cuda",
    model_name: str = "MossFormer2_SS_16K",
    model_root: Optional[str | Path] = None,
    allow_legacy_fallback: bool = True,
) -> ClearVoiceModelHandle:
    if model_name not in SEPARATION_MODELS:
        raise ValueError(f"Unsupported clearvoice separation model: {model_name}")
    return _build_clearvoice_model(
        task="speech_separation",
        model_name=model_name,
        device=device,
        model_root=model_root,
        allow_legacy_fallback=allow_legacy_fallback,
    )


def _stack_speaker_outputs(
    per_channel_outputs: Sequence[list[np.ndarray]],
    *,
    num_speakers: int,
    input_sr: int,
    model_sr: int,
    input_length: int,
) -> Dict[str, torch.Tensor]:
    outputs = {}
    for speaker_idx in range(num_speakers):
        speaker_channels = [channel_outputs[speaker_idx] for channel_outputs in per_channel_outputs]
        speaker_audio = np.stack(speaker_channels, axis=0).astype(np.float32, copy=False)
        outputs[f"speaker_{speaker_idx + 1}"] = ensure_channel_first_tensor(
            trim_or_pad(
                resample_channels(speaker_audio, model_sr, input_sr) if model_sr != input_sr else speaker_audio,
                input_length,
            )
        )
    return outputs


def run_clearvoice_separation(
    audio: AudioInput,
    model: ClearVoiceModelHandle | ClearVoice,
    sample_rate: Optional[int] = None,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Dict[str, torch.Tensor] | int]:
    handle = _coerce_handle(model, expected_task="speech_separation")
    wav, input_sr, input_path = load_audio_input(audio, sample_rate)
    model_wav = resample_channels(wav, input_sr, handle.sample_rate)
    num_speakers = int(handle.speech_model.args.num_spks)

    per_channel_outputs = []
    for channel_audio in model_wav:
        processed, scalar = _normalize_channel_input(channel_audio, handle.model_name)
        decoded = _decode_channel(handle, processed)
        if not isinstance(decoded, list) or len(decoded) != num_speakers:
            raise RuntimeError(
                f"clearvoice separation expected {num_speakers} speakers, got {type(decoded)!r}"
            )
        per_channel_outputs.append(
            [
                _postprocess_single_output(
                    speaker_audio,
                    scalar=scalar,
                    input_length=processed.shape[0],
                )
                for speaker_audio in decoded
            ]
        )

    finalized_outputs = _stack_speaker_outputs(
        per_channel_outputs,
        num_speakers=num_speakers,
        input_sr=input_sr,
        model_sr=handle.sample_rate,
        input_length=wav.shape[1],
    )
    write_named_outputs(
        finalized_outputs,
        sample_rate=input_sr,
        output_path=output_path,
        input_path=input_path,
    )
    return {"sr": input_sr, "outputs": finalized_outputs}


def build_se_model(
    device: str | torch.device = "cuda",
    model_name: str = "MossFormer2_SE_48K",
    model_root: Optional[str | Path] = None,
    allow_legacy_fallback: bool = True,
) -> ClearVoiceModelHandle:
    return build_clearvoice_enhancer(
        device=device,
        model_name=model_name,
        model_root=model_root,
        allow_legacy_fallback=allow_legacy_fallback,
    )


def run_se_model(
    audio_file: AudioInput,
    se_model: ClearVoiceModelHandle | ClearVoice,
    target_sample_rate: int = 48000,
) -> Dict[str, Dict[str, torch.Tensor] | int]:
    del target_sample_rate
    result = run_clearvoice_enhancement(audio_file, se_model, include_residual=True)
    outputs = result["outputs"]
    return {
        "separated": {
            "vocal": outputs["enhanced"],
            "accompy": outputs["residual"],
        },
        "sr": result["sr"],
    }


__all__ = [
    "ClearVoiceModelHandle",
    "build_clearvoice_enhancer",
    "run_clearvoice_enhancement",
    "build_clearvoice_separator",
    "run_clearvoice_separation",
    "build_se_model",
    "run_se_model",
]
