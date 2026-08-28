from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from data_gen.sr.audiosr.pipeline import build_model, make_batch_for_super_resolution
from data_gen.sr.audiosr.audiosr_utils import seed_everything
from utils.commons.os_utils import load_env_local

os.environ["TOKENIZERS_PARALLELISM"] = "true"
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)

AudioLike = Union[str, Path, np.ndarray, torch.Tensor]
InputLayout = Optional[str]
VALID_INPUT_LAYOUTS = {None, "batch", "channels", "time_channels"}


@dataclass
class PreparedAudio:
    audio: torch.Tensor  # [B, C, T] at self.sample_rate, normalized per item/channel.
    lengths: List[int]
    names: List[Optional[str]]
    channel_count: int


@dataclass
class ChunkInfo:
    flat_index: int
    start_sample: int
    end_sample: int
    current_chunk_len: int
    chunk_waveform: torch.Tensor  # [1, T]


@dataclass
class ChunkState:
    total_samples: int
    overlap_samples: int
    final_waveform: torch.Tensor  # [T]
    overlap_contribution_map: torch.Tensor  # [T]
    fade_in: torch.Tensor
    fade_out: torch.Tensor


class AudioSR:
    def __init__(
        self,
        model_name: str = "basic",
        device: str = "auto",
        save_path: str = "./output",
        ddim_steps: int = 50,
        guidance_scale: float = 3.5,
        seed: int = 42,
        suffix: str = "_AudioSR_Processed_48K",
        chunking: bool = True,
        chunk_duration: int = 15,
        overlap_duration: int = 2,
        model_root: Optional[Union[str, Path]] = 'pretrained_models/sr/audiosr',
        allow_download: bool = False,
        precision: str = "fp32",
        verbose: bool = False,
    ) -> None:
        """AudioSR inference API.

        The public infer path accepts path/list/array/tensor inputs, normalizes them
        into [B, C, T], runs each channel independently, and returns [B, C, T].
        """
        load_env_local()
        torch.set_float32_matmul_precision("high")
        self.model_name = model_name
        self.device = device
        self.save_path = save_path
        self.ddim_steps = ddim_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.suffix = suffix
        self.chunking = chunking
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.model_root = model_root
        self.allow_download = allow_download
        self.sample_rate = 48000
        self.precision = precision
        self.verbose = verbose
        self.model = build_model(
            model_name=self.model_name,
            device=self.device,
            model_root=self.model_root,
            allow_download=self.allow_download,
        )

    def infer(
        self,
        audio: Union[AudioLike, Sequence[AudioLike]],
        sr: Optional[int] = None,
        sr_list: Optional[Sequence[int]] = None,
        names: Optional[Union[str, Sequence[Optional[str]]]] = None,
        output_path: Optional[Union[str, Path]] = None,
        batch_size: int = 4,
        chunking: Optional[bool] = None,
        chunk_duration: Optional[int] = None,
        overlap_duration: Optional[int] = None,
        ddim_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        return_tensor: bool = True,
        input_layout: InputLayout = None,
    ) -> Dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if input_layout not in VALID_INPUT_LAYOUTS:
            raise ValueError(f"input_layout must be one of {sorted(x for x in VALID_INPUT_LAYOUTS if x is not None)}.")

        seed = self.seed if seed is None else seed
        seed_everything(int(seed))
        ddim_steps = self.ddim_steps if ddim_steps is None else ddim_steps
        guidance_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        chunking = self.chunking if chunking is None else chunking
        chunk_duration = self.chunk_duration if chunk_duration is None else chunk_duration
        overlap_duration = self.overlap_duration if overlap_duration is None else overlap_duration

        prepared = self._prepare_input_audio(audio, sr=sr, sr_list=sr_list, names=names, input_layout=input_layout)
        flat_audio, flat_lengths = self._flatten_channels(prepared.audio, prepared.lengths)
        flat_output = self._run_flat_audio(
            flat_audio=flat_audio,
            flat_lengths=flat_lengths,
            batch_size=batch_size,
            chunking=chunking,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
        )
        output_audio = self._unflatten_channels(
            flat_output=flat_output,
            batch_size=prepared.audio.shape[0],
            channel_count=prepared.channel_count,
            lengths=prepared.lengths,
        )
        saved_files = self._save_outputs(output_audio, prepared.lengths, prepared.names, output_path)

        result_audio: Union[torch.Tensor, np.ndarray]
        if return_tensor:
            result_audio = output_audio
        else:
            result_audio = output_audio.detach().cpu().numpy()
        return {
            "sr": self.sample_rate,
            "audio": result_audio,
            "lengths": prepared.lengths,
            "names": prepared.names,
            "saved_files": saved_files,
        }

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        return self.infer(*args, **kwargs)

    def _prepare_input_audio(
        self,
        audio: Union[AudioLike, Sequence[AudioLike]],
        *,
        sr: Optional[int],
        sr_list: Optional[Sequence[int]],
        names: Optional[Union[str, Sequence[Optional[str]]]],
        input_layout: InputLayout,
    ) -> PreparedAudio:
        items, item_srs, default_names = self._collect_audio_items(
            audio,
            sr=sr,
            sr_list=sr_list,
            names=names,
            input_layout=input_layout,
        )
        resolved_names = self._resolve_names(names, default_names, len(items))

        resampled_items = []
        lengths = []
        channel_counts = []
        for item, item_sr in zip(items, item_srs):
            resampled = self._resample_audio(item, item_sr, self.sample_rate)
            normalized = self._normalize_waveform(resampled)
            resampled_items.append(normalized)
            lengths.append(int(normalized.shape[-1]))
            channel_counts.append(int(normalized.shape[0]))

        if len(set(channel_counts)) != 1:
            raise ValueError(f"All batch items must have the same channel count, got {channel_counts}.")

        target_device = self._resolve_batch_device(resampled_items)
        resampled_items = [
            item.to(target_device) if item.device != target_device else item
            for item in resampled_items
        ]

        channel_count = channel_counts[0]
        max_length = max(lengths)
        padded_items = [
            F.pad(item, (0, max_length - item.shape[-1])) if item.shape[-1] < max_length else item
            for item in resampled_items
        ]
        return PreparedAudio(
            audio=torch.stack(padded_items, dim=0),
            lengths=lengths,
            names=resolved_names,
            channel_count=channel_count,
        )

    def _collect_audio_items(
        self,
        audio: Union[AudioLike, Sequence[AudioLike]],
        *,
        sr: Optional[int],
        sr_list: Optional[Sequence[int]],
        names: Optional[Union[str, Sequence[Optional[str]]]],
        input_layout: InputLayout,
    ) -> tuple[List[torch.Tensor], List[int], List[Optional[str]]]:
        if self._is_path(audio):
            waveform, item_sr = torchaudio.load(str(audio))
            return [waveform.float()], [int(item_sr)], [Path(audio).stem]

        if isinstance(audio, (np.ndarray, torch.Tensor)):
            if sr is None:
                raise ValueError("Array/tensor input requires sr.")
            if sr_list is not None:
                raise ValueError("Batch array/tensor input accepts a single sr, not sr_list.")
            items = self._split_array_or_tensor(audio, input_layout=input_layout, names=names)
            return items, [int(sr)] * len(items), self._default_names(len(items))

        if isinstance(audio, Sequence):
            audio_list = list(audio)
            if not audio_list:
                raise ValueError("audio list must not be empty.")

            if all(self._is_path(item) for item in audio_list):
                items = []
                item_srs = []
                default_names = []
                for item in audio_list:
                    waveform, item_sr = torchaudio.load(str(item))
                    items.append(waveform.float())
                    item_srs.append(int(item_sr))
                    default_names.append(Path(item).stem)
                return items, item_srs, default_names

            if all(isinstance(item, (np.ndarray, torch.Tensor)) for item in audio_list):
                if sr is None and sr_list is None:
                    raise ValueError("Array/tensor list input requires sr or sr_list.")
                if sr is not None and sr_list is not None:
                    raise ValueError("Provide either sr or sr_list, not both.")
                if sr_list is not None and len(sr_list) != len(audio_list):
                    raise ValueError("sr_list length must match audio list length.")

                items = []
                for item in audio_list:
                    parsed_items = self._split_array_or_tensor(item, input_layout=input_layout, names=None)
                    if len(parsed_items) != 1:
                        raise ValueError("Array/tensor list items must be single audio samples, not batches.")
                    items.append(parsed_items[0])
                item_srs = [int(sr)] * len(items) if sr is not None else [int(x) for x in sr_list]
                return items, item_srs, self._default_names(len(items))

        raise TypeError(f"Unsupported audio input type: {type(audio)!r}")

    def _split_array_or_tensor(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        *,
        input_layout: InputLayout,
        names: Optional[Union[str, Sequence[Optional[str]]]],
    ) -> List[torch.Tensor]:
        tensor = self._as_float_tensor(audio)
        if tensor.ndim == 1:
            return [tensor.unsqueeze(0)]

        if tensor.ndim == 2:
            if input_layout == "batch":
                return [tensor[idx : idx + 1] for idx in range(tensor.shape[0])]
            if input_layout == "channels":
                return [tensor]
            if input_layout == "time_channels":
                return [tensor.transpose(0, 1).contiguous()]

            name_count = self._names_count(names)
            if tensor.shape[1] <= 8 and tensor.shape[0] > tensor.shape[1]:
                return [tensor.transpose(0, 1).contiguous()]
            if name_count == tensor.shape[0] and tensor.shape[0] > 1:
                return [tensor[idx : idx + 1] for idx in range(tensor.shape[0])]
            if tensor.shape[0] <= 8:
                return [tensor]
            return [tensor[idx : idx + 1] for idx in range(tensor.shape[0])]

        if tensor.ndim == 3:
            if tensor.shape[2] <= 8 and tensor.shape[1] > tensor.shape[2]:
                tensor = tensor.transpose(1, 2).contiguous()
            return [tensor[idx] for idx in range(tensor.shape[0])]

        raise ValueError(f"Audio tensor must be 1D, 2D, or 3D, got shape={tuple(tensor.shape)}.")

    @staticmethod
    def _as_float_tensor(audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(audio, torch.Tensor):
            return audio.detach().float()
        return torch.from_numpy(np.asarray(audio)).float()

    @staticmethod
    def _is_path(audio: Any) -> bool:
        return isinstance(audio, (str, Path))

    @staticmethod
    def _names_count(names: Optional[Union[str, Sequence[Optional[str]]]]) -> Optional[int]:
        if names is None or isinstance(names, str):
            return None
        return len(names)

    @staticmethod
    def _default_names(count: int) -> List[Optional[str]]:
        return [f"item_{idx:06d}" for idx in range(count)]

    def _resolve_names(
        self,
        names: Optional[Union[str, Sequence[Optional[str]]]],
        default_names: Sequence[Optional[str]],
        count: int,
    ) -> List[Optional[str]]:
        if names is None:
            return list(default_names)
        if isinstance(names, str):
            if count != 1:
                raise ValueError("String name can only be used for single-item input.")
            return [names]
        resolved = list(names)
        if len(resolved) != count:
            raise ValueError("names length must match batch size.")
        return resolved

    @staticmethod
    def _resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        if orig_sr == target_sr:
            return audio
        return torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=target_sr)

    @staticmethod
    def _normalize_waveform(audio: torch.Tensor) -> torch.Tensor:
        mean = audio.mean(dim=-1, keepdim=True)
        centered = audio - mean
        peak = centered.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        return centered / peak * 0.5

    @staticmethod
    def _resolve_batch_device(items: Sequence[torch.Tensor]) -> torch.device:
        non_cpu_devices = {item.device for item in items if item.device.type != "cpu"}
        if len(non_cpu_devices) > 1:
            raise ValueError(f"All tensor inputs must be on the same non-CPU device, got {sorted(map(str, non_cpu_devices))}.")
        if non_cpu_devices:
            return next(iter(non_cpu_devices))
        return items[0].device

    @staticmethod
    def _flatten_channels(audio: torch.Tensor, lengths: Sequence[int]) -> tuple[torch.Tensor, List[int]]:
        batch_size, channel_count, total_length = audio.shape
        flat_audio = audio.reshape(batch_size * channel_count, total_length)
        flat_lengths = []
        for length in lengths:
            flat_lengths.extend([int(length)] * channel_count)
        return flat_audio, flat_lengths

    def _run_flat_audio(
        self,
        *,
        flat_audio: torch.Tensor,
        flat_lengths: Sequence[int],
        batch_size: int,
        chunking: bool,
        chunk_duration: int,
        overlap_duration: int,
        guidance_scale: float,
        ddim_steps: int,
    ) -> torch.Tensor:
        chunk_states, all_chunk_infos = self._prepare_chunk_states(
            flat_audio=flat_audio,
            flat_lengths=flat_lengths,
            chunking=chunking,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
        )

        for start_idx in range(0, len(all_chunk_infos), batch_size):
            current_chunk_infos = all_chunk_infos[start_idx : start_idx + batch_size]
            waveform_batch = self._run_waveform_batch_inference(
                chunk_waveforms=[chunk_info.chunk_waveform for chunk_info in current_chunk_infos],
                guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
            )
            if not isinstance(waveform_batch, torch.Tensor):
                waveform_batch = torch.from_numpy(waveform_batch)

            for batch_idx, chunk_info in enumerate(current_chunk_infos):
                self._accumulate_chunk_result(
                    chunk_state=chunk_states[chunk_info.flat_index],
                    chunk_info=chunk_info,
                    processed_chunk=waveform_batch[batch_idx : batch_idx + 1],
                )

        outputs = [self._finalize_chunk_state(state) for state in chunk_states]
        max_length = max(output.shape[-1] for output in outputs)
        outputs = [
            F.pad(output, (0, max_length - output.shape[-1])) if output.shape[-1] < max_length else output
            for output in outputs
        ]
        return torch.stack(outputs, dim=0)

    def _prepare_chunk_states(
        self,
        *,
        flat_audio: torch.Tensor,
        flat_lengths: Sequence[int],
        chunking: bool,
        chunk_duration: int,
        overlap_duration: int,
    ) -> tuple[List[ChunkState], List[ChunkInfo]]:
        if chunk_duration <= 0:
            raise ValueError("chunk_duration must be greater than 0.")
        if overlap_duration < 0:
            raise ValueError("overlap_duration must be greater than or equal to 0.")
        if chunking and chunk_duration <= overlap_duration:
            raise ValueError("chunk_duration must be greater than overlap_duration.")

        chunk_samples = int(chunk_duration * self.sample_rate)
        overlap_samples = int(overlap_duration * self.sample_rate) if chunking else 0
        step_samples = chunk_samples - overlap_samples if chunking else chunk_samples
        if chunk_samples <= 0 or step_samples <= 0:
            raise ValueError("chunk_duration and overlap_duration produce invalid chunk sizes.")

        chunk_states: List[ChunkState] = []
        all_chunk_infos: List[ChunkInfo] = []
        for flat_index, total_samples in enumerate(flat_lengths):
            total_samples = int(total_samples)
            if total_samples <= 100:
                raise ValueError(f"Audio is too short for AudioSR inference: {total_samples} samples.")

            source = flat_audio[flat_index, :total_samples]
            final_waveform = source.new_zeros(total_samples)
            contribution_map = source.new_zeros(total_samples)
            if overlap_samples > 0:
                fade_window = torch.hann_window(2 * overlap_samples, periodic=False, device=source.device)
                fade_in = fade_window[:overlap_samples]
                fade_out = fade_window[overlap_samples:]
            else:
                fade_in = source.new_empty(0)
                fade_out = source.new_empty(0)

            state = ChunkState(
                total_samples=total_samples,
                overlap_samples=overlap_samples,
                final_waveform=final_waveform,
                overlap_contribution_map=contribution_map,
                fade_in=fade_in,
                fade_out=fade_out,
            )
            chunk_infos = self._make_chunk_infos_for_flat_audio(
                flat_index=flat_index,
                source=source,
                chunking=chunking,
                chunk_samples=chunk_samples,
                overlap_samples=overlap_samples,
                step_samples=step_samples,
            )
            self._merge_short_last_chunk(
                chunk_infos=chunk_infos,
                source=source,
                chunk_samples=chunk_samples,
                overlap_samples=overlap_samples,
            )
            chunk_states.append(state)
            all_chunk_infos.extend(chunk_infos)

        return chunk_states, all_chunk_infos

    def _make_chunk_infos_for_flat_audio(
        self,
        *,
        flat_index: int,
        source: torch.Tensor,
        chunking: bool,
        chunk_samples: int,
        overlap_samples: int,
        step_samples: int,
    ) -> List[ChunkInfo]:
        total_samples = int(source.shape[-1])
        if not chunking:
            return [
                ChunkInfo(
                    flat_index=flat_index,
                    start_sample=0,
                    end_sample=total_samples,
                    current_chunk_len=total_samples,
                    chunk_waveform=source.unsqueeze(0),
                )
            ]

        chunk_infos = []
        for start_sample in range(0, total_samples, step_samples):
            end_sample = start_sample + chunk_samples
            chunk_waveform = source[start_sample:end_sample].unsqueeze(0)
            current_chunk_len = int(chunk_waveform.shape[-1])
            if current_chunk_len < chunk_samples:
                chunk_waveform = F.pad(chunk_waveform, (0, chunk_samples - current_chunk_len))
            chunk_infos.append(
                ChunkInfo(
                    flat_index=flat_index,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    current_chunk_len=current_chunk_len,
                    chunk_waveform=chunk_waveform,
                )
            )
        return chunk_infos

    def _merge_short_last_chunk(
        self,
        *,
        chunk_infos: List[ChunkInfo],
        source: torch.Tensor,
        chunk_samples: int,
        overlap_samples: int,
    ) -> None:
        if overlap_samples <= 0 or len(chunk_infos) < 2:
            return
        last_chunk_info = chunk_infos[-1]
        if last_chunk_info.current_chunk_len > overlap_samples * 2:
            return

        prev_chunk_info = chunk_infos[-2]
        merged_start = prev_chunk_info.start_sample
        merged_end = int(source.shape[-1])
        merged_waveform = source[merged_start:merged_end].unsqueeze(0)
        merged_len = int(merged_waveform.shape[-1])
        if merged_len < chunk_samples:
            merged_waveform = F.pad(merged_waveform, (0, chunk_samples - merged_len))
        chunk_infos[-2] = ChunkInfo(
            flat_index=prev_chunk_info.flat_index,
            start_sample=merged_start,
            end_sample=merged_end,
            current_chunk_len=merged_len,
            chunk_waveform=merged_waveform,
        )
        chunk_infos.pop()

    def _run_waveform_batch_inference(
        self,
        *,
        chunk_waveforms: Sequence[torch.Tensor],
        guidance_scale: float,
        ddim_steps: int,
    ) -> torch.Tensor:
        single_batches = []
        for chunk_waveform in chunk_waveforms:
            batch, _ = make_batch_for_super_resolution(
                None,
                waveform=chunk_waveform.detach().cpu().numpy(),
                normalize_waveform=False,
            )
            single_batches.append(batch)

        merged_batch = self._merge_single_batches(single_batches)
        autocast_enabled = self.precision in ("fp16", "float16", "bf16", "bfloat16")
        autocast_dtype = torch.bfloat16 if self.precision in ("bf16", "bfloat16") else torch.float16
        with torch.no_grad(), torch.autocast(device_type=self.model.device.type if hasattr(self.model.device, 'type') else 'cuda', dtype=autocast_dtype, enabled=autocast_enabled):
            waveform = self.model.generate_batch(
                merged_batch,
                unconditional_guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
                normalize_output=False,
                verbose=self.verbose,
            )
        return waveform

    def _merge_single_batches(self, single_batches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        merged_batch: Dict[str, Any] = {"sampling_rate": self.sample_rate}
        tensor_keys = ["waveform", "stft", "log_mel_spec", "waveform_lowpass", "lowpass_mel"]
        for key in tensor_keys:
            tensors = [batch[key] for batch in single_batches]
            merged_batch[key] = self._pad_and_stack_tensors(key, tensors)
        return merged_batch

    @staticmethod
    def _pad_and_stack_tensors(key: str, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        if key in {"waveform", "waveform_lowpass"}:
            max_length = max(tensor.shape[-1] for tensor in tensors)
            padded = [F.pad(tensor, (0, max_length - tensor.shape[-1])) for tensor in tensors]
        else:
            max_frames = max(tensor.shape[1] for tensor in tensors)
            padded = [F.pad(tensor, (0, 0, 0, max_frames - tensor.shape[1])) for tensor in tensors]
        return torch.cat(padded, dim=0)

    def _accumulate_chunk_result(
        self,
        *,
        chunk_state: ChunkState,
        chunk_info: ChunkInfo,
        processed_chunk: torch.Tensor,
    ) -> None:
        processed = processed_chunk[0, 0, : chunk_info.current_chunk_len].to(chunk_state.final_waveform.device)
        start_sample = chunk_info.start_sample
        target_end_sample = min(chunk_info.end_sample, chunk_state.total_samples)
        valid_len = target_end_sample - start_sample
        processed = processed[:valid_len]

        overlap_samples = chunk_state.overlap_samples
        if overlap_samples > 0 and start_sample > 0:
            processed[:overlap_samples] *= chunk_state.fade_in[: min(overlap_samples, processed.shape[-1])]
        if overlap_samples > 0 and chunk_info.end_sample < chunk_state.total_samples:
            processed[-overlap_samples:] *= chunk_state.fade_out[-min(overlap_samples, processed.shape[-1]) :]

        chunk_state.final_waveform[start_sample:target_end_sample] += processed

        contribution = chunk_state.final_waveform.new_ones(valid_len)
        if overlap_samples > 0 and start_sample > 0:
            contribution[:overlap_samples] = chunk_state.fade_in[: min(overlap_samples, valid_len)]
        if overlap_samples > 0 and chunk_info.end_sample < chunk_state.total_samples:
            contribution[-overlap_samples:] = chunk_state.fade_out[-min(overlap_samples, valid_len) :]
        chunk_state.overlap_contribution_map[start_sample:target_end_sample] += contribution

    @staticmethod
    def _finalize_chunk_state(chunk_state: ChunkState) -> torch.Tensor:
        contribution = chunk_state.overlap_contribution_map.clone()
        contribution[contribution == 0] = 1.0
        waveform = chunk_state.final_waveform / contribution
        waveform = waveform - waveform.mean(dim=-1, keepdim=True)
        peak = waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        waveform = waveform / peak * 0.5
        return torch.clamp(waveform, -1.0, 1.0)

    @staticmethod
    def _unflatten_channels(
        *,
        flat_output: torch.Tensor,
        batch_size: int,
        channel_count: int,
        lengths: Sequence[int],
    ) -> torch.Tensor:
        max_length = max(lengths)
        output = flat_output.new_zeros(batch_size, channel_count, max_length)
        flat_output = flat_output.reshape(batch_size, channel_count, flat_output.shape[-1])
        for sample_idx, length in enumerate(lengths):
            output[sample_idx, :, :length] = flat_output[sample_idx, :, :length]
        return output

    def _save_outputs(
        self,
        audio: torch.Tensor,
        lengths: Sequence[int],
        names: Sequence[Optional[str]],
        output_path: Optional[Union[str, Path]],
    ) -> List[str]:
        if output_path is None:
            return []

        output_path = Path(output_path)
        if output_path.suffix and audio.shape[0] != 1:
            raise ValueError("File output_path is only supported for single-item output.")

        saved_files = []
        used_names: Dict[str, int] = {}
        for idx in range(audio.shape[0]):
            if output_path.suffix:
                target_path = output_path
            else:
                output_path.mkdir(parents=True, exist_ok=True)
                base_name = self._safe_output_name(names[idx] or f"item_{idx:06d}")
                count = used_names.get(base_name, 0)
                used_names[base_name] = count + 1
                if count:
                    base_name = f"{base_name}_{count}"
                target_path = output_path / f"{base_name}.wav"

            target_path.parent.mkdir(parents=True, exist_ok=True)
            length = int(lengths[idx])
            data = audio[idx, :, :length].detach().cpu().numpy().T
            sf.write(target_path, data, self.sample_rate)
            saved_files.append(str(target_path))
        return saved_files

    @staticmethod
    def _safe_output_name(name: str) -> str:
        name = Path(str(name)).stem
        return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name) or "audio"


if __name__ == '__main__':
    from utils.commons.os_utils import load_env_local
    load_env_local()

    audiosr = AudioSR(
        model_name="basic",
        device='cuda',
        # model_root="/path/to/pretrained_models/sr/audiosr",
        allow_download=True,
        precision='bf16'
    )

    audiosr.infer(
        audio='user/prompts/audio/251124_24k/0_video_2c6be891d49e20c746a26b109d07539446d797ac_s43__2896_3028.wav',
        output_path='infer_out/sr/audiosr/0_video_2c6be891d49e20c746a26b109d07539446d797ac_s43__2896_3028.wav'
    )

