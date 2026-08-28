from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from torch.nn.utils.rnn import pad_sequence

from .audio import PathLike, load_audio_path, prepare_waveform
from .checkpoint import (
    default_content_checkpoint,
    default_content_style_checkpoint,
    download_tokenizer_snapshot,
    load_dispatch_checkpoint,
    load_state_dict_checkpoint,
    resolve_content_checkpoint_path,
    resolve_content_style_checkpoint_path,
)
from .models import RepCodec, VevoRepCodec


AudioInput = Union[PathLike, np.ndarray, torch.Tensor]
AudioBatchInput = Union[
    AudioInput,
    Sequence[PathLike],
    Sequence[Union[np.ndarray, torch.Tensor]],
]
VectorType = str
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
CONTENT_CONFIG_PATH = ASSETS_DIR / "hubert_large_l18_c32.yaml"
HUBERT_STATS_PATH = ASSETS_DIR / "hubert_large_l18_mean_std.npz"
CPU_DEVICE = torch.device("cpu")


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _duration_reduce(tokens: torch.Tensor, n_gram: int = 1) -> torch.Tensor:
    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1D tensor")
    if tokens.numel() == 0:
        return tokens
    if n_gram <= 0:
        raise ValueError("n_gram must be positive")
    if tokens.numel() < n_gram:
        return tokens

    n_gram_seq = tokens.unfold(0, n_gram, 1)
    mask = torch.all(n_gram_seq[1:] != n_gram_seq[:-1], dim=1)
    return torch.cat((n_gram_seq[0, :n_gram], n_gram_seq[1:, -1][mask]))


def _validate_vector_type(vector_type: VectorType) -> VectorType:
    valid_vector_types = {"content", "content_style", "both"}
    if vector_type not in valid_vector_types:
        raise ValueError(
            f"vector_type must be one of {sorted(valid_vector_types)}, got {vector_type!r}"
        )
    return vector_type


def _is_path_like(value: object) -> bool:
    return isinstance(value, (str, Path))


def _is_waveform(value: object) -> bool:
    return isinstance(value, (np.ndarray, torch.Tensor))


def _normalize_integer_values(
    values: Union[int, Sequence[int], np.ndarray, torch.Tensor],
    expected_length: int,
    name: str,
    allow_broadcast_scalar: bool = False,
) -> list[int]:
    if isinstance(values, (int, np.integer)):
        if not allow_broadcast_scalar:
            raise ValueError(f"{name} must provide one value per audio item")
        normalized = [int(values)] * expected_length
    elif isinstance(values, torch.Tensor):
        normalized = [int(item) for item in values.reshape(-1).tolist()]
    elif isinstance(values, np.ndarray):
        normalized = [int(item) for item in values.reshape(-1).tolist()]
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, Path)):
        normalized = [int(item) for item in values]
    else:
        raise TypeError(f"{name} must be an int, tensor, ndarray, or sequence of ints")

    if len(normalized) != expected_length:
        raise ValueError(f"{name} must have length {expected_length}, got {len(normalized)}")
    return normalized


def _concat_token_chunks(chunks: Sequence[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty(0, dtype=np.int64)
    if len(chunks) == 1:
        return chunks[0].astype(np.int64, copy=False)
    return np.concatenate(chunks).astype(np.int64, copy=False)


@dataclass
class VevoTokenResult:
    content_ids: Optional[np.ndarray]
    content_style_ids: Optional[np.ndarray]
    source_sample_rate: int
    vector_type: VectorType = "both"
    content_reduced: bool = True
    content_style_reduced: bool = False
    token_sample_rate: int = 16000

    def to_npz(self, path: PathLike) -> Path:
        output_path = Path(path)
        payload = dict(
            source_sample_rate=np.int64(self.source_sample_rate),
            token_sample_rate=np.int64(self.token_sample_rate),
            vector_type=np.array(self.vector_type),
            content_reduced=np.bool_(self.content_reduced),
            content_style_reduced=np.bool_(self.content_style_reduced),
        )
        if self.content_ids is not None:
            payload["content_ids"] = self.content_ids
        if self.content_style_ids is not None:
            payload["content_style_ids"] = self.content_style_ids
        np.savez(output_path, **payload)
        return output_path


@dataclass(frozen=True)
class _AudioRecord:
    audio_index: int
    source_sample_rate: int
    waveform_16k: torch.Tensor


@dataclass(frozen=True)
class _ChunkRecord:
    audio_index: int
    chunk_index: int
    core_len_16k: int
    infer_len_16k: int
    target_token_len: int
    waveform_16k: torch.Tensor


class VevoTokenExtractor:
    load_sample_rate = 24000
    token_sample_rate = 16000
    hubert_output_layer = 18
    chunk_duration_seconds = 8.0
    chunk_size_16k = 128000
    token_hop_16k = 320
    hubert_alignment_16k = 80

    def __init__(
        self,
        content_tokenizer: VevoRepCodec,
        content_style_tokenizer: RepCodec,
        hubert_model: torch.nn.Module,
        hubert_feat_norm_mean: torch.Tensor,
        hubert_feat_norm_std: torch.Tensor,
        device: torch.device,
    ):
        self.content_tokenizer = content_tokenizer
        self.content_style_tokenizer = content_style_tokenizer
        self.hubert_model = hubert_model
        self.hubert_feat_norm_mean = hubert_feat_norm_mean.to(device)
        self.hubert_feat_norm_std = hubert_feat_norm_std.to(device)
        self.device = device

        self.content_style_codebook_size = self.content_style_tokenizer.codebook_size
        self.content_codebook_size = self.content_tokenizer.quantizer.codebook.codebook_size

    @classmethod
    def from_pretrained(
        cls,
        cache_dir: Optional[PathLike] = None,
        device: Optional[Union[str, torch.device]] = None,
        download: bool = True,
        repo_id: str = "amphion/Vevo",
        content_ckpt_path: Optional[PathLike] = None,
        content_style_ckpt_path: Optional[PathLike] = None,
    ) -> "VevoTokenExtractor":
        device = _resolve_device(device)

        if content_ckpt_path is None or content_style_ckpt_path is None:
            if download:
                snapshot_root = download_tokenizer_snapshot(cache_dir=cache_dir, repo_id=repo_id)
            elif cache_dir is not None:
                snapshot_root = Path(cache_dir).expanduser()
            else:
                raise ValueError(
                    "cache_dir or explicit checkpoint paths are required when download=False"
                )
        else:
            snapshot_root = None

        if content_ckpt_path is None:
            content_ckpt_path = default_content_checkpoint(snapshot_root)
            if not Path(content_ckpt_path).exists():
                content_ckpt_path = Path(snapshot_root) / "tokenizer" / "vq32"
        if content_style_ckpt_path is None:
            content_style_ckpt_path = default_content_style_checkpoint(snapshot_root)

        content_ckpt_path = resolve_content_checkpoint_path(content_ckpt_path)
        content_style_ckpt_path = resolve_content_style_checkpoint_path(content_style_ckpt_path)

        hubert_model = cls._build_hubert_model(device)
        hubert_feat_norm_mean, hubert_feat_norm_std = cls._load_hubert_stats(device)
        content_tokenizer = cls._load_content_tokenizer(content_ckpt_path, device)
        content_style_tokenizer = cls._load_content_style_tokenizer(
            content_style_ckpt_path,
            device,
        )
        return cls(
            content_tokenizer=content_tokenizer,
            content_style_tokenizer=content_style_tokenizer,
            hubert_model=hubert_model,
            hubert_feat_norm_mean=hubert_feat_norm_mean,
            hubert_feat_norm_std=hubert_feat_norm_std,
            device=device,
        )

    @staticmethod
    def _build_hubert_model(device: torch.device) -> torch.nn.Module:
        hubert = torchaudio.pipelines.HUBERT_LARGE.get_model()
        hubert.eval()
        hubert.to(device)
        return hubert

    @staticmethod
    def _load_hubert_stats(device: torch.device):
        stat = np.load(HUBERT_STATS_PATH)
        mean = torch.tensor(stat["mean"], dtype=torch.float32, device=device)
        std = torch.tensor(stat["std"], dtype=torch.float32, device=device)
        return mean, std

    @staticmethod
    def _load_content_tokenizer(
        checkpoint_path: PathLike,
        device: torch.device,
    ) -> VevoRepCodec:
        with CONTENT_CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        model = VevoRepCodec(**config)
        model.eval()
        model.to(device)
        load_state_dict_checkpoint(
            model,
            checkpoint_path,
            candidate_paths=(("model", "repcodec"), ("state_dict",), ()),
            strict=True,
        )
        model.quantizer.initial()
        return model

    @staticmethod
    def _load_content_style_tokenizer(
        checkpoint_path: PathLike,
        device: torch.device,
    ) -> RepCodec:
        model = RepCodec(
            codebook_size=8192,
            hidden_size=1024,
            codebook_dim=8,
            vocos_dim=384,
            vocos_intermediate_dim=2048,
            vocos_num_layers=12,
        )
        model.eval()
        model.to(device)

        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            model = load_dispatch_checkpoint(model, checkpoint_path)
        else:
            load_state_dict_checkpoint(
                model,
                checkpoint_path,
                candidate_paths=(("model",), ("state_dict",), ()),
                strict=True,
            )
        model.eval()
        return model

    def _prepare_audio_record_from_path(
        self,
        audio_path: PathLike,
        audio_index: int,
    ) -> _AudioRecord:
        prepared = load_audio_path(
            audio_path,
            device=CPU_DEVICE,
            load_sample_rate=self.load_sample_rate,
            token_sample_rate=self.token_sample_rate,
        )
        waveform_16k = prepared.waveform_16k.squeeze(0).detach().cpu()
        if waveform_16k.numel() == 0:
            raise ValueError(f"audio at {audio_path!s} is empty after resampling")
        return _AudioRecord(
            audio_index=audio_index,
            source_sample_rate=prepared.source_sample_rate,
            waveform_16k=waveform_16k,
        )

    def _prepare_audio_record_from_waveform(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        audio_index: int,
    ) -> _AudioRecord:
        prepared = prepare_waveform(
            waveform,
            sample_rate=sample_rate,
            device=CPU_DEVICE,
            load_sample_rate=self.load_sample_rate,
            token_sample_rate=self.token_sample_rate,
        )
        waveform_16k = prepared.waveform_16k.squeeze(0).detach().cpu()
        if waveform_16k.numel() == 0:
            raise ValueError("audio is empty after resampling")
        return _AudioRecord(
            audio_index=audio_index,
            source_sample_rate=prepared.source_sample_rate,
            waveform_16k=waveform_16k,
        )

    def _normalize_audio_inputs(
        self,
        audio: AudioBatchInput,
        sample_rate: Optional[int] = None,
        sample_rates: Optional[Union[int, Sequence[int], np.ndarray, torch.Tensor]] = None,
        audio_lengths: Optional[Union[Sequence[int], np.ndarray, torch.Tensor]] = None,
    ) -> tuple[list[_AudioRecord], bool]:
        if _is_path_like(audio):
            return [self._prepare_audio_record_from_path(audio, audio_index=0)], False

        if _is_waveform(audio):
            if isinstance(audio, (np.ndarray, torch.Tensor)) and audio.ndim == 2:
                if audio_lengths is not None:
                    if sample_rate is None:
                        raise ValueError("sample_rate is required for batched [B, T] audio")
                    batch_size = int(audio.shape[0])
                    lengths = _normalize_integer_values(
                        audio_lengths,
                        expected_length=batch_size,
                        name="audio_lengths",
                    )
                    max_length = int(audio.shape[1])
                    records = []
                    for audio_index, length in enumerate(lengths):
                        if length <= 0 or length > max_length:
                            raise ValueError(
                                f"audio_lengths[{audio_index}] must be in [1, {max_length}], got {length}"
                            )
                        records.append(
                            self._prepare_audio_record_from_waveform(
                                audio[audio_index, :length],
                                sample_rate=sample_rate,
                                audio_index=audio_index,
                            )
                        )
                    return records, True

            if sample_rate is None:
                raise ValueError("sample_rate is required when audio is a waveform")
            return [
                self._prepare_audio_record_from_waveform(
                    audio,
                    sample_rate=sample_rate,
                    audio_index=0,
                )
            ], False

        if isinstance(audio, Sequence) and not isinstance(audio, (str, bytes, Path)):
            audio_items = list(audio)
            if not audio_items:
                raise ValueError("audio list must not be empty")

            all_paths = all(_is_path_like(item) for item in audio_items)
            all_waveforms = all(_is_waveform(item) for item in audio_items)
            if not (all_paths or all_waveforms):
                raise ValueError("audio list must be homogenous: all paths or all waveforms")

            if all_paths:
                return [
                    self._prepare_audio_record_from_path(item, audio_index=index)
                    for index, item in enumerate(audio_items)
                ], True

            if sample_rates is None and sample_rate is not None:
                sample_rates = sample_rate
            if sample_rates is None:
                raise ValueError("sample_rates is required for waveform lists")

            normalized_sample_rates = _normalize_integer_values(
                sample_rates,
                expected_length=len(audio_items),
                name="sample_rates",
                allow_broadcast_scalar=True,
            )
            return [
                self._prepare_audio_record_from_waveform(
                    item,
                    sample_rate=normalized_sample_rates[index],
                    audio_index=index,
                )
                for index, item in enumerate(audio_items)
            ], True

        raise TypeError("unsupported audio input type")

    def _target_token_len_from_core_len(self, core_len_16k: int) -> int:
        return core_len_16k // self.token_hop_16k

    def _required_infer_len_from_target_tokens(self, target_token_len: int) -> int:
        return target_token_len * self.token_hop_16k + self.hubert_alignment_16k

    def _build_chunk_records_for_audio(self, audio_record: _AudioRecord) -> list[_ChunkRecord]:
        waveform = audio_record.waveform_16k
        total_len = int(waveform.shape[0])
        chunk_records = []
        chunk_index = 0

        for chunk_start in range(0, total_len, self.chunk_size_16k):
            chunk_end = min(total_len, chunk_start + self.chunk_size_16k)
            core_waveform = waveform[chunk_start:chunk_end]
            core_len_16k = int(core_waveform.shape[0])
            is_last_chunk = chunk_end >= total_len
            target_token_len = self._target_token_len_from_core_len(core_len_16k)

            if not is_last_chunk and core_len_16k == self.chunk_size_16k:
                lookahead_waveform = waveform[
                    chunk_end : min(total_len, chunk_end + self.hubert_alignment_16k)
                ]
                if int(lookahead_waveform.shape[0]) < self.hubert_alignment_16k:
                    lookahead_waveform = F.pad(
                        lookahead_waveform,
                        (0, self.hubert_alignment_16k - int(lookahead_waveform.shape[0])),
                    )
                infer_waveform = torch.cat([core_waveform, lookahead_waveform], dim=0)
            else:
                required_infer_len = self._required_infer_len_from_target_tokens(target_token_len)
                pad_right = max(0, required_infer_len - core_len_16k)
                infer_waveform = F.pad(core_waveform, (0, pad_right))

            chunk_records.append(
                _ChunkRecord(
                    audio_index=audio_record.audio_index,
                    chunk_index=chunk_index,
                    core_len_16k=core_len_16k,
                    infer_len_16k=int(infer_waveform.shape[0]),
                    target_token_len=target_token_len,
                    waveform_16k=infer_waveform,
                )
            )
            chunk_index += 1

        return chunk_records

    def _extract_hubert_features(
        self,
        wavs: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
    ):
        if wav_lens is None:
            wav_lens = torch.tensor([wavs.shape[1]] * wavs.shape[0], device=wavs.device).int()
        feats, feat_lengths = self.hubert_model.extract_features(
            wavs,
            lengths=wav_lens,
            num_layers=self.hubert_output_layer,
        )
        return feats[-1], feat_lengths

    def _extract_content_ids_from_hubert_batch(
        self,
        feats: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> list[np.ndarray]:
        x = self.content_tokenizer.encoder(feats.transpose(1, 2))
        z = self.content_tokenizer.projector(x)
        _quantized, indices = self.content_tokenizer.quantizer.codebook.forward_index(
            z.transpose(2, 1)
        )
        batch_token_ids = indices[0]
        outputs = []
        for batch_index, token_len in enumerate(token_lengths.tolist()):
            effective_len = min(int(token_len), int(batch_token_ids.shape[1]))
            outputs.append(
                batch_token_ids[batch_index, :effective_len]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
        return outputs

    def _extract_content_style_ids_from_hubert_batch(
        self,
        feats: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> list[np.ndarray]:
        normalized = (feats - self.hubert_feat_norm_mean.to(feats)) / self.hubert_feat_norm_std.to(
            feats
        )
        token_ids, _ = self.content_style_tokenizer.quantize(normalized)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim == 3 and token_ids.shape[0] == 1:
            token_ids = token_ids.squeeze(0)
        if token_ids.ndim != 2:
            raise ValueError(f"unexpected content-style token shape: {tuple(token_ids.shape)}")

        outputs = []
        for batch_index, token_len in enumerate(token_lengths.tolist()):
            effective_len = min(int(token_len), int(token_ids.shape[1]))
            outputs.append(
                token_ids[batch_index, :effective_len]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
        return outputs

    def _extract_content_ids_from_hubert(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        reduce_content: bool = True,
        reduce_n_gram: int = 1,
    ) -> np.ndarray:
        token_ids = self._extract_content_ids_from_hubert_batch(feats, feat_lengths)[0]
        if not reduce_content:
            return token_ids
        reduced = _duration_reduce(torch.from_numpy(token_ids), n_gram=reduce_n_gram)
        return reduced.cpu().numpy().astype(np.int64, copy=False)

    def _extract_content_style_ids_from_hubert(
        self,
        feats: torch.Tensor,
        feat_lengths: torch.Tensor,
        reduce_content_style: bool = False,
        reduce_n_gram: int = 1,
    ) -> np.ndarray:
        token_ids = self._extract_content_style_ids_from_hubert_batch(feats, feat_lengths)[0]
        if not reduce_content_style:
            return token_ids
        reduced = _duration_reduce(torch.from_numpy(token_ids), n_gram=reduce_n_gram)
        return reduced.cpu().numpy().astype(np.int64, copy=False)

    def _collate_chunk_batch(
        self,
        chunk_records: Sequence[_ChunkRecord],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wav_lens = torch.tensor(
            [chunk_record.infer_len_16k for chunk_record in chunk_records],
            dtype=torch.int,
            device=self.device,
        )
        wavs = pad_sequence(
            [chunk_record.waveform_16k for chunk_record in chunk_records],
            batch_first=True,
            padding_value=0.0,
        ).to(self.device)
        return wavs, wav_lens

    def _extract_chunk_tokens(
        self,
        chunk_records: Sequence[_ChunkRecord],
        num_audios: int,
        vector_type: VectorType,
        batch_size: int,
    ) -> tuple[Optional[list[list[np.ndarray]]], Optional[list[list[np.ndarray]]]]:
        content_chunks = [[] for _ in range(num_audios)] if vector_type in {"content", "both"} else None
        content_style_chunks = (
            [[] for _ in range(num_audios)] if vector_type in {"content_style", "both"} else None
        )

        for batch_start in range(0, len(chunk_records), batch_size):
            batch_records = list(chunk_records[batch_start : batch_start + batch_size])
            wavs, wav_lens = self._collate_chunk_batch(batch_records)
            token_lengths = torch.tensor(
                [chunk_record.target_token_len for chunk_record in batch_records],
                dtype=torch.long,
                device=self.device,
            )

            with torch.no_grad():
                feats, feat_lengths = self._extract_hubert_features(wavs, wav_lens)
                effective_token_lengths = torch.minimum(token_lengths, feat_lengths.to(torch.long))

                batch_content = None
                batch_content_style = None
                if content_chunks is not None:
                    batch_content = self._extract_content_ids_from_hubert_batch(
                        feats,
                        effective_token_lengths,
                    )
                if content_style_chunks is not None:
                    batch_content_style = self._extract_content_style_ids_from_hubert_batch(
                        feats,
                        effective_token_lengths,
                    )

            for batch_index, chunk_record in enumerate(batch_records):
                if content_chunks is not None and batch_content is not None:
                    content_chunks[chunk_record.audio_index].append(batch_content[batch_index])
                if content_style_chunks is not None and batch_content_style is not None:
                    content_style_chunks[chunk_record.audio_index].append(
                        batch_content_style[batch_index]
                    )

        return content_chunks, content_style_chunks

    def _assemble_results(
        self,
        audio_records: Sequence[_AudioRecord],
        content_chunks: Optional[list[list[np.ndarray]]],
        content_style_chunks: Optional[list[list[np.ndarray]]],
        vector_type: VectorType,
        reduce_content: bool,
        reduce_n_gram: int,
        reduce_content_style: bool,
        reduce_content_style_n_gram: int,
    ) -> list[VevoTokenResult]:
        results = []
        for audio_record in audio_records:
            content_ids = None
            content_style_ids = None

            if content_chunks is not None:
                content_ids = _concat_token_chunks(content_chunks[audio_record.audio_index])
                if reduce_content:
                    reduced = _duration_reduce(
                        torch.from_numpy(content_ids),
                        n_gram=reduce_n_gram,
                    )
                    content_ids = reduced.cpu().numpy().astype(np.int64, copy=False)

            if content_style_chunks is not None:
                content_style_ids = _concat_token_chunks(
                    content_style_chunks[audio_record.audio_index]
                )
                if reduce_content_style:
                    reduced = _duration_reduce(
                        torch.from_numpy(content_style_ids),
                        n_gram=reduce_content_style_n_gram,
                    )
                    content_style_ids = reduced.cpu().numpy().astype(np.int64, copy=False)

            results.append(
                VevoTokenResult(
                    content_ids=content_ids,
                    content_style_ids=content_style_ids,
                    source_sample_rate=audio_record.source_sample_rate,
                    vector_type=vector_type,
                    content_reduced=reduce_content,
                    content_style_reduced=reduce_content_style,
                    token_sample_rate=self.token_sample_rate,
                )
            )

        return results

    def _extract_from_audio_records(
        self,
        audio_records: Sequence[_AudioRecord],
        vector_type: VectorType = "both",
        batch_size: int = 1,
        reduce_content: bool = True,
        reduce_n_gram: int = 1,
        reduce_content_style: bool = False,
        reduce_content_style_n_gram: int = 1,
    ) -> list[VevoTokenResult]:
        vector_type = _validate_vector_type(vector_type)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not audio_records:
            raise ValueError("audio input must not be empty")

        chunk_records = []
        for audio_record in audio_records:
            chunk_records.extend(self._build_chunk_records_for_audio(audio_record))

        content_chunks, content_style_chunks = self._extract_chunk_tokens(
            chunk_records=chunk_records,
            num_audios=len(audio_records),
            vector_type=vector_type,
            batch_size=batch_size,
        )
        return self._assemble_results(
            audio_records=audio_records,
            content_chunks=content_chunks,
            content_style_chunks=content_style_chunks,
            vector_type=vector_type,
            reduce_content=reduce_content,
            reduce_n_gram=reduce_n_gram,
            reduce_content_style=reduce_content_style,
            reduce_content_style_n_gram=reduce_content_style_n_gram,
        )

    def _extract_from_prepared_audio(
        self,
        waveform_16k: torch.Tensor,
        source_sample_rate: int,
        vector_type: VectorType = "both",
        batch_size: int = 1,
        reduce_content: bool = True,
        reduce_n_gram: int = 1,
        reduce_content_style: bool = False,
        reduce_content_style_n_gram: int = 1,
    ) -> VevoTokenResult:
        waveform_16k = waveform_16k.squeeze(0).detach().cpu()
        audio_record = _AudioRecord(
            audio_index=0,
            source_sample_rate=source_sample_rate,
            waveform_16k=waveform_16k,
        )
        return self._extract_from_audio_records(
            [audio_record],
            vector_type=vector_type,
            batch_size=batch_size,
            reduce_content=reduce_content,
            reduce_n_gram=reduce_n_gram,
            reduce_content_style=reduce_content_style,
            reduce_content_style_n_gram=reduce_content_style_n_gram,
        )[0]

    def extract_batch(
        self,
        audio: AudioBatchInput,
        sample_rate: Optional[int] = None,
        sample_rates: Optional[Union[int, Sequence[int], np.ndarray, torch.Tensor]] = None,
        audio_lengths: Optional[Union[Sequence[int], np.ndarray, torch.Tensor]] = None,
        batch_size: int = 1,
        vector_type: VectorType = "both",
        reduce_content: bool = True,
        reduce_n_gram: int = 1,
        reduce_content_style: bool = False,
        reduce_content_style_n_gram: int = 1,
    ) -> list[VevoTokenResult]:
        audio_records, _is_batch_input = self._normalize_audio_inputs(
            audio,
            sample_rate=sample_rate,
            sample_rates=sample_rates,
            audio_lengths=audio_lengths,
        )
        return self._extract_from_audio_records(
            audio_records,
            vector_type=vector_type,
            batch_size=batch_size,
            reduce_content=reduce_content,
            reduce_n_gram=reduce_n_gram,
            reduce_content_style=reduce_content_style,
            reduce_content_style_n_gram=reduce_content_style_n_gram,
        )

    def extract_from_16k_batch(
        self,
        wavs: torch.Tensor,
        wav_lengths: Optional[Union[Sequence[int], np.ndarray, torch.Tensor]] = None,
        vector_type: VectorType = "both",
        reduce_content: bool = True,
        reduce_n_gram: int = 1,
        reduce_content_style: bool = False,
        reduce_content_style_n_gram: int = 1,
    ) -> list[VevoTokenResult]:
        vector_type = _validate_vector_type(vector_type)
        if not isinstance(wavs, torch.Tensor):
            raise TypeError("wavs must be a torch.Tensor")
        if wavs.ndim != 2:
            raise ValueError(f"wavs must have shape [B, T], got {tuple(wavs.shape)}")
        if wavs.shape[0] == 0:
            raise ValueError("wavs batch must not be empty")

        wavs = wavs.to(self.device, dtype=torch.float32)
        if wav_lengths is None:
            wav_lengths = torch.full(
                (wavs.shape[0],),
                wavs.shape[1],
                dtype=torch.int,
                device=self.device,
            )
        else:
            wav_lengths = torch.as_tensor(wav_lengths, device=self.device, dtype=torch.int)
            wav_lengths = wav_lengths.reshape(-1)
            if wav_lengths.numel() != wavs.shape[0]:
                raise ValueError(
                    f"wav_lengths must have length {wavs.shape[0]}, got {wav_lengths.numel()}"
                )
            if torch.any(wav_lengths <= 0) or torch.any(wav_lengths > wavs.shape[1]):
                raise ValueError(f"wav_lengths values must be in [1, {wavs.shape[1]}]")

        with torch.no_grad():
            feats, feat_lengths = self._extract_hubert_features(wavs, wav_lengths)
            token_lengths = feat_lengths.to(torch.long)
            batch_content = (
                self._extract_content_ids_from_hubert_batch(feats, token_lengths)
                if vector_type in {"content", "both"}
                else None
            )
            batch_content_style = (
                self._extract_content_style_ids_from_hubert_batch(feats, token_lengths)
                if vector_type in {"content_style", "both"}
                else None
            )

        results = []
        for index in range(wavs.shape[0]):
            content_ids = batch_content[index] if batch_content is not None else None
            content_style_ids = (
                batch_content_style[index] if batch_content_style is not None else None
            )

            if content_ids is not None and reduce_content:
                content_ids = (
                    _duration_reduce(torch.from_numpy(content_ids), n_gram=reduce_n_gram)
                    .cpu()
                    .numpy()
                    .astype(np.int64, copy=False)
                )
            if content_style_ids is not None and reduce_content_style:
                content_style_ids = (
                    _duration_reduce(
                        torch.from_numpy(content_style_ids),
                        n_gram=reduce_content_style_n_gram,
                    )
                    .cpu()
                    .numpy()
                    .astype(np.int64, copy=False)
                )

            results.append(
                VevoTokenResult(
                    content_ids=content_ids,
                    content_style_ids=content_style_ids,
                    source_sample_rate=self.token_sample_rate,
                    vector_type=vector_type,
                    content_reduced=reduce_content,
                    content_style_reduced=reduce_content_style,
                    token_sample_rate=self.token_sample_rate,
                )
            )
        return results

    def extract_from_16k_batch_chunked(
        self,
        wavs: torch.Tensor,
        wav_lengths: Optional[Union[Sequence[int], np.ndarray, torch.Tensor]] = None,
        batch_size: int = 1,
        vector_type: VectorType = "both",
        reduce_content: bool = True,
        reduce_n_gram: int = 1,
        reduce_content_style: bool = False,
        reduce_content_style_n_gram: int = 1,
    ) -> list[VevoTokenResult]:
        if not isinstance(wavs, torch.Tensor):
            raise TypeError("wavs must be a torch.Tensor")
        if wavs.ndim != 2:
            raise ValueError(f"wavs must have shape [B, T], got {tuple(wavs.shape)}")
        if wavs.shape[0] == 0:
            raise ValueError("wavs batch must not be empty")

        wavs = wavs.to(self.device, dtype=torch.float32)
        if wav_lengths is None:
            normalized_wav_lengths = [int(wavs.shape[1])] * int(wavs.shape[0])
        else:
            normalized_wav_lengths = _normalize_integer_values(
                wav_lengths,
                expected_length=int(wavs.shape[0]),
                name="wav_lengths",
            )

        max_length = int(wavs.shape[1])
        audio_records = []
        for audio_index, length in enumerate(normalized_wav_lengths):
            if length <= 0 or length > max_length:
                raise ValueError(
                    f"wav_lengths[{audio_index}] must be in [1, {max_length}], got {length}"
                )
            audio_records.append(
                _AudioRecord(
                    audio_index=audio_index,
                    source_sample_rate=self.token_sample_rate,
                    waveform_16k=wavs[audio_index, :length].detach(),
                )
            )

        return self._extract_from_audio_records(
            audio_records,
            vector_type=vector_type,
            batch_size=batch_size,
            reduce_content=reduce_content,
            reduce_n_gram=reduce_n_gram,
            reduce_content_style=reduce_content_style,
            reduce_content_style_n_gram=reduce_content_style_n_gram,
        )

    def extract_from_path(
        self,
        audio_path: PathLike,
        vector_type: VectorType = "both",
        batch_size: int = 1,
        reduce_content: bool = True,
        reduce_n_gram: int = 1,
        reduce_content_style: bool = False,
        reduce_content_style_n_gram: int = 1,
    ) -> VevoTokenResult:
        return self.extract_batch(
            [audio_path],
            batch_size=batch_size,
            vector_type=vector_type,
            reduce_content=reduce_content,
            reduce_n_gram=reduce_n_gram,
            reduce_content_style=reduce_content_style,
            reduce_content_style_n_gram=reduce_content_style_n_gram,
        )[0]

    def extract_from_waveform(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        vector_type: VectorType = "both",
        batch_size: int = 1,
        reduce_content: bool = True,
        reduce_n_gram: int = 1,
        reduce_content_style: bool = False,
        reduce_content_style_n_gram: int = 1,
    ) -> VevoTokenResult:
        return self.extract_batch(
            [waveform],
            sample_rates=[sample_rate],
            batch_size=batch_size,
            vector_type=vector_type,
            reduce_content=reduce_content,
            reduce_n_gram=reduce_n_gram,
            reduce_content_style=reduce_content_style,
            reduce_content_style_n_gram=reduce_content_style_n_gram,
        )[0]

    def extract_content_ids(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        batch_size: int = 1,
        reduce_content: bool = True,
        reduce_n_gram: int = 1,
    ) -> np.ndarray:
        if isinstance(audio, (str, Path)):
            return self.extract_from_path(
                audio,
                vector_type="content",
                batch_size=batch_size,
                reduce_content=reduce_content,
                reduce_n_gram=reduce_n_gram,
            ).content_ids
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio is a waveform")
        return self.extract_from_waveform(
            audio,
            sample_rate=sample_rate,
            vector_type="content",
            batch_size=batch_size,
            reduce_content=reduce_content,
            reduce_n_gram=reduce_n_gram,
        ).content_ids

    def extract_content_style_ids(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        batch_size: int = 1,
        reduce_content_style: bool = False,
        reduce_n_gram: int = 1,
    ) -> np.ndarray:
        if isinstance(audio, (str, Path)):
            return self.extract_from_path(
                audio,
                vector_type="content_style",
                batch_size=batch_size,
                reduce_content_style=reduce_content_style,
                reduce_content_style_n_gram=reduce_n_gram,
            ).content_style_ids
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio is a waveform")
        return self.extract_from_waveform(
            audio,
            sample_rate=sample_rate,
            vector_type="content_style",
            batch_size=batch_size,
            reduce_content_style=reduce_content_style,
            reduce_content_style_n_gram=reduce_n_gram,
        ).content_style_ids


def build_vevo_token_model(
    cache_dir: Optional[PathLike] = 'pretrained_models/Vevo',
    device: Optional[Union[str, torch.device]] = None,
    download: bool = True,
    repo_id: str = "amphion/Vevo",
    content_ckpt_path: Optional[PathLike] = None,
    content_style_ckpt_path: Optional[PathLike] = None,
) -> VevoTokenExtractor:
    return VevoTokenExtractor.from_pretrained(
        cache_dir=cache_dir,
        device=device,
        download=download,
        repo_id=repo_id,
        content_ckpt_path=content_ckpt_path,
        content_style_ckpt_path=content_style_ckpt_path,
    )


def run_vevo_token_model(
    model: VevoTokenExtractor,
    audio: AudioBatchInput,
    sample_rate: Optional[int] = None,
    sample_rates: Optional[Union[int, Sequence[int], np.ndarray, torch.Tensor]] = None,
    audio_lengths: Optional[Union[Sequence[int], np.ndarray, torch.Tensor]] = None,
    batch_size: int = 1,
    vector_type: VectorType = "both",
    reduce_content: bool = True,
    reduce_n_gram: int = 1,
    reduce_content_style: bool = False,
    reduce_content_style_n_gram: int = 1,
) -> Union[VevoTokenResult, list[VevoTokenResult]]:
    vector_type = _validate_vector_type(vector_type)

    if _is_path_like(audio):
        return model.extract_from_path(
            audio,
            vector_type=vector_type,
            batch_size=batch_size,
            reduce_content=reduce_content,
            reduce_n_gram=reduce_n_gram,
            reduce_content_style=reduce_content_style,
            reduce_content_style_n_gram=reduce_content_style_n_gram,
        )

    if _is_waveform(audio):
        if isinstance(audio, (np.ndarray, torch.Tensor)) and audio.ndim == 2 and audio_lengths is not None:
            return model.extract_batch(
                audio,
                sample_rate=sample_rate,
                audio_lengths=audio_lengths,
                batch_size=batch_size,
                vector_type=vector_type,
                reduce_content=reduce_content,
                reduce_n_gram=reduce_n_gram,
                reduce_content_style=reduce_content_style,
                reduce_content_style_n_gram=reduce_content_style_n_gram,
            )

        if sample_rate is None:
            raise ValueError("sample_rate is required when audio is a waveform")
        return model.extract_from_waveform(
            audio,
            sample_rate=sample_rate,
            vector_type=vector_type,
            batch_size=batch_size,
            reduce_content=reduce_content,
            reduce_n_gram=reduce_n_gram,
            reduce_content_style=reduce_content_style,
            reduce_content_style_n_gram=reduce_content_style_n_gram,
        )

    if isinstance(audio, Sequence) and not isinstance(audio, (str, bytes, Path)):
        return model.extract_batch(
            audio,
            sample_rate=sample_rate,
            sample_rates=sample_rates,
            audio_lengths=audio_lengths,
            batch_size=batch_size,
            vector_type=vector_type,
            reduce_content=reduce_content,
            reduce_n_gram=reduce_n_gram,
            reduce_content_style=reduce_content_style,
            reduce_content_style_n_gram=reduce_content_style_n_gram,
        )

    raise TypeError("unsupported audio input type")


build_vevo_model = build_vevo_token_model
run_vevo_model = run_vevo_token_model
