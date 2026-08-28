import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
import math
import unicodedata
import json

import numpy as np
import soundfile as sf
import torch
import torchaudio

from utils.commons.hparams import set_hparams
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.dataset_utils import collate_xd, pad_or_cut_xd
from utils.audio.transform import float_range_normalize, float_range_normalize_torch, batch_resample, to_mono

AudioLike = Union[
    str,                                      # wav path
    Tuple[Union[np.ndarray, torch.Tensor], int],  # (waveform, sr)
]
BatchedAudioLike = Tuple[
    torch.Tensor,                              # wavs: already collated, on device
    Union[torch.Tensor, List[int], np.ndarray] # wav_lens
]
MaybeList = Union[Any, List[Any]]


def ensure_list(x: MaybeList) -> List[Any]:
    return x if isinstance(x, list) else [x]


def _compile_ignore_regex(
    ignore_literals: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> Optional[re.Pattern]:
    """
    返回一个把 ignore_literals / ignore_patterns OR 起来的 regex。
    - ignore_literals：按“字面匹配”处理（会 re.escape），适合你说的“名单字符串”
    - ignore_patterns：按“正则”处理，适合复杂规则
    """
    parts: List[str] = []

    if ignore_literals:
        for s in ignore_literals:
            if s is None:
                continue
            s = str(s)
            if not s:
                continue
            parts.append(re.escape(s))

    if ignore_patterns:
        for p in ignore_patterns:
            if p is None:
                continue
            p = str(p)
            if not p:
                continue
            parts.append(f"(?:{p})")

    if not parts:
        return None

    return re.compile("|".join(parts), flags=re.IGNORECASE | re.DOTALL)


def _split_by_ignore_regex(text: str, ignore_regex: re.Pattern) -> List[Dict[str, str]]:
    """
    把 text 切成若干段：
    - {"kind": "text", "content": "..."}：参与对齐
    - {"kind": "ignore", "content": "..."}：不参与对齐，但需要记录位置
    """
    items: List[Dict[str, str]] = []
    last = 0

    for m in ignore_regex.finditer(text):
        if m.start() > last:
            items.append({"kind": "text", "content": text[last:m.start()]})
        items.append({"kind": "ignore", "content": m.group(0)})
        last = m.end()

    if last < len(text):
        items.append({"kind": "text", "content": text[last:]})

    return items


def _merge_markers_into_json_items(
    json_items: List[Dict[str, Any]],
    markers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    把 markers 按 token_pos 插入到 json_items 序列中，形成一个“统一 list”，marker 会带 is_ignored=True。
    这里不会给 marker 填 start_time/end_time/conf（它们为 None），避免误用。
    """
    if not markers:
        return json_items

    merged: List[Dict[str, Any]] = []
    m_i = 0
    markers_sorted = sorted(markers, key=lambda x: int(x.get("token_pos", 0)))

    def emit_markers(before_token_pos: int):
        nonlocal m_i
        while m_i < len(markers_sorted) and int(markers_sorted[m_i].get("token_pos", 0)) <= before_token_pos:
            mk = dict(markers_sorted[m_i])
            mk.setdefault("is_ignored", True)
            mk.setdefault("start_time", None)
            mk.setdefault("end_time", None)
            mk.setdefault("conf", None)
            merged.append(mk)
            m_i += 1

    for token_pos, it in enumerate(json_items):
        emit_markers(token_pos)
        merged.append(it)

    emit_markers(len(json_items))
    return merged


def _merge_markers_into_align_result(
    result: "ForcedAlignResultWithConf",
    markers: List[Dict[str, Any]],
) -> "ForcedAlignResultWithConf":
    """把 markers 插入到 align result 的 items 序列里（不需要 return_json）。"""
    if not markers:
        return result

    merged: List[ForcedAlignItemWithConf] = []
    m_i = 0
    markers_sorted = sorted(markers, key=lambda x: int(x.get("token_pos", 0)))

    def emit_markers(before_token_pos: int):
        nonlocal m_i
        while m_i < len(markers_sorted) and int(markers_sorted[m_i].get("token_pos", 0)) <= before_token_pos:
            mk = markers_sorted[m_i]
            merged.append(
                ForcedAlignItemWithConf(
                    text=str(mk.get("text", "")),
                    start_time=None,
                    end_time=None,
                    conf=None,
                    is_ignored=True,
                    token_pos=int(mk.get("token_pos", 0)),
                )
            )
            m_i += 1

    items = list(result.items)
    for token_pos, it in enumerate(items):
        emit_markers(token_pos)
        merged.append(it)

    emit_markers(len(items))
    return ForcedAlignResultWithConf(items=merged)


def _linspace_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, max_n).astype(np.int64)


def _save_alignment_debug_artifacts(
    save_dir: str,
    index: int,
    *,
    wav: np.ndarray,
    sr: int,
    tokens: List[str],
    word_start_times: np.ndarray,
    word_end_times: np.ndarray,
    word_conf: np.ndarray,
    debug: Optional[Dict[str, Any]] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, f"{index:06d}")

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    sf.write(base + "_audio.wav", wav, sr)

    with open(base + "_tokens.txt", "w", encoding="utf-8") as f:
        for i, tok in enumerate(tokens):
            st = float(word_start_times[i]) if i < len(word_start_times) else None
            ed = float(word_end_times[i]) if i < len(word_end_times) else None
            cf = float(word_conf[i]) if i < len(word_conf) else None
            f.write(f"{i}\t{tok}\t{st}\t{ed}\t{cf}\n")

    meta: Dict[str, Any] = {
        "sr": int(sr),
        "tokens": list(tokens),
        "word_start_times": [float(x) for x in word_start_times.tolist()],
        "word_end_times": [float(x) for x in word_end_times.tolist()],
        "word_conf": [float(x) for x in word_conf.tolist()],
    }
    if isinstance(debug, dict):
        meta["debug_keys"] = sorted(list(debug.keys()))
        meta["frame_sec"] = float(debug.get("frame_sec", 0.0))
        meta["audio_len_frames"] = int(debug.get("audio_len_frames", 0))
        meta["word_len"] = int(debug.get("word_len", len(tokens)))

    with open(base + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    # waveform
    t = np.arange(len(wav), dtype=np.float32) / float(sr)
    fig = plt.figure(figsize=(14, 3))
    ax = plt.gca()
    ax.plot(t, wav, linewidth=0.6, color="black")

    for i in range(min(len(tokens), len(word_start_times), len(word_end_times))):
        st = float(word_start_times[i])
        ed = float(word_end_times[i])
        ax.axvline(st, color="green", linewidth=0.6, alpha=0.8)
        ax.axvline(ed, color="red", linewidth=0.6, alpha=0.8)
        if len(tokens) <= 80:
            ax.text((st + ed) * 0.5, 0.9, tokens[i], transform=ax.get_xaxis_transform(),
                    fontsize=7, rotation=90, ha="center", va="top")

    ax.set_xlabel("time (s)")
    ax.set_title(f"{index:06d} waveform + boundaries")
    fig.tight_layout()
    fig.savefig(base + "_wave.png", dpi=160)
    plt.close(fig)

    if not isinstance(debug, dict):
        return

    frame_sec = float(debug.get("frame_sec", 0.0)) or (1.0 / 50.0)

    dp = debug.get("dp", None)
    states = debug.get("states", None)
    probs_words = debug.get("probs_words", None)

    if isinstance(dp, torch.Tensor):
        dp = dp.detach().cpu().numpy()
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    if isinstance(probs_words, torch.Tensor):
        probs_words = probs_words.detach().cpu().numpy()

    # dp heatmap
    if isinstance(dp, np.ndarray) and dp.ndim == 2 and dp.size > 0:
        dp_vis = dp.astype(np.float32)
        dp_vis = np.where(np.isfinite(dp_vis), dp_vis, -1e4)

        dp_vis = dp_vis - dp_vis.max(axis=1, keepdims=True)
        dp_vis = np.clip(dp_vis, -50.0, 0.0)

        Ti, Si = dp_vis.shape
        t_idx = _linspace_indices(Ti, 1200)
        s_idx = _linspace_indices(Si, 400)
        dp_small = dp_vis[np.ix_(t_idx, s_idx)]

        fig = plt.figure(figsize=(14, 6))
        ax = plt.gca()
        im = ax.imshow(dp_small.T, aspect="auto", origin="lower", cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.set_xlabel("time (downsampled frames)")
        ax.set_ylabel("state (downsampled)")
        ax.set_title(f"{index:06d} DP log-score (normalized per frame)")
        fig.tight_layout()
        fig.savefig(base + "_dp.png", dpi=160)
        plt.close(fig)

    # probs heatmap + viterbi path
    if isinstance(probs_words, np.ndarray) and probs_words.ndim == 2 and probs_words.size > 0:
        pw = probs_words.astype(np.float32)
        Ti, Wi = pw.shape
        t_idx = _linspace_indices(Ti, 1200)
        pw_small = pw[t_idx, :]

        fig = plt.figure(figsize=(14, 6))
        ax = plt.gca()
        im = ax.imshow(pw_small.T, aspect="auto", origin="lower", cmap="magma")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        if isinstance(states, np.ndarray) and states.ndim == 1 and len(states) >= max(t_idx[-1] + 1, 1):
            st_small = states[t_idx]
            w_path = np.where((st_small % 2) == 1, (st_small - 1) // 2, -1).astype(np.float32)
            w_path[w_path < 0] = np.nan
            ax.plot(np.arange(len(w_path)), w_path, color="cyan", linewidth=1.0, alpha=0.9)

        if len(tokens) <= 80 and Wi <= 80:
            ax.set_yticks(np.arange(Wi))
            ax.set_yticklabels(tokens[:Wi], fontsize=7)

        ax.set_xlabel("time (downsampled frames)")
        ax.set_ylabel("word index")
        ax.set_title(f"{index:06d} p(word) heatmap + viterbi path")
        fig.tight_layout()
        fig.savefig(base + "_probs.png", dpi=160)
        plt.close(fig)


@dataclass(frozen=True)
class ForcedAlignItemWithConf:
    """
    One aligned item span.

    Attributes:
        text (str):
            The aligned unit (cjk character or word) produced by the forced aligner processor.
        start_time (float | None):
            Start time in seconds. None means this item is a marker (not aligned to audio).
        end_time (float | None):
            End time in seconds. None means this item is a marker (not aligned to audio).
        conf (float | None):
            Alignment confidence. None for marker items.
        is_ignored (bool):
            True for marker items that were ignored during alignment.
        token_pos (int | None):
            Marker insertion position in token space (0..N). None for normal aligned items.
    """
    text: str
    start_time: Optional[float]
    end_time: Optional[float]
    conf: Optional[float] = 0.0
    is_ignored: bool = False
    token_pos: Optional[int] = None


@dataclass(frozen=True)
class ForcedAlignResultWithConf:
    """
    Forced alignment output for one sample.

    Attributes:
        items (List[ForcedAlignItemWithConf]):
            Aligned token spans.
    """
    items: List[ForcedAlignItemWithConf]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> ForcedAlignItemWithConf:
        return self.items[idx]
    

class ForceAlignerTextProcessor():
    """
    Adapted from Qwen3 Forced Aligner
    """
    def is_kept_char(self, ch: str) -> bool:
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            return True
        return False

    def clean_token(self, token: str) -> str:
        return "".join(ch for ch in token if self.is_kept_char(ch))

    def is_cjk_char(self, ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF   # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # Extension A
            or 0x20000 <= code <= 0x2A6DF  # Extension B
            or 0x2A700 <= code <= 0x2B73F  # Extension C
            or 0x2B740 <= code <= 0x2B81F  # Extension D
            or 0x2B820 <= code <= 0x2CEAF  # Extension E
            or 0xF900 <= code <= 0xFAFF    # Compatibility Ideographs
        )

    def split_segment_with_chinese(self, seg: str) -> List[str]:
        tokens: List[str] = []
        buf: List[str] = []

        def flush_buf():
            nonlocal buf
            if buf:
                tokens.append("".join(buf))
                buf = []

        for ch in seg:
            if self.is_cjk_char(ch):
                flush_buf()
                tokens.append(ch)
            else:
                buf.append(ch)

        flush_buf()
        return tokens

    def tokenize_space_lang(self, text: str) -> List[str]:
        tokens: List[str] = []
        for seg in text.split():
            cleaned = self.clean_token(seg)
            if cleaned:
                tokens.extend(self.split_segment_with_chinese(cleaned))
        return tokens


def _build_aligner_model_by_hparams(hp, attn_implementation: str, *, init_pretrained: bool):
    model_version = str(hp.get("model_version", "aligner_2tower"))
    if model_version == "aligner_mdm":
        from modules.asr.forced_align.aligner_mdm import build_aligner_model as build_aligner_model_impl
    else:
        from modules.asr.forced_align.aligner_2tower import build_aligner_model as build_aligner_model_impl

    return build_aligner_model_impl(hp, attn_implementation, init_pretrained=init_pretrained)


class ForcedAlignerInfer:
    def __init__(
        self,
        ckpt: str,
        device: Union[str, torch.device] = 'cuda',
        precision: torch.dtype = torch.float16,
        attn_implementation: str = 'flash_attention_2',
    ):
        if ckpt.endswith('.ckpt'):
            hp = set_hparams(os.path.join(Path(ckpt).parent, 'config.yaml'), global_hparams=False)
        else:
            hp = set_hparams(os.path.join(ckpt, 'config.yaml'), global_hparams=False)
        self.hp = hp
        self.model_version = str(hp.get('model_version', 'aligner_2tower'))
        self.device = device
        self.precision = precision

        model, text_tokenizer = _build_aligner_model_by_hparams(
            hp,
            attn_implementation,
            init_pretrained=False,
        )
        load_ckpt(model, ckpt, 'model', strict=False)
        model.to(device, dtype=precision)
        model.eval()
        self.model = model
        self.text_tokenizer = text_tokenizer
        self.text_processor = ForceAlignerTextProcessor()
        self.resamplers = {}

    def _autocast_kwargs(self) -> Dict[str, Any]:
        device_type = "cuda" if (isinstance(self.device, str) and self.device.startswith("cuda")) else "cpu"
        return {
            "device_type": device_type,
            "dtype": self.precision,
            "enabled": (device_type == "cuda"),
        }

    def _to_json(self, results: List[ForcedAlignResultWithConf]) -> List[List[Dict[str, Any]]]:
        out: List[List[Dict[str, Any]]] = []
        for res in results:
            items: List[Dict[str, Any]] = []
            for it in res:
                items.append(
                    {
                        "start_time": it.start_time,
                        "end_time": it.end_time,
                        "text": it.text,
                        "conf": it.conf,
                    }
                )
            out.append(items)
        return out

    @torch.inference_mode()
    def align(
        self,
        audio: Union[AudioLike, List[AudioLike], BatchedAudioLike],
        text: Union[str, List[str]],
        emit_temp: float = 1.0,
        trans_temp: float = 1.0,
        return_conf: bool = True,
        return_json: bool = False,
        *,
        debug_save_dir: Optional[str] = None,
        debug_start_index: int = 0,
    ) -> Union[List[ForcedAlignResultWithConf], List[List[Dict[str, Any]]]]:
        hparams = self.hp
        fm_wav = int(hparams["frames_multiple"]) * int(hparams["hop_size"])
        tgt_sr = int(hparams["audio_sample_rate"])

        want_debug = debug_save_dir is not None
        if want_debug:
            os.makedirs(str(debug_save_dir), exist_ok=True)

        texts = ensure_list(text)

        target_device = self.device if isinstance(self.device, torch.device) else torch.device(str(self.device))

        def _maybe_get_batched_audio(x: Any) -> Optional[Tuple[torch.Tensor, Any]]:
            if not (isinstance(x, tuple) and len(x) == 2 and torch.is_tensor(x[0])):
                return None

            wav_lens_like = x[1]

            if isinstance(wav_lens_like, (int, np.integer, float, np.floating)):
                return None

            if torch.is_tensor(wav_lens_like) or isinstance(wav_lens_like, (list, tuple, np.ndarray)):
                return x[0], wav_lens_like

            return None

        batched = _maybe_get_batched_audio(audio)
        if batched is None and isinstance(audio, list) and len(audio) == 1:
            batched = _maybe_get_batched_audio(audio[0])

        wavs_used_np: Optional[List[np.ndarray]] = None

        if batched is not None:
            wavs, wav_lens_in = batched

            if wavs.device != target_device:
                wavs = wavs.to(target_device, non_blocking=(wavs.device.type == "cpu"))

            if wavs.dim() == 1:
                wavs = wavs.unsqueeze(0)
            elif wavs.dim() == 3 and wavs.shape[1] == 1:
                wavs = wavs[:, 0, :]
            elif wavs.dim() != 2:
                raise ValueError(
                    f"Prebatched wavs must be [B, T] (or [B, 1, T]/[T]), got shape={tuple(wavs.shape)}"
                )

            if not torch.is_floating_point(wavs):
                wavs = wavs.float()

            if torch.is_tensor(wav_lens_in):
                wav_lens = wav_lens_in.to(
                    device=target_device,
                    dtype=torch.long,
                    non_blocking=(wav_lens_in.device.type == "cpu"),
                )
            elif isinstance(wav_lens_in, np.ndarray):
                wav_lens = torch.from_numpy(wav_lens_in).to(device=target_device, dtype=torch.long, non_blocking=True)
            else:
                wav_lens = torch.tensor(list(wav_lens_in), device=target_device, dtype=torch.long)

            wavs = float_range_normalize_torch(wavs, wav_lens)

            B = int(wavs.shape[0])
            if int(wav_lens.numel()) != B:
                raise ValueError(f"Batch size mismatch: wavs.shape[0]={B}, wav_lens.numel()={int(wav_lens.numel())}")

            T = int(wavs.shape[-1])
            wav_lens = wav_lens.clamp(min=0, max=T)

            if fm_wav > 0:
                wav_lens_nonzero = wav_lens.clamp(min=1)
                wav_lens = ((wav_lens_nonzero + fm_wav - 1) // fm_wav) * fm_wav

                tgt_T = int(math.ceil(max(T, 1) / fm_wav) * fm_wav)
                if tgt_T != T:
                    wavs = pad_or_cut_xd(wavs, tgt_T, dim=-1)

            if len(texts) == 1 and B > 1:
                texts = texts * B
            if len(texts) != B:
                raise ValueError(f"Batch size mismatch: audio_batch={B}, text_batch={len(texts)}")

            if want_debug:
                wavs_used_np = []
                wavs_cpu = wavs.detach().to("cpu")
                wav_lens_cpu = wav_lens.detach().to("cpu")
                for b in range(B):
                    L = int(wav_lens_cpu[b].item())
                    wavs_used_np.append(wavs_cpu[b, :L].numpy().astype(np.float32, copy=False))

        else:
            audios = ensure_list(audio)

            if len(texts) == 1 and len(audios) > 1:
                texts = texts * len(audios)
            if len(texts) != len(audios):
                raise ValueError(f"Batch size mismatch: audio_batch={len(audios)}, text_batch={len(texts)}")

            wavs_np: List[np.ndarray] = []
            srs: List[int] = []
            for audio_item in audios:
                if isinstance(audio_item, str):
                    wav_t, orig_sr = torchaudio.load(audio_item)  # [C, T]
                    wav_np = wav_t.mean(dim=0).detach().cpu().numpy()
                else:
                    wav_raw, orig_sr = audio_item
                    if isinstance(wav_raw, torch.Tensor):
                        wav_np = wav_raw.detach().cpu().numpy()
                    else:
                        wav_np = np.asarray(wav_raw)
                    wav_np = to_mono(wav_np)

                wav_np = float_range_normalize(wav_np)
                wavs_np.append(wav_np.astype(np.float32, copy=False))
                srs.append(int(orig_sr))

            wavs_np = batch_resample(
                wavs_np,
                srs,
                tgt_sr=tgt_sr,
                resamplers=self.resamplers,
                device=self.device,
            )

            wavs_list: List[torch.Tensor] = []
            wavs_used_np = [] if want_debug else None
            for w in wavs_np:
                wt = torch.from_numpy(w).float()
                if fm_wav > 0:
                    tgt_len = int(math.ceil(max(int(wt.numel()), 1) / fm_wav) * fm_wav)
                    wt = pad_or_cut_xd(wt, tgt_len)
                wavs_list.append(wt)
                if wavs_used_np is not None:
                    wavs_used_np.append(wt.detach().cpu().numpy().astype(np.float32, copy=False))

            wav_lens = torch.LongTensor([int(x.numel()) for x in wavs_list]).to(self.device)
            wavs = collate_xd(wavs_list).to(self.device)

        word_lists: List[List[str]] = []
        txt_tokens_list: List[torch.Tensor] = []

        mdm_mask_positions: Optional[List[List[int]]] = [] if self.model_version == 'aligner_mdm' else None
        mdm_mask_id: Optional[int] = None
        if mdm_mask_positions is not None:
            mdm_mask_id = int(getattr(self.model, 'mask_id', self.text_tokenizer.encode('<MASK>')[0]))

        for t in texts:
            word_list = self.text_processor.tokenize_space_lang(str(t))
            word_lists.append(word_list)

            if self.model_version == 'aligner_mdm':
                if len(word_list) > 0:
                    input_text = ''.join([w + '<MASK><MASK>' for w in word_list])
                else:
                    input_text = '<MASK>'

                ids = self.text_tokenizer.encode(input_text)
                ids_t = torch.tensor(ids, dtype=torch.long)
                if mdm_mask_positions is not None:
                    mask_pos = torch.nonzero(ids_t == mdm_mask_id, as_tuple=False).squeeze(-1).tolist()
                    mdm_mask_positions.append(mask_pos)
                txt_tokens_list.append(ids_t)

            else:
                if len(word_list) > 0:
                    input_text = '<|wbd|>'.join(word_list) + '<|wbd|>'
                else:
                    input_text = '<|wbd|>'

                ids = self.text_tokenizer.encode(input_text)
                txt_tokens_list.append(torch.tensor(ids, dtype=torch.long))

        txt_lens = torch.LongTensor([int(x.numel()) for x in txt_tokens_list]).to(self.device)
        txt_tokens = collate_xd(txt_tokens_list).to(self.device)

        with torch.autocast(**self._autocast_kwargs()):
            if self.model_version == 'aligner_mdm':
                infer_timesteps = int(hparams.get('infer_timesteps', 100))
                infer_temperature = float(hparams.get('infer_temperature', 0.7))
                infer_token_topk = int(hparams.get('infer_token_topk', 5))

                raw_results = self.model.inference(
                    wavs,
                    txt_tokens,
                    wav_lens,
                    txt_lens,
                    timesteps=infer_timesteps,
                    temperature=infer_temperature,
                    token_topk=infer_token_topk,
                    return_scores=True,
                    return_debug=want_debug,
                )
            else:
                raw_results = self.model.inference(
                    wavs,
                    txt_tokens,
                    wav_lens,
                    txt_lens,
                    emit_temp=emit_temp,
                    trans_temp=trans_temp,
                    return_debug=want_debug,
                )

        results: List[ForcedAlignResultWithConf] = []

        if self.model_version == 'aligner_mdm':
            pred = raw_results.get('pred', None) if isinstance(raw_results, dict) else None
            if not torch.is_tensor(pred):
                pred = torch.empty((len(word_lists), 0), dtype=torch.long, device='cpu')
            pred = pred.detach().to('cpu')

            token_conf = raw_results.get('token_conf', None) if isinstance(raw_results, dict) else None
            if torch.is_tensor(token_conf):
                token_conf = token_conf.detach().to('cpu')
            else:
                token_conf = None

            ts_start = float(getattr(self.text_tokenizer, 'timestamp_start', 0.0))
            ts_step = float(getattr(self.text_tokenizer, 'timestamp_step', 0.08))
            ts_start_id = int(getattr(self.text_tokenizer, 'timestamp_start_id', self.text_tokenizer.encode('<|TS0.00|>')[0]))
            ts_end_id = int(getattr(self.text_tokenizer, 'timestamp_end_id', self.text_tokenizer.encode('<|TS300.00|>')[0]))

            for b in range(len(word_lists)):
                word_list = word_lists[b]
                mask_pos: List[int] = []
                if mdm_mask_positions is not None and b < len(mdm_mask_positions):
                    mask_pos = list(mdm_mask_positions[b])

                Wb = int(min(len(word_list), len(mask_pos) // 2))

                st_vals: List[float] = []
                ed_vals: List[float] = []
                cf_vals: List[float] = []

                items: List[ForcedAlignItemWithConf] = []
                for j, token_text in enumerate(word_list):
                    if j < Wb and b < pred.shape[0]:
                        p_st = int(mask_pos[2 * j])
                        p_ed = int(mask_pos[2 * j + 1])

                        id_st = int(pred[b, p_st].item()) if p_st < pred.shape[1] else ts_start_id
                        id_ed = int(pred[b, p_ed].item()) if p_ed < pred.shape[1] else ts_start_id

                        valid_st = (ts_start_id <= id_st <= ts_end_id)
                        valid_ed = (ts_start_id <= id_ed <= ts_end_id)

                        st_id = min(max(id_st, ts_start_id), ts_end_id)
                        ed_id = min(max(id_ed, ts_start_id), ts_end_id)

                        start_time = ts_start + float(st_id - ts_start_id) * ts_step
                        end_time = ts_start + float(ed_id - ts_start_id) * ts_step
                        if end_time < start_time:
                            end_time = start_time

                        start_time_r = float(round(float(start_time), 3))
                        end_time_r = float(round(float(end_time), 3))

                        conf_st = 0.0
                        conf_ed = 0.0
                        if token_conf is not None and b < token_conf.shape[0]:
                            if p_st < token_conf.shape[1]:
                                conf_st = float(token_conf[b, p_st].item())
                            if p_ed < token_conf.shape[1]:
                                conf_ed = float(token_conf[b, p_ed].item())

                        if not valid_st:
                            conf_st = 0.0
                        if not valid_ed:
                            conf_ed = 0.0

                        conf_val = float(min(conf_st, conf_ed))
                        conf = conf_val if return_conf else None

                        st_vals.append(start_time_r)
                        ed_vals.append(end_time_r)
                        cf_vals.append(conf_val)
                    else:
                        start_time_r = None
                        end_time_r = None
                        conf = None

                    items.append(
                        ForcedAlignItemWithConf(
                            text=str(token_text),
                            start_time=start_time_r,
                            end_time=end_time_r,
                            conf=conf,
                        )
                    )

                results.append(ForcedAlignResultWithConf(items=items))

                if want_debug and wavs_used_np is not None:
                    _save_alignment_debug_artifacts(
                        str(debug_save_dir),
                        debug_start_index + b,
                        wav=wavs_used_np[b],
                        sr=tgt_sr,
                        tokens=word_list,
                        word_start_times=np.asarray(st_vals[:Wb], dtype=np.float32),
                        word_end_times=np.asarray(ed_vals[:Wb], dtype=np.float32),
                        word_conf=np.asarray(cf_vals[:Wb], dtype=np.float32),
                        debug=None,
                    )

        else:
            for b in range(min(len(word_lists), len(raw_results))):
                word_list = word_lists[b]
                rr = raw_results[b]

                st = rr.get('word_start_times', None)
                ed = rr.get('word_end_times', None)
                cf = rr.get('word_conf', None)

                if st is None or ed is None:
                    st = torch.zeros((0,), device=wavs.device)
                    ed = torch.zeros((0,), device=wavs.device)
                if cf is None:
                    cf = torch.zeros_like(st)

                st = st.detach().to("cpu")
                ed = ed.detach().to("cpu")
                cf = cf.detach().to("cpu")

                Wb = int(min(len(word_list), int(st.numel())))

                items: List[ForcedAlignItemWithConf] = []
                for j, token_text in enumerate(word_list):
                    if j < Wb:
                        start_time = float(round(float(st[j].item()), 3))
                        end_time = float(round(float(ed[j].item()), 3))
                        conf = float(cf[j].item()) if return_conf else None
                    else:
                        start_time = None
                        end_time = None
                        conf = None

                    items.append(
                        ForcedAlignItemWithConf(
                            text=str(token_text),
                            start_time=start_time,
                            end_time=end_time,
                            conf=conf,
                        )
                    )

                results.append(ForcedAlignResultWithConf(items=items))

                if want_debug and wavs_used_np is not None:
                    dbg = rr.get("debug", None)
                    _save_alignment_debug_artifacts(
                        str(debug_save_dir),
                        debug_start_index + b,
                        wav=wavs_used_np[b],
                        sr=tgt_sr,
                        tokens=word_list,
                        word_start_times=st[:Wb].numpy(),
                        word_end_times=ed[:Wb].numpy(),
                        word_conf=cf[:Wb].numpy(),
                        debug=dbg,
                    )

        if return_json:
            return self._to_json(results)
        return results

    def align_with_ignored_patterns(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        text: Union[str, List[str]],
        emit_temp: float = 1.0,
        trans_temp: float = 1.0,
        *,
        ignore_literals: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        return_conf: bool = True,
        return_json: bool = False,
        merge_ignored_into_json: bool = True,
        merge_ignored_into_results: bool = True,
        include_pieces_in_meta: bool = True,
        debug_save_dir: Optional[str] = None,
        debug_start_index: int = 0,
    ) -> Dict[str, Any]:
        """
        对齐时忽略指定字符串/模式（例如 speaker tags），并返回 marker 信息。

        返回 dict：
        - results：与 align(...) 相同的返回（list），但 text 使用剔除 ignore 后的版本
        - meta：每条样本的 marker 信息（token_pos 基于 tokenize_space_lang 的 token 计数）
        - results_with_markers：当 return_json=False 且 merge_ignored_into_results=True 时提供
        - results_with_markers_json：当 return_json=True 且 merge_ignored_into_json=True 时提供
        """
        ignore_regex = _compile_ignore_regex(ignore_literals=ignore_literals, ignore_patterns=ignore_patterns)

        texts = ensure_list(text)
        audios = ensure_list(audio)

        if len(texts) == 1 and len(audios) > 1:
            texts = texts * len(audios)

        stripped_texts: List[str] = []
        meta: List[Dict[str, Any]] = []

        for t in texts:
            t = str(t)
            if ignore_regex is None:
                stripped_texts.append(t)
                meta.append(
                    {
                        "original_text": t,
                        "stripped_text": t,
                        "markers": [],
                        "ignore_literals": ignore_literals or [],
                        "ignore_patterns": ignore_patterns or [],
                        "pieces": None,
                    }
                )
                continue

            pieces = _split_by_ignore_regex(t, ignore_regex)
            stripped = "".join(p["content"] for p in pieces if p["kind"] == "text")

            token_pos = 0
            markers: List[Dict[str, Any]] = []
            for p in pieces:
                if p["kind"] == "text":
                    seg_text = p["content"]
                    if seg_text:
                        token_pos += len(self.text_processor.tokenize_space_lang(seg_text))
                else:
                    markers.append(
                        {
                            "text": p["content"],
                            "token_pos": int(token_pos),
                            "is_ignored": True,
                        }
                    )

            stripped_texts.append(stripped)
            meta_item: Dict[str, Any] = {
                "original_text": t,
                "stripped_text": stripped,
                "markers": markers,
                "ignore_literals": ignore_literals or [],
                "ignore_patterns": ignore_patterns or [],
            }
            meta_item["pieces"] = pieces if include_pieces_in_meta else None
            meta.append(meta_item)

        results = self.align(
            audio=audios,
            text=stripped_texts,
            emit_temp=emit_temp,
            trans_temp=trans_temp,
            return_conf=return_conf,
            return_json=return_json,
            debug_save_dir=debug_save_dir,
            debug_start_index=debug_start_index,
        )

        out: Dict[str, Any] = {
            "results": results,
            "meta": meta,
        }

        if (not return_json) and merge_ignored_into_results:
            merged_results: List[ForcedAlignResultWithConf] = []
            for res, m in zip(results, meta):
                merged_results.append(_merge_markers_into_align_result(res, list(m.get("markers") or [])))
            out["results_with_markers"] = merged_results

        if return_json and merge_ignored_into_json:
            merged_json: List[List[Dict[str, Any]]] = []
            for res_items, m in zip(results, meta):
                merged_json.append(_merge_markers_into_json_items(list(res_items), list(m.get("markers") or [])))
            out["results_with_markers_json"] = merged_json

        return out


if __name__ == '__main__':
    from utils.commons.os_utils import load_env_local
    load_env_local()

    # ckpt = 'checkpoints/260206_aligner_2tower',
    # ckpt = 'checkpoints/260211_aligner_mdm',
    # ckpt = 'checkpoints/260212_aligner_mdm',
    # ckpt = 'checkpoints/260213_aligner_mdm',
    # ckpt = 'checkpoints/260219_aligner_mdm',
    # ckpt = 'checkpoints/260219_aligner_2tower_v2',
    # ckpt = 'checkpoints/260224_aligner_2tower_v4',
    # ckpt = 'checkpoints/260225_aligner_2tower_v4'
    # ckpt = 'checkpoints/260225_aligner_2tower_v5'
    # ckpt = 'checkpoints/260225_aligner_2tower_v5'
    ckpt = 'checkpoints/260304_aligner_2tower_v6'
    # ckpt = 'checkpoints/260311_aligner_mdm'

    exp_name = Path(ckpt).stem

    aligner = ForcedAlignerInfer(
        ckpt=ckpt,
        # precision=torch.bfloat16,
        # attn_implementation='sdpa',
    )

    results = aligner.align_with_ignored_patterns(
        # audio="infer_out/tts_dialogue/infer_once/260108_prompt_base_dialogue/step84000/0-我家这餐馆，菜味儿正宗分量也足，怎么饭点.wav",
        # text="<S1>我家这餐馆，菜味儿正宗分量也足，怎么饭点总坐不满啊？ </S1><S2>你试过做抖音团购套餐吗？ </S2><S1>抖音团购？哎呦，那是不是得花钱推广，还怕没人买亏了本啊？ </S1><S2>不用花推广费，套餐定价灵活还能吸引新客。 </S2><S1>真不用花钱？这靠谱吗？ </S1><S2>现在点击视频下方链接，就能零元上架抖音团购套餐，官方还会给本地流量推荐，帮你把附近想吃的人都引过来。像双人餐、家庭餐都能做，还能设置到店核销，不怕跑单。 </S2><S1>这听着行啊，那团购套餐咋设计啊？ </S1><S2>平台有现成的套餐模板，你按自家招牌菜搭配就行，还能自动生成图文海报，就算不会做宣传，用户刷到直接就能下单，用完都说好还会带朋友来。 </S2><S1>那我现在就去弄这个团购套餐！ </S1><S2>你们做餐饮的也赶紧点下方链接，上架抖音团购，让店里天天满座！</S2>",
        # audio="/mnt/bn/sa-ag-data/liruiqi/data/speech/Emilia_small2/EN/EN-B000005/EN_B00000_S01555_W000003.mp3",
        # text=" Did you guys want to talk about the podcast that you were on?",
        # audio="/mnt/bn/genai-data2/liruiqi/code/ScriptSpeech/user/prompts/dzq_enhanced.wav",
        # text="什么等下等下，我看到一个留言说什么邓紫曦都四十二岁了，你算错了吧！我十年前上我是歌手的时候，二十二岁啊兄弟，十年啊，二十二加十，三十二，懂吗？懂吗？三十二不是四十二，你算错了，谁教你的数学谁是你的数学老师。",
        audio="user/assets/text.wav",
        text="外宝吃的贼香，贼香那个烟熏果木烤肉啊，现在在哈尔滨开新店了，自助的模式，预售价格九十三块钱实现上吃啊。你们赶紧的预售的价格是真香。",
        ignore_literals=["<S1>", "</S1>", "<S2>", "</S2>", "<S3>", "</S3>", "<S4>", "</S4>"],
        return_json=True,
        # debug_save_dir=f"infer_out/asr/aligner/{exp_name}",
        # emit_temp=0.5,
        # trans_temp=1.0,
        # debug_start_index=0,
    )
    print(results)
    with open(f'infer_out/asr/aligner/{exp_name}/results.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# CUDA_VISIBLE_DEVICES=3 python inference/asr/aligner_infer.py
