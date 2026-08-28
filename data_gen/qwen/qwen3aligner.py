import base64
import io
import json
import os
import re
import urllib.request
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
import math

import numpy as np
import soundfile as sf
import torch
from langdetect import detect as classify_language, LangDetectException

from data_gen.qwen.qwen_asr_compat.audio_io import (
    AudioLike,
    ensure_list,
    normalize_audios,
)
from data_gen.qwen.qwen_asr_compat.forced_aligner import Qwen3ForcedAligner

from utils.text import remove_pause_punct


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_json_value(value):
    if torch.is_tensor(value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    return str(value)


def _audio_meta(audio):
    try:
        arr = np.asarray(audio)
        meta = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "num_samples": int(arr.size),
            "duration_sec": round(float(arr.size) / 16000.0, 6),
            "finite": bool(np.isfinite(arr).all()) if arr.size else True,
        }
        if arr.size:
            meta.update({
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "abs_max": float(np.max(np.abs(arr))),
                "mean": float(np.mean(arr)),
            })
        return meta
    except Exception as exc:
        return {"error": str(exc)}


def _inputs_meta(inputs, audio_token_id=None, timestamp_token_id=None):
    if inputs is None:
        return None
    meta = {}
    try:
        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            meta["input_ids"] = _safe_json_value(input_ids)
            if audio_token_id is not None:
                meta["audio_token_counts"] = (input_ids == int(audio_token_id)).sum(dim=-1).detach().cpu().tolist()
            if timestamp_token_id is not None:
                meta["timestamp_token_counts"] = (input_ids == int(timestamp_token_id)).sum(dim=-1).detach().cpu().tolist()
        if "attention_mask" in inputs:
            meta["attention_mask"] = _safe_json_value(inputs["attention_mask"])
            meta["attention_lengths"] = inputs["attention_mask"].sum(dim=-1).detach().cpu().tolist()
        if "input_features" in inputs:
            meta["input_features"] = _safe_json_value(inputs["input_features"])
        if "feature_attention_mask" in inputs:
            meta["feature_attention_mask"] = _safe_json_value(inputs["feature_attention_mask"])
            meta["feature_attention_lengths"] = inputs["feature_attention_mask"].sum(dim=-1).detach().cpu().tolist()
    except Exception as exc:
        meta["error"] = str(exc)
    return meta


def _resolve_audio_token_id(model, processor):
    """Resolve audio token id across wrapper and thinker configs."""
    candidates = [
        getattr(getattr(model, "thinker", None), "config", None),
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "thinker_config", None),
        getattr(processor, "tokenizer", None),
    ]
    for candidate in candidates:
        token_id = getattr(candidate, "audio_token_id", None)
        if token_id is not None:
            return int(token_id)

    tokenizer = getattr(processor, "tokenizer", None)
    audio_token = getattr(processor, "audio_token", None) or getattr(tokenizer, "audio_token", None)
    if tokenizer is not None and audio_token is not None:
        token_id = tokenizer.convert_tokens_to_ids(audio_token)
        if token_id is not None:
            return int(token_id)

    raise AttributeError(
        "Cannot resolve Qwen3 ASR audio_token_id from model.thinker.config, "
        "model.config, model.config.thinker_config, or processor tokenizer."
    )


def _write_debug_dump(
    *,
    stage,
    audios=None,
    texts=None,
    languages=None,
    word_lists=None,
    aligner_input_texts=None,
    inputs=None,
    audio_token_id=None,
    timestamp_token_id=None,
    error=None,
):
    if not (
        _env_flag("SCRIPTSPEECH_QWEN_ALIGNER_DUMP_BAD_INPUT")
        or (stage == "pre_forward" and _env_flag("SCRIPTSPEECH_QWEN_ALIGNER_DEBUG"))
    ):
        return
    dump_dir = os.environ.get("SCRIPTSPEECH_QWEN_ALIGNER_DUMP_DIR") or os.path.join("user", "temp", "source_data_debug", "qwen_aligner")
    dump_wav = _env_flag("SCRIPTSPEECH_QWEN_ALIGNER_DUMP_WAV")
    try:
        os.makedirs(dump_dir, exist_ok=True)
        dump_id = f"{int(time.time() * 1000)}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        records = []
        audios = audios or []
        texts = texts or []
        languages = languages or []
        word_lists = word_lists or []
        aligner_input_texts = aligner_input_texts or []
        n = max(len(audios), len(texts), len(languages), 1)
        for idx in range(n):
            audio = audios[idx] if idx < len(audios) else None
            wav_path = None
            if dump_wav and audio is not None:
                wav_path = os.path.join(dump_dir, f"{dump_id}_{idx}.wav")
                sf.write(wav_path, np.asarray(audio, dtype=np.float32), 16000)
            text = texts[idx] if idx < len(texts) else None
            aligner_input_text = aligner_input_texts[idx] if idx < len(aligner_input_texts) else None
            record = {
                "dump_id": dump_id,
                "stage": stage,
                "pid": os.getpid(),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sample_idx": idx,
                "language": languages[idx] if idx < len(languages) else None,
                "text_len": None if text is None else len(str(text)),
                "text_prefix": None if text is None else str(text)[:300],
                "word_count": None if idx >= len(word_lists) else len(word_lists[idx]),
                "word_prefix": None if idx >= len(word_lists) else word_lists[idx][:50],
                "aligner_input_text_len": None if aligner_input_text is None else len(str(aligner_input_text)),
                "aligner_input_text_prefix": None if aligner_input_text is None else str(aligner_input_text)[:300],
                "audio": None if audio is None else _audio_meta(audio),
                "wav_path": wav_path,
                "inputs": _inputs_meta(inputs, audio_token_id=audio_token_id, timestamp_token_id=timestamp_token_id),
                "error_type": None if error is None else type(error).__name__,
                "error": None if error is None else str(error)[-2000:],
            }
            records.append(record)
        jsonl_path = os.path.join(dump_dir, "qwen_aligner_bad_inputs.jsonl")
        with open(jsonl_path, "a") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, default=_safe_json_value) + "\n")
    except Exception:
        return


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


class Qwen3ForcedAlignerWithConf(Qwen3ForcedAligner):
    def _to_structured_items(self, timestamp_output: List[Dict[str, Any]]) -> ForcedAlignResultWithConf:
        items: List[ForcedAlignItemWithConf] = []
        for it in timestamp_output:
            items.append(
                ForcedAlignItemWithConf(
                    text=str(it.get("text", "")),
                    start_time=float(it.get("start_time", 0)),
                    end_time=float(it.get("end_time", 0)),
                    conf=float(it.get("conf", 0.0)),
                )
            )
        return ForcedAlignResultWithConf(items=items)

    @torch.inference_mode()
    def align(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        text: Union[str, List[str]],
        language: Union[str, List[str]],
        return_conf: bool = True,
    ) -> List[ForcedAlignResultWithConf]:
        """
        Run forced alignment for batch or single sample.

        Args:
            audio:
                Audio input(s). Each item supports:
                  - local path / https URL / base64 string
                  - (np.ndarray, sr)
                All audios will be converted into mono 16k float32 arrays in [-1, 1].
            text:
                Transcript(s) for alignment.
            language:
                Language(s) for each sample (e.g., "Chinese", "English").

        Returns:
            List[ForcedAlignResultWithConf]:
                One result per sample. Each result contains `items`, and each token can be accessed via
                `.text`, `.start_time`, `.end_time`.
        """
        texts = ensure_list(text)
        languages = ensure_list(language)
        audios = normalize_audios(audio)

        if len(languages) == 1 and len(audios) > 1:
            languages = languages * len(audios)

        if not (len(audios) == len(texts) == len(languages)):
            raise ValueError(
                f"Batch size mismatch: audio={len(audios)}, text={len(texts)}, language={len(languages)}"
            )

        word_lists = []
        aligner_input_texts = []
        for t, lang in zip(texts, languages):
            word_list, aligner_input_text = self.aligner_processor.encode_timestamp(t, lang)
            word_lists.append(word_list)
            aligner_input_texts.append(aligner_input_text)
        audio_token_id = _resolve_audio_token_id(self.model, self.processor)

        try:
            inputs = self.processor(
                text=aligner_input_texts,
                audio=audios,
                return_tensors="pt",
                padding=True,
            )
        except Exception as exc:
            _write_debug_dump(
                stage="processor",
                audios=audios,
                texts=texts,
                languages=languages,
                word_lists=word_lists,
                aligner_input_texts=aligner_input_texts,
                audio_token_id=audio_token_id,
                timestamp_token_id=self.timestamp_token_id,
                error=exc,
            )
            raise

        try:
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            if _env_flag("SCRIPTSPEECH_QWEN_ALIGNER_DEBUG"):
                _write_debug_dump(
                    stage="pre_forward",
                    audios=audios,
                    texts=texts,
                    languages=languages,
                    word_lists=word_lists,
                    aligner_input_texts=aligner_input_texts,
                    inputs=inputs,
                    audio_token_id=audio_token_id,
                    timestamp_token_id=self.timestamp_token_id,
                )
            logits = self.model.thinker(**inputs).logits  # [B, T, V]
        except Exception as exc:
            _write_debug_dump(
                stage="model_forward",
                audios=audios,
                texts=texts,
                languages=languages,
                word_lists=word_lists,
                aligner_input_texts=aligner_input_texts,
                inputs=inputs,
                audio_token_id=audio_token_id,
                timestamp_token_id=self.timestamp_token_id,
                error=exc,
            )
            raise
        output_ids = logits.argmax(dim=-1)            # [B, T]

        results: List[ForcedAlignResultWithConf] = []
        for b, (input_id, output_id, word_list) in enumerate(zip(inputs["input_ids"], output_ids, word_lists)):
            timestamp_mask = input_id == self.timestamp_token_id

            masked_output_id = output_id[timestamp_mask]  # [K]，你这里 K=2*N
            timestamp_ms = (masked_output_id * self.timestamp_segment_time).to("cpu").numpy()

            timestamp_output = self.aligner_processor.parse_timestamp(word_list, timestamp_ms)

            token_conf: Optional[torch.Tensor] = None
            if return_conf:
                masked_logits = logits[b, timestamp_mask, :]  # [K, V]

                masked_logits_f32 = masked_logits.float()
                log_denom = torch.logsumexp(masked_logits_f32, dim=-1)  # [K]
                log_num = masked_logits_f32.gather(-1, masked_output_id.unsqueeze(-1)).squeeze(-1)  # [K]
                masked_conf = (log_num - log_denom).exp()  # [K]，top1 概率

                if masked_conf.numel() == 2 * len(timestamp_output) and masked_conf.numel() % 2 == 0:
                    conf_pair = masked_conf.view(len(timestamp_output), 2)
                    token_conf = conf_pair.min(dim=-1).values
                elif masked_conf.numel() == len(timestamp_output):
                    token_conf = masked_conf
                else:
                    token_conf = None

            for j, it in enumerate(timestamp_output):
                it["start_time"] = round(it["start_time"] / 1000.0, 3)
                it["end_time"] = round(it["end_time"] / 1000.0, 3)
                if token_conf is not None:
                    it["conf"] = float(token_conf[j].detach().cpu())

            results.append(self._to_structured_items(timestamp_output))

        return results


def sentence_conf_geomean(items, eps=1e-6, weight_by_duration=True):
    logs = []
    weights = []
    for it in items:
        if getattr(it, "is_ignored", False):
            continue
        if getattr(it, "conf", None) is None:
            continue
        c = max(float(it.conf), eps)
        if weight_by_duration:
            if getattr(it, "start_time", None) is None or getattr(it, "end_time", None) is None:
                w = 1.0
            else:
                w = max(float(it.end_time) - float(it.start_time), 0.0)
                if w <= 0:
                    w = 1.0
        else:
            w = 1.0
        logs.append(math.log(c) * w)
        weights.append(w)
    return math.exp(sum(logs) / max(sum(weights), eps))


def sentence_conf_min(items):
    confs = [float(it.conf) for it in items if (not getattr(it, "is_ignored", False)) and getattr(it, "conf", None) is not None]
    return min(confs) if confs else 0.0


def sentence_conf_p10(items):
    confs = sorted(
        float(it.conf)
        for it in items
        if (not getattr(it, "is_ignored", False)) and getattr(it, "conf", None) is not None
    )
    if not confs:
        return 0.0
    k = int(0.1 * (len(confs) - 1))
    return confs[k]


def detect_language_for_qwen3aligner(text: str):
    try:
        lang = classify_language(text)
    except LangDetectException:
        lang = 'zh'
    if lang == 'ko':
        return 'korean'
    elif lang == 'ja':
        return 'japanese'
    elif lang == 'en':
        return 'english'
    else:
        return 'chinese'
    

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


class Qwen3AlignerInfer:
    def __init__(self, model_path="pretrained_models/Qwen3-ForcedAligner-0.6B", device='cuda:0', precision=torch.bfloat16, attn_implementation='sdpa'):
        if isinstance(device, torch.device):
            device_index = device.index
            device = device.type
            if device_index is not None:
                device = f"{device}:{device_index}"
        
        self.device = device
        self.precision = precision

        self.aligner = Qwen3ForcedAlignerWithConf.from_pretrained(
            model_path,
            dtype=self.precision,
            device_map=device,
            attn_implementation=attn_implementation,
            local_files_only=True
        )

    def process(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        text: Union[str, List[str]],
        language: Union[str, List[str]],
        return_conf: bool = True,
        return_json: bool = False
    ):
        with torch.autocast(device_type='cpu' if self.device == 'cpu' else 'cuda', dtype=self.precision):
            results = self.aligner.align(
                audio=audio,
                text=text,
                language=language,
                return_conf=return_conf
            )

        if return_json:
            results_ = []
            for result in results:
                result_ = []
                for item in result:
                    result_.append({
                        "start_time": item.start_time,
                        "end_time": item.end_time,
                        "text": item.text,
                        "conf": item.conf
                    })
                results_.append(result_)
            results = results_

        return results
    
    def process_with_ignored_patterns(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        text: Union[str, List[str]],
        language: Union[str, List[str]],
        *,
        ignore_literals: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        return_conf: bool = True,
        return_json: bool = False,
        merge_ignored_into_json: bool = True,
        merge_ignored_into_results: bool = True,
        include_pieces_in_meta: bool = True,
    ) -> Dict[str, Any]:
        """对齐时忽略指定字符串/模式（例如 speaker tags），并返回 marker 信息。

        参数：
        - ignore_literals：字面匹配“名单字符串”，例如 ["<S1>", "</S1>", "<S2>", "</S2>"]
        - ignore_patterns：正则名单，例如 [r"</?S\\d+>"]

        返回 dict：
        - results：与 process(...) 相同的返回（list），但 text 使用剔除 ignore 后的版本
        - meta：每条样本的 marker 信息（token_pos 基于 aligner_processor.encode_timestamp 的 token 计数）
        - results_with_markers：当 return_json=False 且 merge_ignored_into_results=True 时提供（更推荐）
        - results_with_markers_json：当 return_json=True 且 merge_ignored_into_json=True 时提供
        """
        ignore_regex = _compile_ignore_regex(ignore_literals=ignore_literals, ignore_patterns=ignore_patterns)

        texts = ensure_list(text)
        languages = ensure_list(language)
        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(texts) != len(languages):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}")

        stripped_texts: List[str] = []
        meta: List[Dict[str, Any]] = []

        for t, lang in zip(texts, languages):
            if ignore_regex is None:
                stripped_texts.append(t)
                meta.append({
                    "original_text": t,
                    "stripped_text": t,
                    "markers": [],
                    "ignore_literals": ignore_literals or [],
                    "ignore_patterns": ignore_patterns or [],
                    "pieces": None,
                })
                continue

            pieces = _split_by_ignore_regex(t, ignore_regex)
            stripped = "".join(p["content"] for p in pieces if p["kind"] == "text")

            token_pos = 0
            markers: List[Dict[str, Any]] = []
            for p in pieces:
                if p["kind"] == "text":
                    seg_text = p["content"]
                    if seg_text:
                        word_list, _ = self.aligner.aligner_processor.encode_timestamp(seg_text, lang)
                        token_pos += len(word_list)
                else:
                    markers.append({
                        "text": p["content"],
                        "token_pos": int(token_pos),
                        "is_ignored": True,
                    })

            stripped_texts.append(stripped)
            meta_item: Dict[str, Any] = {
                "original_text": t,
                "stripped_text": stripped,
                "markers": markers,
                "ignore_literals": ignore_literals or [],
                "ignore_patterns": ignore_patterns or [],
            }
            if include_pieces_in_meta:
                meta_item["pieces"] = pieces
            else:
                meta_item["pieces"] = None
            meta.append(meta_item)

        results = self.process(
            audio=audio,
            text=stripped_texts,
            language=languages,
            return_conf=return_conf,
            return_json=return_json,
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
            merged_json = []
            for res_items, m in zip(results, meta):
                merged_json.append(_merge_markers_into_json_items(list(res_items), list(m.get("markers") or [])))
            out["results_with_markers_json"] = merged_json

        return out


if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')

    from utils.commons.io import json_dumps

    aligner = Qwen3AlignerInfer()

    # results = aligner.process(
    #     audio="user/temp/audio.wav",
    #     text="延安，我是 taka，我是黄瓜酱，我是小刘，欢迎大家来到奥特电波，欢迎大家。今天呢是我们之前跟大家讲的那个歌友会，新年歌友会啊，朋友们，然后请上了一位唱歌非常好听的男士。哈喽哈喽，哈哈哈，所以你是，所以我是大人。啊，欢迎欢迎欢迎。那谢谢大家嘛，新年歌友会不得热闹一点，然后就是希望给大家带来一个，就是说比较有这个祥和喜庆的氛围在里面，是吧？大英，嗯，所以说我们就先来给大家送上一首好运来，好不好？那这样我们就一人接一句，好不好？那我先开始啊，我要用那个高八九的声音，可以吗？可以啊。明天之河飞一 个红飘带，愿你健康春常在。哈哈哈。快点继续去，快点继续去。新年歌友会嘛，为嘛小众要开？哪个中国节？对呀，我就是那个红飘带。嗯，今天这个就是呃歌友会呢，我们就是会简单的用比较短的时间来回顾一下我们在千禧年的时候，嗯，对，我们今天游戏环节呢分为三种 。第一种呢是先听前奏，然后猜这首歌是什么。嗯，然后我们选择的歌呢，一定都是起码百分之八九十的人都听过的这首歌。而且都是很古早的歌。",
    #     language="Chinese",
    #     return_conf=True,
    #     return_json=True
    # )
    # print(json_dumps(results))

    results = aligner.process_with_ignored_patterns(
        audio="infer_out/tts_dialogue/infer_once/260108_prompt_base_dialogue/step84000/0-我家这餐馆，菜味儿正宗分量也足，怎么饭点.wav",
        text="<S1>我家这餐馆，菜味儿正宗分量也足，怎么饭点总坐不满啊？ </S1><S2>你试过做抖音团购套餐吗？ </S2><S1>抖音团购？哎呦，那是不是得花钱推广，还怕没人买亏了本啊？ </S1><S2>不用花推广费，套餐定价灵活还能吸引新客。 </S2><S1>真不用花钱？这靠谱吗？ </S1><S2>现在点击视频下方链接，就能零元上架抖音团购套餐，官方还会给本地流量推荐，帮你把附近想吃的人都引过来。像双人餐、家庭餐都能做，还能设置到店核销，不怕跑单。 </S2><S1>这听着行啊，那团购套餐咋设计啊？ </S1><S2>平台有现成的套餐模板，你按自家招牌菜搭配就行，还能自动生成图文海报，就算不会做宣传，用户刷到直接就能下单，用完都说好还会带朋友来。 </S2><S1>那我现在就去弄这个团购套餐！ </S1><S2>你们做餐饮的也赶紧点下方链接，上架抖音团购，让店里天天满座！</S2>",
        # audio="/mnt/bn/sa-ag-data/liruiqi/data/speech/Emilia_small2/EN/EN-B000005/EN_B00000_S01555_W000003.mp3",
        # text=" Did you guys want to talk about the podcast that you were on?",
        language="Chinese",
        ignore_literals=["<S1>", "</S1>", "<S2>", "</S2>", "<S3>", "</S3>", "<S4>", "</S4>"],
        return_json=True
    )
    # print(results)
    import json
    with open('infer_out/asr/aligner/qwen3aligner_infer.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
