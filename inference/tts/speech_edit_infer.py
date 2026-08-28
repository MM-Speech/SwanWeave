import os
import random
import re
import tempfile
from datetime import datetime, timedelta
import collections
import collections.abc
import math
import unicodedata
import numpy as np
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
from attrdictionary import AttrDict
from typing import Optional, Dict, List, Any, Tuple
import socket
from contextlib import closing
import yaml
from argparse import ArgumentParser
import torch.distributed as dist
import torch
import soundfile as sf
import librosa
import shutil
import time

os.environ.setdefault("MODELSCOPE_CACHE", "/mnt/bn/sa-ag-data/liruiqi/code/modelscope")

from multiprocessing import Process, set_start_method
from utils.commons.os_utils import kill_void
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint, torch_load_dist
from utils.commons.hparams import set_hparams, hparams
from modules.asr.sensevoice.sensevoice_api import build_asr_model, run_asr_model
# from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin, SemanticLMBuildModelMixin, build_vae
from modules.tts.scriptspeech.build_model_utils import SemanticLMBuildModelMixin, build_vae
from modules.tts.scriptspeech.dit_edit import DiTBuildModelMixin

from utils.commons.upload_tos_utils import send_file_to_tos
import json
# from tasks.tts.task_utils.prompttts_task_utils import build_audio_mask_from_ids
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text.zh_text_norm import num2chn
from utils.text import is_chinese
from utils.text.cosyvoice2_tokenizer import CosyVoice2Tokenizer
# from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import build_spk_mask_from_text_tokens, _get_sx_token_patterns, augment_text_with_pinyin_s1s2_safe


# from utils.commons.dataset_utils import collate_xd, pad_or_cut_xd
# from utils.audio.transform import float_range_normalize, float_range_normalize_torch, batch_resample, to_mono
from inference.asr.aligner_infer import ForcedAlignerInfer

from data_gen.source_separation.uvr.uvr_api import build_uvr_model, run_uvr_model
from data_gen.qwen.sensevoice import build_vad_model, run_vad_model

from utils.audio.transform import batch_resample, to_mono

from utils.commons.seq_utils import seq_match

# from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import build_spk_mask_from_text_tokens
def build_spk_mask_from_text_tokens(text_tokens: torch.LongTensor, sx_patterns: dict):
    """
    仅针对 <Sx> 和 </Sx> 都是单 token 的情况做快速实现：
    - 全程 torch CPU
    - 用 diff + cumsum 构造区间并一次性赋值
    语义与原实现一致：包含 open/close token 本身。
    """
    if text_tokens is None or text_tokens.numel() == 0:
        return torch.zeros((0,), dtype=torch.long)

    # 确保在 CPU（你的 text_tokens 本来就是 CPU）
    t = text_tokens.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    L = t.numel()

    out = torch.zeros((L,), dtype=torch.long)

    # 固定顺序，和你原来 for tag, info in sx_patterns.items() 可能不同，
    # 但通常你希望 S1..S4 顺序确定；如需要“后写覆盖前写”，按这个顺序即可
    for spk_id in (1, 2, 3, 4):
        tag = f"S{spk_id}"
        info = sx_patterns.get(tag)
        if not info:
            continue

        open_ids = info["open"]
        close_ids = info["close"]
        if len(open_ids) != 1 or len(close_ids) != 1:
            # 不是单 token 的，回退到你原来的实现（或 rolling-hash 版）
            # 这里直接 continue 也行，但会丢功能
            continue

        open_id = int(open_ids[0])
        close_id = int(close_ids[0])

        open_pos = (t == open_id).nonzero(as_tuple=False).flatten()
        if open_pos.numel() == 0:
            continue
        close_pos = (t == close_id).nonzero(as_tuple=False).flatten()
        if close_pos.numel() == 0:
            continue

        # 为每个 open 找到后面最近的 close
        # 规则和你 numpy searchsorted 对齐：close >= open+1
        idx = torch.searchsorted(close_pos, open_pos + 1, right=False)
        valid = idx < close_pos.numel()
        if not torch.any(valid):
            continue

        s = open_pos[valid]
        e = close_pos[idx[valid]]  # inclusive

        # diff：长度 L+1，区间 [s, e] => diff[s]+=1, diff[e+1]-=1
        diff = torch.zeros((L + 1,), dtype=torch.int32)
        diff.index_add_(0, s, torch.ones_like(s, dtype=torch.int32))

        e1 = e + 1
        in_range = e1 <= L
        if torch.any(in_range):
            diff.index_add_(0, e1[in_range], -torch.ones_like(e1[in_range], dtype=torch.int32))

        inside = diff.cumsum(0)[:-1] > 0
        out[inside] = spk_id

    return out

# from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import _get_sx_token_patterns
def _get_sx_token_patterns(tokenizer):
    """
    预计算 <S1></S1> ... <S4></S4> 的 token pattern。
    兼容：这些 tag 可能是单 token，也可能被 BPE 拆成多 token。
    """
    patterns = {}
    for i in range(1, 5):
        tag = f"S{i}"
        patterns[tag] = {
            "open": _encode_tag_pattern(tokenizer, f"<{tag}>"),
            "close": _encode_tag_pattern(tokenizer, f"</{tag}>"),
            "id": i,
        }
    return patterns
def _encode_tag_pattern(tokenizer, s: str):
    """返回 tokenizer.encode(s) 的 token id 序列（list[int]）"""
    ids = tokenizer.encode(s)
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    return list(map(int, ids))


def splice_silence(
    wav: np.ndarray,
    sr: int = 24000,
    sil_sec: float = 0.5,
    mode: str = "both",  # "in" / "out" / "both"
) -> np.ndarray:
    """
    在音频前/后补静音（0），不做淡入淡出。
    - mode="in": 只在开头补
    - mode="out": 只在结尾补
    - mode="both": 两边都补
    """
    if wav is None:
        return wav
    wav = np.asarray(wav)
    if wav.ndim != 1 or wav.size == 0:
        return wav

    n = int(sr * float(sil_sec))
    if n <= 0:
        return wav.copy()

    zeros = np.zeros((n,), dtype=wav.dtype)

    head = zeros if mode in ("in", "both") else np.zeros((0,), dtype=wav.dtype)
    tail = zeros if mode in ("out", "both") else np.zeros((0,), dtype=wav.dtype)

    return np.concatenate([head, wav, tail], axis=0)

def flatten_grouped_samples(samples_cfg):
    """
    将分组格式的 samples 展平为 list，并写入：
      - __group__: 组名
      - __group_idx__: 组内序号（从 0 开始）
    """
    if not isinstance(samples_cfg, dict):
        return samples_cfg or []

    flat = []
    for group_name, group_list in samples_cfg.items():
        if not isinstance(group_list, (list, tuple)):
            continue
        for gi, sample in enumerate(group_list):
            if sample is None:
                sample = {}
            s = dict(sample)
            s["__group__"] = group_name
            s["__group_idx__"] = gi
            flat.append(s)
    return flat


def raw_text_process(txt, wav=None, wav_len=None, check_len=False):
    """
    训练侧同款清洗规则（简化 skip 打印），默认不做长度过滤。
    """
    if not isinstance(txt, str):
        return ""

    txt = txt.strip()
    if txt.startswith('sil '):
        txt = txt[4:]
    txt = txt.replace(' sil ', ' ')
    txt = remove_spaces_between_chinese(txt)

    if txt and txt[-1] not in '.,?!;。，？！；、':
        if is_chinese(txt):
            txt = txt + '。'
        else:
            txt = txt + '.'

    if wav is not None:
        wav_len = wav.shape[0]

    # 仅当你未来真的想做“推理侧长度保护”才打开
    if check_len and wav_len is not None:
        try:
            if len(get_word_list(txt)) > wav_len // hparams['hop_size'] // 4:
                return None
        except Exception:
            # hparams 未就绪/异常时不做过滤
            pass

    return txt



def _norm_spaces_caption(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u3000", " ")
    s = _SPACE_RE.sub(" ", s)
    return s.strip()

def extract_s_text(caption: str) -> str:
    """
    从 caption 中按顺序提取 <S1>...</S1> 和 <S2>...</S2>，
    并把相邻的同标签片段合并成一个 <S1>... ...</S1> 或 <S2>... ...</S2>。
    若没有任何 S1/S2 片段则返回 ""。
    返回结果带有 S1/S2 包裹的合并版本。
    """
    if not isinstance(caption, str) or not caption.strip():
        return ""

    segments = []
    for tag, inner in _S1S2_TAG_RE.findall(caption):
        inner_norm = _norm_spaces_caption(inner)
        if not inner_norm:
            continue
        segments.append([tag.upper(), inner_norm])

    if not segments:
        return ""

    merged = []
    for tag, content in segments:
        if merged and tag == merged[-1][0]:
            merged[-1][1] = merged[-1][1] + ' ' + content
        else:
            merged.append([tag, content])

    out = ''.join(f"<{tag}>{content}</{tag}>" for tag, content in merged)
    return out if out.strip() else ""

def raw_text_process_s1s2_tagged(txt, wav=None, wav_len=None, check_len=False):
    """
    对含 <S1>/<S2> 的 text 做 raw_text_process 等价清洗，
    只处理标签内部文本，保留标签结构与顺序。
    """
    if not isinstance(txt, str) or not txt.strip():
        return ""

    # 没标签就走普通清洗
    if _S1S2_TEXT_RE.search(txt) is None:
        return raw_text_process(txt, wav=wav, wav_len=wav_len, check_len=check_len)

    if wav is not None:
        wav_len = wav.shape[0]

    # 可选总长度检查（默认关闭）
    if check_len and wav_len is not None:
        plain_parts = []
        for m in _S1S2_TEXT_RE.finditer(txt):
            inner = _norm_spaces_caption(m.group(2))
            if inner:
                plain_parts.append(inner)
        plain = " ".join(plain_parts)

        if plain:
            plain_norm = raw_text_process(plain, wav=None, wav_len=None, check_len=False) or ""
            try:
                if len(get_word_list(plain_norm)) > wav_len // hparams['hop_size'] // 4:
                    return None
            except Exception:
                pass

    # 分段清洗并重建
    def _repl(m):
        tag = m.group(1).upper()
        inner = m.group(2)
        inner_proc = raw_text_process(inner, wav=None, wav_len=None, check_len=False)
        if inner_proc is None:
            inner_proc = ""
        return f"<{tag}>{inner_proc}</{tag}>"

    out = _S1S2_TEXT_RE.sub(_repl, txt)
    out = _SPACE_RE.sub(' ', out).strip()
    return out

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===== 文本清洗辅助正则（与数据处理侧对齐） =====
_SPACE_RE = re.compile(r"\s+")

_S1S2_TAG_RE = re.compile(
    r'<\s*(S[1-4])\s*>(.*?)</\s*S[1-4]\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)

_S1S2_TEXT_RE = re.compile(
    r'<\s*(S[1-4])\s*>(.*?)</\s*\1\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)


def _num_string_to_price_zh(num_str: str) -> str:
    ns = (num_str or "").strip()
    if not ns:
        return ""

    # 已经是中文价格读法时保持原样，避免重复处理。
    if not re.search(r"\d", ns):
        return ns

    ns = ns.replace("．", ".").replace("。", ".")
    ns = re.sub(r"[,\s]", "", ns)

    if not re.fullmatch(r"\d+(?:\.\d+)?", ns):
        return num_str

    if "." in ns:
        int_part, frac_part = ns.split(".", 1)
        int_part = int_part or "0"
        frac_part = frac_part.rstrip("0")

        zh_int = num2chn(int_part, alt_two=False)
        if not frac_part:
            return zh_int

        if len(frac_part) == 1:
            return f"{zh_int}块{num2chn(frac_part, use_units=False, alt_two=False)}"

        mao_digit = frac_part[0]
        tail_digits = frac_part[1:]
        zh_tail = "".join(
            num2chn(d, use_units=False, alt_two=False) for d in tail_digits
        )
        if mao_digit == "0":
            return f"{zh_int}块零{zh_tail}"
        if len(tail_digits) == 1 and tail_digits == "0":
            return f"{zh_int}块{num2chn(mao_digit, use_units=False, alt_two=False)}毛"
        return (
            f"{zh_int}块"
            f"{num2chn(mao_digit, use_units=False, alt_two=False)}毛"
            f"{zh_tail}"
        )

    return num2chn(ns, alt_two=False)

def _fmt_mb(x: int) -> str:
    return f"{x / 1024**2:,.1f} MB"

def _cuda_snapshot(device=None):
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.cuda.current_device()
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(idx)
    free, total = torch.cuda.mem_get_info(idx)  # driver 视角 free/total
    alloc = torch.cuda.memory_allocated(idx)    # pytorch 实际分配
    reserved = torch.cuda.memory_reserved(idx)  # pytorch 缓存池占用
    max_alloc = torch.cuda.max_memory_allocated(idx)
    max_reserved = torch.cuda.max_memory_reserved(idx)
    return {
        "idx": idx,
        "free": free, "total": total,
        "alloc": alloc, "reserved": reserved,
        "max_alloc": max_alloc, "max_reserved": max_reserved,
    }

def log_cuda_mem(tag: str, device=None, prev=None):
    snap = _cuda_snapshot(device)
    if snap is None:
        print(f"[MEM][CPU] {tag} (cuda not available)")
        return snap

    line = (
        f"[MEM][CUDA:{snap['idx']}] {tag} | "
        f"free {_fmt_mb(snap['free'])} / total {_fmt_mb(snap['total'])} | "
        f"alloc {_fmt_mb(snap['alloc'])} | reserved {_fmt_mb(snap['reserved'])} | "
        f"max_alloc {_fmt_mb(snap['max_alloc'])} | max_reserved {_fmt_mb(snap['max_reserved'])}"
    )
    if prev is not None:
        da = snap["alloc"] - prev["alloc"]
        dr = snap["reserved"] - prev["reserved"]
        df = snap["free"] - prev["free"]
        line += f" | Δalloc {_fmt_mb(da)} Δreserved {_fmt_mb(dr)} Δfree {_fmt_mb(df)}"
    print(line)
    return snap

def module_footprint_bytes(m: torch.nn.Module):
    # 参数 + buffer 的字节数（按当前 dtype 计算）
    p_bytes = 0
    b_bytes = 0
    for p in m.parameters(recurse=True):
        p_bytes += p.numel() * p.element_size()
    for b in m.buffers(recurse=True):
        b_bytes += b.numel() * b.element_size()
    n_params = sum(p.numel() for p in m.parameters(recurse=True))
    dtype = None
    try:
        dtype = next(m.parameters()).dtype
    except StopIteration:
        dtype = None
    return p_bytes, b_bytes, n_params, dtype

def print_module_mem(name: str, m: torch.nn.Module):
    p_bytes, b_bytes, n_params, dtype = module_footprint_bytes(m)
    print(
        f"[MODULE] {name}: params={n_params/1e6:.2f}M "
        f"dtype={dtype} param={_fmt_mb(p_bytes)} buffer={_fmt_mb(b_bytes)} total={_fmt_mb(p_bytes+b_bytes)}"
    )

from dataclasses import dataclass, field


@dataclass
class SpeechEditResult:
    method: str
    wav_path: Optional[str] = None
    out_path: Optional[str] = None
    out_file: Optional[str] = None
    output_wav_path: Optional[str] = None

    sample_rate: int = 24000

    whole_text_src: Optional[str] = None
    whole_text_tgt: Optional[str] = None

    # 每项表示一个最终实际编辑段；如有 anchor 扩展或字段合并，写入最终结果。
    edits: List[Dict[str, Any]] = field(default_factory=list)

    # 预留给全局调试信息，例如原始 spans、merged_mask_spans 等。
    debug: Dict[str, Any] = field(default_factory=dict)

    def add_edit(
        self,
        text_src: Optional[str] = None,
        text_tgt: Optional[str] = None,
        time_tag: Optional[Tuple[float, float]] = None,
        time_tag_samples: Optional[Tuple[int, int]] = None,
        merged: bool = False,
        merged_fields: Optional[List[Dict[str, Any]]] = None,
        merge_reason: Optional[str] = None,
        anchor_expand: Optional[Dict[str, int]] = None,
        anchor_conf_before: Optional[Dict[str, float]] = None,
        anchor_conf_after: Optional[Dict[str, float]] = None,
        anchor_text_expanded: Optional[str] = None,
        text_replaced: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        item = {
            "text_src": text_src,
            "text_tgt": text_tgt,
            "time_tag": time_tag,
            "time_tag_samples": time_tag_samples,
            "merged": merged,
            "merged_fields": merged_fields or [],
            "merge_reason": merge_reason,
            "anchor_expand": anchor_expand,
            "anchor_conf_before": anchor_conf_before,
            "anchor_conf_after": anchor_conf_after,
            "anchor_text_expanded": anchor_text_expanded,
            "text_replaced": text_replaced,
        }
        if extra:
            item.update(extra)
        self.edits.append(item)


def _merge_reasons_to_text(reasons) -> Optional[str]:
    if not reasons:
        return None
    uniq = []
    for reason in reasons:
        if reason and reason not in uniq:
            uniq.append(reason)
    return ",".join(uniq) if uniq else None


def _build_step_debug_lookup(span_debug: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    lookup = {}
    for item in span_debug:
        if "step" in item:
            lookup[int(item["step"])] = item
    return lookup


def _append_result_edits_from_merged_spans(
    result: SpeechEditResult,
    merged_spans: List[Dict[str, Any]],
    span_debug: List[Dict[str, Any]],
    text_src: List[str],
    text_tgt: List[str],
    sample_rate: int,
) -> None:
    step_debug_lookup = _build_step_debug_lookup(span_debug)
    for span in merged_spans:
        steps = list(span.get("steps", []))
        merged_fields = []
        for step in steps:
            base_item = {
                "step": step,
                "text_src": text_src[step] if step < len(text_src) else None,
                "text_tgt": text_tgt[step] if step < len(text_tgt) else None,
            }
            debug_item = step_debug_lookup.get(step)
            if debug_item is not None:
                field_item = dict(debug_item)
                field_item.setdefault("text_src", base_item["text_src"])
                field_item.setdefault("text_tgt", base_item["text_tgt"])
                field_item.setdefault("step", step)
            else:
                field_item = base_item
            merged_fields.append(field_item)

        text_src_item = None
        text_tgt_item = None
        anchor_expand = None
        anchor_conf_before = None
        anchor_conf_after = None
        anchor_text_expanded = None
        text_replaced = None

        if len(steps) == 1:
            step = steps[0]
            debug_item = step_debug_lookup.get(step, {})
            text_src_item = debug_item.get("text_src", text_src[step] if step < len(text_src) else None)
            text_tgt_item = debug_item.get("text_tgt", text_tgt[step] if step < len(text_tgt) else None)
            anchor_expand = debug_item.get("anchor_expand")
            anchor_conf_before = debug_item.get("anchor_conf_before")
            anchor_conf_after = debug_item.get("anchor_conf_after")
            anchor_text_expanded = debug_item.get("anchor_text_expanded")
            text_replaced = text_tgt_item

        start_sample = int(span["start_sample"])
        end_sample = int(span["end_sample"])
        result.add_edit(
            text_src=text_src_item,
            text_tgt=text_tgt_item,
            time_tag=(start_sample / sample_rate, end_sample / sample_rate),
            time_tag_samples=(start_sample, end_sample),
            merged=len(steps) > 1,
            merged_fields=merged_fields,
            merge_reason=_merge_reasons_to_text(span.get("merge_reasons")),
            anchor_expand=anchor_expand,
            anchor_conf_before=anchor_conf_before,
            anchor_conf_after=anchor_conf_after,
            anchor_text_expanded=anchor_text_expanded,
            text_replaced=text_replaced,
            extra={"steps": steps},
        )


def parse_seq(whole_text_src, whole_text_tgt):
    whole_text_src = raw_text_process_s1s2_tagged(_norm_spaces_caption(whole_text_src))
    whole_text_tgt = raw_text_process_s1s2_tagged(_norm_spaces_caption(whole_text_tgt))

    if not whole_text_src and not whole_text_tgt:
        return [], [], ""
    if not whole_text_src or not whole_text_tgt:
        # Pure insertion/deletion: caller should decide how to handle it.
        return [whole_text_src], [whole_text_tgt], whole_text_tgt

    # Character-level alignment. This is an aligner, not a diff; we post-process the track into edit spans.
    _, track = seq_match(whole_text_src, whole_text_tgt, score_fn=lambda a, b: a == b, metric="mean", priority=0)

    # Build per-step ops from the alignment track.
    # We follow the same interpretation as print_seq_match(): each coord consumes either (src,tgt), (src, gap), or (gap,tgt).
    ops = []  # list[(src_char|None, tgt_char|None, kind)]
    prev_i, prev_j = -1, -1
    for i, j in track:
        di, dj = int(i) - int(prev_i), int(j) - int(prev_j)
        if di == 1 and dj == 1:
            a = whole_text_src[int(i)]
            b = whole_text_tgt[int(j)]
            kind = "eq" if a == b else "rep"
        elif di == 1 and dj == 0:
            a = whole_text_src[int(i)]
            b = None
            kind = "del"
        elif di == 0 and dj == 1:
            a = None
            b = whole_text_tgt[int(j)]
            kind = "ins"
        else:
            raise RuntimeError(f"unexpected alignment step: {(prev_i, prev_j)} -> {(i, j)}")
        ops.append((a, b, kind))
        prev_i, prev_j = int(i), int(j)

    # Group consecutive non-equal ops into edit hunks.
    hunks = []
    h_start = None
    for idx, (_, _, kind) in enumerate(ops):
        if kind != "eq":
            if h_start is None:
                h_start = idx
        else:
            if h_start is not None:
                hunks.append((h_start, idx - 1))
                h_start = None
    if h_start is not None:
        hunks.append((h_start, len(ops) - 1))

    # Merge nearby hunks separated by a short equal-gap, which is more suitable for speech editing.
    # Rule:
    # - merge if the equal-gap length <= 2
    # - do not merge across strong boundary punctuations
    # - allow a slightly larger gap (<= 3) for number/price-related edits
    # - avoid producing overly long fields (max length <= 12)
    _BOUNDARY_PUNCT = set("，。！？；：,.!?;:")
    _NUMERIC_CHARS = set("零一二三四五六七八九十百千万亿块毛分点")

    def _has_numeric(s: str) -> bool:
        if not s:
            return False
        for ch in s:
            if ch.isdigit() or ch in _NUMERIC_CHARS:
                return True
        return False

    def _hunk_parts(hs, he):
        sp = "".join(ch for ch, _, _ in ops[hs:he + 1] if ch is not None)
        tp = "".join(ch for _, ch, _ in ops[hs:he + 1] if ch is not None)
        return sp, tp

    merged_hunks = []
    if hunks:
        cur_s, cur_e = hunks[0]
        for next_s, next_e in hunks[1:]:
            gap_ops = ops[cur_e + 1:next_s]
            gap_eq_len = sum(1 for _, _, k in gap_ops if k == "eq")
            gap_text = "".join(ch for ch, _, _ in gap_ops if ch is not None)

            cur_src, cur_tgt = _hunk_parts(cur_s, cur_e)
            next_src, next_tgt = _hunk_parts(next_s, next_e)

            # Do not merge across strong punctuation boundaries.
            crosses_boundary = any(ch in _BOUNDARY_PUNCT for ch in gap_text)

            # Relax the gap threshold for number/price-related edits.
            numeric_related = _has_numeric(cur_src + cur_tgt + gap_text + next_src + next_tgt)
            gap_th = 3 if numeric_related else 2

            # Compute merged lengths if we were to merge.
            merged_src, merged_tgt = _hunk_parts(cur_s, next_e)
            too_long = max(len(merged_src), len(merged_tgt)) > 12

            if (not crosses_boundary) and (gap_eq_len <= gap_th) and (not too_long):
                cur_e = next_e
            else:
                merged_hunks.append((cur_s, cur_e))
                cur_s, cur_e = next_s, next_e
        merged_hunks.append((cur_s, cur_e))
        hunks = merged_hunks

    edit_items = []

    for s, e in hunks:
        src_part = "".join(ch for ch, _, _ in ops[s:e + 1] if ch is not None)
        tgt_part = "".join(ch for _, ch, _ in ops[s:e + 1] if ch is not None)

        # For insertion/deletion-only hunks, one side can be empty. Anchor with a neighboring aligned character
        # so downstream callers (e.g., edit-by-replace) have a non-empty src/tgt span.
        if src_part == "" or tgt_part == "":
            left_anchor = ops[s - 1] if s - 1 >= 0 and ops[s - 1][2] == "eq" else None
            right_anchor = ops[e + 1] if e + 1 < len(ops) and ops[e + 1][2] == "eq" else None

            # Prefer left anchor unless it is punctuation; in that case, use the right anchor when available.
            prefer_right_anchor = False
            if left_anchor is not None and s > 0:
                left_anchor_ch = left_anchor[0] or ""
                if left_anchor_ch in _BOUNDARY_PUNCT and right_anchor is not None:
                    prefer_right_anchor = True

            if left_anchor is not None and s > 0 and not prefer_right_anchor:
                anchor_ch = left_anchor[0]
                src_part = (anchor_ch or "") + src_part
                tgt_part = (anchor_ch or "") + tgt_part
            elif right_anchor is not None:
                anchor_ch = right_anchor[0]
                src_part = src_part + (anchor_ch or "")
                tgt_part = tgt_part + (anchor_ch or "")

        # As a last resort, fall back to the full strings (still a meaningful "replace").
        if src_part == "" or tgt_part == "":
            src_part = whole_text_src
            tgt_part = whole_text_tgt

        if src_part == tgt_part:
            continue

        edit_items.append({
            "text_src": src_part,
            "text_tgt": tgt_part,
        })

    text_src = []
    text_tgt = []
    filtered_whole_text_tgt = whole_text_tgt
    cursor = 0

    for item in edit_items:
        src_part = item["text_src"]
        tgt_part = item["text_tgt"]

        if abs(len(src_part) - len(tgt_part)) > 3:
            pos = filtered_whole_text_tgt.find(tgt_part, cursor)
            if pos >= 0:
                filtered_whole_text_tgt = (
                    filtered_whole_text_tgt[:pos] +
                    src_part +
                    filtered_whole_text_tgt[pos + len(tgt_part):]
                )
                cursor = pos + len(src_part)
            print(
                f"[WARNING] parse_seq 忽略字段：text_src={src_part!r}, text_tgt={tgt_part!r}，"
                f"字数差为 {abs(len(src_part) - len(tgt_part))}，超过 3。"
            )
            continue

        pos = filtered_whole_text_tgt.find(tgt_part, cursor)
        if pos >= 0:
            cursor = pos + len(tgt_part)

        text_src.append(src_part)
        text_tgt.append(tgt_part)

    print(f"[parse_seq] {text_src =}, {text_tgt =}")
    return text_src, text_tgt, filtered_whole_text_tgt


def _space_uppercase_letters_in_targets(whole_text_tgt, text_tgt_list):
    def _space_one(text):
        if re.fullmatch(r"[A-Z]+", text):
            return " ".join(list(text))
        return text

    new_text_tgt_list = []
    new_whole_text_tgt = whole_text_tgt
    for tgt in text_tgt_list:
        spaced_tgt = _space_one(tgt)
        new_text_tgt_list.append(spaced_tgt)
        if spaced_tgt != tgt:
            new_whole_text_tgt = new_whole_text_tgt.replace(tgt, spaced_tgt)
    return new_whole_text_tgt, new_text_tgt_list


class SpeechEditInfer(DiTBuildModelMixin):
    def __init__(self, device, dit_ckpt,
                vae_ckpt=None,
                merge_ckpt=None, merge_weight=0.5,
                use_sa_front: bool = False,
                g2p_model: str = 'qwen'):
        self.device = device
        self.precision = torch.bfloat16
        self.build_model(
            dit_ckpt,
            vae_ckpt=vae_ckpt,
            merge_ckpt=merge_ckpt,
            merge_weight=merge_weight,
        )

        self.sr = 24000

    
    def build_model(self, dit_ckpt,
                    vae_ckpt=None,
                    merge_ckpt=None, merge_weight=0.5,):
        self.asr_model = None

        # 建议：每次 build 前清一下峰值统计，方便看 max_alloc
        # if torch.cuda.is_available():
        #     idx = torch.device(self.device).index if torch.device(self.device).index is not None else torch.cuda.current_device()
        #     torch.cuda.reset_peak_memory_stats(idx)

        snap = log_cuda_mem("enter build_model()", self.device)

        # ====== hparams & config ======
        if os.path.isfile(dit_ckpt):
            set_hparams(config=os.path.join(os.path.dirname(dit_ckpt), 'config.yaml'),
                        print_hparams=False, global_hparams=True)
        else:
            set_hparams(config=os.path.join(dit_ckpt, 'config.yaml'),
                        print_hparams=False, global_hparams=True)
        hparams["exp_name"] = 'infer'
        self.config = AttrDict(hparams)

        snap = log_cuda_mem("after set_hparams()", self.device, prev=snap)

        # ====== VAE & audio tokenizer ======
        vae_ckpt_path = vae_ckpt or hparams.get('vae_ckpt')
        self.vae, self.hp_vae = build_vae(vae_ckpt_path)
        print_module_mem("VAE (on CPU)", self.vae)
        snap = log_cuda_mem("after build_vae() (still CPU)", self.device, prev=snap)

        # 搬到 GPU
        self.vae.to(self.device)
        print_module_mem("VAE (on GPU)", self.vae)
        snap = log_cuda_mem("after vae.to(device)", self.device, prev=snap)

        # ====== DiT & 文本 tokenizer ======
        self.dit_text_tokenizer, self.dit_vocab_size = self.build_dit_text_tokenizer()
        self._sx_patterns = _get_sx_token_patterns(self.dit_text_tokenizer)
        snap = log_cuda_mem("after build_dit_text_tokenizer()", self.device, prev=snap)

        self.dit = self.build_dit(hparams)
        print_module_mem("DiT (on CPU, before load_ckpt)", self.dit)
        snap = log_cuda_mem("after build_dit() (still CPU)", self.device, prev=snap)

        load_ckpt(self.dit, dit_ckpt, 'dit', strict=False)
        print_module_mem("DiT (on CPU, after load_ckpt)", self.dit)
        snap = log_cuda_mem("after load_ckpt(dit) (still CPU)", self.device, prev=snap)

        if merge_ckpt is not None:
            self.dit = merge_model_weights(self.dit, merge_ckpt,
                                        ignore_module=['cross', 'caption_proj'],
                                        weight=merge_weight)
            print_module_mem("DiT (on CPU, after merge)", self.dit)
            snap = log_cuda_mem("after merge_model_weights() (still CPU)", self.device, prev=snap)

        self.vae.eval(); self.vae.to(self.device, dtype=self.precision)
        self.dit.eval(); self.dit.to(self.device, dtype=self.precision)
        print_module_mem("DiT (on GPU)", self.dit)
        snap = log_cuda_mem("after dit.to(device)", self.device, prev=snap)

        # ====== caption 相关 encoder ======
        self.use_caption = hparams.get('use_caption', False)
        print(f"[INFO] use_caption={self.use_caption}, model_size={hparams.get('model_size', 'base')}")

        if self.use_caption:
            raise NotImplementedError("use_caption is not supported")
        else:
            self.sd_text_encoder = None
            self.goku_text_encoder = None

        log_cuda_mem("leave build_model()", self.device, prev=snap)
    

    def _tokenize_dit_text(self, text: str):
        """
        兼容 HuggingFace tokenizer 和 CosyVoice2Tokenizer：
        - HF: BatchEncoding 支持 .to(self.device)
        - Cosy: 返回 dict，没有 .to 方法，需要手动搬 tensor
        """

        if isinstance(self.dit_text_tokenizer, CosyVoice2Tokenizer):
            # CosyVoice2Tokenizer 分支（参考你给的 process_text_seg）
            text_inputs = self.dit_text_tokenizer(
                text,
                padding=True,
                return_tensors='pt',
            )
            txt_tokens = text_inputs['input_ids'].to(self.device)
            txt_mask = text_inputs['attention_mask'].bool().to(self.device)
        else:
            # 原来的 Qwen / HF 分支
            text_inputs = self.dit_text_tokenizer(
                text,
                padding=True,
                return_tensors='pt',
            ).to(self.device)
            txt_tokens = text_inputs['input_ids'].clone()
            txt_mask = text_inputs['attention_mask'].bool()

        # 把 padding 位置替换成 cfg 用的 mask token
        txt_tokens[~txt_mask] = self.cfg_mask_text_token
        txt_lens = txt_mask.long().sum(-1)

        # ====== spk_mask: [B, T], 0/1/2/3/4 ======
        B, T = txt_tokens.shape
        masks = []
        for b in range(B):
            L = int(txt_lens[b].item())
            # 只在有效 token 范围内找 tag，避免 pad 区域误匹配
            m = build_spk_mask_from_text_tokens(txt_tokens[b, :L].detach().cpu(), self._sx_patterns)
            if L < T:
                m = torch.cat([m, torch.zeros((T - L,), dtype=torch.long)], dim=0)
            masks.append(m)
        spk_mask = torch.stack(masks, dim=0).to(self.device)

        return txt_tokens, txt_mask, txt_lens, spk_mask


    @torch.no_grad()
    def forward(self,
                text,
                ref_audio=None,
                ref_text=None,
                prompt=None,          # 推理时唯一的 caption（语义上：有 ref 时是 local，无 ref 时是 full）
                cfg_w=None,
                negative_prompt=None, # 仍然保留接口，但实际不用
                infer_length=5.0,
                start_time=0.2,
                end_time=0.2,
                num_step=50,
                timestep_annealing_w=(1.0,0.0,1.0),
                use_amo_sampler=False,
                use_sway=False,
                cpm=300.0,
                gen_wav_start=None,
                gen_wav_end=None,
                overlap_time: tuple=None,
                extend_mask: float=1.0):
        
        '''
        必须得有ref_audio，假设这里已经是mask过的ref_audio
        overlap_time: 重叠时间，单位秒。注意边界检查。现在overlap左右连接不太连贯。如果overlap_time=None，不进行overlap。
        想法是在加了overlap的wav上预测，然后把没有overlap的

        - 由于case的字数变化较大，缩放一下音频的被mask的区域，避免时间太短太长
        '''
        assert ref_audio is not None, "ref_audio is None"
        assert gen_wav_start is not None, "gen_wav_start is None"
        assert gen_wav_end is not None, "gen_wav_end is None"

        # 扩展mask区域
        if extend_mask != 1.0:
            duration = gen_wav_end - gen_wav_start
            scale_end = int(gen_wav_start + duration * extend_mask)
            delta = scale_end - gen_wav_end

            if delta > 0:
                pad = np.zeros((delta,), dtype=ref_audio.dtype)
                ref_audio = np.concatenate([ref_audio[:gen_wav_end], pad, ref_audio[gen_wav_end:]], axis=0)
            elif delta < 0:
                cut_end = scale_end
                ref_audio = np.concatenate([ref_audio[:cut_end], ref_audio[gen_wav_end:]], axis=0)
            gen_wav_end = scale_end
            # 确保编辑段为0
            ref_audio[gen_wav_start:gen_wav_end] = 0.0


        exact_ref_audio = ref_audio.copy()
        exact_start_time = gen_wav_start
        exact_end_time = gen_wav_end
        if overlap_time is not None:
            overlap_start, overlap_end = overlap_time
            overlap_start = int(overlap_start * self.sr)
            overlap_end = int(overlap_end * self.sr)

            gen_wav_start = max(0, gen_wav_start - overlap_start)
            gen_wav_end = min(ref_audio.shape[0], gen_wav_end + overlap_end)

            ref_audio[gen_wav_start:gen_wav_end] = 0.0
            

        speech = len(text) > 0
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']

        ref_wav = torch.from_numpy(
            np.concatenate([ref_audio, np.zeros(0, dtype=np.float16)])
        )[None, :].to(self.device)

        ref_wav_lens = torch.LongTensor(
            [ref_wav.shape[1] // fm_wav * fm_wav]
        ).to(self.device)
        ref_wav = ref_wav[:, :ref_wav_lens[0]]

        if overlap_time is not None:
            exact_ref_wav = torch.from_numpy(
                np.concatenate([exact_ref_audio, np.zeros(0, dtype=np.float16)])
            )[None, :].to(self.device)
            exact_ref_wav = exact_ref_wav[:, :ref_wav_lens[0]]

        if ref_text is None:
            '''
            暂时一定要给出ref_text
            '''
            raise ValueError("ref_text is None")
        
        # 需要的推理时长是确定的
        # infer_length ? 这个原本是秒数时长，拿来生成gen_lat_len的
        # 新生成的 latent 长度应该是确定的
        # gen_lat_len ?

        # 没有caption
        caption = prompt
        # bgm_flag_val = _extract_bgm_flag_from_caption(caption)
        bgm_flag_val = 0

        use_caption = getattr(self, "use_caption", False) and (caption is not None)
        caption_emb = None
        caption_lens = None
        caption_text_mark = None

        # ====== 推断 latent_dim ======
        latent_dim = None
        if hasattr(self.vae, "latent_dim"):
            latent_dim = getattr(self.vae, "latent_dim")
        if latent_dim is None and hasattr(self.dit, "hp") and hasattr(self.dit.hp, "in_channels"):
            latent_dim = self.dit.hp.in_channels
        if latent_dim is None:
            latent_dim = 32  # 兜底

        if hasattr(self.vae, "hop_size"):
            hop_size = getattr(self.vae, "hop_size")
        else:
            hop_size = 240
        if hasattr(self.vae, "vae_stride"):
            vae_stride = getattr(self.vae, "vae_stride")
        else:
            vae_stride = 4

        mel_len = ref_wav.shape[-1] // hop_size
        # ====== VAE 编码参考音频，构造 lat_ctx / ctx_mask ======
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=self.precision):
                lat_ctx_ref = self.vae.encode_latent(ref_wav)  # [1, L_ref, C]
                if overlap_time is not None:
                    lat_ctx_exact_ref = self.vae.encode_latent(exact_ref_wav)  # [1, L_ref, C]
            latent_dim = lat_ctx_ref.size(-1)

            lat_len = lat_ctx_ref.size(1)
            tgt_len = lat_len

            # ctx_mask_ref = torch.ones_like(lat_ctx_ref[:, :, 0:1])

            ### 这里把lat 挖空的部分设为0
            lat = lat_ctx_ref
            # ctx_mask = ctx_mask_ref
            fm = hparams['frames_multiple']
            gen_start = int(gen_wav_start / hop_size) 
            gen_end = int(gen_wav_end / hop_size)
            gen_start = (gen_start // int(fm)) * int(fm)
            gen_end = (gen_end // int(fm)) * int(fm)

            ctx_mask_mel = torch.ones((int(mel_len), 1), dtype=torch.float32)
            ctx_mask_mel[gen_start:gen_end] = 0.0
            ctx_mask = ctx_mask_mel[::vae_stride]
            ctx_mask = ctx_mask.unsqueeze(0).to(self.device)

            if overlap_time is not None:
                exact_start = int(exact_start_time / hop_size)
                exact_end = int(exact_end_time / hop_size)
                exact_start = (exact_start // int(fm)) * int(fm)
                exact_end = (exact_end // int(fm)) * int(fm)
                exact_ctx_mask_mel = torch.ones((int(mel_len), 1), dtype=torch.float32)
                exact_ctx_mask_mel[exact_start:exact_end] = 0.0
                exact_ctx_mask = exact_ctx_mask_mel[::vae_stride]
                exact_ctx_mask = exact_ctx_mask.unsqueeze(0).to(self.device)


            # print(f"ctx_mask: {ctx_mask}")
            # print(f"ctx_mask.shape: {ctx_mask.shape}")
            # print(f"ctx_mask_mel: {ctx_mask_mel}")
            # import pdb; pdb.set_trace()


            ### full_text 应该是什么呢？
            ## 就是输入的完整的文本text，这里是修改过后的text
            full_text = text

        # 文本 token
        txt_tokens, txt_mask, txt_lens, spk_mask = self._tokenize_dit_text(full_text)

        # ====== VAD mask（只当作条件传进去）======
        vad_mask = torch.zeros_like(lat[:, :, :1])
        if not self.config.get('drop_vad', False) and speech:
            vad_mask[:, int(start_time * 25):-int(end_time * 25)] = 1.0

        # 3 路 CFG：在 batch 维度上复制 VAD
        vad_mask = torch.cat([vad_mask] * 3, dim=0)

        # ====== 文本 CFG======
        txt_tokens = torch.cat([
            txt_tokens,
            txt_tokens,
            torch.full_like(txt_tokens, self.cfg_mask_text_token),
        ], dim=0)
        txt_mask = torch.cat([txt_mask] * 3, dim=0)
        txt_lens = torch.cat([txt_lens] * 3, dim=0)

        if spk_mask is not None:
            spk_mask = torch.cat([
                spk_mask,                          # 路1：真实文本
                spk_mask,                          # 路2：真实文本
                torch.zeros_like(spk_mask),        # 路3：文本全 mask
            ], dim=0)

        # ====== latent / ctx_mask 也复制 3 路 ======
        lat = torch.cat([
            lat,
            torch.zeros_like(lat),
            torch.zeros_like(lat),
        ], dim=0)
        ctx_mask = torch.cat([ctx_mask] * 3, dim=0)

        batch_size = lat.shape[0]
        # ====== 组装 Diffusion.inference 所需 inputs ======
        inputs = {
            'txt_tokens': txt_tokens,
            'spk_mask': spk_mask,
            'txt_mask': txt_mask,
            'txt_lens': txt_lens,
            'ctx_mask': ctx_mask,
            'lat_ctx': lat,
            'caption_emb': caption_emb,
            'caption_lens': caption_lens,
            'caption_text_mark': caption_text_mark,
            'vad_mask': vad_mask,
            'tgt_len': torch.full(
                (batch_size,),
                tgt_len,
                dtype=torch.long,
                device=self.device,
            ),
        }
        if hasattr(self.dit, 'hp') and bool(getattr(self.dit.hp, 'use_bgm_flag', False)):
            if batch_size % 3 == 0:
                b0 = batch_size // 3
                base = torch.full(
                    (b0,),
                    int(bgm_flag_val),
                    dtype=torch.long,
                    device=self.device,
                )
                inputs['bgm_flag'] = torch.cat([
                    base,
                    torch.full_like(base, 2),
                    torch.full_like(base, 2),
                ], dim=0)
            else:
                inputs['bgm_flag'] = torch.full(
                    (batch_size,),
                    int(bgm_flag_val),
                    dtype=torch.long,
                    device=self.device,
                )

        # ====== 调 Diffusion.inference 生成 latent，再用 VAE decode 回波形 ======
        with torch.autocast(device_type='cuda', dtype=self.precision):
            x = self.dit.inference(inputs, timesteps=num_step, seq_cfg_w=cfg_w, timestep_annealing_w=timestep_annealing_w, use_amo_sampler=use_amo_sampler, use_sway=use_sway)

            # 把前面 ref 的部分替换为真实 prompt latent
            # if lat_ctx_ref.shape[1] > 0:
            #     x[:, :lat_ctx_ref.shape[1]] = lat_ctx_ref

            hop_size = self.hp_vae['hop_size']
            vae_stride = self.hp_vae['vae_stride']

            ref_lat = lat_ctx_ref.size(1) if lat_ctx_ref is not None else 0
            # gen_lat = ?

            # 0.5 秒 overlap -> latent 帧数
            overlap_sec = 0.5
            overlap_lat = int(overlap_sec * 24000 / hop_size / vae_stride)

            # 原来：只解码 [ref_lat - overlap_lat, ref_lat + gen_lat]
            # 新：解码所有
            # start_lat = 0
            # end_lat = ref_lat
            # x_dec = x # [1, L_dec, C]
            if overlap_time is None:
                x_dec = lat_ctx_ref * ctx_mask + x * (1 - ctx_mask)
            else:
                print(f"使用overlap_time: {overlap_time}")
                blend_mask_mel = torch.ones((int(mel_len), 1), dtype=torch.float32)
 
                exact_start_mel = int(exact_start_time / hop_size)
                exact_end_mel = int(exact_end_time / hop_size)
 
                gs = max(0, min(int(gen_start), int(mel_len)))
                ge = max(0, min(int(gen_end), int(mel_len)))
                es = max(gs, min(int(exact_start_mel), ge))
                ee = max(es, min(int(exact_end_mel), ge))
 
                blend_mask_mel[es:ee] = 0.0
 
                if es > gs:
                    n = es - gs
                    blend_mask_mel[gs:es, 0] = torch.linspace(1.0, 0.0, steps=n, dtype=torch.float32)
 
                if ge > ee:
                    n = ge - ee
                    blend_mask_mel[ee:ge, 0] = torch.linspace(0.0, 1.0, steps=n, dtype=torch.float32)
 
                blend_mask = blend_mask_mel[::vae_stride].unsqueeze(0).to(self.device)
                if blend_mask.shape[1] > lat_ctx_exact_ref.shape[1]:
                    blend_mask = blend_mask[:, :lat_ctx_exact_ref.shape[1]]
                elif blend_mask.shape[1] < lat_ctx_exact_ref.shape[1]:
                    pad = torch.ones((1, lat_ctx_exact_ref.shape[1] - blend_mask.shape[1], 1), device=blend_mask.device, dtype=blend_mask.dtype)
                    blend_mask = torch.cat([blend_mask, pad], dim=1)
 
                x_dec = lat_ctx_exact_ref * blend_mask + x * (1 - blend_mask)
            # import pdb; pdb.set_trace()

            # 解码
            with torch.autocast(device_type='cuda', dtype=self.precision):
                wav_dec = self.vae.decode(x_dec)[0, 0].to(torch.float32)

            # 需要在整体的wav前后加overlap吗？
            # 丢掉 overlap 对应的 wav（以及 ref 部分）
            # drop_lat = ref_lat - start_lat               # <= overlap_lat
            # drop_wav = drop_lat * vae_stride * hop_size  # samples
            # wav_pred = wav_dec[drop_wav:]                # 现在基本就是 “gen wav”
            wav_pred = wav_dec

            # clip 防止溢出
            if wav_pred.abs().max() > 1:
                print('Wav amplitude exceed 1, clip it.')
                wav_pred = wav_pred / (wav_pred.abs().max())

            wav_pred = wav_pred.cpu().numpy()

            # wav_pred = splice_silence(wav_pred, sr=24000, sil_sec=0.2, mode="both")

        return wav_pred


    @torch.no_grad()
    def forward_multimask(
        self,
        text,
        ref_audio=None,
        ref_text=None,
        prompt=None,          # 推理时唯一的 caption（语义上：有 ref 时是 local，无 ref 时是 full）
        cfg_w=None,
        negative_prompt=None, # 仍然保留接口，但实际不用
        infer_length=5.0,
        start_time=0.2,
        end_time=0.2,
        num_step=50,
        timestep_annealing_w=(1.0,0.0,1.0),
        use_amo_sampler=False,
        use_sway=False,
        cpm=300.0,
        gen_wav_start=None,
        gen_wav_end=None,
        overlap_time: tuple=None,
        extend_mask: float=1.0):
        
        '''
        支持多段 mask 的 ref_audio 推理。
        注意：这里保留 extend_mask 参数接口，但不实现 extend_mask 的逻辑，不改音频总长度。
        '''
        assert ref_audio is not None, "ref_audio is None"
        assert gen_wav_start is not None, "gen_wav_start is None"
        assert gen_wav_end is not None, "gen_wav_end is None"

        if isinstance(gen_wav_start, (list, tuple, np.ndarray)):
            gen_wav_start_lst = [int(x) for x in gen_wav_start]
        else:
            gen_wav_start_lst = [int(gen_wav_start)]
        if isinstance(gen_wav_end, (list, tuple, np.ndarray)):
            gen_wav_end_lst = [int(x) for x in gen_wav_end]
        else:
            gen_wav_end_lst = [int(gen_wav_end)]

        if len(gen_wav_start_lst) != len(gen_wav_end_lst):
            raise ValueError(f"gen_wav_start 和 gen_wav_end 长度不一致: {len(gen_wav_start_lst)} != {len(gen_wav_end_lst)}")
        if len(gen_wav_start_lst) == 0:
            raise ValueError("gen_wav_start 为空")

        # If a span is too close to the beginning, left-overlap will be clipped by max(0, s-overlap_start).
        # Pad a short silence at the beginning so that overlap can take effect without contaminating the first phoneme.
        pad_samples = 0
        orig_ref_len = int(ref_audio.shape[0])
        if overlap_time is not None:
            overlap_start_samples = int(float(overlap_time[0]) * self.sr)
            min_start = min(int(x) for x in gen_wav_start_lst)
            if min_start < int(overlap_start_samples):
                pad_samples = int(0.16 * self.sr)
                if pad_samples > 0:
                    print("[WARNING] 由于待修改字符太靠前，在句首pad了0.16s静音。")
                    ref_audio = np.concatenate(
                        [np.zeros((pad_samples,), dtype=ref_audio.dtype), ref_audio],
                        axis=0,
                    )
                    gen_wav_start_lst = [int(x) + int(pad_samples) for x in gen_wav_start_lst]
                    gen_wav_end_lst = [int(x) + int(pad_samples) for x in gen_wav_end_lst]

        spans = []
        for s, e in zip(gen_wav_start_lst, gen_wav_end_lst):
            s = max(0, int(s))
            e = min(ref_audio.shape[0], int(e))
            if e <= s:
                raise ValueError(f"非法编辑区间: {(s, e)}")
            spans.append((s, e))

        exact_ref_audio = ref_audio.copy()
        exact_spans = list(spans)
        if overlap_time is not None:
            overlap_start, overlap_end = overlap_time
            overlap_start = int(overlap_start * self.sr)
            overlap_end = int(overlap_end * self.sr)

            spans = [
                (
                    max(0, s - overlap_start),
                    min(ref_audio.shape[0], e + overlap_end),
                )
                for s, e in spans
            ]
            for s, e in spans:
                ref_audio[s:e] = 0.0

        speech = len(text) > 0
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']

        ref_wav = torch.from_numpy(
            np.concatenate([ref_audio, np.zeros(0, dtype=np.float16)])
        )[None, :].to(self.device)

        ref_wav_lens = torch.LongTensor(
            [ref_wav.shape[1] // fm_wav * fm_wav]
        ).to(self.device)
        ref_wav = ref_wav[:, :ref_wav_lens[0]]

        if overlap_time is not None:
            exact_ref_wav = torch.from_numpy(
                np.concatenate([exact_ref_audio, np.zeros(0, dtype=np.float16)])
            )[None, :].to(self.device)
            exact_ref_wav = exact_ref_wav[:, :ref_wav_lens[0]]

        if ref_text is None:
            raise ValueError("ref_text is None")
        
        caption = prompt
        bgm_flag_val = 0

        use_caption = getattr(self, "use_caption", False) and (caption is not None)
        caption_emb = None
        caption_lens = None
        caption_text_mark = None

        latent_dim = None
        if hasattr(self.vae, "latent_dim"):
            latent_dim = getattr(self.vae, "latent_dim")
        if latent_dim is None and hasattr(self.dit, "hp") and hasattr(self.dit.hp, "in_channels"):
            latent_dim = self.dit.hp.in_channels
        if latent_dim is None:
            latent_dim = 32

        if hasattr(self.vae, "hop_size"):
            hop_size = getattr(self.vae, "hop_size")
        else:
            hop_size = 240
        if hasattr(self.vae, "vae_stride"):
            vae_stride = getattr(self.vae, "vae_stride")
        else:
            vae_stride = 4

        mel_len = ref_wav.shape[-1] // hop_size
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=self.precision):
                lat_ctx_ref = self.vae.encode_latent(ref_wav)
                if overlap_time is not None:
                    lat_ctx_exact_ref = self.vae.encode_latent(exact_ref_wav)
            latent_dim = lat_ctx_ref.size(-1)

            lat_len = lat_ctx_ref.size(1)
            tgt_len = lat_len

            lat = lat_ctx_ref
            fm = hparams['frames_multiple']

            span_mels = []
            ctx_mask_mel = torch.ones((int(mel_len), 1), dtype=torch.float32)
            for s, e in spans:
                ms = int(s / hop_size)
                me = int(e / hop_size)
                ms = (ms // int(fm)) * int(fm)
                me = (me // int(fm)) * int(fm)
                ms = max(0, min(int(ms), int(mel_len)))
                me = max(0, min(int(me), int(mel_len)))
                if me <= ms:
                    continue
                ctx_mask_mel[ms:me] = 0.0
                span_mels.append((ms, me))
            ctx_mask = ctx_mask_mel[::vae_stride]
            ctx_mask = ctx_mask.unsqueeze(0).to(self.device)

            if overlap_time is not None:
                exact_span_mels = []
                exact_ctx_mask_mel = torch.ones((int(mel_len), 1), dtype=torch.float32)
                for s, e in exact_spans:
                    ms = int(s / hop_size)
                    me = int(e / hop_size)
                    ms = (ms // int(fm)) * int(fm)
                    me = (me // int(fm)) * int(fm)
                    ms = max(0, min(int(ms), int(mel_len)))
                    me = max(0, min(int(me), int(mel_len)))
                    if me <= ms:
                        continue
                    exact_ctx_mask_mel[ms:me] = 0.0
                    exact_span_mels.append((ms, me))
                exact_ctx_mask = exact_ctx_mask_mel[::vae_stride]
                exact_ctx_mask = exact_ctx_mask.unsqueeze(0).to(self.device)

            full_text = text

        txt_tokens, txt_mask, txt_lens, spk_mask = self._tokenize_dit_text(full_text)

        vad_mask = torch.zeros_like(lat[:, :, :1])
        if not self.config.get('drop_vad', False) and speech:
            vad_mask[:, int(start_time * 25):-int(end_time * 25)] = 1.0

        vad_mask = torch.cat([vad_mask] * 3, dim=0)

        txt_tokens = torch.cat([
            txt_tokens,
            txt_tokens,
            torch.full_like(txt_tokens, self.cfg_mask_text_token),
        ], dim=0)
        txt_mask = torch.cat([txt_mask] * 3, dim=0)
        txt_lens = torch.cat([txt_lens] * 3, dim=0)

        if spk_mask is not None:
            spk_mask = torch.cat([
                spk_mask,
                spk_mask,
                torch.zeros_like(spk_mask),
            ], dim=0)

        lat = torch.cat([
            lat,
            torch.zeros_like(lat),
            torch.zeros_like(lat),
        ], dim=0)
        ctx_mask = torch.cat([ctx_mask] * 3, dim=0)

        batch_size = lat.shape[0]
        inputs = {
            'txt_tokens': txt_tokens,
            'spk_mask': spk_mask,
            'txt_mask': txt_mask,
            'txt_lens': txt_lens,
            'ctx_mask': ctx_mask,
            'lat_ctx': lat,
            'caption_emb': caption_emb,
            'caption_lens': caption_lens,
            'caption_text_mark': caption_text_mark,
            'vad_mask': vad_mask,
            'tgt_len': torch.full(
                (batch_size,),
                tgt_len,
                dtype=torch.long,
                device=self.device,
            ),
        }
        if hasattr(self.dit, 'hp') and bool(getattr(self.dit.hp, 'use_bgm_flag', False)):
            if batch_size % 3 == 0:
                b0 = batch_size // 3
                base = torch.full(
                    (b0,),
                    int(bgm_flag_val),
                    dtype=torch.long,
                    device=self.device,
                )
                inputs['bgm_flag'] = torch.cat([
                    base,
                    torch.full_like(base, 2),
                    torch.full_like(base, 2),
                ], dim=0)
            else:
                inputs['bgm_flag'] = torch.full(
                    (batch_size,),
                    int(bgm_flag_val),
                    dtype=torch.long,
                    device=self.device,
                )

        with torch.autocast(device_type='cuda', dtype=self.precision):
            x = self.dit.inference(inputs, timesteps=num_step, seq_cfg_w=cfg_w, timestep_annealing_w=timestep_annealing_w, use_amo_sampler=use_amo_sampler, use_sway=use_sway)

            if overlap_time is None:
                x_dec = lat_ctx_ref * ctx_mask + x * (1 - ctx_mask)
            else:
                print(f"使用overlap_time: {overlap_time}")
                blend_mask_mel = torch.ones((int(mel_len), 1), dtype=torch.float32)

                for (gs, ge), (es, ee) in zip(span_mels, exact_span_mels):
                    blend_mask_mel[es:ee] = 0.0

                    if es > gs:
                        n = es - gs
                        blend_mask_mel[gs:es, 0] = torch.linspace(1.0, 0.0, steps=n, dtype=torch.float32)

                    if ge > ee:
                        n = ge - ee
                        blend_mask_mel[ee:ge, 0] = torch.linspace(0.0, 1.0, steps=n, dtype=torch.float32)

                blend_mask = blend_mask_mel[::vae_stride].unsqueeze(0).to(self.device)
                if blend_mask.shape[1] > lat_ctx_exact_ref.shape[1]:
                    blend_mask = blend_mask[:, :lat_ctx_exact_ref.shape[1]]
                elif blend_mask.shape[1] < lat_ctx_exact_ref.shape[1]:
                    pad = torch.ones((1, lat_ctx_exact_ref.shape[1] - blend_mask.shape[1], 1), device=blend_mask.device, dtype=blend_mask.dtype)
                    blend_mask = torch.cat([blend_mask, pad], dim=1)

                x_dec = lat_ctx_exact_ref * blend_mask + x * (1 - blend_mask)

            with torch.autocast(device_type='cuda', dtype=self.precision):
                wav_dec = self.vae.decode(x_dec)[0, 0].to(torch.float32)

            wav_pred = wav_dec

            if wav_pred.abs().max() > 1:
                print('Wav amplitude exceed 1, clip it.')
                wav_pred = wav_pred / (wav_pred.abs().max())

            wav_pred = wav_pred.cpu().numpy()

            if pad_samples > 0:
                wav_pred = wav_pred[int(pad_samples):int(pad_samples) + int(orig_ref_len)]

        return wav_pred

class SpeechEditInferWrapper:
    def __init__(self, infer: SpeechEditInfer):
        self.base_infer = infer
        self.device = torch.device('cuda:0')

        self.asr_model = None
        self.aligner = None
        self.uvr_model = None
        self.vad_model = None

    def _strip_for_alignment_with_map(self, s: str):
        s = "" if s is None else str(s)
        keep_chars = []
        idx_map = []
        i = 0
        n = len(s)

        while i < n:
            tag_match = re.match(r"<\s*/?\s*S[1-4]\s*>", s[i:], flags=re.IGNORECASE)
            if tag_match is not None:
                i += tag_match.end()
                continue

            ch = s[i]
            if ch.isspace() or unicodedata.category(ch).startswith("P"):
                i += 1
                continue

            keep_chars.append(ch)
            idx_map.append(i)
            i += 1

        return ''.join(keep_chars), idx_map

    def _strip_ws_with_map(self, s: str):
        s = "" if s is None else str(s)
        keep_chars = []
        idx_map = []
        for i, ch in enumerate(s):
            if ch.isspace():
                continue
            keep_chars.append(ch)
            idx_map.append(i)
        return ''.join(keep_chars), idx_map
 
    def _replace_unique_occurrence(self, haystack: str, needle: str, repl: str) -> str:
        haystack = "" if haystack is None else str(haystack)
        needle = "" if needle is None else str(needle)
        repl = "" if repl is None else str(repl)

        needle = needle.strip()
        if needle == "":
            raise ValueError("text_src is empty")

        hay_n, hay_map = self._strip_ws_with_map(haystack)
        ndl_n, _ = self._strip_ws_with_map(needle)

        if ndl_n == "":
            raise ValueError("text_src becomes empty after whitespace normalization")

        hits = []
        start = 0
        while True:
            pos = hay_n.find(ndl_n, start)
            if pos < 0:
                break
            hits.append(pos)
            start = pos + 1

        if len(hits) != 1:
            print(f"asr结果: {haystack}, 待替换文本：{needle}")
            raise ValueError(f"期望在asr结果中找到唯一的代替换文本{needle}，但是实际出现{len(hits)}次。")

        pos = hits[0]
        start_orig = hay_map[pos]
        end_orig = hay_map[pos + len(ndl_n) - 1] + 1
        return haystack[:start_orig] + repl + haystack[end_orig:]

    def _replace_first_occurrence_for_alignment(self, haystack: str, needle: str, repl: str) -> str:
        haystack = "" if haystack is None else str(haystack)
        needle = "" if needle is None else str(needle)
        repl = "" if repl is None else str(repl)

        hay_n, hay_map = self._strip_for_alignment_with_map(haystack)
        ndl_n, _ = self._strip_for_alignment_with_map(needle)
        if ndl_n == "":
            raise ValueError("text_src becomes empty after alignment normalization")

        pos = hay_n.find(ndl_n)
        if pos < 0:
            raise ValueError(f"无法在当前文本中找到待替换片段: {needle}")

        start_orig = hay_map[pos]
        end_orig = hay_map[pos + len(ndl_n) - 1] + 1
        return haystack[:start_orig] + repl + haystack[end_orig:]

    def _find_ordered_occurrences(self, whole_text: str, parts: list):
        whole_n, _ = self._strip_for_alignment_with_map(whole_text)
        spans = []
        cursor = 0

        for part in parts:
            part = "" if part is None else str(part)
            part_n, _ = self._strip_for_alignment_with_map(part)
            if part_n == "":
                raise ValueError("text_src 中存在空片段")

            pos = whole_n.find(part_n, cursor)
            if pos < 0:
                raise ValueError(f"无法在文本 = {whole_text} 中按顺序找到待编辑片段: {part}")

            spans.append((pos, pos + len(part_n) - 1, part))
            cursor = pos + len(part_n)

        return spans

    def _expand_all_occurrences_for_alignment(self, whole_text_src: str, text_src: list, text_tgt: list):
        whole_n, whole_map = self._strip_for_alignment_with_map(whole_text_src)
        if whole_n == "":
            raise ValueError("whole_text_src 在对齐归一化后为空")

        matches = []
        for idx, (src_part, tgt_part) in enumerate(zip(text_src, text_tgt)):
            src_n, _ = self._strip_for_alignment_with_map(src_part)
            if src_n == "":
                raise ValueError("text_src 中存在归一化后为空的片段")

            start = 0
            hit_count = 0
            while True:
                pos = whole_n.find(src_n, start)
                if pos < 0:
                    break
                matches.append({
                    "start_n": pos,
                    "end_n": pos + len(src_n) - 1,
                    "pair_idx": idx,
                    "text_src": src_part,
                    "text_tgt": tgt_part,
                })
                hit_count += 1
                start = pos + len(src_n)

            if hit_count == 0:
                raise ValueError(f"无法在 whole_text_src 中找到待替换字段: {src_part}")
            if hit_count > 1:
                print(f"[WARNING] 字段 {src_part} 在文本中出现了 {hit_count} 次，将全部修改。")

        matches.sort(key=lambda x: (x["start_n"], x["end_n"]))
        for i in range(1, len(matches)):
            prev = matches[i - 1]
            curr = matches[i]
            if curr["start_n"] <= prev["end_n"]:
                raise ValueError(
                    "存在重叠或歧义的待替换字段，无法执行全部替换："
                    f"{prev['text_src']} 与 {curr['text_src']}"
                )

        pieces = []
        cursor_orig = 0
        expanded_text_src = []
        expanded_text_tgt = []
        expanded_debug = []
        for match in matches:
            start_orig = whole_map[match["start_n"]]
            end_orig = whole_map[match["end_n"]] + 1
            pieces.append(whole_text_src[cursor_orig:start_orig])
            pieces.append(match["text_tgt"])
            cursor_orig = end_orig

            expanded_text_src.append(match["text_src"])
            expanded_text_tgt.append(match["text_tgt"])
            expanded_debug.append({
                "text_src": match["text_src"],
                "text_tgt": match["text_tgt"],
                "char_span": (match["start_n"], match["end_n"]),
            })

        pieces.append(whole_text_src[cursor_orig:])
        whole_text_tgt = "".join(pieces)
        return whole_text_tgt, expanded_text_src, expanded_text_tgt, expanded_debug

    def _build_aligner_char_index(self, aligner_results: list):
        chars = []
        item_indices = []

        for item_idx, item in enumerate(aligner_results):
            text = "" if item.get("text") is None else str(item.get("text"))
            for ch in text:
                if ch.isspace() or unicodedata.category(ch).startswith("P"):
                    continue
                chars.append(ch)
                item_indices.append(item_idx)

        return "".join(chars), item_indices

    def _extract_aligner_span_text(self, aligner_results: list, start_item_idx: int, end_item_idx: int) -> str:
        parts = []
        for item in aligner_results[start_item_idx:end_item_idx + 1]:
            text = "" if item.get("text") is None else str(item.get("text"))
            if text:
                parts.append(text)
        return "".join(parts).strip()

    def model_infer(self, cfg, whole_text_tgt: str, wav_path, time_tag, out_path, out_file=None, overlap_time:tuple=None, extend_mask=1.0, return_wav=False, num_step=20):
        '''标准的推理格式'''
        # set_seed(42)
        os.makedirs(out_path, exist_ok=True)
        negative_prompt = cfg.get('negative_prompt', None)

        caption = ""
        caption = _norm_spaces_caption(caption)

        if out_file is None:
            out_file = f'{whole_text_tgt[:5]}.wav'
            # out_file = os.path.join(out_path, f'{text[:5]}.wav')
        
        whole_text_tgt = _norm_spaces_caption(whole_text_tgt)
        whole_text_tgt = raw_text_process_s1s2_tagged(whole_text_tgt)

        audio_raw, audio_raw_sr = librosa.load(wav_path, sr=None)
        audio = batch_resample(
            wavs=[audio_raw.astype(np.float32, copy=False)],
            sample_rates=audio_raw_sr,
            tgt_sr=self.base_infer.sr,
            resamplers=None,
            device=self.device,
        )[0]
        # audio, _ = librosa.load(wav_path, sr=24000)

        gen_wav_start, gen_wav_end = time_tag
        audio[gen_wav_start:gen_wav_end] = 0.0

        

        wav = self.base_infer.forward(
            whole_text_tgt,
            ref_audio=audio,
            ref_text=whole_text_tgt,
            prompt=caption,
            cfg_w=cfg.get('cfg_w', None),
            infer_length=cfg.get('infer_length', 0),
            num_step=num_step,
            start_time=cfg.get('vad_len', 0.2),
            end_time=cfg.get('vad_len', 0.2),
            negative_prompt=negative_prompt,
            timestep_annealing_w=cfg.get('timestep_annealing_w', (1.0, 0.0, 1.0)),
            use_amo_sampler=cfg.get('use_amo_sampler', False),
            use_sway=cfg.get('use_sway', False),
            cpm=cfg.get('cpm', 300.0),
            gen_wav_start=gen_wav_start,
            gen_wav_end=gen_wav_end,
            overlap_time=overlap_time,
            extend_mask=extend_mask,
        )
        
        if return_wav:
            return wav
        else:
            output_wav_path = os.path.join(out_path, out_file)
            sf.write(output_wav_path, wav, self.base_infer.sr, "PCM_16")

            result = SpeechEditResult(
                method="model_infer",
                wav_path=wav_path,
                out_path=out_path,
                out_file=out_file,
                output_wav_path=output_wav_path,
                sample_rate=self.base_infer.sr,
                whole_text_src=None,
                whole_text_tgt=whole_text_tgt,
            )
            result.add_edit(
                text_src=None,
                text_tgt=whole_text_tgt,
                time_tag=(gen_wav_start / self.base_infer.sr, gen_wav_end / self.base_infer.sr),
                time_tag_samples=(gen_wav_start, gen_wav_end),
                text_replaced=whole_text_tgt,
                extra={
                    "overlap_time": overlap_time,
                    "extend_mask": extend_mask,
                },
            )
            return result

    def model_infer2(self, cfg, text_src, text_tgt, wav_path, time_tag, out_path, out_file=None, overlap_time:tuple=None, extend_enalbe=False):
        '''输入的是src音频，修改前的文本，修改后的文本，时间标签
        ——需要做asr，然后替换修改的文本，传给标准的推理格式model_infer'''
        
        time_tag = (int(time_tag[0]), int(time_tag[1]))
        # Allow Arabic numerals in text_tgt (e.g. "19.9") and convert to Chinese price reading.
        text_tgt = _num_string_to_price_zh(_norm_spaces_caption(text_tgt))

        device = self.device
        if self.asr_model is None:
            self.asr_model = build_asr_model(device) 
        audio, _ = librosa.load(wav_path, sr=24000)
        wav_16k = librosa.resample(
            audio.astype(np.float32),
            orig_sr=24000,
            target_sr=16000
        )
        asr_out = run_asr_model([wav_16k], self.asr_model, with_segments=False)
        asr_text = asr_out[0]['text_normed']

        text_replaced = self._replace_unique_occurrence(asr_text, text_src, text_tgt)

        # import pdb; pdb.set_trace()

        if extend_enalbe:
            extend_mask = len(text_tgt) / len(text_src)
        else:
            extend_mask = 1.0

        result = self.model_infer(
            cfg=cfg,
            whole_text_tgt=text_replaced,
            wav_path=wav_path,
            time_tag=time_tag,
            out_path=out_path,
            out_file=out_file,
            overlap_time=overlap_time,
            extend_mask=extend_mask,
        )
        if isinstance(result, SpeechEditResult):
            result.method = "model_infer2"
            result.whole_text_src = asr_text
            result.whole_text_tgt = text_replaced
            result.debug.update({
                "asr_text": asr_text,
            })
            if len(result.edits) > 0:
                result.edits[0]["text_src"] = text_src
                result.edits[0]["text_tgt"] = text_tgt
                result.edits[0]["text_replaced"] = text_replaced
                result.edits[0]["merged_fields"] = [{
                    "step": 0,
                    "text_src": text_src,
                    "text_tgt": text_tgt,
                }]
        return result

    def model_infer3(self, cfg, text_src, text_tgt, wav_path, out_path, out_file=None, overlap_time:tuple=None, debug=False):
        '''输入的是src音频，修改前的文本，修改后的文本，无时间标签
        ——需要做asr，然后替换修改的文本，
        ——还需要做aligner，找到替换的时间点，传给标准的推理格式model_infer
        --aligner如果置信度太低，先做一遍uvr，再做align
        --对于字数变化较大的，调整空白音频时长'''

        device = self.device
        if self.asr_model is None:
            self.asr_model = build_asr_model(device) 
        audio, _ = librosa.load(wav_path, sr=24000)
        wav_16k = librosa.resample(
            audio.astype(np.float32),
            orig_sr=24000,
            target_sr=16000
        )
        asr_out = run_asr_model([wav_16k], self.asr_model, with_segments=False)
        asr_text = asr_out[0]['text_normed']

        text_replaced = self._replace_unique_occurrence(asr_text, text_src, text_tgt)
        # 该函数里确保了存在唯一的替换文本

        if self.aligner is None:
            self.aligner = ForcedAlignerInfer(
                ckpt='checkpoints/260304_aligner_2tower_v6',
            )

        aligner_results = self.aligner.align_with_ignored_patterns(
            audio=wav_path,
            text=asr_text,
            ignore_literals=["<S1>", "</S1>", "<S2>", "</S2>", "<S3>", "</S3>", "<S4>", "</S4>"],
            return_json=True,
            # debug_save_dir=f"infer_out/asr/aligner/{exp_name}",
            # emit_temp=0.5,
            # trans_temp=1.0,
            # debug_start_index=0,
        )

        ### 解析aligner结果，得到替换的时间点
        aligner_results = aligner_results['results'][0]
        start_time_lst = [item['start_time'] for item in aligner_results]  # 从aligner结果中提取start_time
        end_time_lst = [item['end_time'] for item in aligner_results]  # 从aligner结果中提取end_time
        text_lst = [item['text'] for item in aligner_results]  # 从aligner结果中提取text
        conf_lst = [item['conf'] for item in aligner_results]  # 从aligner结果中提取conf

        text_src_lst = list(text_src)
        m = len(text_src_lst)
        for i in range(0, len(text_lst) - m + 1):
            if text_lst[i:i+m] == text_src_lst:
                # bug fix：这里的end_time_lst[i+m-1]是因为aligner的end_time是开区间，所以需要减1
                time_tag = (float(start_time_lst[i]), float(end_time_lst[i+m-1]))
                conf = conf_lst[i], conf_lst[i+m]
                break
        
        if conf[0] < 0.5 or conf[1] < 0.5:
            ## 需要做一遍uvr 然后重新align
            print(f"[WARNING] confidence too low: {conf}, {wav_path =}")
            if self.uvr_model is None:
                self.uvr_model = build_uvr_model(self.device) 

            tmp_dir = "/mnt/bn/sa-ag-data/leike/tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(wav_path))[0]
            tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            uvr_base = os.path.join(tmp_dir, f"{stem}_{tag}.wav")
            vocals_path = os.path.join(tmp_dir, f"{stem}_{tag}_vocals.wav")
 
            run_uvr_model(wav_path, self.uvr_model, output_path=uvr_base)

            aligner_results = self.aligner.align_with_ignored_patterns(
                audio=vocals_path,
                text=asr_text,
                ignore_literals=["<S1>", "</S1>", "<S2>", "</S2>", "<S3>", "</S3>", "<S4>", "</S4>"],
                return_json=True,
            )

            ### 解析aligner结果，得到替换的时间点
            aligner_results = aligner_results['results'][0]
            start_time_lst = [item['start_time'] for item in aligner_results]  # 从aligner结果中提取start_time
            end_time_lst = [item['end_time'] for item in aligner_results]  # 从aligner结果中提取end_time
            text_lst = [item['text'] for item in aligner_results]  # 从aligner结果中提取text
            conf_lst = [item['conf'] for item in aligner_results]  # 从aligner结果中提取conf

            text_src_lst = list(text_src)
            m = len(text_src_lst)
            for i in range(0, len(text_lst) - m + 1):
                if text_lst[i:i+m] == text_src_lst:
                    # bug fix：这里的end_time_lst[i+m-1]是因为aligner的end_time是开区间，所以需要减1
                    time_tag = (float(start_time_lst[i]), float(end_time_lst[i+m-1]))
                    conf = conf_lst[i], conf_lst[i+m]
                    break

            

        time_tag = (int(time_tag[0]*self.base_infer.sr), int(time_tag[1]*self.base_infer.sr))
        # import pdb; pdb.set_trace()

        extend_mask = len(text_tgt) / len(text_src)

        if debug:
            print(f"| [INFO] asr_text: {asr_text}")
            print(f"| [INFO] time_tag: {[x/self.base_infer.sr for x in time_tag]}")
            print(f"| [INFO] conf: {conf}")
            print(f"| [INFO] extend_mask: {extend_mask}")


        return self.model_infer(
            cfg=cfg,
            whole_text_tgt=text_replaced,
            wav_path=wav_path,
            time_tag=time_tag,
            out_path=out_path,
            out_file=out_file,
            overlap_time=overlap_time,
            extend_mask=extend_mask,
        )

    def model_infer4(self, cfg, text_src, text_tgt, wav_path, out_path, out_file=None, overlap_time:tuple=None, debug=False,
                     uvr_enable=True, vad_enable=True, extend_enable=False, anchor_enable=True, anchor_threshold=0.8):
        '''260408，接口与model_infer3相同，增加默认uvr，vad，aligner anchor'''

        device = self.device
        if self.asr_model is None:
            self.asr_model = build_asr_model(device) 
        audio, _ = librosa.load(wav_path, sr=24000)
        wav_16k = librosa.resample(
            audio.astype(np.float32),
            orig_sr=24000,
            target_sr=16000
        )
        asr_out = run_asr_model([wav_16k], self.asr_model, with_segments=False)
        asr_text = asr_out[0]['text_normed']

        text_replaced = self._replace_unique_occurrence(asr_text, text_src, text_tgt)
        # 该函数里确保了存在唯一的替换文本

        if self.aligner is None:
            self.aligner = ForcedAlignerInfer(
                ckpt='checkpoints/260304_aligner_2tower_v6',
            )
        
        if uvr_enable and self.uvr_model is None:
            self.uvr_model = build_uvr_model(self.device) 
        if vad_enable and self.vad_model is None:
            self.vad_model = build_vad_model(self.device)
        
        tmp_dir = "tmp/260409"
        os.makedirs(tmp_dir, exist_ok=True)

        aligner_wav_path = wav_path
        wav_to_edit = wav_path
        vocals_path = None
        instrumental_path = None
        vad_path = None
        vad_value: list = None

        stem = os.path.splitext(os.path.basename(wav_path))[0]
        tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if uvr_enable:
            if debug:
                print(f"| [INFO] processing uvr on {wav_path}")
            uvr_base = os.path.join(tmp_dir, f"{stem}_{tag}.wav")
            ### 预知的保存路径
            vocals_path = os.path.join(tmp_dir, f"{stem}_{tag}_vocals.wav")
            instrumental_path = os.path.join(tmp_dir, f"{stem}_{tag}_instrumental.wav")

            run_uvr_model(wav_path, self.uvr_model, output_path=uvr_base)

            aligner_wav_path = vocals_path
            instrumental_path = instrumental_path

        ### 做vad
        if vad_enable:
            if debug:
                print(f"| [INFO] processing vad on {aligner_wav_path}")
            vad_path = os.path.join(tmp_dir, f"{stem}_{tag}_vad.wav") # 保存vad截取结果的路径

            audio, _ = librosa.load(aligner_wav_path, sr=24000) # vad的采样率需要是16k
            wav_16k = librosa.resample(
                audio.astype(np.float32),
                orig_sr=24000,
                target_sr=16000
            )
            res = self.vad_model.generate(
                input=wav_16k,
                batch_size_s=300,
                merge_vad=False,
                merge_length_s=10,
            )
            vad_value = res[0]["value"] # 单位是ms
            print(f"vad发现 {len(vad_value)} 段声音")

            chunks = []
            for s_ms, e_ms in vad_value:
                s = int(round(s_ms * 24000 / 1000.0))
                e = int(round(e_ms * 24000 / 1000.0))
                s = max(0, s)
                e = min(len(audio), e)
                if e <= s:
                    continue
                chunks.append(audio[s:e])
            
            if len(chunks) > 0:
                wav_vad = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
                sf.write(vad_path, wav_vad, 24000, "PCM_16")

            else:
                raise ValueError("vad没有发现有效声音")
            wav_to_edit = vad_path


            aligner_wav_path = vad_path
        
        aligner_results = self.aligner.align_with_ignored_patterns(
            audio=aligner_wav_path,
            text=asr_text,
            ignore_literals=["<S1>", "</S1>", "<S2>", "</S2>", "<S3>", "</S3>", "<S4>", "</S4>"],
            return_json=True,
        )

        ### 解析aligner结果，得到替换的时间点
        aligner_results = aligner_results['results'][0]
        start_time_lst = [item['start_time'] for item in aligner_results]  # 从aligner结果中提取start_time
        end_time_lst = [item['end_time'] for item in aligner_results]  # 从aligner结果中提取end_time
        text_lst = [item['text'] for item in aligner_results]  # 从aligner结果中提取text
        conf_lst = [item['conf'] for item in aligner_results]  # 从aligner结果中提取conf


        text_src_lst = list(text_src)
        m = len(text_src_lst)
        match_start_idx = None
        match_end_idx = None
        for i in range(0, len(text_lst) - m + 1):
            # bug fix：这里的end_time_lst[i+m-1]是因为aligner的end_time是开区间，所以需要减1
            if text_lst[i:i+m] == text_src_lst:
                match_start_idx = i
                match_end_idx = i + m - 1
                break

        if match_start_idx is None or match_end_idx is None:
            raise ValueError(f"无法在aligner结果中找到待编辑文本: {text_src}")

        anchor_left_idx = match_start_idx
        anchor_right_idx = match_end_idx
        anchor_left_expand = 0
        anchor_right_expand = 0
        anchor_conf_before = (
            float(conf_lst[match_start_idx]),
            float(conf_lst[match_end_idx]),
        )

        if anchor_enable:
            while float(conf_lst[anchor_left_idx]) < anchor_threshold:
                if anchor_left_idx == 0:
                    print(
                        f"[WARNING] 左侧anchor无法继续扩展：已到达最左边界，"
                        f"当前conf={float(conf_lst[anchor_left_idx]):.4f} < threshold={anchor_threshold}"
                    )
                    break
                anchor_left_idx -= 1
                anchor_left_expand += 1

            while float(conf_lst[anchor_right_idx]) < anchor_threshold:
                if anchor_right_idx == len(conf_lst) - 1:
                    print(
                        f"[WARNING] 右侧anchor无法继续扩展：已到达最右边界，"
                        f"当前conf={float(conf_lst[anchor_right_idx]):.4f} < threshold={anchor_threshold}"
                    )
                    break
                anchor_right_idx += 1
                anchor_right_expand += 1

            if anchor_left_expand > 5:
                print(f"[WARNING] 左侧anchor扩展过多：{anchor_left_expand} 个字")
            if anchor_right_expand > 5:
                print(f"[WARNING] 右侧anchor扩展过多：{anchor_right_expand} 个字")

        time_tag = (
            float(start_time_lst[anchor_left_idx]),
            float(end_time_lst[anchor_right_idx]),
        )
        conf = (
            float(conf_lst[anchor_left_idx]),
            float(conf_lst[anchor_right_idx]),
        )
        anchor_conf_after = conf

        time_tag = (int(time_tag[0]*self.base_infer.sr), int(time_tag[1]*self.base_infer.sr))

        if debug:
            print(f"| [INFO] asr_text: {asr_text}")
            print(f"| [INFO] time_tag: {[x/self.base_infer.sr for x in time_tag]}")
            print(f"| [INFO] conf: {conf}")
            print(f"| [INFO] anchor_enable: {anchor_enable}")
            print(f"| [INFO] anchor_threshold: {anchor_threshold}")
            print(f"| [INFO] anchor_conf_before: {anchor_conf_before}")
            print(f"| [INFO] anchor_conf_after: {anchor_conf_after}")
            print(f"| [INFO] anchor_expand: left={anchor_left_expand}, right={anchor_right_expand}")
            print(f"| [INFO] vad: {vad_value}")

        if extend_enable:
            raise NotImplementedError("extend_enable not implemented in infer4")
        
        edit_wav = self.model_infer(
            cfg=cfg,
            whole_text_tgt=text_replaced,
            wav_path=wav_to_edit,
            time_tag=time_tag,
            out_path=out_path,
            out_file=out_file,
            overlap_time=overlap_time,
            return_wav=True,
        )

        ### 根据vad拆开wav
        if vad_enable:
            assert vad_value is not None
            assert edit_wav is not None
            seg_lens = [e_ms - s_ms for s_ms, e_ms in vad_value]
            ## check下这里，vad_reverse应该是zeros吧？
            vad_reverse = np.array(audio, copy=True) 

            edit_start = 0
            for s_ms, e_ms in vad_value:
                s = int(round(s_ms * 24000 / 1000.0))
                e = int(round(e_ms * 24000 / 1000.0))

                seg_len = int(e - s)
                seg_edit = edit_wav[edit_start:edit_start+seg_len]
                # if int(seg_edit.shape[0]) != seg_len:
                #     raise ValueError(f"edit_wav长度错误，s={s}, e={e}, seg_len={seg_len}, seg_edit.shape={seg_edit.shape}")
                actual_seg_len = len(seg_edit)
                vad_reverse[s:s+actual_seg_len] = seg_edit
                edit_start += e - s
                if debug:
                    print(f"| [INFO] 音频长度变化：{edit_start - len(edit_wav)}")
            
            edit_wav = vad_reverse
        
        ### 如果做了uvr，合并音效
        if uvr_enable:
            instrumental_wav, _ = librosa.load(instrumental_path, sr=44100)
            edit_wav = librosa.resample(
                edit_wav.astype(np.float32),
                orig_sr=24000,
                target_sr=44100
            )
            assert -100 < len(instrumental_wav) - len(edit_wav) < 100, f"{instrumental_wav.shape} != {edit_wav.shape}"
            if len(instrumental_wav) > len(edit_wav):
                instrumental_wav = instrumental_wav[:len(edit_wav)]
            else:
                edit_wav = edit_wav[:len(instrumental_wav)]


            if debug:
                print(f"| [INFO] add instrumental to vocals")
            edit_wav = np.add(edit_wav, instrumental_wav)
            output_wav_path = os.path.join(out_path, out_file)
            sf.write(output_wav_path, edit_wav, 44100, "PCM_16")

        else:
            output_wav_path = os.path.join(out_path, out_file)
            sf.write(output_wav_path, edit_wav, 24000, "PCM_16")

        result = SpeechEditResult(
            method="model_infer4",
            wav_path=wav_path,
            out_path=out_path,
            out_file=out_file,
            output_wav_path=output_wav_path,
            sample_rate=self.base_infer.sr,
            whole_text_src=asr_text,
            whole_text_tgt=text_replaced,
            debug={
                "asr_text": asr_text,
                "conf": conf,
                "anchor_enable": anchor_enable,
                "anchor_threshold": anchor_threshold,
                "vad": vad_value,
                "uvr_enable": uvr_enable,
                "vad_enable": vad_enable,
            },
        )
        result.add_edit(
            text_src=text_src,
            text_tgt=text_tgt,
            time_tag=(time_tag[0] / self.base_infer.sr, time_tag[1] / self.base_infer.sr),
            time_tag_samples=time_tag,
            merged=False,
            merged_fields=[{
                "step": 0,
                "text_src": text_src,
                "text_tgt": text_tgt,
                "time_tag": (time_tag[0] / self.base_infer.sr, time_tag[1] / self.base_infer.sr),
            }],
            anchor_expand={
                "left": anchor_left_expand,
                "right": anchor_right_expand,
            },
            anchor_conf_before={
                "left": anchor_conf_before[0],
                "right": anchor_conf_before[1],
            },
            anchor_conf_after={
                "left": anchor_conf_after[0],
                "right": anchor_conf_after[1],
            },
            text_replaced=text_replaced,
            extra={
                "overlap_time": overlap_time,
            },
        )
        return result

    def model_infer_multi(self, cfg, text_src:list, text_tgt:list, wav_path, out_path, out_file=None, overlap_time:tuple=None, debug=False,
                     uvr_enable=True, vad_enable=True, extend_enable=False, anchor_enable=True, anchor_threshold=0.8, num_step=20):
        '''260412，大体和model_infer4相同，但是可以对多段mask进行推理
            需要做asr'''
        timing = {
            'start': 0, 'before_asr': 0, 'after_asr': 0, 'before_uvr': 0, 'after_uvr': 0,
            'before_vad': 0, 'after_vad': 0, 'before_aligner': 0, 'after_aligner': 0, 'before_dit': 0, 'after_dit': 0, 'end': 0
        }
        timing['start'] = time.time()
        if text_src == "" or text_src is None:
            raise ValueError("text_src 不能为空")
        if text_tgt == "" or text_tgt is None:
            raise ValueError("text_tgt 不能为空")
        if overlap_time is not None and (overlap_time[0] > 0.31 or overlap_time[1] > 0.31):
            print('[WARNING] overlap 时长可能过大')

        if not isinstance(text_src, (list, tuple)):
            text_src = [text_src]
        if not isinstance(text_tgt, (list, tuple)):
            text_tgt = [text_tgt]
        if len(text_src) == 0:
            raise ValueError("text_src 不能为空")
        if len(text_src) != len(text_tgt):
            raise ValueError(f"text_src 和 text_tgt 长度不一致: {len(text_src)} != {len(text_tgt)}")

        text_src = [_norm_spaces_caption(x) for x in text_src]
        # Allow Arabic numerals in text_tgt (e.g. "78.9") and convert to Chinese price reading.
        text_tgt = [_num_string_to_price_zh(_norm_spaces_caption(x)) for x in text_tgt]
        if any(x == "" for x in text_src):
            raise ValueError("text_src 中存在空片段")
        if any(x == "" for x in text_tgt):
            raise ValueError("text_tgt 中存在空片段")

        if self.asr_model is None:
            self.asr_model = build_asr_model(self.device)
        audio_raw, audio_raw_sr = librosa.load(wav_path, sr=None)
        audio = batch_resample(
            wavs=[audio_raw.astype(np.float32, copy=False)],
            sample_rates=audio_raw_sr,
            tgt_sr=24000,
            resamplers=None,
            device=self.device,
        )[0]
        wav_16k = batch_resample(
            wavs=[audio.astype(np.float32, copy=False)],
            sample_rates=24000,
            tgt_sr=16000,
            resamplers=None,
            device=self.device,
        )[0]
        # wav_16k = librosa.resample(
        #     audio.astype(np.float32),
        #     orig_sr=24000,
        #     target_sr=16000
        # )
        timing['before_asr'] = time.time()
        asr_out = run_asr_model([wav_16k], self.asr_model, with_segments=False) # 这里单纯包个autocast没用
        timing['after_asr'] = time.time()
        asr_text = asr_out[0]['text_normed']

        whole_text_tgt, text_src, text_tgt, expanded_debug = self._expand_all_occurrences_for_alignment(
            whole_text_src=asr_text,
            text_src=text_src,
            text_tgt=text_tgt,
        )

        if self.aligner is None:
            self.aligner = ForcedAlignerInfer(
                ckpt='checkpoints/260304_aligner_2tower_v6',
            )

        if uvr_enable and self.uvr_model is None:
            self.uvr_model = build_uvr_model(self.device, precision='fp16')
        if vad_enable and self.vad_model is None:
            self.vad_model = build_vad_model(self.device)

        os.makedirs(out_path, exist_ok=True)
        tmp_dir = "/dev/shm/260415"
        os.makedirs(tmp_dir, exist_ok=True)

        aligner_wav_path = wav_path
        wav_to_edit = np.array(audio, copy=True) # 由path改为wav array
        vocals_path = None
        instrumental_path = None
        vad_path = None
        vad_value: list = None
        vocals_wav = None
        instrumental_wav = None
        aligner_wav_16k = wav_16k  # 改为直接传给aligner array，避免I/O

        stem = os.path.splitext(os.path.basename(wav_path))[0]
        tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        timing['before_uvr'] = time.time()
        if uvr_enable:
            if debug:
                print(f"| [INFO] processing uvr on {wav_path}")
            uvr_base = os.path.join(tmp_dir, f"{stem}_{tag}.wav")
            vocals_path = os.path.join(tmp_dir, f"{stem}_{tag}_vocals.wav")
            instrumental_path = os.path.join(tmp_dir, f"{stem}_{tag}_instrumental.wav")

            uvr_ret = run_uvr_model(wav_path, self.uvr_model, batch_size = 16) #, output_path=uvr_base)
            vocals_wav_raw = uvr_ret['outputs']['vocals'].detach().cpu().numpy()
            vocals_wav_raw = to_mono(vocals_wav_raw)
            instrumental_wav = uvr_ret['outputs']['instrumental'].detach().cpu().numpy()
            instrumental_wav = to_mono(instrumental_wav)

            vocals_wav = batch_resample(wavs=[vocals_wav_raw.astype(np.float32, copy=False)], sample_rates=audio_raw_sr, tgt_sr=24000, resamplers=None, device=self.device)[0]
            wav_to_edit = np.array(vocals_wav, copy=True)

            aligner_wav_path = vocals_path
            aligner_wav_16k = batch_resample(wavs=[vocals_wav_raw.astype(np.float32, copy=False)], sample_rates=audio_raw_sr, tgt_sr=16000, resamplers=None, device=self.device)[0]
        timing['after_uvr'] = time.time()

        # 这里没必要重新load
        if uvr_enable:
            # reverse_base_audio, _ = librosa.load(aligner_wav_path, sr=24000)
            reverse_base_audio = np.array(vocals_wav, copy=True)
        else:
            reverse_base_audio = np.array(audio, copy=True)

        timing['before_vad'] = time.time()
        if vad_enable:
            if debug:
                print(f"| [INFO] processing vad on {aligner_wav_path}")
            vad_path = os.path.join(tmp_dir, f"{stem}_{tag}_vad.wav")

            # vad_audio, _ = librosa.load(aligner_wav_path, sr=24000)
            vad_audio = np.array(reverse_base_audio, copy=True)

            wav_16k = batch_resample(wavs=[vad_audio.astype(np.float32, copy=False)], sample_rates=24000, tgt_sr=16000, resamplers=None, device=self.device)[0]
            # wav_16k = librosa.resample(
            #     vad_audio.astype(np.float32),
            #     orig_sr=24000,
            #     target_sr=16000
            # )
            res = self.vad_model.generate(
                input=wav_16k,
                batch_size_s=300,
                merge_vad=False,
                merge_length_s=10,
            )
            vad_value = res[0]["value"]
            print(f"vad发现 {len(vad_value)} 段声音")

            chunks = []
            for s_ms, e_ms in vad_value:
                s = int(round(s_ms * 24000 / 1000.0))
                e = int(round(e_ms * 24000 / 1000.0))
                s = max(0, s)
                e = min(len(vad_audio), e)
                if e <= s:
                    continue
                chunks.append(vad_audio[s:e])

            if len(chunks) > 0:
                wav_vad = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
                sf.write(vad_path, wav_vad, 24000, "PCM_16")
            else:
                raise ValueError("vad没有发现有效声音")

            wav_to_edit = wav_vad
            aligner_wav_path = vad_path
            aligner_wav_16k = batch_resample(wavs=[wav_vad.astype(np.float32, copy=False)], sample_rates=24000, tgt_sr=16000, resamplers=None, device=self.device)[0]
        timing['after_vad'] = time.time()

        timing['before_aligner'] = time.time()
        # 这个aligner需要16k的音频
        aligner_results = self.aligner.align_with_ignored_patterns(
            # audio=aligner_wav_path,
            audio=(aligner_wav_16k, 16000),
            text=asr_text,
            ignore_literals=["<S1>", "</S1>", "<S2>", "</S2>", "<S3>", "</S3>", "<S4>", "</S4>"],
            return_json=True,
        )
        timing['after_aligner'] = time.time()


        aligner_results = aligner_results['results'][0]
        aligned_text, align_item_indices = self._build_aligner_char_index(aligner_results)
        src_spans = self._find_ordered_occurrences(aligned_text, text_src)
        conf_lst = [float(item['conf']) for item in aligner_results]

        anchored_spans = []
        span_debug = []
        for idx, (char_start, char_end, src_part) in enumerate(src_spans):
            start_item_idx = align_item_indices[char_start]
            end_item_idx = align_item_indices[char_end]
            anchor_left_idx = start_item_idx
            anchor_right_idx = end_item_idx
            anchor_left_expand = 0
            anchor_right_expand = 0
            anchor_conf_before = (
                float(conf_lst[start_item_idx]),
                float(conf_lst[end_item_idx]),
            )

            if anchor_enable:
                while float(conf_lst[anchor_left_idx]) < anchor_threshold:
                    if anchor_left_idx == 0:
                        print(
                            f"[WARNING] 第{idx+1}段左侧anchor无法继续扩展：已到达最左边界，"
                            f"当前conf={float(conf_lst[anchor_left_idx]):.4f} < threshold={anchor_threshold}"
                        )
                        break
                    anchor_left_idx -= 1
                    anchor_left_expand += 1

                while float(conf_lst[anchor_right_idx]) < anchor_threshold:
                    if anchor_right_idx == len(conf_lst) - 1:
                        print(
                            f"[WARNING] 第{idx+1}段右侧anchor无法继续扩展：已到达最右边界，"
                            f"当前conf={float(conf_lst[anchor_right_idx]):.4f} < threshold={anchor_threshold}"
                        )
                        break
                    anchor_right_idx += 1
                    anchor_right_expand += 1

                if anchor_left_expand > 5:
                    print(f"[WARNING] 第{idx+1}段左侧anchor扩展过多：{anchor_left_expand} 个字")
                if anchor_right_expand > 5:
                    print(f"[WARNING] 第{idx+1}段右侧anchor扩展过多：{anchor_right_expand} 个字")

            anchor_conf_after = (
                float(conf_lst[anchor_left_idx]),
                float(conf_lst[anchor_right_idx]),
            )
            anchor_triggered = (anchor_left_expand > 0) or (anchor_right_expand > 0)
            anchor_text_expanded = None
            if anchor_triggered:
                anchor_text_expanded = self._extract_aligner_span_text(
                    aligner_results,
                    anchor_left_idx,
                    anchor_right_idx,
                )
            start_time = float(aligner_results[anchor_left_idx]['start_time'])
            end_time = float(aligner_results[anchor_right_idx]['end_time'])
            conf_slice = [
                float(aligner_results[item_idx]['conf'])
                for item_idx in sorted(set(align_item_indices[char_start:char_end + 1]))
            ]

            anchored_spans.append({
                "left_idx": anchor_left_idx,
                "right_idx": anchor_right_idx,
                "start_sample": int(round(start_time * self.base_infer.sr)),
                "end_sample": int(round(end_time * self.base_infer.sr)),
                "steps": [idx],
                "merge_reasons": [],
            })
            span_debug.append({
                "step": idx,
                "text_src": src_part,
                "text_tgt": text_tgt[idx],
                "time_tag": (start_time, end_time),
                "conf_min": min(conf_slice) if len(conf_slice) > 0 else None,
                "anchor_conf_before": anchor_conf_before,
                "anchor_conf_after": anchor_conf_after,
                "anchor_expand": {
                    "left": anchor_left_expand,
                    "right": anchor_right_expand,
                },
                "anchor_text_expanded": anchor_text_expanded,
            })

        merged_spans = []
        for span in anchored_spans:
            if not merged_spans:
                merged_spans.append(dict(span))
                continue

            prev = merged_spans[-1]
            if span["left_idx"] <= prev["right_idx"]:
                print(
                    f"[WARNING] anchor扩展后，第{span['steps'][0]+1}段与前面段发生重叠，"
                    f"将mask区间合并。"
                )
                prev["left_idx"] = min(prev["left_idx"], span["left_idx"])
                prev["right_idx"] = max(prev["right_idx"], span["right_idx"])
                prev["start_sample"] = min(prev["start_sample"], span["start_sample"])
                prev["end_sample"] = max(prev["end_sample"], span["end_sample"])
                prev["steps"].extend(span["steps"])
                prev["merge_reasons"] = list(dict.fromkeys(prev.get("merge_reasons", []) + ["anchor_overlap"] + span.get("merge_reasons", [])))
            else:
                merged_spans.append(dict(span))

        if overlap_time is not None and len(merged_spans) > 1:
            overlap_start = int(overlap_time[0] * self.base_infer.sr)
            overlap_end = int(overlap_time[1] * self.base_infer.sr)
            overlap_merged_spans = [dict(merged_spans[0])]
            for span in merged_spans[1:]:
                prev = overlap_merged_spans[-1]
                prev_overlap_end = prev["end_sample"] + overlap_end
                curr_overlap_start = span["start_sample"] - overlap_start
                if curr_overlap_start <= prev_overlap_end:
                    print(
                        f"[WARNING] overlap_time导致mask区间重叠：第{span['steps'][0]+1}段与前面段将合并。"
                    )
                    prev["left_idx"] = min(prev["left_idx"], span["left_idx"])
                    prev["right_idx"] = max(prev["right_idx"], span["right_idx"])
                    prev["start_sample"] = min(prev["start_sample"], span["start_sample"])
                    prev["end_sample"] = max(prev["end_sample"], span["end_sample"])
                    prev["steps"].extend(span["steps"])
                    prev["merge_reasons"] = list(dict.fromkeys(prev.get("merge_reasons", []) + ["overlap_time_overlap"] + span.get("merge_reasons", [])))
                else:
                    overlap_merged_spans.append(dict(span))
            merged_spans = overlap_merged_spans

        gen_wav_start = [span["start_sample"] for span in merged_spans]
        gen_wav_end = [span["end_sample"] for span in merged_spans]
        merged_span_debug = [
            {
                "steps": span["steps"],
                "time_tag": (
                    span["start_sample"] / self.base_infer.sr,
                    span["end_sample"] / self.base_infer.sr,
                ),
            }
            for span in merged_spans
        ]
        merged_field_debug = [
            {
                "steps": span["steps"],
                "text_src": [text_src[i] for i in span["steps"]],
                "text_tgt": [text_tgt[i] for i in span["steps"]],
                "time_tag": (
                    span["start_sample"] / self.base_infer.sr,
                    span["end_sample"] / self.base_infer.sr,
                ),
            }
            for span in merged_spans
        ]

        if debug:
            print(f"| [INFO] asr_text: {asr_text}")
            print(f"| [INFO] whole_text_tgt: {whole_text_tgt}")
            print(f"| [INFO] multi_expanded_fields: {expanded_debug}")
            print(f"| [INFO] spans: {span_debug}")
            print(f"| [INFO] merged_mask_spans: {merged_span_debug}")
            print(f"| [INFO] merged_fields: {merged_field_debug}")
            print(f"| [INFO] vad: {vad_value}")

        if extend_enable:
            raise NotImplementedError("extend_enable not implemented in infer5")

        negative_prompt = cfg.get('negative_prompt', None)
        caption = _norm_spaces_caption("")

        timing['before_dit'] = time.time()
        edit_wav = self.base_infer.forward_multimask(
            whole_text_tgt,
            ref_audio=wav_to_edit, # librosa.load(wav_to_edit, sr=24000)[0],
            ref_text=whole_text_tgt,
            prompt=caption,
            cfg_w=cfg.get('cfg_w', None),
            infer_length=cfg.get('infer_length', 0),
            num_step=num_step,
            start_time=cfg.get('vad_len', 0.2),
            end_time=cfg.get('vad_len', 0.2),
            negative_prompt=negative_prompt,
            timestep_annealing_w=cfg.get('timestep_annealing_w', (1.0, 0.0, 1.0)),
            use_amo_sampler=cfg.get('use_amo_sampler', False),
            use_sway=cfg.get('use_sway', False),
            cpm=cfg.get('cpm', 300.0),
            gen_wav_start=gen_wav_start,
            gen_wav_end=gen_wav_end,
            overlap_time=overlap_time,
        )
        timing['after_dit'] = time.time()

        if vad_enable:
            assert vad_value is not None
            assert edit_wav is not None
            vad_reverse = np.array(reverse_base_audio, copy=True)

            edit_start = 0
            for s_ms, e_ms in vad_value:
                s = int(round(s_ms * 24000 / 1000.0))
                e = int(round(e_ms * 24000 / 1000.0))

                seg_len = int(e - s)
                seg_edit = edit_wav[edit_start:edit_start + seg_len]
                actual_seg_len = len(seg_edit)
                vad_reverse[s:s + actual_seg_len] = seg_edit
                edit_start += e - s
                if debug:
                    print(f"| [INFO] 音频长度变化：{edit_start - len(edit_wav)}")

            edit_wav = vad_reverse

        if out_file is None:
            out_file = f'{str(whole_text_tgt)[:5]}.wav'

        if uvr_enable:
            # instrumental_wav, _ = librosa.load(instrumental_path, sr=44100)
            instrumental_wav = np.array(instrumental_wav, copy=True)
            edit_wav = batch_resample(wavs=[edit_wav.astype(np.float32, copy=False)], sample_rates=24000, tgt_sr=audio_raw_sr, resamplers=None, device=self.device)[0]
            # edit_wav = librosa.resample(
            #     edit_wav.astype(np.float32),
            #     orig_sr=24000,
            #     target_sr=audio_raw_sr
            # )
            assert -100 < len(instrumental_wav) - len(edit_wav) < 100, f"{instrumental_wav.shape} != {edit_wav.shape}"
            if len(instrumental_wav) > len(edit_wav):
                instrumental_wav = instrumental_wav[:len(edit_wav)]
            else:
                edit_wav = edit_wav[:len(instrumental_wav)]

            if debug:
                print(f"| [INFO] add instrumental to vocals")
            edit_wav = np.add(edit_wav, instrumental_wav)
            output_wav_path = os.path.join(out_path, out_file)
            sf.write(output_wav_path, edit_wav, audio_raw_sr, "PCM_16")

        else:
            output_wav_path = os.path.join(out_path, out_file)
            sf.write(output_wav_path, edit_wav, 24000, "PCM_16")

        result = SpeechEditResult(
            method="model_infer_multi",
            wav_path=wav_path,
            out_path=out_path,
            out_file=out_file,
            output_wav_path=output_wav_path,
            sample_rate=audio_raw_sr if uvr_enable else self.base_infer.sr,
            whole_text_src=asr_text,
            whole_text_tgt=whole_text_tgt,
            debug={
                "asr_text": asr_text,
                "multi_expanded_fields": expanded_debug,
                "spans": span_debug,
                "merged_mask_spans": merged_span_debug,
                "merged_fields": merged_field_debug,
                "vad": vad_value,
                "uvr_enable": uvr_enable,
                "vad_enable": vad_enable,
                "anchor_enable": anchor_enable,
                "anchor_threshold": anchor_threshold,
                "asr_device": str(self.device),
                "vad_device": str(self.device) if vad_enable else None,
            },
        )
        _append_result_edits_from_merged_spans(
            result=result,
            merged_spans=merged_spans,
            span_debug=span_debug,
            text_src=text_src,
            text_tgt=text_tgt,
            sample_rate=self.base_infer.sr,
        )
        timing['end'] = time.time()
        if debug:
            print(f"| [INFO] timing: {timing}")
            return result, timing
        else:
            ## 这里新加了删除中间临时文件
            for temp_path in [locals().get("uvr_base"), vocals_path, instrumental_path, vad_path]:
                if temp_path and os.path.isfile(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError as e:
                        print(f"[WARNING] 删除临时文件失败: {temp_path}, err={e}")
            return result

    def model_infer_multi2(self, cfg, whole_text_src, whole_text_tgt, text_src:list, text_tgt:list, wav_path, out_path, out_file=None, overlap_time:tuple=None,
                           debug=False, anchor_enable=False, anchor_threshold=0.8, num_step=20):
        '''
        whole_text_src和whole_text_tgt是完整的文本，text_src和text_tgt是单独提出来修改的文本，不需要asr；
        同时处理多段mask；
        '''
        timing = {
            'start': 0, 'before_uvr': 0, 'after_uvr': 0, 'before_aligner': 0, 'after_aligner': 0, 'before_dit': 0, 'after_dit': 0, 'end': 0,
        }
        timing['start'] = time.time()
        if whole_text_src == whole_text_tgt:
            print("输入输出文本相同，无需修改")
            return
        if text_src == "" or text_src == None:
            raise ValueError("text_src 不能为空")
        if text_tgt == "" or text_tgt == None:
            raise ValueError("text_tgt 不能为空")
        if overlap_time is not None and (overlap_time[0] > 0.31 or overlap_time[1] > 0.31):
            print('[WARNING] overlap 时长可能过大')

        os.makedirs(out_path, exist_ok=True)
        negative_prompt = cfg.get('negative_prompt', None)
        caption = _norm_spaces_caption("")

        if not isinstance(text_src, (list, tuple)):
            text_src = [text_src]
        if not isinstance(text_tgt, (list, tuple)):
            text_tgt = [text_tgt]
        if len(text_src) == 0:
            raise ValueError("text_src 不能为空")
        # if len(text_src) != len(text_tgt):
        #     raise ValueError(f"text_src 和 text_tgt 长度不一致: {len(text_src)} != {len(text_tgt)}")
        if len(text_src) != len(text_tgt):
            raise ValueError(f"text_src 和 text_tgt 长度不一致: {len(text_src)} != {len(text_tgt)}")

        if out_file is None:
            out_file = f'{str(whole_text_tgt)[:5]}.wav'

        whole_text_src = raw_text_process_s1s2_tagged(_norm_spaces_caption(whole_text_src))
        whole_text_tgt = raw_text_process_s1s2_tagged(_norm_spaces_caption(whole_text_tgt))
        text_src = [_norm_spaces_caption(x) for x in text_src]
        text_tgt = [_norm_spaces_caption(x) for x in text_tgt]
        whole_text_tgt, text_tgt = _space_uppercase_letters_in_targets(whole_text_tgt, text_tgt)
        if any(x == "" for x in text_src):
            raise ValueError("text_src 中存在空片段")
        if any(x == "" for x in text_tgt):
            raise ValueError("text_tgt 中存在空片段")

        if self.aligner is None:
            self.aligner = ForcedAlignerInfer(
                ckpt='checkpoints/260304_aligner_2tower_v6',
            )
        if self.uvr_model is None:
            self.uvr_model = build_uvr_model(self.device, precision='fp16')

        ignore_literals = ["<S1>", "</S1>", "<S2>", "</S2>", "<S3>", "</S3>", "<S4>", "</S4>"]
        tmp_dir = "/dev/shm/260415"
        # tmp_dir = "tmp/260410"
        os.makedirs(tmp_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        _uvr_base = os.path.join(tmp_dir, f"{stem}_{tag}.wav")
        _vocals_path = os.path.join(tmp_dir, f"{stem}_{tag}_vocals.wav")
        _instrumental_path = os.path.join(tmp_dir, f"{stem}_{tag}_instrumental.wav")

        _audio_raw, audio_raw_sr = librosa.load(wav_path, sr=None)
        if debug:
            print(f"| [INFO] processing uvr on {wav_path}")
        timing['before_uvr'] = time.time()
        uvr_ret = run_uvr_model(wav_path, self.uvr_model, batch_size=16)
        # run_uvr_model(wav_path, self.uvr_model, output_path=_uvr_base)
        timing['after_uvr'] = time.time()

        vocals_wav_raw = uvr_ret['outputs']['vocals'].detach().cpu().numpy()
        vocals_wav_raw = to_mono(vocals_wav_raw)
        instrumental_wav = uvr_ret['outputs']['instrumental'].detach().cpu().numpy()
        instrumental_wav = to_mono(instrumental_wav)

        audio = batch_resample(
            wavs=[vocals_wav_raw.astype(np.float32, copy=False)],
            sample_rates=audio_raw_sr,
            tgt_sr=24000,
            resamplers=None,
            device=self.device,
        )[0]
        aligner_audio_16k = batch_resample(
            wavs=[vocals_wav_raw.astype(np.float32, copy=False)],
            sample_rates=audio_raw_sr,
            tgt_sr=16000,
            resamplers=None,
            device=self.device,
        )[0]
        # audio, _ = librosa.load(_vocals_path, sr=24000)
        span_debug = []

        timing['before_aligner'] = time.time()
        aligner_results = self.aligner.align_with_ignored_patterns(
            audio=(aligner_audio_16k, 16000),
            # audio=(audio, self.base_infer.sr),
            text=whole_text_src,
            ignore_literals=ignore_literals,
            return_json=True,
        )
        timing['after_aligner'] = time.time()
        aligner_results = aligner_results['results'][0]

        aligned_text, align_item_indices = self._build_aligner_char_index(aligner_results)
        src_spans = self._find_ordered_occurrences(aligned_text, text_src)
        conf_lst = [float(item['conf']) for item in aligner_results]
        anchored_spans = []
        for idx, (char_start, char_end, src_part) in enumerate(src_spans):
            start_item_idx = align_item_indices[char_start]
            end_item_idx = align_item_indices[char_end]
            anchor_left_idx = start_item_idx
            anchor_right_idx = end_item_idx
            anchor_left_expand = 0
            anchor_right_expand = 0
            anchor_conf_before = (
                float(conf_lst[start_item_idx]),
                float(conf_lst[end_item_idx]),
            )

            if anchor_enable:
                while float(conf_lst[anchor_left_idx]) < anchor_threshold:
                    if anchor_left_idx == 0:
                        print(
                            f"[WARNING] 第{idx+1}段左侧anchor无法继续扩展：已到达最左边界，"
                            f"当前conf={float(conf_lst[anchor_left_idx]):.4f} < threshold={anchor_threshold}"
                        )
                        break
                    anchor_left_idx -= 1
                    anchor_left_expand += 1

                while float(conf_lst[anchor_right_idx]) < anchor_threshold:
                    if anchor_right_idx == len(conf_lst) - 1:
                        print(
                            f"[WARNING] 第{idx+1}段右侧anchor无法继续扩展：已到达最右边界，"
                            f"当前conf={float(conf_lst[anchor_right_idx]):.4f} < threshold={anchor_threshold}"
                        )
                        break
                    anchor_right_idx += 1
                    anchor_right_expand += 1

                if anchor_left_expand > 5:
                    print(f"[WARNING] 第{idx+1}段左侧anchor扩展过多：{anchor_left_expand} 个字")
                if anchor_right_expand > 5:
                    print(f"[WARNING] 第{idx+1}段右侧anchor扩展过多：{anchor_right_expand} 个字")

            anchor_conf_after = (
                float(conf_lst[anchor_left_idx]),
                float(conf_lst[anchor_right_idx]),
            )
            start_time = float(aligner_results[anchor_left_idx]['start_time'])
            end_time = float(aligner_results[anchor_right_idx]['end_time'])

            conf_slice = [
                float(aligner_results[item_idx]['conf'])
                for item_idx in sorted(set(align_item_indices[char_start:char_end + 1]))
            ]
            start_sample = int(round(start_time * self.base_infer.sr))
            end_sample = int(round(end_time * self.base_infer.sr))

            anchored_spans.append({
                "left_idx": anchor_left_idx,
                "right_idx": anchor_right_idx,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "steps": [idx],
                "merge_reasons": [],
            })
            span_debug.append({
                "step": idx,
                "text_src": src_part,
                "text_tgt": text_tgt[idx],
                "time_tag": (start_time, end_time),
                "conf_min": min(conf_slice) if len(conf_slice) > 0 else None,
                "anchor_conf_before": anchor_conf_before,
                "anchor_conf_after": anchor_conf_after,
                "anchor_expand": {
                    "left": anchor_left_expand,
                    "right": anchor_right_expand,
                },
            })

        merged_spans = []
        for span in anchored_spans:
            if not merged_spans:
                merged_spans.append(dict(span))
                continue

            prev = merged_spans[-1]
            if span["left_idx"] <= prev["right_idx"]:
                print(
                    f"[WARNING] anchor扩展后，第{span['steps'][0]+1}段与前面段发生重叠，"
                    f"将mask区间合并。"
                )
                prev["left_idx"] = min(prev["left_idx"], span["left_idx"])
                prev["right_idx"] = max(prev["right_idx"], span["right_idx"])
                prev["start_sample"] = min(prev["start_sample"], span["start_sample"])
                prev["end_sample"] = max(prev["end_sample"], span["end_sample"])
                prev["steps"].extend(span["steps"])
                prev["merge_reasons"] = list(dict.fromkeys(prev.get("merge_reasons", []) + ["anchor_overlap"] + span.get("merge_reasons", [])))
            else:
                merged_spans.append(dict(span))

        if overlap_time is not None and len(merged_spans) > 1:
            overlap_start = int(overlap_time[0] * self.base_infer.sr)
            overlap_end = int(overlap_time[1] * self.base_infer.sr)
            overlap_merged_spans = [dict(merged_spans[0])]
            for span in merged_spans[1:]:
                prev = overlap_merged_spans[-1]
                prev_overlap_end = prev["end_sample"] + overlap_end
                curr_overlap_start = span["start_sample"] - overlap_start
                if curr_overlap_start <= prev_overlap_end:
                    print(
                        f"[WARNING] overlap_time导致mask区间重叠：第{span['steps'][0]+1}段与前面段将合并。"
                    )
                    prev["left_idx"] = min(prev["left_idx"], span["left_idx"])
                    prev["right_idx"] = max(prev["right_idx"], span["right_idx"])
                    prev["start_sample"] = min(prev["start_sample"], span["start_sample"])
                    prev["end_sample"] = max(prev["end_sample"], span["end_sample"])
                    prev["steps"].extend(span["steps"])
                    prev["merge_reasons"] = list(dict.fromkeys(prev.get("merge_reasons", []) + ["overlap_time_overlap"] + span.get("merge_reasons", [])))
                else:
                    overlap_merged_spans.append(dict(span))
            merged_spans = overlap_merged_spans

        gen_wav_start = [span["start_sample"] for span in merged_spans]
        gen_wav_end = [span["end_sample"] for span in merged_spans]
        for start_sample, end_sample in zip(gen_wav_start, gen_wav_end):
            audio[start_sample:end_sample] = 0.0

        merged_span_debug = [
            {
                "steps": span["steps"],
                "time_tag": (
                    span["start_sample"] / self.base_infer.sr,
                    span["end_sample"] / self.base_infer.sr,
                ),
            }
            for span in merged_spans
        ]
        merged_field_debug = [
            {
                "steps": span["steps"],
                "text_src": [text_src[i] for i in span["steps"]],
                "text_tgt": [text_tgt[i] for i in span["steps"]],
                "time_tag": (
                    span["start_sample"] / self.base_infer.sr,
                    span["end_sample"] / self.base_infer.sr,
                ),
            }
            for span in merged_spans
        ]

        timing['before_dit'] = time.time()
        edit_wav = self.base_infer.forward_multimask(
            whole_text_tgt,
            ref_audio=audio,
            ref_text=whole_text_tgt,
            prompt=caption,
            cfg_w=cfg.get('cfg_w', None),
            infer_length=cfg.get('infer_length', 0),
            num_step=num_step,
            start_time=cfg.get('vad_len', 0.2),
            end_time=cfg.get('vad_len', 0.2),
            negative_prompt=negative_prompt,
            timestep_annealing_w=cfg.get('timestep_annealing_w', (1.0, 0.0, 1.0)),
            use_amo_sampler=cfg.get('use_amo_sampler', False),
            use_sway=cfg.get('use_sway', False),
            cpm=cfg.get('cpm', 300.0),
            gen_wav_start=gen_wav_start,
            gen_wav_end=gen_wav_end,
            overlap_time=overlap_time,
        )
        timing['after_dit'] = time.time()

        if debug:
            print(f"| [INFO] whole_text_src: {whole_text_src}")
            print(f"| [INFO] whole_text_tgt: {whole_text_tgt}")
            print(f"| [INFO] spans: {span_debug}")
            print(f"| [INFO] merged_mask_spans: {merged_span_debug}")
            print(f"| [INFO] merged_fields: {merged_field_debug}")
            print(f"| [INFO] add instrumental to vocals")
            print(f"| [INFO] timing: {timing}")

        instrumental_wav = np.array(instrumental_wav, copy=True)
        edit_wav = batch_resample(
            wavs=[edit_wav.astype(np.float32, copy=False)],
            sample_rates=24000,
            tgt_sr=audio_raw_sr,
            resamplers=None,
            device=self.device,
        )[0]
        # instrumental_wav, _ = librosa.load(_instrumental_path, sr=44100)
        # edit_wav = librosa.resample(
        #     edit_wav.astype(np.float32),
        #     orig_sr=24000,
        #     target_sr=44100
        # )
        assert -0.01 * len(edit_wav) < len(instrumental_wav) - len(edit_wav) < 0.01*len(edit_wav), f"{instrumental_wav.shape} != {edit_wav.shape}，长度差异过大"
        if len(instrumental_wav) > len(edit_wav):
            instrumental_wav = instrumental_wav[:len(edit_wav)]
        else:
            edit_wav = edit_wav[:len(instrumental_wav)]

        edit_wav = np.add(edit_wav, instrumental_wav)
        output_wav_path = os.path.join(out_path, out_file)
        sf.write(output_wav_path, edit_wav, audio_raw_sr, "PCM_16")
        result = SpeechEditResult(
            method="model_infer_multi2",
            wav_path=wav_path,
            out_path=out_path,
            out_file=out_file,
            output_wav_path=output_wav_path,
            sample_rate=audio_raw_sr,
            whole_text_src=whole_text_src,
            whole_text_tgt=whole_text_tgt,
            debug={
                "spans": span_debug,
                "merged_mask_spans": merged_span_debug,
                "merged_fields": merged_field_debug,
                "anchor_enable": anchor_enable,
                "anchor_threshold": anchor_threshold,
                "uvr_device": str(self.device),
                "timing": timing,
            },
        )
        _append_result_edits_from_merged_spans(
            result=result,
            merged_spans=merged_spans,
            span_debug=span_debug,
            text_src=text_src,
            text_tgt=text_tgt,
            sample_rate=self.base_infer.sr,
        )
        timing['end'] = time.time()
        if debug:
            return result, timing
        else:
            for temp_path in [locals().get("_uvr_base"), locals().get("_vocals_path"), locals().get("_instrumental_path")]:
                if temp_path and os.path.isfile(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError as e:
                        print(f"[WARNING] 删除临时文件失败: {temp_path}, err={e}")
            return result

    def model_infer_multi3(self, cfg, whole_text_src, whole_text_tgt, text_src:list, text_tgt:list, gen_wav_start, gen_wav_end, wav_path, out_path, out_file=None, overlap_time:tuple=None, debug=False, num_step=20):
        '''
        与model_infer_multi2相比，手动输入各段的起止时间，而不是靠aligner；
        260413，虽然不需要anchor机制，但是增加了overlap_time触发的重叠合并；
        '''
        timing = {
            'start': 0, 'before_uvr': 0, 'after_uvr': 0, 'before_dit': 0, 'after_dit': 0, 'end': 0,
        }
        timing['start'] = time.time()
        if whole_text_src == whole_text_tgt:
            print("输入输出文本相同，无需修改")
            return

        if text_src == "" or text_src == None:
            raise ValueError("text_src 不能为空")
        if text_tgt == "" or text_tgt == None:
            raise ValueError("text_tgt 不能为空")
        if overlap_time is not None and (overlap_time[0] > 0.31 or overlap_time[1] > 0.31):
            print('[WARNING] overlap 时长可能过大')
        os.makedirs(out_path, exist_ok=True)
        negative_prompt = cfg.get('negative_prompt', None)
        caption = _norm_spaces_caption("")

        if not isinstance(text_src, (list, tuple)):
            text_src = [text_src]
        if not isinstance(text_tgt, (list, tuple)):
            text_tgt = [text_tgt]
        # if len(text_src) != len(text_tgt):
        #     raise ValueError(f"text_src 和 text_tgt 长度不一致: {len(text_src)} != {len(text_tgt)}")
        if len(text_src) == 0:
            raise ValueError("text_src 不能为空")
        if len(text_src) != len(text_tgt):
            raise ValueError(f"text_src 和 text_tgt 长度不一致: {len(text_src)} != {len(text_tgt)}")

        if out_file is None:
            out_file = f'{str(whole_text_tgt)[:5]}.wav'

        whole_text_src = raw_text_process_s1s2_tagged(_norm_spaces_caption(whole_text_src))
        whole_text_tgt = raw_text_process_s1s2_tagged(_norm_spaces_caption(whole_text_tgt))
        text_src = [_norm_spaces_caption(x) for x in text_src]
        text_tgt = [_norm_spaces_caption(x) for x in text_tgt]
        if any(x == "" for x in text_src):
            raise ValueError("text_src 中存在空片段")
        if any(x == "" for x in text_tgt):
            raise ValueError("text_tgt 中存在空片段")

        if isinstance(gen_wav_start, (list, tuple, np.ndarray)):
            gen_wav_start_lst = [int(x) for x in gen_wav_start]
        else:
            gen_wav_start_lst = [int(gen_wav_start)]
        if isinstance(gen_wav_end, (list, tuple, np.ndarray)):
            gen_wav_end_lst = [int(x) for x in gen_wav_end]
        else:
            gen_wav_end_lst = [int(gen_wav_end)]

        if len(gen_wav_start_lst) != len(gen_wav_end_lst):
            raise ValueError(f"gen_wav_start 和 gen_wav_end 长度不一致: {len(gen_wav_start_lst)} != {len(gen_wav_end_lst)}")
        if any(start_sample >= end_sample for start_sample, end_sample in zip(gen_wav_start_lst, gen_wav_end_lst)):
            raise ValueError("gen_wav_start 中有元素大于等于 gen_wav_end 中的元素")
        if any(gen_wav_start_lst[i] > gen_wav_start_lst[i + 1] for i in range(len(gen_wav_start_lst) - 1)):
            raise ValueError("gen_wav_start/gen_wav_end 需要按时间顺序输入")
        if len(gen_wav_start_lst) != len(text_src):
            raise ValueError(
                f"待编辑字段数量和时间段数量不一致: text_src={len(text_src)}, spans={len(gen_wav_start_lst)}"
            )

        if self.uvr_model is None:
            self.uvr_model = build_uvr_model(self.device, precision='fp16')

        audio_raw, audio_raw_sr = librosa.load(wav_path, sr=None)
        if debug:
            print(f"| [INFO] processing uvr on {wav_path}")
        timing['before_uvr'] = time.time()
        uvr_ret = run_uvr_model(wav_path, self.uvr_model, batch_size=16)
        timing['after_uvr'] = time.time()

        vocals_wav_raw = uvr_ret['outputs']['vocals'].detach().cpu().numpy()
        vocals_wav_raw = to_mono(vocals_wav_raw)
        instrumental_wav = uvr_ret['outputs']['instrumental'].detach().cpu().numpy()
        instrumental_wav = to_mono(instrumental_wav)

        # Edit on vocals @ 24k, then resample back and add instrumental (same as multi2).
        audio = batch_resample(
            wavs=[vocals_wav_raw.astype(np.float32, copy=False)],
            sample_rates=audio_raw_sr,
            tgt_sr=24000,
            resamplers=None,
            device=self.device,
        )[0]
        span_debug = []
        spans = []
        for idx, (start_sample, end_sample) in enumerate(zip(gen_wav_start_lst, gen_wav_end_lst)):
            start_sample = max(0, int(start_sample))
            end_sample = min(audio.shape[0], int(end_sample))
            if end_sample <= start_sample:
                raise ValueError(f"非法编辑区间: {(start_sample, end_sample)}")
            spans.append({
                "start_sample": start_sample,
                "end_sample": end_sample,
                "steps": [idx],
                "merge_reasons": [],
            })
            span_debug.append({
                "step": idx,
                "text_src": text_src[idx],
                "text_tgt": text_tgt[idx],
                "time_tag": (start_sample / self.base_infer.sr, end_sample / self.base_infer.sr),
            })

        merged_spans = [dict(spans[0])]
        for span in spans[1:]:
            prev = merged_spans[-1]
            if span["start_sample"] <= prev["end_sample"]:
                print(
                    f"[WARNING] 手动输入的mask区间重叠：第{span['steps'][0]+1}段与前面段将合并。"
                )
                prev["start_sample"] = min(prev["start_sample"], span["start_sample"])
                prev["end_sample"] = max(prev["end_sample"], span["end_sample"])
                prev["steps"].extend(span["steps"])
                prev["merge_reasons"] = list(dict.fromkeys(prev.get("merge_reasons", []) + ["manual_overlap"] + span.get("merge_reasons", [])))
            else:
                merged_spans.append(dict(span))

        if overlap_time is not None and len(merged_spans) > 1:
            overlap_start = int(overlap_time[0] * self.base_infer.sr)
            overlap_end = int(overlap_time[1] * self.base_infer.sr)
            overlap_merged_spans = [dict(merged_spans[0])]
            for span in merged_spans[1:]:
                prev = overlap_merged_spans[-1]
                prev_overlap_end = prev["end_sample"] + overlap_end
                curr_overlap_start = span["start_sample"] - overlap_start
                if curr_overlap_start <= prev_overlap_end:
                    print(
                        f"[WARNING] overlap_time导致mask区间重叠：第{span['steps'][0]+1}段与前面段将合并。"
                    )
                    prev["start_sample"] = min(prev["start_sample"], span["start_sample"])
                    prev["end_sample"] = max(prev["end_sample"], span["end_sample"])
                    prev["steps"].extend(span["steps"])
                    prev["merge_reasons"] = list(dict.fromkeys(prev.get("merge_reasons", []) + ["overlap_time_overlap"] + span.get("merge_reasons", [])))
                else:
                    overlap_merged_spans.append(dict(span))
            merged_spans = overlap_merged_spans

        gen_wav_start_lst = [span["start_sample"] for span in merged_spans]
        gen_wav_end_lst = [span["end_sample"] for span in merged_spans]
        for start_sample, end_sample in zip(gen_wav_start_lst, gen_wav_end_lst):
            audio[start_sample:end_sample] = 0.0

        merged_span_debug = [
            {
                "steps": span["steps"],
                "time_tag": (
                    span["start_sample"] / self.base_infer.sr,
                    span["end_sample"] / self.base_infer.sr,
                ),
            }
            for span in merged_spans
        ]
        merged_field_debug = [
            {
                "steps": span["steps"],
                "text_src": [text_src[i] for i in span["steps"]],
                "text_tgt": [text_tgt[i] for i in span["steps"]],
                "time_tag": (
                    span["start_sample"] / self.base_infer.sr,
                    span["end_sample"] / self.base_infer.sr,
                ),
            }
            for span in merged_spans
        ]

        timing['before_dit'] = time.time()
        edit_wav = self.base_infer.forward_multimask(
            whole_text_tgt,
            ref_audio=audio,
            ref_text=whole_text_tgt,
            prompt=caption,
            cfg_w=cfg.get('cfg_w', None),
            infer_length=cfg.get('infer_length', 0),
            num_step=num_step,
            start_time=cfg.get('vad_len', 0.2),
            end_time=cfg.get('vad_len', 0.2),
            negative_prompt=negative_prompt,
            timestep_annealing_w=cfg.get('timestep_annealing_w', (1.0, 0.0, 1.0)),
            use_amo_sampler=cfg.get('use_amo_sampler', False),
            use_sway=cfg.get('use_sway', False),
            cpm=cfg.get('cpm', 300.0),
            gen_wav_start=gen_wav_start_lst,
            gen_wav_end=gen_wav_end_lst,
            overlap_time=overlap_time,
        )
        timing['after_dit'] = time.time()

        if debug:
            print(f"| [INFO] whole_text_src: {whole_text_src}")
            print(f"| [INFO] whole_text_tgt: {whole_text_tgt}")
            print(f"| [INFO] spans: {span_debug}")
            print(f"| [INFO] merged_mask_spans: {merged_span_debug}")
            print(f"| [INFO] merged_fields: {merged_field_debug}")
            print(f"| [INFO] add instrumental to vocals")
            print(f"| [INFO] timing: {timing}")

        instrumental_wav = np.array(instrumental_wav, copy=True)
        edit_wav = batch_resample(
            wavs=[edit_wav.astype(np.float32, copy=False)],
            sample_rates=24000,
            tgt_sr=audio_raw_sr,
            resamplers=None,
            device=self.device,
        )[0]
        assert -0.01 * len(edit_wav) < len(instrumental_wav) - len(edit_wav) < 0.01 * len(edit_wav), (
            f"{instrumental_wav.shape} != {edit_wav.shape}，长度差异过大"
        )
        if len(instrumental_wav) > len(edit_wav):
            instrumental_wav = instrumental_wav[:len(edit_wav)]
        else:
            edit_wav = edit_wav[:len(instrumental_wav)]
        edit_wav = np.add(edit_wav, instrumental_wav)

        output_wav_path = os.path.join(out_path, out_file)
        sf.write(output_wav_path, edit_wav, audio_raw_sr, "PCM_16")
        result = SpeechEditResult(
            method="model_infer_multi3",
            wav_path=wav_path,
            out_path=out_path,
            out_file=out_file,
            output_wav_path=output_wav_path,
            sample_rate=audio_raw_sr,
            whole_text_src=whole_text_src,
            whole_text_tgt=whole_text_tgt,
            debug={
                "spans": span_debug,
                "merged_mask_spans": merged_span_debug,
                "merged_fields": merged_field_debug,
                "timing": timing,
            },
        )
        _append_result_edits_from_merged_spans(
            result=result,
            merged_spans=merged_spans,
            span_debug=span_debug,
            text_src=text_src,
            text_tgt=text_tgt,
            sample_rate=self.base_infer.sr,
        )
        timing['end'] = time.time()
        if debug:
            return result, timing
        return result

    def model_infer_multi4(self, cfg, whole_text_src, text_src:list, text_tgt:list, wav_path, out_path, out_file=None, overlap_time:tuple=None,
                           debug=False, anchor_enable=False, anchor_threshold=0.8):
        '''
        多字段推理接口，输入完整原始文本和字段替换列表，自动将 text_src 的所有出现都替换为 text_tgt，
        然后转调 model_infer_multi2。
        '''
        if whole_text_src is None or whole_text_src == "":
            raise ValueError("whole_text_src 不能为空")
        if text_src == "" or text_src is None:
            raise ValueError("text_src 不能为空")
        if text_tgt == "" or text_tgt is None:
            raise ValueError("text_tgt 不能为空")

        if not isinstance(text_src, (list, tuple)):
            text_src = [text_src]
        if not isinstance(text_tgt, (list, tuple)):
            text_tgt = [text_tgt]
        if len(text_src) == 0:
            raise ValueError("text_src 不能为空")
        if len(text_src) != len(text_tgt):
            raise ValueError(f"text_src 和 text_tgt 长度不一致: {len(text_src)} != {len(text_tgt)}")

        whole_text_src = raw_text_process_s1s2_tagged(_norm_spaces_caption(whole_text_src))
        text_src = [_norm_spaces_caption(x) for x in text_src]
        text_tgt = [
            _num_string_to_price_zh(_norm_spaces_caption(x)) for x in text_tgt
        ]
        if any(x == "" for x in text_src):
            raise ValueError("text_src 中存在空片段")
        if any(x == "" for x in text_tgt):
            raise ValueError("text_tgt 中存在空片段")

        whole_text_tgt, expanded_text_src, expanded_text_tgt, expanded_debug = self._expand_all_occurrences_for_alignment(
            whole_text_src=whole_text_src,
            text_src=text_src,
            text_tgt=text_tgt,
        )

        if debug:
            print(f"| [INFO] multi4 expanded_fields: {expanded_debug}")
            print(f"| [INFO] multi4 whole_text_src: {whole_text_src}")
            print(f"| [INFO] multi4 whole_text_tgt: {whole_text_tgt}")

        inner_ret = self.model_infer_multi2(
            cfg=cfg,
            whole_text_src=whole_text_src,
            whole_text_tgt=whole_text_tgt,
            text_src=expanded_text_src,
            text_tgt=expanded_text_tgt,
            wav_path=wav_path,
            out_path=out_path,
            out_file=out_file,
            overlap_time=overlap_time,
            debug=debug,
            anchor_enable=anchor_enable,
            anchor_threshold=anchor_threshold,
        )

        timing = None
        result = inner_ret
        if debug and isinstance(inner_ret, tuple) and len(inner_ret) == 2:
            result, timing = inner_ret

        if isinstance(result, SpeechEditResult):
            result.method = "model_infer_multi4"
            if result.debug is None:
                result.debug = {}
            result.debug["multi4_expanded_fields"] = expanded_debug
            result.debug["multi4_whole_text_src"] = whole_text_src
            result.debug["multi4_whole_text_tgt"] = whole_text_tgt

        if debug and timing is not None:
            return result, timing
        return result

    def model_infer_multi5(self, cfg, whole_text_src, whole_text_tgt, wav_path, out_path, out_file=None, overlap_time:tuple=None,
                           debug=False, anchor_enable=False, anchor_threshold=0.8):
        '''
        输入完整原始文本和完整目标文本，自动提取待编辑字段，再转调 model_infer_multi2。
        '''
        if whole_text_src is None or whole_text_src == "":
            raise ValueError("whole_text_src 不能为空")
        if whole_text_tgt is None or whole_text_tgt == "":
            raise ValueError("whole_text_tgt 不能为空")

        whole_text_src = raw_text_process_s1s2_tagged(_norm_spaces_caption(whole_text_src))
        whole_text_tgt = raw_text_process_s1s2_tagged(_norm_spaces_caption(whole_text_tgt))

        if whole_text_src == whole_text_tgt:
            print("输入输出文本相同，无需修改")
            return

        text_src, text_tgt, whole_text_tgt = parse_seq(whole_text_src, whole_text_tgt)
        if len(text_src) == 0 or len(text_tgt) == 0:
            raise ValueError("无法从 whole_text_src 和 whole_text_tgt 中提取有效编辑字段")
        if len(text_src) != len(text_tgt):
            raise ValueError(
                f"parse_seq 解析出的 text_src 和 text_tgt 长度不一致: {len(text_src)} != {len(text_tgt)}"
            )
        if any(x == "" for x in text_src):
            raise ValueError("parse_seq 解析出的 text_src 中存在空片段")
        if any(x == "" for x in text_tgt):
            raise ValueError("parse_seq 解析出的 text_tgt 中存在空片段")

        inner_ret = self.model_infer_multi2(
            cfg=cfg,
            whole_text_src=whole_text_src,
            whole_text_tgt=whole_text_tgt,
            text_src=text_src,
            text_tgt=text_tgt,
            wav_path=wav_path,
            out_path=out_path,
            out_file=out_file,
            overlap_time=overlap_time,
            debug=debug,
            anchor_enable=anchor_enable,
            anchor_threshold=anchor_threshold,
        )

        timing = None
        result = inner_ret
        if debug and isinstance(inner_ret, tuple) and len(inner_ret) == 2:
            result, timing = inner_ret

        if isinstance(result, SpeechEditResult):
            result.method = "model_infer_multi5"
            if result.debug is None:
                result.debug = {}
            result.debug["multi5_whole_text_src"] = whole_text_src
            result.debug["multi5_whole_text_tgt"] = whole_text_tgt
            result.debug["multi5_parsed_text_src"] = text_src
            result.debug["multi5_parsed_text_tgt"] = text_tgt

        if debug and timing is not None:
            return result, timing
        return result

    def model_infer_multi6(self, cfg, span_text_src, span_text_tgt, coarse_time_tag, wav_path, out_path, out_file=None, overlap_time:tuple=None,
                           debug=False, anchor_enable=False, anchor_threshold=0.8, num_step=20):
        '''
        粗粒度秒级时间 + 单段原始文本/目标文本：
        1. 按 coarse_time_tag（单位秒）直接截取局部音频，不做扩窗；
        2. 将 span_text_src/span_text_tgt 视为该局部音频对应的完整原始/目标文本；
        3. 参照 multi5 -> multi2 的思路，在局部完成多 mask 编辑；
        4. 将局部编辑结果与局部 instrumental 合成，再拼回整段 full mix。
        '''
        timing = {
            'start': 0, 'before_uvr': 0, 'after_uvr': 0,
            'before_aligner': 0, 'after_aligner': 0, 'before_dit': 0, 'after_dit': 0, 'end': 0,
        }
        timing['start'] = time.time()

        if span_text_src == "" or span_text_src is None:
            raise ValueError("span_text_src 不能为空")
        if span_text_tgt == "" or span_text_tgt is None:
            raise ValueError("span_text_tgt 不能为空")
        if coarse_time_tag is None or len(coarse_time_tag) != 2:
            raise ValueError("coarse_time_tag 需要是长度为 2 的 (start_sec, end_sec)")
        if overlap_time is not None and (overlap_time[0] > 0.31 or overlap_time[1] > 0.31):
            print('[WARNING] overlap 时长可能过大')

        os.makedirs(out_path, exist_ok=True)
        negative_prompt = cfg.get('negative_prompt', None)
        caption = _norm_spaces_caption("")

        whole_text_src = raw_text_process_s1s2_tagged(_norm_spaces_caption(span_text_src))
        whole_text_tgt = raw_text_process_s1s2_tagged(_num_string_to_price_zh(_norm_spaces_caption(span_text_tgt)))
        if whole_text_src == whole_text_tgt:
            print("输入输出文本相同，无需修改")
            return
        if out_file is None:
            out_file = f'{str(whole_text_tgt)[:5]}.wav'

        text_src, text_tgt, whole_text_tgt = parse_seq(whole_text_src, whole_text_tgt)
        if len(text_src) == 0 or len(text_tgt) == 0:
            raise ValueError(
                f"multi6 在局部文本中没有得到有效编辑字段，"
                f"span_text_src={span_text_src!r}, span_text_tgt={span_text_tgt!r}"
            )
        if len(text_src) != len(text_tgt):
            raise ValueError(f"multi6 解析出的 text_src 和 text_tgt 长度不一致: {len(text_src)} != {len(text_tgt)}")
        if any(x == "" for x in text_src):
            raise ValueError("multi6 解析出的 text_src 中存在空片段")
        if any(x == "" for x in text_tgt):
            raise ValueError("multi6 解析出的 text_tgt 中存在空片段")
        whole_text_tgt, text_tgt = _space_uppercase_letters_in_targets(whole_text_tgt, text_tgt)

        if self.aligner is None:
            self.aligner = ForcedAlignerInfer(
                ckpt='checkpoints/260304_aligner_2tower_v6',
            )
        if self.uvr_model is None:
            self.uvr_model = build_uvr_model(self.device, precision='fp16')

        audio_raw, audio_raw_sr = librosa.load(wav_path, sr=None)
        timing['before_uvr'] = time.time()
        uvr_ret = run_uvr_model(wav_path, self.uvr_model, batch_size=16)
        timing['after_uvr'] = time.time()

        vocals_wav_raw = to_mono(uvr_ret['outputs']['vocals'].detach().cpu().numpy())
        instrumental_wav = to_mono(uvr_ret['outputs']['instrumental'].detach().cpu().numpy())
        base_len = min(len(vocals_wav_raw), len(instrumental_wav))
        vocals_wav_raw = vocals_wav_raw[:base_len]
        instrumental_wav = instrumental_wav[:base_len]
        full_mix_base = np.add(vocals_wav_raw, instrumental_wav)

        coarse_start_sec = float(coarse_time_tag[0])
        coarse_end_sec = float(coarse_time_tag[1])
        if coarse_end_sec <= coarse_start_sec:
            raise ValueError(f"非法 coarse_time_tag: {coarse_time_tag}")
        crop_start_sec = max(0.0, coarse_start_sec)
        crop_end_sec = min(base_len / float(audio_raw_sr), coarse_end_sec)
        crop_start_raw = int(round(crop_start_sec * audio_raw_sr))
        crop_end_raw = int(round(crop_end_sec * audio_raw_sr))
        if crop_end_raw <= crop_start_raw:
            raise ValueError(f"截取区间为空: {coarse_time_tag}")

        local_vocals_raw = np.array(vocals_wav_raw[crop_start_raw:crop_end_raw], copy=True)
        local_instrumental_raw = np.array(instrumental_wav[crop_start_raw:crop_end_raw], copy=True)
        local_audio_24k = batch_resample(
            wavs=[local_vocals_raw.astype(np.float32, copy=False)],
            sample_rates=audio_raw_sr,
            tgt_sr=self.base_infer.sr,
            resamplers=None,
            device=self.device,
        )[0]
        local_audio_16k = batch_resample(
            wavs=[local_vocals_raw.astype(np.float32, copy=False)],
            sample_rates=audio_raw_sr,
            tgt_sr=16000,
            resamplers=None,
            device=self.device,
        )[0]

        ignore_literals = ["<S1>", "</S1>", "<S2>", "</S2>", "<S3>", "</S3>", "<S4>", "</S4>"]
        span_debug = []
        timing['before_aligner'] = time.time()
        aligner_results = self.aligner.align_with_ignored_patterns(
            audio=(local_audio_16k, 16000),
            text=whole_text_src,
            ignore_literals=ignore_literals,
            return_json=True,
        )
        timing['after_aligner'] = time.time()
        aligner_results = aligner_results['results'][0]

        aligned_text, align_item_indices = self._build_aligner_char_index(aligner_results)
        src_spans = self._find_ordered_occurrences(aligned_text, text_src)
        conf_lst = [float(item['conf']) for item in aligner_results]
        anchored_spans = []
        for idx, (char_start, char_end, src_part) in enumerate(src_spans):
            start_item_idx = align_item_indices[char_start]
            end_item_idx = align_item_indices[char_end]
            anchor_left_idx = start_item_idx
            anchor_right_idx = end_item_idx
            anchor_left_expand = 0
            anchor_right_expand = 0
            anchor_conf_before = (
                float(conf_lst[start_item_idx]),
                float(conf_lst[end_item_idx]),
            )

            if anchor_enable:
                while float(conf_lst[anchor_left_idx]) < anchor_threshold:
                    if anchor_left_idx == 0:
                        print(
                            f"[WARNING] 第{idx+1}段左侧anchor无法继续扩展：已到达最左边界，"
                            f"当前conf={float(conf_lst[anchor_left_idx]):.4f} < threshold={anchor_threshold}"
                        )
                        break
                    anchor_left_idx -= 1
                    anchor_left_expand += 1

                while float(conf_lst[anchor_right_idx]) < anchor_threshold:
                    if anchor_right_idx == len(conf_lst) - 1:
                        print(
                            f"[WARNING] 第{idx+1}段右侧anchor无法继续扩展：已到达最右边界，"
                            f"当前conf={float(conf_lst[anchor_right_idx]):.4f} < threshold={anchor_threshold}"
                        )
                        break
                    anchor_right_idx += 1
                    anchor_right_expand += 1

            anchor_conf_after = (
                float(conf_lst[anchor_left_idx]),
                float(conf_lst[anchor_right_idx]),
            )
            anchor_triggered = (anchor_left_expand > 0) or (anchor_right_expand > 0)
            anchor_text_expanded = None
            if anchor_triggered:
                anchor_text_expanded = self._extract_aligner_span_text(
                    aligner_results,
                    anchor_left_idx,
                    anchor_right_idx,
                )
            start_time = float(aligner_results[anchor_left_idx]['start_time'])
            end_time = float(aligner_results[anchor_right_idx]['end_time'])
            conf_slice = [
                float(aligner_results[item_idx]['conf'])
                for item_idx in sorted(set(align_item_indices[char_start:char_end + 1]))
            ]
            start_sample = int(round(start_time * self.base_infer.sr))
            end_sample = int(round(end_time * self.base_infer.sr))
            anchored_spans.append({
                "left_idx": anchor_left_idx,
                "right_idx": anchor_right_idx,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "steps": [idx],
                "merge_reasons": [],
            })
            span_debug.append({
                "step": idx,
                "text_src": src_part,
                "text_tgt": text_tgt[idx],
                "time_tag": (start_time, end_time),
                "conf_min": min(conf_slice) if len(conf_slice) > 0 else None,
                "anchor_conf_before": anchor_conf_before,
                "anchor_conf_after": anchor_conf_after,
                "anchor_expand": {
                    "left": anchor_left_expand,
                    "right": anchor_right_expand,
                },
                "anchor_text_expanded": anchor_text_expanded,
            })

        merged_spans = []
        for span in anchored_spans:
            if not merged_spans:
                merged_spans.append(dict(span))
                continue
            prev = merged_spans[-1]
            if span["left_idx"] <= prev["right_idx"]:
                print(
                    f"[WARNING] anchor扩展后，第{span['steps'][0]+1}段与前面段发生重叠，"
                    f"将mask区间合并。"
                )
                prev["left_idx"] = min(prev["left_idx"], span["left_idx"])
                prev["right_idx"] = max(prev["right_idx"], span["right_idx"])
                prev["start_sample"] = min(prev["start_sample"], span["start_sample"])
                prev["end_sample"] = max(prev["end_sample"], span["end_sample"])
                prev["steps"].extend(span["steps"])
                prev["merge_reasons"] = list(dict.fromkeys(prev.get("merge_reasons", []) + ["anchor_overlap"] + span.get("merge_reasons", [])))
            else:
                merged_spans.append(dict(span))

        if overlap_time is not None and len(merged_spans) > 1:
            overlap_start = int(overlap_time[0] * self.base_infer.sr)
            overlap_end = int(overlap_time[1] * self.base_infer.sr)
            overlap_merged_spans = [dict(merged_spans[0])]
            for span in merged_spans[1:]:
                prev = overlap_merged_spans[-1]
                prev_overlap_end = prev["end_sample"] + overlap_end
                curr_overlap_start = span["start_sample"] - overlap_start
                if curr_overlap_start <= prev_overlap_end:
                    print(
                        f"[WARNING] overlap_time导致mask区间重叠：第{span['steps'][0]+1}段与前面段将合并。"
                    )
                    prev["left_idx"] = min(prev["left_idx"], span["left_idx"])
                    prev["right_idx"] = max(prev["right_idx"], span["right_idx"])
                    prev["start_sample"] = min(prev["start_sample"], span["start_sample"])
                    prev["end_sample"] = max(prev["end_sample"], span["end_sample"])
                    prev["steps"].extend(span["steps"])
                    prev["merge_reasons"] = list(dict.fromkeys(prev.get("merge_reasons", []) + ["overlap_time_overlap"] + span.get("merge_reasons", [])))
                else:
                    overlap_merged_spans.append(dict(span))
            merged_spans = overlap_merged_spans

        local_edit_audio = np.array(local_audio_24k, copy=True)
        gen_wav_start = [span["start_sample"] for span in merged_spans]
        gen_wav_end = [span["end_sample"] for span in merged_spans]
        for start_sample, end_sample in zip(gen_wav_start, gen_wav_end):
            local_edit_audio[start_sample:end_sample] = 0.0

        merged_span_debug = [
            {
                "steps": span["steps"],
                "time_tag": (
                    span["start_sample"] / self.base_infer.sr,
                    span["end_sample"] / self.base_infer.sr,
                ),
            }
            for span in merged_spans
        ]
        merged_field_debug = [
            {
                "steps": span["steps"],
                "text_src": [text_src[i] for i in span["steps"]],
                "text_tgt": [text_tgt[i] for i in span["steps"]],
                "time_tag": (
                    span["start_sample"] / self.base_infer.sr,
                    span["end_sample"] / self.base_infer.sr,
                ),
            }
            for span in merged_spans
        ]

        timing['before_dit'] = time.time()
        edit_local_vocals = self.base_infer.forward_multimask(
            whole_text_tgt,
            ref_audio=local_edit_audio,
            ref_text=whole_text_tgt,
            prompt=caption,
            cfg_w=cfg.get('cfg_w', None),
            infer_length=cfg.get('infer_length', 0),
            num_step=num_step,
            start_time=cfg.get('vad_len', 0.2),
            end_time=cfg.get('vad_len', 0.2),
            negative_prompt=negative_prompt,
            timestep_annealing_w=cfg.get('timestep_annealing_w', (1.0, 0.0, 1.0)),
            use_amo_sampler=cfg.get('use_amo_sampler', False),
            use_sway=cfg.get('use_sway', False),
            cpm=cfg.get('cpm', 300.0),
            gen_wav_start=gen_wav_start,
            gen_wav_end=gen_wav_end,
            overlap_time=overlap_time,
        )
        timing['after_dit'] = time.time()

        edit_local_vocals = batch_resample(
            wavs=[edit_local_vocals.astype(np.float32, copy=False)],
            sample_rates=self.base_infer.sr,
            tgt_sr=audio_raw_sr,
            resamplers=None,
            device=self.device,
        )[0]
        target_local_len = crop_end_raw - crop_start_raw
        length_diff = int(len(edit_local_vocals) - target_local_len)
        max_len_diff = int(cfg.get('multi6_max_len_diff_samples', max(8, round(0.01 * audio_raw_sr))))
        if abs(length_diff) > max_len_diff:
            raise ValueError(
                f"multi6 局部编辑长度漂移过大: edited={len(edit_local_vocals)}, "
                f"target={target_local_len}, diff={length_diff}"
            )
        if length_diff > 0:
            edit_local_vocals = edit_local_vocals[:target_local_len]
        elif length_diff < 0:
            edit_local_vocals = np.pad(edit_local_vocals, (0, -length_diff))
        if len(local_instrumental_raw) > len(edit_local_vocals):
            local_instrumental_raw = local_instrumental_raw[:len(edit_local_vocals)]
        else:
            edit_local_vocals = edit_local_vocals[:len(local_instrumental_raw)]
        local_edit_mix = np.add(edit_local_vocals, local_instrumental_raw)
        if len(local_edit_mix) != target_local_len:
            raise ValueError(
                f"multi6 局部拼接长度异常: local_edit_mix={len(local_edit_mix)}, "
                f"target={target_local_len}"
            )
        final_wav = np.concatenate(
            [
                full_mix_base[:crop_start_raw],
                local_edit_mix,
                full_mix_base[crop_end_raw:],
            ],
            axis=0,
        )
        if len(final_wav) != len(full_mix_base):
            raise ValueError(
                f"multi6 最终输出长度异常: final={len(final_wav)}, base={len(full_mix_base)}"
            )

        output_wav_path = os.path.join(out_path, out_file)
        sf.write(output_wav_path, final_wav, audio_raw_sr, "PCM_16")

        crop_start_24k = int(round(crop_start_raw * self.base_infer.sr / float(audio_raw_sr)))
        merged_spans_global = []
        for span in merged_spans:
            span_global = dict(span)
            span_global["start_sample"] = span["start_sample"] + crop_start_24k
            span_global["end_sample"] = span["end_sample"] + crop_start_24k
            merged_spans_global.append(span_global)

        result = SpeechEditResult(
            method="model_infer_multi6",
            wav_path=wav_path,
            out_path=out_path,
            out_file=out_file,
            output_wav_path=output_wav_path,
            sample_rate=audio_raw_sr,
            whole_text_src=whole_text_src,
            whole_text_tgt=whole_text_tgt,
            debug={
                "coarse_time_tag": coarse_time_tag,
                "crop_time_tag": (crop_start_raw / audio_raw_sr, crop_end_raw / audio_raw_sr),
                "local_whole_text_src": whole_text_src,
                "local_whole_text_tgt": whole_text_tgt,
                "target_local_len": target_local_len,
                "local_edit_len_diff": length_diff,
                "spans": span_debug,
                "merged_mask_spans": merged_span_debug,
                "merged_fields": merged_field_debug,
                "anchor_enable": anchor_enable,
                "anchor_threshold": anchor_threshold,
                "timing": timing,
                "uvr_device": str(self.device),
            },
        )
        _append_result_edits_from_merged_spans(
            result=result,
            merged_spans=merged_spans_global,
            span_debug=span_debug,
            text_src=text_src,
            text_tgt=text_tgt,
            sample_rate=self.base_infer.sr,
        )
        timing['end'] = time.time()
        if debug:
            return result, timing
        return result

    def inference(
        self,
        cfg,
        wav_path,
        out_path,
        out_file=None,
        text_src=None,
        text_tgt=None,
        whole_text_src=None,
        whole_text_tgt=None,
        time_tag=None,
        gen_wav_start=None,
        gen_wav_end=None,
        overlap_time: tuple = (0.16, 0.16),
        debug=False,
        extend_mask=1.0,
        return_wav=False,
        uvr_enable=True,
        vad_enable=True,
        extend_enable=False,
        anchor_enable=True,
        anchor_threshold=0.8,
    ):
        '''
        统一推理分发入口，优先使用多字段方法。

        推荐传参组合：
        1. 单字段手工时间：
           whole_text_tgt + time_tag
           调用model_infer
        
        2. 单字段手工时间，但是只给要改的原字段和目标字段，如果发现原本的文本有残留，可以手动扩展下预测的time tag，改用手工时间：
           text_src + text_tgt + time_tag
           调用model_infer2

        3. 单字段自动定位：
           text_src(str) + text_tgt(str), not list
           由多字段自动定位覆盖掉了，实际调用model_infer_multi，不再是model_infer4

        4. 多字段，待修改字段和目标字段：
           text_src(list) + text_tgt(list)
           调用model_infer_multi，会将每个 text_src 的所有出现都替换为对应 text_tgt，text_tgt可以传数字。
           如果某个字段出现多次，会输出 [WARNING] 提示出现次数，并全部修改。

        5. 多字段完整原始、目标文本，待修改字段和目标字段 + aligner：
           whole_text_src + whole_text_tgt + text_src(list) + text_tgt(list)
           调用model_infer_multi2。里面对英语字母做了空格处理。

        6. 多字段完整原始、目标文本，待修改字段和目标字段 + 手工时间：
           whole_text_src + whole_text_tgt + text_src(list) + text_tgt(list)
           + gen_wav_start + gen_wav_end
           调用model_infer_multi3。whole_text_src允许为空
        
        7. 多字段原始文本，待修改字段和目标字段 + aligner
            whole_text_src + text_src(list) + text_tgt(list)
            调用model_infer_multi4，修改所有出现的text_src，text_tgt可以传数字

        new:
        8. 多字段原始文本、目标文本
            whole_text_src + whole_text_tgt
            调用model_infer_multi5

        9. 单字段原始文本、目标文本 + 粗粒度秒级时间
            text_src + text_tgt + time_tag
            当 cfg.use_multi6=True 时调用 model_infer_multi6
            time_tag 视为 coarse_time_tag（单位秒）

        说明：
        - 单字段直接目标文本模式下，whole_text_tgt 表示最终完整目标文本
        - 多字段模式下，text_src/text_tgt 需要按文本或时间顺序输入
        - gen_wav_start/gen_wav_end 的单位为 sample
        '''

        def _is_seq(x):
            return isinstance(x, (list, tuple, np.ndarray))

        def _as_list(x):
            if _is_seq(x):
                return list(x)
            return [x]

        has_text_pair = (text_src is not None) or (text_tgt is not None)
        has_whole_text_src = whole_text_src is not None
        has_whole_text_tgt = whole_text_tgt is not None
        has_multi_time = (gen_wav_start is not None) or (gen_wav_end is not None)
        use_multi6 = bool(cfg.get('use_multi6', False))
        prefer_multi = _is_seq(text_src) or _is_seq(text_tgt) or has_whole_text_src or has_multi_time or time_tag is None

        if has_text_pair and (text_src is None or text_tgt is None):
            raise ValueError("text_src 和 text_tgt 需要同时提供")
        if has_whole_text_tgt and has_whole_text_src is False and has_multi_time:
            raise ValueError("手工多字段时间模式下 whole_text_src 和 whole_text_tgt 需要同时提供")
        if has_multi_time and (gen_wav_start is None or gen_wav_end is None):
            raise ValueError("gen_wav_start 和 gen_wav_end 需要同时提供")

        if has_whole_text_tgt and not has_whole_text_src and not has_text_pair and not has_multi_time:
            if time_tag is None:
                raise ValueError("whole_text_tgt 模式需要提供 time_tag")
            return self.model_infer(
                cfg=cfg,
                whole_text_tgt=whole_text_tgt,
                wav_path=wav_path,
                time_tag=time_tag,
                out_path=out_path,
                out_file=out_file,
                overlap_time=overlap_time,
                extend_mask=extend_mask,
                return_wav=return_wav,
            )

        # multi5: only whole_text_src + whole_text_tgt (no text_src/text_tgt, no manual spans)
        if has_whole_text_src and has_whole_text_tgt and not has_text_pair and not has_multi_time:
            if time_tag is not None:
                raise ValueError("multi5 不使用 time_tag")
            return self.model_infer_multi5(
                cfg=cfg,
                whole_text_src=whole_text_src,
                whole_text_tgt=whole_text_tgt,
                wav_path=wav_path,
                out_path=out_path,
                out_file=out_file,
                overlap_time=overlap_time,
                debug=debug,
                anchor_enable=anchor_enable,
                anchor_threshold=anchor_threshold,
            )

        if not has_text_pair:
            raise ValueError(
                "无法判断使用哪个推理接口，请至少提供以下组合之一："
                "whole_text_tgt+time_tag；text_src+text_tgt；"
                "whole_text_src+whole_text_tgt；whole_text_src+whole_text_tgt+text_src+text_tgt；"
                "whole_text_src+whole_text_tgt+text_src+text_tgt+gen_wav_start+gen_wav_end"
            )

        if prefer_multi:
            text_src_multi = _as_list(text_src)
            text_tgt_multi = _as_list(text_tgt)

            if return_wav:
                raise ValueError("多字段接口当前不支持 return_wav=True")
            if extend_mask != 1.0:
                raise ValueError("多字段接口当前不支持自定义 extend_mask")
            if time_tag is not None:
                raise ValueError("多字段分发不使用 time_tag，请改用 gen_wav_start/gen_wav_end")

            if has_multi_time:
                if not has_whole_text_src:
                    raise ValueError("手工多字段时间模式需要同时提供 whole_text_src 和 whole_text_tgt")
                return self.model_infer_multi3(
                    cfg=cfg,
                    whole_text_src=whole_text_src,
                    whole_text_tgt=whole_text_tgt,
                    text_src=text_src_multi,
                    text_tgt=text_tgt_multi,
                    gen_wav_start=gen_wav_start,
                    gen_wav_end=gen_wav_end,
                    wav_path=wav_path,
                    out_path=out_path,
                    out_file=out_file,
                    overlap_time=overlap_time,
                    debug=debug,
                )

            if has_whole_text_src:
                if not has_whole_text_tgt:
                    return self.model_infer_multi4(
                        cfg=cfg,
                        whole_text_src=whole_text_src,
                        text_src=text_src_multi,
                        text_tgt=text_tgt_multi,
                        wav_path=wav_path,
                        out_path=out_path,
                        out_file=out_file,
                        overlap_time=overlap_time,
                        debug=debug,
                        anchor_enable=anchor_enable,
                        anchor_threshold=anchor_threshold,
                    )
                return self.model_infer_multi2(
                    cfg=cfg,
                    whole_text_src=whole_text_src,
                    whole_text_tgt=whole_text_tgt,
                    text_src=text_src_multi,
                    text_tgt=text_tgt_multi,
                    wav_path=wav_path,
                    out_path=out_path,
                    out_file=out_file,
                    overlap_time=overlap_time,
                    debug=debug,
                    anchor_enable=anchor_enable,
                    anchor_threshold=anchor_threshold,
                )

            return self.model_infer_multi(
                cfg=cfg,
                text_src=text_src_multi,
                text_tgt=text_tgt_multi,
                wav_path=wav_path,
                out_path=out_path,
                out_file=out_file,
                overlap_time=overlap_time,
                debug=debug,
                uvr_enable=uvr_enable,
                vad_enable=vad_enable,
                extend_enable=extend_enable,
                anchor_enable=anchor_enable,
                anchor_threshold=anchor_threshold,
            )

        if time_tag is not None:
            if use_multi6:
                return self.model_infer_multi6(
                    cfg=cfg,
                    span_text_src=text_src,
                    span_text_tgt=text_tgt,
                    coarse_time_tag=time_tag,
                    wav_path=wav_path,
                    out_path=out_path,
                    out_file=out_file,
                    overlap_time=overlap_time,
                    debug=debug,
                    anchor_enable=anchor_enable,
                    anchor_threshold=anchor_threshold,
                )
            return self.model_infer2(
                cfg=cfg,
                text_src=text_src,
                text_tgt=text_tgt,
                wav_path=wav_path,
                time_tag=time_tag,
                out_path=out_path,
                out_file=out_file,
                overlap_time=overlap_time,
            )

        if return_wav:
            raise ValueError("单字段自动定位模式当前不支持 return_wav=True，请改用 whole_text_tgt+time_tag 组合")
        if extend_mask != 1.0:
            raise ValueError("单字段自动定位模式当前不支持自定义 extend_mask")

        return self.model_infer4(
            cfg=cfg,
            text_src=text_src,
            text_tgt=text_tgt,
            wav_path=wav_path,
            out_path=out_path,
            out_file=out_file,
            overlap_time=overlap_time,
            debug=debug,
            uvr_enable=uvr_enable,
            vad_enable=vad_enable,
            extend_enable=extend_enable,
            anchor_enable=anchor_enable,
            anchor_threshold=anchor_threshold,
        )


def single_infer(args, cfg, out_path):

    device = torch.device('cuda:0')

    dit_ckpt = args.dit_ckpt
    infer_ins = SpeechEditInfer(
        device,
        dit_ckpt=dit_ckpt,
        vae_ckpt=args.vae_ckpt,
        merge_ckpt=args.merge_ckpt,
        merge_weight=args.merge_weight,
        use_sa_front=cfg.get('use_sa_front', False),
    )
    os.makedirs(out_path, exist_ok=True)
    negative_prompt = cfg.get('negative_prompt', None)

    caption = ""
    caption = _norm_spaces_caption(caption)

    whole_text_tgt = "万宝吃的贼香贼香烟那个熏果木烤肉，现在在哈尔滨开新店了，自主的模式，预售的价格二十二块钱，轻松实现暂时哦，你们赶紧的，预售的价格是真香。"
    out_file = os.path.join(out_path, 'test_gen—22-pingjie.wav')

    text_lst = [
        "万宝吃的贼香贼香烟那个熏果木烤肉，现在在哈尔滨开新店了，自主的模式，预售的价格二十二块钱，轻松实现暂时哦，你们赶紧的，预售的价格是真香。",
        "万宝吃的贼香贼香烟那个熏果木烤肉，现在在哈尔滨开新店了，自主的模式，预售的价格五十八块钱，轻松实现暂时哦，你们赶紧的，预售的价格是真香。",
        "万宝吃的贼香贼香烟那个熏果木烤肉，现在在哈尔滨开新店了，自主的模式，预售的价格七十六块钱，轻松实现暂时哦，你们赶紧的，预售的价格是真香。",
        "万宝吃的贼香贼香烟那个熏果木烤肉，现在在哈尔滨开新店了，自主的模式，预售的价格十七块钱，轻松实现暂时哦，你们赶紧的，预售的价格是真香。",
        "万宝吃的贼香贼香烟那个熏果木烤肉，现在在哈尔滨开新店了，自主的模式，预售的价格九十三块钱，轻松实现暂时哦，你们赶紧的，预售的价格是真香。",
        "万宝吃的贼香贼香烟那个熏果木烤肉，现在在哈尔滨开新店了，自主的模式，预售的价格四十块钱，轻松实现暂时哦，你们赶紧的，预售的价格是真香。",

    ]

    out_file_lst = [
        # os.path.join(out_path, 'test_gen—22-pingjie.wav'),
        # os.path.join(out_path, 'test_gen—58-pingjie.wav'),
        # os.path.join(out_path, 'test_gen—76-pingjie.wav'),
        # os.path.join(out_path, 'test_gen—17-pingjie.wav'),
        # os.path.join(out_path, 'test_gen—93-pingjie.wav'),
        # os.path.join(out_path, 'test_gen—40-pingjie.wav'),

        os.path.join(out_path, 'test_gen—22-overlap-pinjie.wav'),
        os.path.join(out_path, 'test_gen—58-overlap-pinjie.wav'),
        os.path.join(out_path, 'test_gen—76-overlap-pinjie.wav'),
        os.path.join(out_path, 'test_gen—17-overlap-pinjie.wav'),
        os.path.join(out_path, 'test_gen—93-overlap-pinjie.wav'),
        os.path.join(out_path, 'test_gen—40-overlap-pinjie.wav'),
    ]

    for text, out_file in zip(text_lst, out_file_lst):

        text = _norm_spaces_caption(text)

        text = raw_text_process_s1s2_tagged(text)

        set_seed(42)

        audio, _ = librosa.load('./user/assets/text.wav', sr=24000)

        # gen_wav_start=int(7.32 * 24000)
        # gen_wav_end=int(7.72 * 24000)
        gen_wav_start=int(7.00 * 24000)
        gen_wav_end=int(8.04 * 24000)

        ref_audio = audio.copy()
        ref_audio[gen_wav_start:gen_wav_end] = 0.0

        num_step = 100

        wav = infer_ins.forward(
            text,
            ref_audio=ref_audio,
            ref_text=text,
            prompt=caption,
            cfg_w=cfg.get('cfg_w', None),
            infer_length=cfg.get('infer_length', 0),
            num_step=num_step,
            start_time=cfg.get('vad_len', 0.2),
            end_time=cfg.get('vad_len', 0.2),
            negative_prompt=negative_prompt,
            timestep_annealing_w=cfg.get('timestep_annealing_w', (1.0, 0.0, 1.0)),
            use_amo_sampler=cfg.get('use_amo_sampler', False),
            use_sway=cfg.get('use_sway', False),
            cpm=cfg.get('cpm', 300.0),
            gen_wav_start=gen_wav_start,
            gen_wav_end=gen_wav_end,
        )

        os.makedirs(out_path, exist_ok=True)
        sf.write(out_file, wav, 24000, "PCM_16")

def infer_test(args, cfg, out_path):
    device = torch.device('cuda:0')

    dit_ckpt = args.dit_ckpt
    infer_ins = SpeechEditInfer(
        device,
        dit_ckpt=dit_ckpt,
        vae_ckpt=args.vae_ckpt,
        merge_ckpt=args.merge_ckpt,
        merge_weight=args.merge_weight,
        use_sa_front=cfg.get('use_sa_front', False),
    )
    infer_ins = SpeechEditInferWrapper(infer_ins)

    text = "万宝吃的贼香贼香烟那个熏果木烤肉，现在在哈尔滨开新店了，自主的模式，预售的价格二十二块钱，轻松实现暂时哦，你们赶紧的，预售的价格是真香。"
    # wav_path = "./user/assets/text.wav"
    # wav_path = "/mnt/bn/sa-ag-data/leike/work/edit/v0200fg10000d5vbmlfog65u50fm29fg.wav"
    # time_tag = (int(7.00 * 24000), int(8.04 * 24000))
    time_tag = (int(7.32 * 24000), int(7.72 * 24000))
    # out_file = os.path.join(out_path, 'test.wav')
    out_file = 'test.wav'

    # infer_ins.model_infer(cfg, text, wav_path, time_tag, out_path, out_file)
    # infer_ins.model_infer2(cfg, '九十三', '十七', wav_path, time_tag, out_path, '17.wav', overlap_time=(0.08, 0.08))
    # infer_ins.model_infer2(cfg, '九十三', '二十二', wav_path, time_tag, out_path, '22.wav', overlap_time=(0.08, 0.08))  
    # infer_ins.model_infer2(cfg, '九十三', '五十八', wav_path, time_tag, out_path, '58.wav', overlap_time=(0.08, 0.08))
    # infer_ins.model_infer2(cfg, '九十三', '六十六', wav_path, time_tag, out_path, '66.wav', overlap_time=(0.08, 0.08))
    # infer_ins.model_infer2(cfg, '九十三', '八十九', wav_path, time_tag, out_path, '89.wav', overlap_time=(0.08, 0.08))
    # infer_ins.model_infer2(cfg, '九十三', '一百零一', wav_path, time_tag, out_path, '101.wav', overlap_time=(0.08, 0.08))
    # infer_ins.model_infer2(cfg, '九十三', '一百九十九', wav_path, time_tag, out_path, '199.wav', overlap_time=(0.08, 0.08))
    
    # infer_ins.model_infer3(cfg, '九十三', '十七', wav_path, out_path+'_aligner', '17.wav', overlap_time=(0.08, 0.08))
    # infer_ins.model_infer3(cfg, '九十三', '二十二', wav_path, out_path+'_aligner', '22.wav', overlap_time=(0.08, 0.08))     
    # infer_ins.model_infer3(cfg, '九十三', '五十八', wav_path, out_path+'_aligner', '58.wav', overlap_time=(0.08, 0.08))
    # infer_ins.model_infer3(cfg, '九十三', '六十六', wav_path, out_path+'_aligner', '66.wav', overlap_time=(0.08, 0.08))
    # infer_ins.model_infer3(cfg, '九十三', '八十九', wav_path, out_path+'_aligner', '89.wav', overlap_time=(0.08, 0.08))
    # infer_ins.model_infer3(cfg, '哈尔滨', '成嘟', wav_path, out_path+'_aligner', '成嘟.wav', overlap_time=(0.08, 0.08)) 
    # infer_ins.model_infer3(cfg, '哈尔滨', '北京', wav_path, out_path+'_aligner', '北京.wav', overlap_time=(0.08, 0.08))
    # infer_ins.model_infer3(cfg, '哈尔滨', '石家庄', wav_path, out_path+'_aligner', '石家庄.wav', overlap_time=(0.08, 0.08)) 
    # infer_ins.model_infer3(cfg, '哈尔滨', '乌鲁木齐', wav_path, out_path+'_aligner', '乌鲁木齐›.wav', overlap_time=(0.08, 0.08))
    # infer_ins.model_infer3(cfg, '十块九', '十一块九', wav_path, out_path+'_aligner', 'v0200fg10000d5vbmlfog65u50fm29fg_11.9.wav', overlap_time=(0.08, 0.08))

    # wav_path = '/mnt/bn/sa-ag-data/leike/work/edit_0326_2/v0345dg10003d71kq2iljht8i87ofhcg.wav'
    # set_seed(42)
    # infer_ins.model_infer3(cfg, '八十九', '一百三十九', wav_path, out_path+'_aligner', 'v0345dg10003d71kq2iljht8i87ofhcg_89_139_032—latest.wav', overlap_time=(0.32, 0.32), debug=True)
    # set_seed(42)
    # infer_ins.model_infer3(cfg, '八十九', '一百三十九', wav_path, out_path+'_aligner', 'v0345dg10003d71kq2iljht8i87ofhcg_89_139_016-latest.wav', overlap_time=(0.16, 0.16), debug=True)
    # set_seed(42)
    # infer_ins.model_infer3(cfg, '八十九', '一百三十九', wav_path, out_path+'_aligner', 'v0345dg10003d71kq2iljht8i87ofhcg_89_139_008-latest.wav', overlap_time=(0.08, 0.08), debug=True)

    # wav_path = '/mnt/bn/sa-ag-data/leike/work/edit_0326_2/v0345dg10003d71ko8qljhtboll5ei60.wav'
    # infer_ins.model_infer3(cfg, '四十多', '五十九块九', wav_path, out_path+'_aligner', 'v0345dg10003d71ko8qljhtboll5ei60_40_59_032-latest.wav', overlap_time=(0.32, 0.32), debug=True)
    # infer_ins.model_infer3(cfg, '四十多', '五十九块九', wav_path, out_path+'_aligner', 'v0345dg10003d71ko8qljhtboll5ei60_40_59_016-latest.wav', overlap_time=(0.16, 0.16), debug=True)
    # infer_ins.model_infer3(cfg, '四十多', '五十九块九', wav_path, out_path+'_aligner', 'v0345dg10003d71ko8qljhtboll5ei60_40_59_008-latest.wav', overlap_time=(0.08, 0.08), debug=True)

    # v0345dg10003d71lirqljht23rbe7o30
    # wav_path = '/mnt/bn/sa-ag-data/leike/work/edit_0326_2/v0345dg10003d71lirqljht23rbe7o30.wav'
    # infer_ins.model_infer3(cfg, '一', '九块九', wav_path, out_path+'_aligner', 'v0345dg10003d71lirqljht23rbe7o30_1_9.9_032-latest.wav', overlap_time=(0.32, 0.32), debug=True)
    # infer_ins.model_infer3(cfg, '一', '九块九', wav_path, out_path+'_aligner', 'v0345dg10003d71lirqljht23rbe7o30_1_9.9_016-latest.wav', overlap_time=(0.16, 0.16), debug=True)
    # infer_ins.model_infer3(cfg, '一', '九块九', wav_path, out_path+'_aligner', 'v0345dg10003d71lirqljht23rbe7o30_1_9.9_008-latest.wav', overlap_time=(0.08, 0.08), debug=True)

    # wav_path = '/mnt/bn/sa-ag-data/leike/work/edit_0326_2/v0345dg10003d71kngaljht7bitns1v0.wav'
    # set_seed(42)
    # infer_ins.model_infer4(cfg, '一百二十八', '一百三十八', wav_path, out_path, 'v0345dg10003d71kngaljht7bitns1v0_128_138_016.wav', overlap_time=(0.16, 0.16), debug=True, anchor_threshold=0.999)

    # wav_path = '/mnt/bn/sa-ag-data/leike/code/ScriptSpeech/infer_out/speech_edit/260408/260407_speechedit_alldata_noise_cfg[1.5,3.0]_step13000_aligner_extend/v0345dg10003d71koq2ljht23rbcmtc0.wav'
    # set_seed(42)
    # infer_ins.model_infer4(cfg, '九十多', '一百零八', wav_path, out_path, 'v0345dg10003d71koq2ljht23rbcmtc0_90_108_016.wav', overlap_time=(0.16, 0.16), debug=True, anchor_threshold=0.999)

    # wav_path = '/mnt/bn/sa-ag-data/leike/code/ScriptSpeech/infer_out/speech_edit/260408/260407_speechedit_alldata_noise_cfg[1.5,3.0]_step13000_aligner_extend/v0345dg10003d71ko8qljhtboll5ei60.wav'
    # set_seed(42)
    # infer_ins.model_infer4(cfg, '四十多', '五十九块九', wav_path, out_path, 'v0345dg10003d71ko8qljhtboll5ei60_40_59.9_016.wav', overlap_time=(0.16, 0.16), debug=True, anchor_threshold=0.999)

    # wav_path = './infer_out/speech_edit/260408/260407_speechedit_alldata_noise_cfg[1.5,3.0]_step13000_aligner_extend/v0345dg10003d71kth2ljht1inruo8ug.wav'
    # set_seed(42)
    # infer_ins.model_infer4(cfg, '九十九', '一百九十九', wav_path, out_path, 'v0345dg10003d71kth2ljht1inruo8ug_99_199_016.wav', overlap_time=(0.16, 0.16), debug=True, anchor_threshold=0.999)



    # wav_path = '/mnt/bn/sa-ag-data/leike/work/edit_0326_2/v0345dg10003d71kngaljht7bitns1v0.wav'
    # set_seed(42)
    # infer_ins.model_infer5(cfg, ['一百二十八', '说再见了'], ['一百三十八', '说拜拜了'], wav_path, out_path, 'v0345dg10003d71kngaljht7bitns1v0_128_138_016.wav', overlap_time=(0.16, 0.16), debug=True)


    # wav_path = '/mnt/bn/sa-ag-data/leike/work/edit_0326/v0345dg10003d71koeiljhtah6aqjjag.wav'
    # set_seed(42)
    # infer_ins.model_infer5(cfg, ['幺三八', '双人套餐'], ['九十八', '三人套餐'], wav_path, out_path, 'v0345dg10003d71koeiljhtah6aqjjag_edit_016.wav', overlap_time=(0.16, 0.16), debug=True)

    # wav_path = '/mnt/bn/sa-ag-data/leike/work/edit_0326_2/v0345dg10003d71ko8qljhtboll5ei60.wav'
    # set_seed(42)
    # infer_ins.model_infer5(cfg, ['四十多', '四楼', '新开', '锅锅', '新店', '开业的'], ['六十八', '五楼', '老字号', '盆盆', '老店', '周年庆'], wav_path, out_path, 'v0345dg10003d71ko8qljhtboll5ei60_edit_016.wav', overlap_time=(0.16, 0.16), debug=True)

    wav_path = 'infer_out/speech_edit/260417/260416_speechedit_alldata_cfg[1.5,3.0]_step31000_aligner/v0345dg10003d71kqi2ljht2drc63c0g.wav'
    set_seed(42)
    infer_ins.inference(cfg=cfg, wav_path=wav_path, out_path=out_path, out_file='test_case2.wav', text_src='五十八块八', text_tgt='19.9', time_tag=[5.36*24000, 6.24*24000], debug=True)

def ads_infer(args, cfg, out_path):
    device = torch.device('cuda:0')

    dit_ckpt = args.dit_ckpt
    infer_ins = SpeechEditInfer(
        device,
        dit_ckpt=dit_ckpt,
        vae_ckpt=args.vae_ckpt,
        merge_ckpt=args.merge_ckpt,
        merge_weight=args.merge_weight,
        use_sa_front=cfg.get('use_sa_front', False),
    )
    infer_ins = SpeechEditInferWrapper(infer_ins)

    # wav_path = '/mnt/bn/sa-ag-data/leike/work/edit/v0200fg10000d5vbmlfog65u50fm29fg.wav'
    # infer_ins.model_infer3(cfg, '十块九', '十一块九', wav_path, out_path+'_aligner', 'v0200fg10000d5vbmlfog65u50fm29fg_11.9_016.wav', overlap_time=(0.16, 0.16))
    # shutil.copy(wav_path, os.path.join(out_path+'_aligner', 'v0200fg10000d5vbmlfog65u50fm29fg.wav'))

    # wav_path = '/mnt/bn/sa-ag-data/leike/work/edit/v2800fgi0000d5tm8b7og65hc3h7kg80.wav'
    # infer_ins.model_infer3(cfg, '八十九', '一百三十九', wav_path, out_path+'_aligner', 'v2800fgi0000d5tm8b7og65hc3h7kg80_139_016.wav', overlap_time=(0.16, 0.16))
    # shutil.copy(wav_path, os.path.join(out_path+'_aligner', 'v2800fgi0000d5tm8b7og65hc3h7kg80.wav'))

    # wav_path = '/mnt/bn/sa-ag-data/leike/work/edit/v2800fgi0000d56f7o7og65l2ps2tsb0.wav'
    # infer_ins.model_infer3(cfg, '九块九', '十三块八', wav_path, out_path+'_aligner', 'v2800fgi0000d56f7o7og65l2ps2tsb0_13.8_016.wav', overlap_time=(0.16, 0.16))
    # shutil.copy(wav_path, os.path.join(out_path+'_aligner', 'v2800fgi0000d56f7o7og65l2ps2tsb0.wav'))

    # wav_path = '/mnt/bn/sa-ag-data/leike/work/edit/v2800fgi0000d60ck87og65sl62s1dlg.wav'
    # infer_ins.model_infer3(cfg, '八十八', '九十八', wav_path, out_path+'_aligner', 'v2800fgi0000d60ck87og65sl62s1dlg_98_016.wav', overlap_time=(0.16, 0.16))
    # shutil.copy(wav_path, os.path.join(out_path+'_aligner', 'v2800fgi0000d60ck87og65sl62s1dlg.wav'))

    wav_path = '/mnt/bn/sa-ag-data/leike/work/edit_0326/v0345dg10003d71kth2ljht1inruo8ug.wav'
    # infer_ins.model_infer3(cfg, '九十九', '一百九十九', wav_path, out_path+'_aligner', 'v0345dg10003d71kth2ljht1inruo8ug_158_016.wav', overlap_time=(0.16, 0.16))

    time_tag = (int(3.08 * 24000), int(3.60 * 24000))
    infer_ins.model_infer2(cfg, '九十九', '一百九十九', wav_path, time_tag, out_path+'_aligner', 'v0345dg10003d71kth2ljht1inruo8ug_158_016_infer2_024.wav', overlap_time=(0.24, 0.24))

    shutil.copy(wav_path, os.path.join(out_path+'_aligner', 'v0345dg10003d71kth2ljht1inruo8ug.wav'))

def inference_usage(args, cfg, out_path):
    device = torch.device('cuda:0')

    dit_ckpt = args.dit_ckpt
    infer_ins = SpeechEditInfer(
        device,
        dit_ckpt=dit_ckpt,
        vae_ckpt=args.vae_ckpt,
        merge_ckpt=args.merge_ckpt,
        merge_weight=args.merge_weight,
        use_sa_front=cfg.get('use_sa_front', False),
    )
    infer_ins = SpeechEditInferWrapper(infer_ins)

    # whole_text_tgt = "万宝吃的贼香贼香烟那个熏果木烤肉，现在在哈尔滨开新店了，自主的模式，预售的价格二十二块钱，轻松实现暂时哦，你们赶紧的，预售的价格是真香。"
    # time_tag = (int(7.32 * 24000), int(7.72 * 24000))

    # wav_path = "./user/assets/text.wav"
    # out_file = 'infer1.wav'
    # edit_result = infer_ins.inference(
    #     cfg=cfg,
    #     wav_path=wav_path,
    #     out_path=out_path,
    #     out_file=out_file,
    #     whole_text_tgt=whole_text_tgt,
    #     time_tag=time_tag,
    #     overlap_time=(0.16, 0.16),
    # )
    # print(edit_result)

    # edit_result = infer_ins.inference(
    #     cfg=cfg,
    #     wav_path=wav_path,
    #     out_path=out_path,
    #     out_file='infer2.wav',
    #     text_src='九十三',
    #     text_tgt='十七',
    #     time_tag=time_tag,
    #     overlap_time=(0.08, 0.08),
    # )
    # print(edit_result)

    wav_path = '/mnt/bn/sa-ag-data/leike/code/ScriptSpeech/infer_out/speech_edit/260420/260416_speechedit_alldata_cfg[1.5,3.0]_step31000_aligner/v0345dg10003d71l8hiljht9n57lp3og.wav'
    set_seed(42)
    edit_result = infer_ins.inference(
        cfg=cfg,
        wav_path=wav_path,
        out_path=out_path,
        out_file='infer4_9.9.wav',
        text_src='九块九',
        text_tgt='十九块九',
        overlap_time=(0.16, 0.16),
        debug=True,
        anchor_enable=False,
        # anchor_threshold=0.9,
    )
    # print(edit_result)

    # wav_path = '/mnt/bn/sa-ag-data/leike/work/edit_0326_2/v0345dg10003d71kngaljht7bitns1v0.wav'
    # set_seed(42)
    # edit_result = infer_ins.inference(
    #     cfg=cfg,
    #     wav_path=wav_path,
    #     out_path=out_path,
    #     out_file='multi1.wav',
    #     text_src=['一百二十八', '说再见了'],
    #     text_tgt=['一百三十八', '说拜拜了'],
    #     overlap_time=(0.16, 0.16),
    #     debug=True,
    # )
    # # print(edit_result)

    # wav_path = '/mnt/bn/sa-ag-data/leike/code/ScriptSpeech/user/work_lk/edit_0409/audio/000001_2458630bdc3a4b2aa6345e4c315bfae9.wav'
    # edit_result = infer_ins.inference(
    #     cfg=cfg,
    #     wav_path=wav_path,
    #     out_path=out_path,
    #     out_file="multi2.wav",
    #     whole_text_tgt='澳宝一分钟焗油护发素，1分钟就能让头发顺滑到打结都难！作为每天忙到脚不沾地的人，我之前总找不到又快又实惠的护发素，直到发现它！它主打秒速渗透技术，1分钟快速修护，洗完头发又顺又亮，还能密集修护5大损伤，营养微粒抗毛躁，清爽顺滑不毛躁。平时价格就很亲民，现在澳宝年末福利大放送，买一瓶到手两大瓶正装，再送四袋旅行装，到手整整六大件，太划算了！之前试过很多护发素，要么贵得肉疼，要么效果不好，这款真的击中我的心巴！香味清新好闻，留香持久，使用后头发顺滑有光泽，不假滑不油腻。澳宝是大品牌，口碑销量都好，还有7天无理由退货，一次性包装未破损可退。',
    #     whole_text_src='澳宝一分钟焗油护发素，1分钟就能让头发顺滑到打结都难！作为每天忙到脚不沾地的人，我之前总找不到又快又华惠的护发素，直到发现它！它主打秒速渗透技术，1分钟快速修护，洗完头发又顺又亮，还能密集修护5大损伤，营养微粒抗毛躁，清爽顺滑不毛躁。平时价格就很亲民，现在澳宝年末福利大放送，买一瓶到手两大瓶正装，再送四袋旅行装，到手整整六大件，太划算了！之前试过很多护发素，要么贵得肉疼，要么效果不好，这款真的击中我的心巴！香味清新好闻，留香持久，使用后头发顺滑有光泽，不假滑不油腻。阿宝是大品牌，口碑销量都好，还有7天无理由退货，一次性包装未破损可退。',
    #     text_src=['华', '阿'],
    #     text_tgt=['实', '澳'],
    #     overlap_time=(0.16, 0.16),
    #     debug=True,
    # )
    # # print(edit_result)

    # edit_result = infer_ins.inference(
    #     cfg=cfg,
    #     wav_path='user/work_lk/edit_0409/audio/000020_65120602bd76417294392aedefe999c6.wav',
    #     out_path=out_path,
    #     out_file="multi3.wav",
    #     whole_text_src='男宝家长举个手！是不是总遇到尿裤前面一大坨、后面不导流的情况？我家娃之前用的尿裤，一泡尿下去前面鼓得像气球，后面还是干的，跑两步就漏，换得我手软！直到找到三只小猪这款！它前面做了芯体加宽10MM的设计，加上双层防侧漏，男宝好动也不怕漏；里面是进口高分子芯体，吸水快还能均匀分散，不结团不起坨；面层底膜都是双热风棉柔结构，软乎乎的跟云朵似的，娃穿一天小屁屁都是干的，再也没红过！而且特别轻薄，有五万个透气微孔，夏天穿也不闷；医护级别的认证，用着特别放心！关键价格还特亲民，性价比绝了，家里有娃的赶紧囤！',
    #     whole_text_tgt='男宝家长举个手！是不是总遇到尿裤前面一大坨、后面不导流的情况？我家娃之前用的尿裤，一泡尿下去前面鼓得像气球，后面还是干的，跑两步就漏，换得我手软！直到找到三只小猪这款！它前面做了芯体加宽十毫米的设计，加上双层防侧漏，男宝好动也不怕漏；里面是进口高分子芯体，吸水快还能均匀分散，不结团不起坨；面层底膜都是双热风棉柔结构，软乎乎的跟云朵似的，娃穿一天小屁屁都是干的，再也没红过！而且特别轻薄，有五万个透气微孔，夏天穿也不闷；医护级别的认证，用着特别放心！关键价格还特亲民，性价比绝了，家里有娃的赶紧囤！',
    #     text_src=['10MM'],
    #     text_tgt=['十毫米'],
    #     gen_wav_start=[14.92*24000], 
    #     gen_wav_end=[15.36*24000],
    #     overlap_time=(0.16, 0.16),
    #     debug=True
    # )
    # # print(edit_result)

    # wav_path = '/mnt/bn/sa-ag-data/leike/code/ScriptSpeech/user/work_lk/edit_0409/audio/000001_2458630bdc3a4b2aa6345e4c315bfae9.wav'
    # edit_result = infer_ins.inference(
    #     cfg=cfg,
    #     wav_path=wav_path,
    #     out_path=out_path,
    #     out_file="multi4.wav",
    #     whole_text_src='澳宝一分钟焗油护发素，1分钟就能让头发顺滑到打结都难！作为每天忙到脚不沾地的人，我之前总找不到又快又华惠的护发素，直到发现它！它主打秒速渗透技术，1分钟快速修护，洗完头发又顺又亮，还能密集修护5大损伤，营养微粒抗毛躁，清爽顺滑不毛躁。平时价格就很亲民，现在澳宝年末福利大放送，买一瓶到手两大瓶正装，再送四袋旅行装，到手整整六大件，太划算了！之前试过很多护发素，要么贵得肉疼，要么效果不好，这款真的击中我的心巴！香味清新好闻，留香持久，使用后头发顺滑有光泽，不假滑不油腻。阿宝是大品牌，口碑销量都好，还有7天无理由退货，一次性包装未破损可退。',
    #     text_src=['1分钟', '每天', '四'],
    #     text_tgt=['两分钟', '每周', '9'],
    #     overlap_time=(0.16, 0.16),
    #     debug=True
    # )
    # # print(edit_result)

def bad_case(args, cfg, out_path):
    device = torch.device('cuda:0')

    dit_ckpt = args.dit_ckpt
    infer_ins = SpeechEditInfer(
        device,
        dit_ckpt=dit_ckpt,
        vae_ckpt=args.vae_ckpt,
        merge_ckpt=args.merge_ckpt,
        merge_weight=args.merge_weight,
        use_sa_front=cfg.get('use_sa_front', False),
    )
    infer_ins = SpeechEditInferWrapper(infer_ins)

    wav_path = 'infer_out/speech_edit/260417/260416_speechedit_alldata_cfg[1.5,3.0]_step31000_aligner/v0345dg10003d71kok2ljhtf5pthsuv0.wav'
    set_seed(42)
    infer_ins.inference(
        cfg=cfg,
        wav_path=wav_path,
        out_path=out_path,
        out_file='bad_case1.wav',
        text_src='二十九',
        text_tgt='八十二',
        # whole_text_src='二十九到一百的，但你竟是真不敢再卖啊，这两天都已经卖爆了，卖完了就下架了。二十九块九当成一百花，相当于二十九在吃火锅。这锅羊毛如果不好的话，就可以打了。上城区火锅好评榜，一直在榜的浅鱼鱼，晚上不定西有歌手助唱，七成以上都是回头客，他们家叫钱鱼鱼啊，就是把贵州和重庆火锅mix到了一起，能吃辣的话，一定要尝试一下他们家这个现烤的牛油锅底大的很巴适不知道点什么的话，可以忙点他们这个一百七十肉的双人套菜，四荤三宿两杯饮料一份小吃，而且连锅底和调料都包含在里面了。肉里呢有一份钱家鲜吊龙、潮汕手工牛肉丸、安克斯、肥牛和一份海贝大咖。肉呢都是现点现切的品质也很不错，这类吃的出了新鲜。这一大桌香香辣辣的专至美味头在东站附近想要吃火锅的，赶紧才上圈过来试试。',
        # whole_text_tgt='八十二到一百的，但你竟是真不敢再卖啊，这两天都已经卖爆了，卖完了就下架了。二十九块九当成一百花，相当于二十九在吃火锅。这锅羊毛如果不好的话，就可以打了。上城区火锅好评榜，一直在榜的浅鱼鱼，晚上不定西有歌手助唱，七成以上都是回头客，他们家叫钱鱼鱼啊，就是把贵州和重庆火锅mix到了一起，能吃辣的话，一定要尝试一下他们家这个现烤的牛油锅底大的很巴适不知道点什么的话，可以忙点他们这个一百七十肉的双人套菜，四荤三宿两杯饮料一份小吃，而且连锅底和调料都包含在里面了。肉里呢有一份钱家鲜吊龙、潮汕手工牛肉丸、安克斯、肥牛和一份海贝大咖。肉呢都是现点现切的品质也很不错，这类吃的出了新鲜。这一大桌香香辣辣的专至美味头在东站附近想要吃火锅的，赶紧才上圈过来试试。',
        # gen_wav_start=[0],
        # gen_wav_end=[0.48*24000],
        overlap_time=(0.16, 0.16),
        debug=True,
    )

    wav_path = 'infer_out/speech_edit/260417/260416_speechedit_alldata_cfg[1.5,3.0]_step31000_aligner/v0345dg10003d71kqmqljht8ovekia70.wav'
    set_seed(42)
    edit_result = infer_ins.inference(
        cfg=cfg,
        wav_path=wav_path,
        out_path=out_path,
        out_file='bad_case2.wav',
        text_src='别人都说我胆子大',
        text_tgt='大家都说我胆子大',
        overlap_time=(0.16, 0.16),
        debug=True,
        anchor_threshold=0.9,
    )

    wav_path = 'infer_out/speech_edit/260417/260416_speechedit_alldata_cfg[1.5,3.0]_step31000_aligner/v0345dg10003d71kqi2ljht2drc63c0g.wav'
    set_seed(42)
    edit_result = infer_ins.inference(
        cfg=cfg,
        wav_path=wav_path,
        out_path=out_path,
        out_file='bad_case3.wav',
        text_src='荆州',
        text_tgt='C B D',
        overlap_time=(0.16, 0.16),
        debug=True,
        anchor_threshold=0.9,
    )

    wav_path = 'user/work_lk/edit_0409/audio/000009_25c256b154394fa0875bc42b74f67ccd.wav'
    set_seed(42)
    edit_result = infer_ins.inference(
        cfg=cfg,
        wav_path=wav_path,
        out_path=out_path,
        out_file='bad_case4.wav',
        whole_text_src='还在等家里的暖气片慢吞吞升温？开久了空气干得脸起皮？这款取暖器直接把你想要的全搞定！我是专门测家电的博主，摸过的取暖器没有一百也有八十，这款是我真心觉得能处的！它用的是石墨烯速热技术，开机几秒就出热风，客厅30秒就能暖起来，再也不用裹着厚被子等半天！还有24小时持续雾化加湿，开一整晚也不会口干舌燥，早上起来脸还是润润的！关键是能语音控制，喊一声小维小维打开取暖器，秒响应！老人小孩用着都方便，不用弯腰找遥控器！外观还带火焰氛围灯，放客厅像个小摆件，朋友来都问链接！还有一搜节能模式，晚上开着也不心疼电费，智能省电又安心！广角摇头设计，采暖面积覆盖31-40㎡，客厅卧室都能暖到，不用挪来挪去！安全方面也超贴心，倾倒断电+过热保护，家里有宝宝也能放心用，3小时无操作自动断电！运行起来还超安静，晚上开着睡觉也不吵，不会吵到宝宝休息！颜值高还不占地方，放哪里都好看，移动也方便，客厅、卧室、书房想放哪放哪！之前双十一都卖爆了，好多姐妹收到都追着我要链接，现在入手超划算，别错过！',
        # whole_text_tgt='还在等家里的暖气片慢吞吞升温？开久了空气干得脸起皮？这款取暖器直接把你想要的全搞定！我是专门测家电的博主，摸过的取暖器没有一百也有八十，这款是我真心觉得能处的！它用的是石墨烯速热技术，开机几秒就出热风，客厅30秒就能暖起来，再也不用裹着厚被子等半天！还有24小时持续雾化加湿，开一整晚也不会口干舌燥，早上起来脸还是润润的！关键是能语音控制，喊一声小维小维打开取暖器，秒响应！老人小孩用着都方便，不用弯腰找遥控器！外观还带火焰氛围灯，放客厅像个小摆件，朋友来都问链接！还有ECO节能模式，晚上开着也不心疼电费，智能省电又安心！广角摇头设计，采暖面积覆盖31-40㎡，客厅卧室都能暖到，不用挪来挪去！安全方面也超贴心，倾倒断电+过热保护，家里有宝宝也能放心用，3小时无操作自动断电！运行起来还超安静，晚上开着睡觉也不吵，不会吵到宝宝休息！颜值高还不占地方，放哪里都好看，移动也方便，客厅、卧室、书房想放哪放哪！之前双十一都卖爆了，好多姐妹收到都追着我要链接，现在入手超划算，别错过！',
        text_src=['一搜'],
        text_tgt=['超级'],
        # gen_wav_start=[38.64*24000],
        # gen_wav_end=[39.04*24000],
        overlap_time=(0.16, 0.16),
        debug=True,
        anchor_threshold=0.9,
    )

    wav_path = 'user/work_lk/edit_0409/audio/000009_25c256b154394fa0875bc42b74f67ccd.wav'
    set_seed(42)
    edit_result = infer_ins.inference(
        cfg=cfg,
        wav_path=wav_path,
        out_path=out_path,
        out_file='bad_case5.wav',
        whole_text_src='还在等家里的暖气片慢吞吞升温？开久了空气干得脸起皮？这款取暖器直接把你想要的全搞定！我是专门测家电的博主，摸过的取暖器没有一百也有八十，这款是我真心觉得能处的！它用的是石墨烯速热技术，开机几秒就出热风，客厅30秒就能暖起来，再也不用裹着厚被子等半天！还有24小时持续雾化加湿，开一整晚也不会口干舌燥，早上起来脸还是润润的！关键是能语音控制，喊一声小维小维打开取暖器，秒响应！老人小孩用着都方便，不用弯腰找遥控器！外观还带火焰氛围灯，放客厅像个小摆件，朋友来都问链接！还有一搜节能模式，晚上开着也不心疼电费，智能省电又安心！广角摇头设计，采暖面积覆盖31-40㎡，客厅卧室都能暖到，不用挪来挪去！安全方面也超贴心，倾倒断电+过热保护，家里有宝宝也能放心用，3小时无操作自动断电！运行起来还超安静，晚上开着睡觉也不吵，不会吵到宝宝休息！颜值高还不占地方，放哪里都好看，移动也方便，客厅、卧室、书房想放哪放哪！之前双十一都卖爆了，好多姐妹收到都追着我要链接，现在入手超划算，别错过！',
        whole_text_tgt='还在等家里的暖气片慢吞吞升温？开久了空气干得脸起皮？这款取暖器直接把你想要的全搞定！我是专门测家电的博主，摸过的取暖器没有一百也有八十，这款是我真心觉得能处的！它用的是石墨烯速热技术，开机几秒就出热风，客厅30秒就能暖起来，再也不用裹着厚被子等半天！还有24小时持续雾化加湿，开一整晚也不会口干舌燥，早上起来脸还是润润的！关键是能语音控制，喊一声小维小维打开取暖器，秒响应！老人小孩用着都方便，不用弯腰找遥控器！外观还带火焰氛围灯，放客厅像个小摆件，朋友来都问链接！还有E C O节能模式，晚上开着也不心疼电费，智能省电又安心！广角摇头设计，采暖面积覆盖31-40㎡，客厅卧室都能暖到，不用挪来挪去！安全方面也超贴心，倾倒断电+过热保护，家里有宝宝也能放心用，3小时无操作自动断电！运行起来还超安静，晚上开着睡觉也不吵，不会吵到宝宝休息！颜值高还不占地方，放哪里都好看，移动也方便，客厅、卧室、书房想放哪放哪！之前双十一都卖爆了，好多姐妹收到都追着我要链接，现在入手超划算，别错过！',
        text_src=['一搜'],
        text_tgt=['E C O'],
        gen_wav_start=[38.64*24000],
        gen_wav_end=[39.04*24000],
        overlap_time=(0.16, 0.16),
        debug=True,
        anchor_threshold=0.9,
    )

    wav_path = 'user/work_lk/edit_0409/audio/000009_25c256b154394fa0875bc42b74f67ccd.wav'
    set_seed(42)
    edit_result = infer_ins.inference(
        cfg=cfg,
        wav_path=wav_path,
        out_path=out_path,
        out_file='bad_case5.wav',
        whole_text_src='还在等家里的暖气片慢吞吞升温？开久了空气干得脸起皮？这款取暖器直接把你想要的全搞定！我是专门测家电的博主，摸过的取暖器没有一百也有八十，这款是我真心觉得能处的！它用的是石墨烯速热技术，开机几秒就出热风，客厅30秒就能暖起来，再也不用裹着厚被子等半天！还有24小时持续雾化加湿，开一整晚也不会口干舌燥，早上起来脸还是润润的！关键是能语音控制，喊一声小维小维打开取暖器，秒响应！老人小孩用着都方便，不用弯腰找遥控器！外观还带火焰氛围灯，放客厅像个小摆件，朋友来都问链接！还有一搜节能模式，晚上开着也不心疼电费，智能省电又安心！广角摇头设计，采暖面积覆盖31-40㎡，客厅卧室都能暖到，不用挪来挪去！安全方面也超贴心，倾倒断电+过热保护，家里有宝宝也能放心用，3小时无操作自动断电！运行起来还超安静，晚上开着睡觉也不吵，不会吵到宝宝休息！颜值高还不占地方，放哪里都好看，移动也方便，客厅、卧室、书房想放哪放哪！之前双十一都卖爆了，好多姐妹收到都追着我要链接，现在入手超划算，别错过！',
        whole_text_tgt='还在等家里的暖气片慢吞吞升温？开久了空气干得脸起皮？这款取暖器直接把你想要的全搞定！我是专门测家电的博主，摸过的取暖器没有一百也有八十，这款是我真心觉得能处的！它用的是石墨烯速热技术，开机几秒就出热风，客厅30秒就能暖起来，再也不用裹着厚被子等半天！还有24小时持续雾化加湿，开一整晚也不会口干舌燥，早上起来脸还是润润的！关键是能语音控制，喊一声小维小维打开取暖器，秒响应！老人小孩用着都方便，不用弯腰找遥控器！外观还带火焰氛围灯，放客厅像个小摆件，朋友来都问链接！还有E C O节能模式，晚上开着也不心疼电费，智能省电又安心！广角摇头设计，采暖面积覆盖31-40㎡，客厅卧室都能暖到，不用挪来挪去！安全方面也超贴心，倾倒断电+过热保护，家里有宝宝也能放心用，3小时无操作自动断电！运行起来还超安静，晚上开着睡觉也不吵，不会吵到宝宝休息！颜值高还不占地方，放哪里都好看，移动也方便，客厅、卧室、书房想放哪放哪！之前双十一都卖爆了，好多姐妹收到都追着我要链接，现在入手超划算，别错过！',
        text_src=['一搜'],
        text_tgt=['E C O'],
        gen_wav_start=[38.64*24000],
        gen_wav_end=[39.04*24000],
        overlap_time=(0.16, 0.16),
        debug=True,
        anchor_threshold=0.9,
    )

def character_test(args, cfg, out_path):
    device = torch.device('cuda:0')

    dit_ckpt = args.dit_ckpt
    infer_ins = SpeechEditInfer(
        device,
        dit_ckpt=dit_ckpt,
        vae_ckpt=args.vae_ckpt,
        merge_ckpt=args.merge_ckpt,
        merge_weight=args.merge_weight,
        use_sa_front=cfg.get('use_sa_front', False),
    )
    infer_ins = SpeechEditInferWrapper(infer_ins)


    wav_path = 'infer_out/speech_edit/260417/260416_speechedit_alldata_cfg[1.5,3.0]_step31000_aligner/v0345dg10003d71kok2ljhtf5pthsuv0.wav'
    set_seed(42)
    infer_ins.inference(
        cfg=cfg,
        wav_path=wav_path,
        out_path=out_path,
        out_file='char_case1.wav',
        text_src='上城区',
        text_tgt='C B D',
        whole_text_src='二十九到一百的，但你竟是真不敢再卖啊，这两天都已经卖爆了，卖完了就下架了。二十九块九当成一百花，相当于二十九在吃火锅。这锅羊毛如果不好的话，就可以打了。上城区火锅好评榜，一直在榜的浅鱼鱼，晚上不定西有歌手助唱，七成以上都是回头客，他们家叫钱鱼鱼啊，就是把贵州和重庆火锅mix到了一起，能吃辣的话，一定要尝试一下他们家这个现烤的牛油锅底大的很巴适不知道点什么的话，可以忙点他们这个一百七十肉的双人套菜，四荤三宿两杯饮料一份小吃，而且连锅底和调料都包含在里面了。肉里呢有一份钱家鲜吊龙、潮汕手工牛肉丸、安克斯、肥牛和一份海贝大咖。肉呢都是现点现切的品质也很不错，这类吃的出了新鲜。这一大桌香香辣辣的专至美味头在东站附近想要吃火锅的，赶紧才上圈过来试试。',
        whole_text_tgt='二十九到一百的，但你竟是真不敢再卖啊，这两天都已经卖爆了，卖完了就下架了。二十九块九当成一百花，相当于二十九在吃火锅。这锅羊毛如果不好的话，就可以打了。C B D 火锅好评榜，一直在榜的浅鱼鱼，晚上不定西有歌手助唱，七成以上都是回头客，他们家叫钱鱼鱼啊，就是把贵州和重庆火锅mix到了一起，能吃辣的话，一定要尝试一下他们家这个现烤的牛油锅底大的很巴适不知道点什么的话，可以忙点他们这个一百七十肉的双人套菜，四荤三宿两杯饮料一份小吃，而且连锅底和调料都包含在里面了。肉里呢有一份钱家鲜吊龙、潮汕手工牛肉丸、安克斯、肥牛和一份海贝大咖。肉呢都是现点现切的品质也很不错，这类吃的出了新鲜。这一大桌香香辣辣的专至美味头在东站附近想要吃火锅的，赶紧才上圈过来试试。',
        # gen_wav_start=[0],
        # gen_wav_end=[0.48*24000],
        overlap_time=(0.16, 0.16),
        debug=True,
    )

    wav_path = 'infer_out/speech_edit/260417/260416_speechedit_alldata_cfg[1.5,3.0]_step31000_aligner/v0345dg10003d71kok2ljhtf5pthsuv0.wav'
    set_seed(42)
    infer_ins.inference(
        cfg=cfg,
        wav_path=wav_path,
        out_path=out_path,
        out_file='char_case2.wav',
        text_src='双人套菜',
        text_tgt='A B C D',
        whole_text_src='二十九到一百的，但你竟是真不敢再卖啊，这两天都已经卖爆了，卖完了就下架了。二十九块九当成一百花，相当于二十九在吃火锅。这锅羊毛如果不好的话，就可以打了。上城区火锅好评榜，一直在榜的浅鱼鱼，晚上不定西有歌手助唱，七成以上都是回头客，他们家叫钱鱼鱼啊，就是把贵州和重庆火锅mix到了一起，能吃辣的话，一定要尝试一下他们家这个现烤的牛油锅底大的很巴适不知道点什么的话，可以忙点他们这个一百七十肉的双人套菜，四荤三宿两杯饮料一份小吃，而且连锅底和调料都包含在里面了。肉里呢有一份钱家鲜吊龙、潮汕手工牛肉丸、安克斯、肥牛和一份海贝大咖。肉呢都是现点现切的品质也很不错，这类吃的出了新鲜。这一大桌香香辣辣的专至美味头在东站附近想要吃火锅的，赶紧才上圈过来试试。',
        whole_text_tgt='二十九到一百的，但你竟是真不敢再卖啊，这两天都已经卖爆了，卖完了就下架了。二十九块九当成一百花，相当于二十九在吃火锅。这锅羊毛如果不好的话，就可以打了。上城区火锅好评榜，一直在榜的浅鱼鱼，晚上不定西有歌手助唱，七成以上都是回头客，他们家叫钱鱼鱼啊，就是把贵州和重庆火锅mix到了一起，能吃辣的话，一定要尝试一下他们家这个现烤的牛油锅底大的很巴适不知道点什么的话，可以忙点他们这个一百七十肉的A B C D，四荤三宿两杯饮料一份小吃，而且连锅底和调料都包含在里面了。肉里呢有一份钱家鲜吊龙、潮汕手工牛肉丸、安克斯、肥牛和一份海贝大咖。肉呢都是现点现切的品质也很不错，这类吃的出了新鲜。这一大桌香香辣辣的专至美味头在东站附近想要吃火锅的，赶紧才上圈过来试试。',
        # gen_wav_start=[0],
        # gen_wav_end=[0.48*24000],
        overlap_time=(0.16, 0.16),
        debug=True,
    )
    
def test_price_zh():
    test_cases = [
        ("1234", "一千二百三十四"),
        ("78.9", "七十八块九"),
        ("1.09", "一块零九"),
        ("1.23", "一块二毛三"),
        ("1.20", "一块二"),
        ("3.40", "三块四"),
        ("十七块", "十七块"),
        ("  9 ", "九"),
    ]

    print("[TEST] _num_string_to_price_zh")
    for src, expected in test_cases:
        actual = _num_string_to_price_zh(src)
        print(f"  {src!r} -> {actual!r}")
        if actual != expected:
            raise AssertionError(
                f"_num_string_to_price_zh({src!r}) = {actual!r}, expected {expected!r}"
            )
    print("[TEST] _num_string_to_price_zh passed")


if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv

        load_dotenv('.env.local')

    kill_void()

    try:
        set_start_method('spawn')  # 多进程启动方式，Linux/Windows 通用
    except RuntimeError:
        pass

    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to YAML config", type=str,
                        default='egs/tts/inference/swan_bench_caption_1spk.yaml')
    parser.add_argument("--dit_ckpt", help="Path to model", type=str,
                        default='checkpoints/260412_speechedit_alldata')
    parser.add_argument("--merge_ckpt", help="Path to merge model", type=str)
    parser.add_argument("--merge_weight", help="Weight to merge model", type=float)
    parser.add_argument("--vae_ckpt", help="Path to VAE ckpt", type=str,
                        default='checkpoints/251120_wavvae_v4_unfreeze')
    args = parser.parse_args()
    # 读取 config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_path = f'/mnt/bn/sa-ag-data/leike/code/ScriptSpeech/tmp/260421/260412_speechedit_alldata'

    # single_infer(args, cfg, out_path)
    # infer_test(args, cfg, out_path)
    # ads_infer(args, cfg, out_path)
    # inference_usage(args, cfg, out_path)
    # test_price_zh()
    bad_case(args, cfg, out_path)
    # character_test(args, cfg, out_path)
