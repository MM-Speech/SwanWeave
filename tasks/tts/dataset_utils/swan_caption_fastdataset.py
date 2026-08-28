import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import os
import random
import re
import traceback
import tempfile
import json
import time
import math
import hashlib
from copy import deepcopy
import torch
import torchaudio
import numpy as np
from utils.commons.base_shm_dataset import get_from_global_stores
from utils.commons.dataset_utils import  SkipLogger
from utils.commons.tos_utils_v2 import TosClient
from tasks.tts.dataset_utils.swan_base_fastdataset import SwanTTSShmDataset
from tasks.tts.dataset_utils.base_fastdataset_v2 import shuffle_speaker_ids, build_speaker_shuffle_map, apply_speaker_shuffle, simple_text_process

DEBUG = False

# ======== precompiled regex patterns ========
_SPACE_RE = re.compile(r'\s+')
_XML_TAG_RE = re.compile(r'<[^>]+>')
_BGM_FIELD_TAG_RE = re.compile(
    r'<\s*BGM\s*>(.*?)</\s*BGM\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)
_BGM_VALUE_PREFIX_RE = re.compile(r'^\s*(?:BGM|背景音乐)\s*[:：]\s*', flags=re.IGNORECASE)
_NO_BGM_VALUES = {"无", "无背景音乐", "None", "no", "null", "0"}
_QUALITY_KEYS = ("pesq", "stoi", "si_sdr", "mos")
_QUALITY_NORMAL_THRESH = {
    "pesq": 2.0,
    "stoi": 0.85,
    "si_sdr": 0.0,
    "mos": 3.0,
}
_QUALITY_HIGH_THRESH = {
    "pesq": 3.0,
    "stoi": 0.90,
    "si_sdr": 5.0,
    "mos": 3.5,
}
QUALITY_FLAG_LOW = 0
QUALITY_FLAG_NORMAL = 1
QUALITY_FLAG_HIGH = 2
QUALITY_FLAG_UNKNOWN = 3

_S1S2_TAG_RE = re.compile(
    r'<\s*(S[1-4])\s*>(.*?)</\s*S[1-4]\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)

_S1S2_TEXT_RE = re.compile(
    r'<\s*(S[1-4])\s*>(.*?)</\s*\1\s*>',
    flags=re.IGNORECASE | re.DOTALL,
)

_I_OPEN_TAG_RE = re.compile(r'<\s*I\s*>', flags=re.IGNORECASE)
_I_CLOSE_TAG_RE = re.compile(r'</\s*I\s*>', flags=re.IGNORECASE)

_S_CLOSE_PERIOD_RE = re.compile(
    r'(</\s*S[1-4]\s*>)\s*(?:。|\.(?!\.))',
    flags=re.IGNORECASE
)

_S_ANY_OPEN_TAG_RE = re.compile(r'<\s*S(\d+)\s*>', flags=re.IGNORECASE)
_SPEAKER_SPLIT_RE = re.compile(r'[;,，、\n]+')

def normalize_sx_periods_keep_layout(s: str) -> str:
    """
    仅对 <S1-4>...</S1-4> 内部做 simple_text_process(补句号等)，
    并删除紧跟在 </Sx> 后面的外部句号（。或单个 .）。
    不会全局压缩空白，因此不会破坏 <SPK> 等其它字段里的换行。
    """
    if not isinstance(s, str) or not s:
        return "" if s is None else str(s)

    if _S1S2_TEXT_RE.search(s) is None:
        return s

    def _repl(m):
        tag = m.group(1).upper()
        inner = m.group(2)
        
        if inner.strip() == "":
            return ""

        inner_proc = simple_text_process(inner, wav=None, wav_len=None) or ""
        inner_proc = inner_proc.strip()  # 去掉 simple_text_process 可能留下的 ". " 尾巴空格

        return f"<{tag}>{inner_proc}</{tag}>"

    out = _S1S2_TEXT_RE.sub(_repl, s)
    out = _S_CLOSE_PERIOD_RE.sub(r'\1', out)  # 删除 </Sx> 后的外部句号
    return out


def _normalize_bgm_value(v) -> str:
    if v is None:
        return ""
    s = v if isinstance(v, str) else str(v)
    s = s.strip()
    if not s:
        return ""

    m = _BGM_FIELD_TAG_RE.search(s)
    if m is not None:
        s = (m.group(1) or "").strip()

    s = _BGM_VALUE_PREFIX_RE.sub("", s).strip()
    return s


def _bgm_flag_from_raw(v) -> bool:
    s = _normalize_bgm_value(v)
    if not s:
        return False
    if s.lower() in _NO_BGM_VALUES:
        return False
    return True


def _safe_float(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _extract_quality_scores(meta: dict) -> dict:
    scores = {}
    quality_score = meta.get("quality_score") if isinstance(meta, dict) else None
    for key in _QUALITY_KEYS:
        value = None
        if isinstance(quality_score, dict):
            value = quality_score.get(key, None)
        if value is None and isinstance(meta, dict):
            value = meta.get(key, None)
        scores[key] = _safe_float(value)
    return scores


def _has_full_quality_scores(scores: dict) -> bool:
    return all(scores.get(key) is not None for key in _QUALITY_KEYS)


def _quality_bucket_from_scores(scores: dict) -> str:
    if not _has_full_quality_scores(scores):
        return "unknown"
    if all(scores[key] >= _QUALITY_HIGH_THRESH[key] for key in _QUALITY_KEYS):
        return "high_quality"
    if all(scores[key] >= _QUALITY_NORMAL_THRESH[key] for key in _QUALITY_KEYS):
        return "normal_quality"
    return "low_quality"


def _quality_flag_from_scores(scores: dict) -> int:
    bucket = _quality_bucket_from_scores(scores)
    if bucket == "high_quality":
        return QUALITY_FLAG_HIGH
    if bucket == "normal_quality":
        return QUALITY_FLAG_NORMAL
    if bucket == "low_quality":
        return QUALITY_FLAG_LOW
    return QUALITY_FLAG_UNKNOWN


def _quality_overall_phrase(bucket: str) -> str:
    if bucket == "high_quality":
        return "high quality"
    if bucket == "normal_quality":
        return "normal quality"
    if bucket == "low_quality":
        return "low quality"
    return ""


def _quality_phrase_for_metric(metric: str, value):
    if value is None:
        return None
    if metric == "stoi":
        if value >= _QUALITY_HIGH_THRESH["stoi"]:
            return "very clear speech"
        if value >= _QUALITY_NORMAL_THRESH["stoi"]:
            return "clear speech"
        return "unclear speech"
    if metric == "si_sdr":
        if value >= _QUALITY_HIGH_THRESH["si_sdr"]:
            return "low noise"
        if value >= _QUALITY_NORMAL_THRESH["si_sdr"]:
            return "some noise"
        return "noticeable noise"
    if metric == "pesq":
        if value >= _QUALITY_HIGH_THRESH["pesq"]:
            return "natural timbre"
        if value >= _QUALITY_NORMAL_THRESH["pesq"]:
            return "fair timbre"
        return "artifacts"
    if metric == "mos":
        if value >= _QUALITY_HIGH_THRESH["mos"]:
            return "good overall quality"
        if value >= _QUALITY_NORMAL_THRESH["mos"]:
            return "fair overall quality"
        return "poor overall quality"
    return None


def _build_quality_caption_from_meta(meta: dict):
    scores = _extract_quality_scores(meta)
    quality_flag = _quality_flag_from_scores(scores)
    bucket = _quality_bucket_from_scores(scores)
    overall = _quality_overall_phrase(bucket)
    if overall == "":
        return "", quality_flag

    phrases = [overall]
    for key in ("stoi", "si_sdr", "pesq", "mos"):
        phrase = _quality_phrase_for_metric(key, scores.get(key))
        if phrase:
            phrases.append(phrase)
    quality_txt = f"Quality: {'; '.join(phrases)}."
    return quality_txt, quality_flag


def quality_bucket_from_flag(flag) -> str:
    try:
        flag = int(flag)
    except (TypeError, ValueError):
        return "unknown"
    if flag == QUALITY_FLAG_HIGH:
        return "high_quality"
    if flag == QUALITY_FLAG_NORMAL:
        return "normal_quality"
    if flag == QUALITY_FLAG_LOW:
        return "low_quality"
    return "unknown"


def build_quality_caption_from_flag(flag):
    bucket = quality_bucket_from_flag(flag)
    overall = _quality_overall_phrase(bucket)
    if overall == "":
        return "", QUALITY_FLAG_UNKNOWN

    bucket_metric_phrase = {
        "high_quality": {
            "stoi": "very clear speech",
            "si_sdr": "low noise",
            "pesq": "natural timbre",
            "mos": "good overall quality",
        },
        "normal_quality": {
            "stoi": "clear speech",
            "si_sdr": "some noise",
            "pesq": "fair timbre",
            "mos": "fair overall quality",
        },
        "low_quality": {
            "stoi": "unclear speech",
            "si_sdr": "noticeable noise",
            "pesq": "artifacts",
            "mos": "poor overall quality",
        },
    }
    phrases = [overall]
    phrases.extend(bucket_metric_phrase[bucket][key] for key in ("stoi", "si_sdr", "pesq", "mos"))
    quality_txt = f"Quality: {'; '.join(phrases)}."
    return quality_txt, int(flag)


def build_quality_caption_from_meta(meta: dict):
    return _build_quality_caption_from_meta(meta)


def quality_flag_from_meta(meta: dict) -> int:
    scores = _extract_quality_scores(meta)
    return _quality_flag_from_scores(scores)


def _count_speaker_tags_in_caption(caption: str) -> int:
    if not isinstance(caption, str) or not caption:
        return 0
    ids = set()
    for x in _S_ANY_OPEN_TAG_RE.findall(caption):
        try:
            ids.add(int(x))
        except Exception:
            continue
    return len(ids)

def _max_speaker_tag_id_in_caption(caption: str) -> int:
    """返回 caption 中出现的最大 <Sx> 的 x；若没有则返回 0。"""
    if not isinstance(caption, str) or not caption:
        return 0
    m = 0
    for x in _S_ANY_OPEN_TAG_RE.findall(caption):
        try:
            v = int(x)
        except Exception:
            continue
        if v > m:
            m = v
    return m


def _count_speakers_field(sc: dict) -> int:
    if not isinstance(sc, dict):
        return 0
    speakers = sc.get('speakers', [])
    if isinstance(speakers, (list, tuple)):
        cleaned = [str(s).strip() for s in speakers if str(s).strip()]
        return len(cleaned)
    if isinstance(speakers, str):
        parts = [p.strip() for p in _SPEAKER_SPLIT_RE.split(speakers) if p.strip()]
        return len(parts)
    return 0


def _encode_tag_pattern(tokenizer, s: str):
    """返回 tokenizer.encode(s) 的 token id 序列（list[int]）"""
    ids = tokenizer.encode(s)
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    return list(map(int, ids))

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

import random
import torch

def add_random_fade_in_out_padding(
    wav: torch.Tensor,
    sr: int,
    fm_wav: int,
    max_sec: float = 1.0,
):
    """
    在 wav 前后各随机加 [0, max_sec] 秒的“静音 padding（全 0）”，且【不修改原始 wav 任意采样点】。
    额外段长度会量化到 fm_wav 的整数倍。

    返回：new_wav, pre_pad_len_samples, post_pad_len_samples
    且保证：new_wav[pre_len : pre_len + len(wav)] 与原 wav 完全相同（逐点相等）
    """
    if wav is None or wav.numel() == 0:
        return wav, 0, 0
    if wav.dim() != 1:
        raise ValueError(f"wav must be 1D, got shape={tuple(wav.shape)}")

    x0 = wav  # 只读引用，绝不 in-place 改它
    N = int(x0.numel())

    # 采样随机时长（秒 -> 采样点数）
    pre_len = int(random.random() * max_sec * sr)
    post_len = int(random.random() * max_sec * sr)

    # 量化到 fm_wav 的整数倍
    if fm_wav and fm_wav > 0:
        pre_len = int(round(pre_len / fm_wav)) * fm_wav
        post_len = int(round(post_len / fm_wav)) * fm_wav

        max_len = int(max_sec * sr)
        max_len = (max_len // fm_wav) * fm_wav
        pre_len = min(pre_len, max_len)
        post_len = min(post_len, max_len)

    device = x0.device
    dtype = x0.dtype

    # 纯静音 padding
    pre = torch.zeros((pre_len,), device=device, dtype=dtype) if pre_len > 0 else None
    post = torch.zeros((post_len,), device=device, dtype=dtype) if post_len > 0 else None

    # concat without touching original wav
    if pre is not None and post is not None:
        out = torch.cat([pre, x0, post], dim=0)
    elif pre is not None:
        out = torch.cat([pre, x0], dim=0)
    elif post is not None:
        out = torch.cat([x0, post], dim=0)
    else:
        out = x0

    # 强保证：中间原始段逐点一致
    if pre_len > 0:
        assert torch.equal(out[pre_len:pre_len + N], x0), "Original wav was modified!"
    else:
        assert torch.equal(out[:N], x0), "Original wav was modified!"

    return out.contiguous(), int(pre_len), int(post_len)

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

def _build_caption_from_swan_caption(meta: dict, quality_txt: str = ""):
    """
    从 meta['swan_caption'] 读取 env/bgm/speakers/local 拼接 caption。
    - env 可选：非空才输出 <ENV>...</ENV>
    - quality 可选：非空则并入 global 尾部
    - bgm 必选：始终输出 <BGM>...</BGM>，缺失/空白则写 '无背景音乐'
    - speakers 可选：<SPK>...</SPK>（支持 list / str）
    - global = ENV(可选) + BGM(必选) + SPK(可选) + Quality(可选)
    - local 不整体包裹，直接拼接在 global 后面
    返回: (caption, global_text, local_text)
    """
    sc = meta.get("swan_caption")
    if not isinstance(sc, dict):
        return "", "", ""

    env = sc.get("env", "")
    bgm = sc.get("bgm", "")
    speakers = sc.get("speakers", [])
    local = sc.get("mapped_local", "") or sc.get("local", "")

    # speakers: list[str] -> "\n".join
    if isinstance(speakers, (list, tuple)):
        speakers_txt = ";".join([str(x) for x in speakers if str(x).strip() != ""])

    parts = []

    # ENV 可选
    if str(env).strip() != "":
        parts.append(f" Environment: {{ {env} }} ")

    # BGM 必选：缺失/空白 -> 随机选一个“无音乐”同义词条
    bgm_str = str(bgm) if bgm is not None else ""
    if bgm_str.strip() == "":
        bgm_str = "无"
    parts.append(f" <BGM>BGM: {{ {bgm_str} }}.</BGM> ")

    # SPK 可选
    if str(speakers_txt).strip() != "":
        parts.append(f" Speaker: {{ {speakers_txt} }}. ")

    if str(quality_txt).strip() != "":
        parts.append(f" {str(quality_txt).strip()} ")

    global_txt = "".join(parts)

    local_txt = _local_to_str(local)
    local_txt = normalize_sx_periods_keep_layout(local_txt)
    local_txt = f"Content: {{ {local_txt} }}."
    caption = global_txt + local_txt
    caption = caption.replace('Subject','Speaker').replace('<I>','<Audio>').replace('</I>','</Audio>').replace('<w>','<S1>').replace('</w>','</S1>')

    return caption, global_txt, local_txt

# ======= Skip 打印辅助 =======
def _print_skip(reason: str, i_worker=None, n_worker=None, item_name: str = None, extra: str = ""):
    if DEBUG is False:
        return
    try:
        worker_info = f"{i_worker}/{n_worker}" if i_worker is not None and n_worker is not None else "-"
        name_info = f", item={item_name}" if item_name else ""
        extra_info = f", {extra}" if extra else ""
        print(f"[SKIP][{worker_info}] {reason}{name_info}{extra_info}")
    except Exception:
        # 避免打印本身异常影响主逻辑
        pass


def _to_jsonable(obj, _depth: int = 0, _max_depth: int = 4, _max_list: int = 50):
    """Best-effort 将 obj 转成可 json dump 的结构，避免把大 tensor/ndarray/wav 写入日志。"""
    if _depth >= _max_depth:
        return str(obj)

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # numpy scalar
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            return str(obj)

    # torch tensor
    if torch.is_tensor(obj):
        try:
            return {
                "__type__": "torch.Tensor",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
                "numel": int(obj.numel()),
            }
        except Exception:
            return {"__type__": "torch.Tensor"}

    # numpy array
    if isinstance(obj, np.ndarray):
        return {
            "__type__": "np.ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "size": int(obj.size),
        }

    if isinstance(obj, bytes):
        return {"__type__": "bytes", "len": len(obj)}

    if isinstance(obj, (list, tuple)):
        out = []
        for i, x in enumerate(obj[:_max_list]):
            out.append(_to_jsonable(x, _depth=_depth + 1))
        if len(obj) > _max_list:
            out.append({"__truncated__": len(obj) - _max_list})
        return out

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # 常见大字段直接做摘要
            if k in {"wav", "audio", "mel", "spec", "pcm", "samples"}:
                out[k] = _to_jsonable(v, _depth=_max_depth)
                continue
            try:
                kk = str(k)
            except Exception:
                kk = repr(k)
            out[kk] = _to_jsonable(v, _depth=_depth + 1)
        return out

    # fallback
    return str(obj)


def _log_skipped_item(
    hparams,
    global_stores,
    i_worker,
    n_worker,
    reason: str,
    item_name: str,
    raw_item,
    extra: str = "",
):
    """将被跳过的样本写入 jsonl 日志。

    通过 hparams 控制：
      - skipped_items_jsonl: str | dict
        - str: 直接当作 path
        - dict: {"path": "...", "enable": true}
    """
    cfg = hparams.get('skipped_items_jsonl', None)
    if not cfg:
        return

    if isinstance(cfg, dict):
        if not cfg.get('enable', True):
            return
        path = cfg.get('path', '') or cfg.get('file', '')
    else:
        path = str(cfg)

    if not path:
        return

    # 多 worker 情况：每个 worker 单独写一个文件，避免并发竞争
    if n_worker is not None and int(n_worker) > 1 and i_worker is not None:
        root, ext = os.path.splitext(path)
        ext = ext or '.jsonl'
        path = f"{root}.w{i_worker}{ext}"

    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    except Exception:
        pass

    rec = {
        "ts": time.time(),
        "reason": reason,
        "item_name": item_name,
        "worker": {
            "i": None if i_worker is None else int(i_worker),
            "n": None if n_worker is None else int(n_worker),
        },
        "extra": extra or "",
        "raw_item": _to_jsonable(raw_item),
    }

    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # 日志写失败不影响主流程
        return


def _norm_spaces_caption(s: str) -> str:
    if not isinstance(s, str):
        return ''
    # 行尾统一
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    # 把所有连续空白压成一个空格（行为与原来 re.sub(r'\s+', ' ', s) 一致）
    s = _SPACE_RE.sub(' ', s)
    return s.strip()


def _local_to_str(val) -> str:
    if val is None:
        return ''
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple)):
        try:
            return ''.join(map(str, val))
        except Exception:
            return ' '.join(map(str, val))
    if isinstance(val, dict):
        if not val:
            return ''
        if 'text' in val:
            return '' if val['text'] is None else str(val['text'])
        if 'local' in val:
            return '' if val['local'] is None else str(val['local'])
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)
    return str(val)


def _is_effectively_empty_local(val) -> bool:
    if isinstance(val, dict) and len(val) == 0:
        return True
    s = _local_to_str(val)
    if s.strip() == '':
        return True
    s_no_tag = _XML_TAG_RE.sub(' ', s)
    return _norm_spaces_caption(s_no_tag) == ''


def augment_text_with_pinyin_s1s2_safe(text: str, hparams):
    """
    只对 <S1>~<S4> 标签内部做 pinyin 混入，保留标签结构。
    若文本不含 S1~S4 标签，则回退到全局 augment（用于纯文本场景）。
    """
    cfg = hparams.get('mix_text_pinyin', {}) or {}
    if not cfg.get('enable', False):
        return text

    from utils.text.pinyin_aug import augment_text_with_pinyin_advanced

    kwargs = dict(
        p_augment=cfg.get('enable_prob', 0.3),
        p_bernoulli_mode=0.1,
        poly_weight_bernoulli=3.0,
        ratio_gamma=3.0,
        poly_weight_ratio=3.0,
        tone3=True,
        pinyin_tokenizer=lambda x: f"<|py_{x}|>"
    )

    if _S1S2_TEXT_RE.search(text) is None:
        return augment_text_with_pinyin_advanced(text, **kwargs)

    def _repl(m):
        tag = m.group(1).upper()
        inner = m.group(2)
        inner_aug = augment_text_with_pinyin_advanced(inner, **kwargs)
        return f"<{tag}>{inner_aug}</{tag}>"

    return _S1S2_TEXT_RE.sub(_repl, text)


def _build_text_from_caption_s1s2(caption: str) -> str:
    """
    从 caption 中按顺序提取 <S1>...</S1> 和 <S2>...</S2>，
    并把相邻的 S1 片段合并成一个 <S1>... ...</S1>。
    若没有任何 S1/S2 片段则返回 ""。
    """
    if not isinstance(caption, str) or not caption.strip():
        return ""

    # 使用预编译的 _S1S2_TAG_RE
    segments = []
    for tag, inner in _S1S2_TAG_RE.findall(caption):
        inner_norm = _norm_spaces_caption(inner)
        if not inner_norm:
            continue
        segments.append([tag.upper(), inner_norm])

    if not segments:
        return ""

    # 合并相邻的 S1 S2段
    merged = []
    for tag, content in segments:
        if merged and tag == merged[-1][0]:
            merged[-1][1] = merged[-1][1] + ' ' + content
        else:
            merged.append([tag, content])

    # 重新拼回成带标签的字符串
    out = ''.join(f"<{tag}>{content}</{tag}>" for tag, content in merged)
    return out if out.strip() else ""

def valid_item_kv(item, k):
    return k in item and item[k] is not None

def merge_A2B(A2B, B_lens):
    token_lens_cumsum = np.cumsum([0] + B_lens[:-1])
    token_lens_cumsum = torch.LongTensor(token_lens_cumsum)
    for i in range(len(B_lens)):
        A2B[i] = A2B[i] + token_lens_cumsum[i]
    A2B = torch.cat(A2B, 0)
    return A2B

class SwanCaptionShmDataset(SwanTTSShmDataset):

    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):

        hop_size = hparams['hop_size']
        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']
        sr = hparams['audio_sample_rate']

        speech_augmentor = None
        if hparams.get('wav_add_noise', False) or hparams.get('wav_add_effect', False):
            from tasks.tts.dataset_utils.augment import SpeechAugment
            speech_augmentor = get_from_global_stores(
                'speech_augmentor', global_stores,
                lambda: SpeechAugment(
                    hparams.get('wav_add_noise', False), hparams.get('wav_add_effect', False), hparams.get('musan_dir', None),
                    noise_prob=hparams.get('wav_add_noise_prob', 0.5), effect_prob=hparams.get('wav_add_effect_prob', 0.5),
                    noise_snr=(6.0, 20.0), with_speech=hparams.get('musan_with_speech', False)
                )
            )

        cosyvoice2_text_tokenizer = None
        if hparams.get('use_cosyvoice2_text_tokenizer', False) and not hparams.get('online_text_alignment_task', False):
            from utils.text.cosyvoice2_tokenizer import get_tokenizer
            cosyvoice2_text_tokenizer = get_from_global_stores(
                'cosyvoice2_text_tokenizer',
                global_stores,
                lambda: get_tokenizer(multilingual=True, num_languages=100)
            )


        skip_logger: SkipLogger = get_from_global_stores(
            'skip_logger', global_stores,
            lambda: SkipLogger([
                'no_score_cnt',
                'no_text_cnt',
                'no_caption_cnt',
                'no_phone_cnt',
            ], interval=1000, i_worker=i_worker, n_worker=n_worker)
        )

        items = processer_fn(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
        if items is None or len(items) == 0:
            _print_skip("processer_fn_returned_none_or_empty", i_worker, n_worker, extra=f"tgt_size={tgt_size}")
            return

        for item_tgt in items:
            # ===== 统一处理 wav：对齐 -> augment -> 对齐 -> (可选) fade+pad -> 更新 wav_len =====
            if hparams.get('load_wav', True):
                # 先对齐，避免 augment 前长度乱
                if fm_wav > 0:
                    item_tgt['wav'] = item_tgt['wav'][: item_tgt['wav'].shape[0] // fm_wav * fm_wav]

                if item_tgt['wav'].numel() == 0:
                    skip_logger.update(1)
                    _print_skip("empty_wav_after_alignment", i_worker, n_worker, item_name=item_tgt.get('item_name', ''))
                    continue

                # augment
                if speech_augmentor is not None:
                    try:
                        item_tgt['wav'] = speech_augmentor(item_tgt['wav'], sr)
                    except Exception as e:
                        _print_skip("speech_augmentor_failed", i_worker, n_worker,
                                    item_name=item_tgt.get('item_name', ''), extra=f"err={str(e)}")

                # augment 后再对齐一次，保证倍数
                if fm_wav > 0:
                    item_tgt['wav'] = item_tgt['wav'][: item_tgt['wav'].shape[0] // fm_wav * fm_wav]

                # 只有 ctx_wav 不存在时，最后做 fade+pad（保证边界干净）
                if item_tgt.get('ctx_wav') is None:
                    max_sec = float(hparams.get('random_fade_pad_max_sec', 1.0))
                    if max_sec > 0 and random.random() < hparams.get('random_fade_pad_prob', 0.0):
                        item_tgt['wav'], pre_pad, post_pad = add_random_fade_in_out_padding(
                            item_tgt['wav'], sr=sr, fm_wav=fm_wav, max_sec=max_sec
                        )

                item_tgt['wav'] = item_tgt['wav'].contiguous()
                item_tgt['wav_len'] = int(item_tgt['wav'].shape[0])

            mel_len_total = item_tgt['wav_len'] // hop_size
            if not (hparams['max_frames'] >= mel_len_total > hparams['min_frames']):
                skip_logger.update(1)
                _print_skip(
                    "frames_out_of_range",
                    i_worker, n_worker,
                    item_name=item_tgt.get('item_name', ''),
                    extra=f"mel_len={mel_len_total}, allowed=({hparams['min_frames']}, {hparams['max_frames']}]"
                )
                continue

            if item_tgt['txt'] is None:
                skip_logger.update(1); continue
            item_tgt['text'] = item_tgt['txt']
            item_tgt['orig_text'] = deepcopy(item_tgt['text'])

            # ===== cosyvoice2 text tokens =====
            if cosyvoice2_text_tokenizer is not None:
                if hparams.get('mix_text_pinyin', {}).get('enable', False):
                    item_tgt['text'] = augment_text_with_pinyin_s1s2_safe(item_tgt['text'], hparams)

                text_tokens = cosyvoice2_text_tokenizer.encode(item_tgt['text'])
                text_tokens = torch.tensor(text_tokens).long()

                # ===== 新增：丢弃 txt_tokens 过长的样本 =====
                # 与训练侧 lat_lens 口径保持一致：wav_len // hop_size // vae_stride
                vae_stride = int(hparams.get('vae_stride', 4))
                latent_len = int(item_tgt['wav_len'] // hop_size // vae_stride)

                if latent_len <= 0 or text_tokens.numel() > latent_len:
                    skip_logger.update(1)
                    _print_skip(
                        "txt_tokens_longer_than_latent",
                        i_worker, n_worker,
                        item_name=item_tgt.get('item_name', ''),
                        extra=f"txt_tokens={text_tokens.numel()}, latent={latent_len}, wav_len={item_tgt['wav_len']}"
                    )
                    continue

                item_tgt['txt_tokens'] = text_tokens

                # ===== spk_mask =====
                sx_patterns = get_from_global_stores(
                    'cosyvoice2_sx_token_patterns',
                    global_stores,
                    lambda: _get_sx_token_patterns(cosyvoice2_text_tokenizer)
                )
                item_tgt['spk_mask'] = build_spk_mask_from_text_tokens(text_tokens, sx_patterns)
                assert item_tgt['spk_mask'].shape == text_tokens.shape, (item_tgt['spk_mask'].shape, text_tokens.shape)

            mel_len = item_tgt['wav'].shape[0] // hop_size

            # 多人样本不参与 zeroshot ref 训练：
            # 保留统一 batch 结构，但显式提供空 ctx 条件，避免后续自动截前缀。
            if bool(item_tgt.get('disable_ref', False)):
                ctx_mask = torch.zeros((mel_len, 1), dtype=torch.float32)
                item_tgt['ctx_mask'] = ctx_mask[::hparams['vae_stride']]

                placeholder_len = min(int(item_tgt['wav'].shape[0]), int(fm_wav)) if int(item_tgt['wav'].shape[0]) > 0 else int(fm_wav)
                placeholder_len = max(1, placeholder_len)
                item_tgt['ctx_wav'] = torch.zeros(
                    (placeholder_len,),
                    dtype=item_tgt['wav'].dtype,
                    device=item_tgt['wav'].device,
                ).contiguous()
                item_tgt['ctx_wav_len'] = 0

            # ctx 相关
            if item_tgt.get('ctx_wav') is None:
                min_idx = max(int(mel_len * 0.1), 200)
                max_idx = min(int(mel_len * 0.9), mel_len - 200)
                if min_idx > max_idx:
                    min_idx = int(mel_len * 0.4)
                    max_idx = int(mel_len * 0.6)
                rand_length = random.randint(min_idx, max_idx) // fm * fm

                ctx_mask = torch.zeros((mel_len, 1))
                ctx_mask[:rand_length] = 1.0
                item_tgt['ctx_mask'] = ctx_mask[::hparams['vae_stride']]

                item_tgt['ctx_wav'] = item_tgt['wav'][:rand_length * hop_size].contiguous()

            item_tgt['len'] = mel_len // int(hparams.get('vae_stride', 4))
            yield item_tgt
            skip_logger.step(1)

def processer_fn_binary(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    """基于 swan_caption 构造 caption 和 text，且不依赖 mel2ph / dur。
    最终每个 item 至少包含: wav/wav_len, txt, caption, item_name, spk_name。
    如果 caption 里既没有 <S1>~<S4>，也没有 <Audio>/<I>，丢弃样本。

    这里额外做两件事：
      1）先把 wav 对齐到 frames_multiple * hop_size 的整数倍，再记录 wav_len；
      2）用裁剪后的 wav_len 计算 mel_len 和 latent_len = mel_len // int(hparams.get('vae_stride', 4))，
         若 phone_encoded 长度 > latent_len，则直接丢掉该 sample。（本函数当前未用 phone_encoded，保持注释一致）
    """
    items = []
    hop_size = hparams['hop_size']
    fm = hparams['frames_multiple']
    fm_wav = fm * hop_size
    sr = hparams['audio_sample_rate']

    for item_ in raw_item:
        try:
            item = {}

            # ========= 1. wav & 长度 =========
            wav = None
            org_sr = sr

            if 'wav' in item_ and item_['wav'] is not None:
                wav = torch.as_tensor(item_['wav'], dtype=torch.float32)
                org_sr = int(item_.get('sr', sr))
            else:
                _print_skip("missing_wav_or_vocal", i_worker, n_worker, item_name=item_.get('item_name'))
                continue

            # 多通道 -> 单通道（假设 [C, T] 或 [T]）
            if wav.dim() > 1:
                wav = wav.mean(dim=0)

            # 重采样到训练采样率
            if org_sr != sr:
                try:
                    wav = torchaudio.functional.resample(
                        wav.unsqueeze(0), orig_freq=org_sr, new_freq=sr
                    )[0]
                except Exception as e:
                    _print_skip(
                        "resample_failed",
                        i_worker, n_worker,
                        item_name=item_.get('item_name'),
                        extra=f"from={org_sr} to={sr}, err={str(e)}"
                    )
                    continue

            # ★ 对齐到 frames_multiple * hop_size 的整数倍
            if fm_wav > 0:
                wav = wav[: (wav.shape[0] // fm_wav) * fm_wav]
            if wav.numel() == 0:
                _print_skip("empty_wav_after_alignment", i_worker, n_worker, item_name=item_.get('item_name'))
                continue

            wav = wav.contiguous()
            item['wav'] = wav
            item['wav_len'] = wav.shape[0]

            # ========= 2. caption: 从 swan_caption 构建 =========
            sc = item_.get('swan_caption', None)
            if not isinstance(sc, dict):
                _print_skip("invalid_swan_caption_type", i_worker, n_worker, item_name=item_.get('item_name'))
                continue
            if len(sc) == 0:
                _print_skip("empty_swan_caption", i_worker, n_worker, item_name=item_.get('item_name'))
                continue

            local_raw = sc.get('mapped_local', '') or sc.get('local', '')
            if _is_effectively_empty_local(local_raw):
                _print_skip("empty_local", i_worker, n_worker, item_name=item_.get('item_name'))
                continue

            quality_txt, quality_flag = _build_quality_caption_from_meta(item_)
            if not bool(hparams.get('use_quality_caption', True)):
                quality_txt = ""

            bgm_raw = sc.get('bgm', '')
            item['bgm_flag'] = _bgm_flag_from_raw(bgm_raw)
            item['quality_flag'] = quality_flag

            caption, global_txt, local_txt = _build_caption_from_swan_caption(item_, quality_txt=quality_txt)
            if not caption:
                _print_skip("no_caption_built", i_worker, n_worker, item_name=item_.get('item_name'))
                continue

            # 若 caption 里出现的 <Sx> 序号超过 speakers 列表长度，则认为标注不一致，直接跳过。
            # 仅当 speakers 字段非空时才启用该校验，避免把没有 speakers 标注的样本全部误杀。
            speakers_cnt = _count_speakers_field(sc)
            if speakers_cnt > 0:
                max_sx = _max_speaker_tag_id_in_caption(caption)
                if max_sx > speakers_cnt:
                    _print_skip(
                        "sx_id_exceeds_speakers",
                        i_worker, n_worker,
                        item_name=item_.get('item_name'),
                        extra=f"max_sx={max_sx}, speakers_cnt={speakers_cnt}",
                    )
                    continue

            max_spk = int(hparams.get('max_speakers_in_caption', 4))
            spk_cnt = max(_count_speaker_tags_in_caption(caption), _count_speakers_field(sc))
            if spk_cnt > max_spk:
                _print_skip(
                    "too_many_speakers",
                    i_worker, n_worker,
                    item_name=item_.get('item_name'),
                    extra=f"spk_cnt={spk_cnt}, max={max_spk}",
                )
                continue

            item['caption'] = caption
            item['txt'] = _build_text_from_caption_s1s2(caption)  # <S1-4>... 拼起来，没有就 ""

            # 新增：global/local
            item['global'] = global_txt   # <ENV><BGM><SPK>
            item['local'] = local_txt     # swan_caption['local'] 原样

            # ========= 多人样本禁用 zeroshot ref（与 processer_fn_jsonl 对齐）=========
            allow_ref = bool(spk_cnt == 1 and isinstance(item['txt'], str) and '<S1>' in item['txt'])
            item['disable_ref'] = not allow_ref
            item['force_ref'] = False

            # ========= 3. 其他元信息 =========
            item['item_name'] = item_.get('item_name', '')
            item['spk_name'] = item_.get('spk_name', item['item_name'])

            if hparams.get('shuffle_spk_ids', True):
                id_map = build_speaker_shuffle_map(item['txt'])
                if id_map:
                    item['txt'] = apply_speaker_shuffle(item['txt'], id_map)
                    item['caption'] = apply_speaker_shuffle(item['caption'], id_map)
                    item['local'] = apply_speaker_shuffle(item['local'], id_map)
                    if item.get('global'):
                        item['global'] = apply_speaker_shuffle(item['global'], id_map)

            items.append(item)
        except Exception as e:
            traceback.print_exc()
            extra = f"err={str(e)}"
            _print_skip(
                "processer_fn_binary_exception",
                i_worker, n_worker,
                item_name=item_.get('item_name', ''),
                extra=extra
            )
            _log_skipped_item(
                hparams, global_stores, i_worker, n_worker,
                reason="processer_fn_binary_exception",
                item_name=item_.get('item_name', ''),
                raw_item=item_,
                extra=extra,
            )
            continue

    return items


def processer_fn_jsonl(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    """
    读取 jsonl 样本，按 subjects/narration 组合 caption，
    从 caption 中抽取 <S1>...</S1> 作为 text（若没有则用 ）。
    若存在 start/end 字段，则只从 wav 中读取对应时间段，而不是整段读完再切。

      1）先把 wav 对齐到 frames_multiple * hop_size 的整数倍，再记录 wav_len；
      2）用裁剪后的 wav_len 计算 mel_len 和 latent_len = mel_len // int(hparams.get('vae_stride', 4))，
         若 phone_encoded 长度 > latent_len，则直接丢掉该 sample。
    """

    def get_tos_client():
        cluster = os.environ.get('CLUSTER', '').lower()
        if cluster == 'va':
            tos_bucket = 'sa-ag-sg-research-sg'
        else:
            tos_bucket = 'humanaigc-ads'
        return TosClient(bucket=tos_bucket)

    def _sha1(s: str) -> str:
        return hashlib.sha1(s.encode('utf-8')).hexdigest()

    def _atomic_write(path: str, data: bytes):
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _get_cache_dir():
        cache_dir = get_from_global_stores(
            'tos_cache_dir', global_stores,
            lambda: hparams.get('tos_cache_dir', '/dev/shm/zyxc_tos_cache')
        )
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass
        return cache_dir

    def _enforce_cache_budget(cache_dir: str):
        max_gb = float(hparams.get('tos_cache_max_gb', 100.0))
        near_ratio = float(hparams.get('tos_cache_near_ratio', 0.90))
        keep_ratio = float(hparams.get('tos_cache_keep_ratio', 0.10))
        try:
            import glob
            max_bytes = int(max_gb * (1024**3))
            if max_bytes <= 0:
                return
            trigger_bytes = int(max_bytes * near_ratio)
            keep_bytes = int(max_bytes * keep_ratio)

            files = glob.glob(os.path.join(cache_dir, "*"))
            if not files:
                return
            sizes = []
            total = 0
            for p in files:
                try:
                    st = os.stat(p)
                    sz = int(st.st_size)
                    mt = float(st.st_mtime)
                    total += sz
                    sizes.append((mt, sz, p))
                except Exception:
                    pass
            if total < trigger_bytes:
                return
            sizes.sort(key=lambda x: x[0])
            for _, sz, p in sizes:
                if total <= keep_bytes:
                    break
                try:
                    os.remove(p)
                    total -= sz
                except Exception:
                    pass
        except Exception:
            return

    def _decode_segment_with_ffmpeg(path: str, start_sec: float, end_sec: float, target_sr: int):
        if not path or not os.path.exists(path):
            return None
        if start_sec is None or end_sec is None or end_sec <= start_sec:
            return None
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-ss", f"{float(start_sec):.6f}", "-to", f"{float(end_sec):.6f}",
                "-i", path,
                "-f", "f32le", "-ac", "1", "-ar", str(int(target_sr)),
                "pipe:1"
            ]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if p.returncode != 0 or p.stdout is None or len(p.stdout) == 0:
                return None
            wav_np = np.frombuffer(p.stdout, dtype=np.float32)
            if wav_np.size == 0:
                return None
            return torch.from_numpy(wav_np)
        except Exception:
            return None

    tos_client: TosClient = get_from_global_stores(
        'tos_client', global_stores,
        get_tos_client
    )

    fm = hparams['frames_multiple']
    hop_size = hparams['hop_size']
    fm_wav = fm * hop_size
    sr = hparams['audio_sample_rate']

    items = []
    for item_ in raw_item:
        temp_wav_path = None
        wav_ref = ''
        try:
            if item_ is None or not isinstance(item_, dict):
                extra = f"type={type(item_)}"
                _print_skip("invalid_jsonl_item", i_worker, n_worker, extra=extra)
                _log_skipped_item(
                    hparams, global_stores, i_worker, n_worker,
                    reason="invalid_jsonl_item",
                    item_name="",
                    raw_item=item_,
                    extra=extra,
                )
                skip_logger.report(1, 'promptjson')
                continue
            # -------- 1. 读 wav，必要时只按 start/end 读一小段 --------
            if 'flag' in item_ and not item_.get('flag'):
                _print_skip("flag_not_true", i_worker, n_worker)
                _log_skipped_item(
                    hparams, global_stores, i_worker, n_worker,
                    reason="flag_not_true",
                    item_name=item_.get('item_name', item_.get('utt_id', item_.get('id', ''))),
                    raw_item=item_,
                )
                continue

            wav_path = item_.get('wav_path')
            wav_k = item_.get('wav_k') or item_.get('wav_key', '')
            wav_ref = wav_path or wav_k

            if not wav_path and wav_k:
                use_cache = bool(hparams.get('tos_cache_enable', True))
                ext = os.path.splitext(wav_k)[1] or '.m4a'
                if use_cache:
                    cache_dir = _get_cache_dir()
                    key_hash = _sha1(wav_k)
                    cached_path = os.path.join(cache_dir, f"{key_hash}{ext}")
                    if not os.path.exists(cached_path) or os.path.getsize(cached_path) == 0:
                        data = tos_client.get_object(wav_k, verbose=False)
                        if data is None:
                            extra = f"wav_k={wav_k}"
                            _print_skip("tos_get_object_none", i_worker, n_worker, extra=extra)
                            _log_skipped_item(
                                hparams, global_stores, i_worker, n_worker,
                                reason="tos_get_object_none",
                                item_name=item_.get('item_name', wav_k),
                                raw_item=item_,
                                extra=extra,
                            )
                            continue
                        try:
                            _atomic_write(cached_path, data)
                        except Exception:
                            cached_path = None
                        _enforce_cache_budget(cache_dir)
                    if cached_path is not None and os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
                        wav_path = cached_path
                    else:
                        data = tos_client.get_object(wav_k, verbose=False)
                        if data is None:
                            extra = f"wav_k={wav_k}"
                            _print_skip("tos_get_object_none", i_worker, n_worker, extra=extra)
                            _log_skipped_item(
                                hparams, global_stores, i_worker, n_worker,
                                reason="tos_get_object_none",
                                item_name=item_.get('item_name', wav_k),
                                raw_item=item_,
                                extra=extra,
                            )
                            continue
                        with tempfile.NamedTemporaryFile(dir='/dev/shm', suffix=ext, delete=False) as f:
                            f.write(data)
                            temp_wav_path = f.name
                        wav_path = temp_wav_path
                else:
                    data = tos_client.get_object(wav_k, verbose=False)
                    if data is None:
                        extra = f"wav_k={wav_k}"
                        _print_skip("tos_get_object_none", i_worker, n_worker, extra=extra)
                        _log_skipped_item(
                            hparams, global_stores, i_worker, n_worker,
                            reason="tos_get_object_none",
                            item_name=item_.get('item_name', wav_k),
                            raw_item=item_,
                            extra=extra,
                        )
                        continue
                    with tempfile.NamedTemporaryFile(dir='/dev/shm', suffix=ext, delete=False) as f:
                        f.write(data)
                        temp_wav_path = f.name
                    wav_path = temp_wav_path

            if not wav_path:
                _print_skip("missing_wav_path", i_worker, n_worker)
                _log_skipped_item(
                    hparams, global_stores, i_worker, n_worker,
                    reason="missing_wav_path",
                    item_name=item_.get('item_name', item_.get('utt_id', item_.get('id', ''))),
                    raw_item=item_,
                )
                continue

            # 有 start/end 的情况：仅解码对应时间段
            if 'start' in item_ and 'end' in item_:
                start = max(float(item_['start']), 0.0)
                end = float(item_['end'])
                if end <= start:
                    extra = f"start={start}, end={end}"
                    _print_skip("invalid_time_range", i_worker, n_worker, extra=extra)
                    _log_skipped_item(
                        hparams, global_stores, i_worker, n_worker,
                        reason="invalid_time_range",
                        item_name=item_.get('item_name', wav_ref),
                        raw_item=item_,
                        extra=extra,
                    )
                    continue

                wav = None
                org_sr = None
                info_err = None
                try:
                    info = torchaudio.info(wav_path)
                    org_sr = int(info.sample_rate)
                except Exception as e:
                    info_err = str(e)

                if org_sr is not None and org_sr > 0:
                    s_idx = int(start * org_sr)
                    e_idx = int(end * org_sr)
                    if e_idx <= s_idx:
                        extra = f"s_idx={s_idx}, e_idx={e_idx}"
                        _print_skip(
                            "time_slice_empty",
                            i_worker, n_worker,
                            item_name=item_.get('item_name', wav_ref),
                            extra=extra,
                        )
                        _log_skipped_item(
                            hparams, global_stores, i_worker, n_worker,
                            reason="time_slice_empty",
                            item_name=item_.get('item_name', wav_ref),
                            raw_item=item_,
                            extra=extra,
                        )
                        continue
                    num_frames = e_idx - s_idx
                    try:
                        wav, org_sr = torchaudio.load(
                            wav_path,
                            frame_offset=s_idx,
                            num_frames=num_frames,
                        )
                        wav = wav.to(torch.float32)
                    except Exception:
                        wav = None

                if wav is None:
                    seg = _decode_segment_with_ffmpeg(wav_path, start, end, sr)
                    if seg is not None:
                        wav = seg[None, :]
                        org_sr = int(sr)

                if wav is None:
                    if org_sr is None:
                        extra = f"path={wav_path}, err={info_err}"
                        _print_skip(
                            "torchaudio_info_failed",
                            i_worker, n_worker,
                            item_name=item_.get('item_name', wav_ref),
                            extra=extra,
                        )
                        _log_skipped_item(
                            hparams, global_stores, i_worker, n_worker,
                            reason="torchaudio_info_failed",
                            item_name=item_.get('item_name', wav_ref),
                            raw_item=item_,
                            extra=extra,
                        )
                        skip_logger.report(1, 'promptjson')
                        continue
                    s_idx2 = int(start * int(org_sr))
                    e_idx2 = int(end * int(org_sr))
                    if e_idx2 <= s_idx2:
                        extra = f"s_idx={s_idx2}, e_idx={e_idx2}, fallback_from=full_decode"
                        _print_skip(
                            "time_slice_empty",
                            i_worker, n_worker,
                            item_name=item_.get('item_name', wav_ref),
                            extra=extra,
                        )
                        _log_skipped_item(
                            hparams, global_stores, i_worker, n_worker,
                            reason="time_slice_empty",
                            item_name=item_.get('item_name', wav_ref),
                            raw_item=item_,
                            extra=extra,
                        )
                        continue
                    try:
                        wav_full, org_sr_full = torchaudio.load(wav_path)
                        wav_full = wav_full.to(torch.float32)
                        org_sr_full = int(org_sr_full)
                        s_idx3 = int(start * org_sr_full)
                        e_idx3 = int(end * org_sr_full)
                        if e_idx3 <= s_idx3:
                            extra = f"s_idx={s_idx3}, e_idx={e_idx3}, fallback_from=full_decode"
                            _print_skip(
                                "time_slice_empty",
                                i_worker, n_worker,
                                item_name=item_.get('item_name', wav_ref),
                                extra=extra,
                            )
                            _log_skipped_item(
                                hparams, global_stores, i_worker, n_worker,
                                reason="time_slice_empty",
                                item_name=item_.get('item_name', wav_ref),
                                raw_item=item_,
                                extra=extra,
                            )
                            continue
                        wav = wav_full[:, s_idx3:e_idx3]
                        org_sr = org_sr_full
                    except Exception as e2:
                        extra = f"path={wav_path}, start={start}, end={end}, err={str(e2)}"
                        _print_skip(
                            "torchaudio_slice_load_failed",
                            i_worker, n_worker,
                            item_name=item_.get('item_name', wav_ref),
                            extra=extra,
                        )
                        _log_skipped_item(
                            hparams, global_stores, i_worker, n_worker,
                            reason="torchaudio_slice_load_failed",
                            item_name=item_.get('item_name', wav_ref),
                            raw_item=item_,
                            extra=extra,
                        )
                        skip_logger.report(1, 'promptjson')
                        continue

            else:
                # 无 start/end：整段加载（与原实现一致）
                try:
                    wav, org_sr = torchaudio.load(wav_path)
                    wav = wav.to(torch.float32)
                except Exception as e:
                    extra = f"path={wav_path}, err={str(e)}"
                    _print_skip("torchaudio_load_failed", i_worker, n_worker, extra=extra)
                    _log_skipped_item(
                        hparams, global_stores, i_worker, n_worker,
                        reason="torchaudio_load_failed",
                        item_name=item_.get('item_name', wav_path),
                        raw_item=item_,
                        extra=extra,
                    )
                    skip_logger.report(1, 'promptjson')
                    continue

            # 多通道 -> 单通道
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # 重采样到训练采样率
            if org_sr != sr:
                try:
                    wav = torchaudio.functional.resample(wav, orig_freq=org_sr, new_freq=sr)
                except Exception as e:
                    extra = f"from={org_sr} to={sr}, err={str(e)}"
                    _print_skip("resample_failed", i_worker, n_worker, extra=extra)
                    _log_skipped_item(
                        hparams, global_stores, i_worker, n_worker,
                        reason="resample_failed",
                        item_name=item_.get('item_name', wav_ref),
                        raw_item=item_,
                        extra=extra,
                    )
                    skip_logger.report(1, 'promptjson')
                    continue

            # 现在 wav: [1, T]
            wav = wav[0]  # -> [T]

            # ★ 对齐到 frames_multiple * hop_size 的整数倍（和 audioset 一致）
            if wav.numel() > 0 and fm_wav > 0:
                wav = wav[: (wav.shape[0] // fm_wav) * fm_wav]
            if wav.numel() == 0:
                _print_skip("empty_wav_after_alignment", i_worker, n_worker)
                _log_skipped_item(
                    hparams, global_stores, i_worker, n_worker,
                    reason="empty_wav_after_alignment",
                    item_name=item_.get('item_name', wav_ref),
                    raw_item=item_,
                )
                continue

            wav = wav.contiguous()

            item = {}
            item['wav'] = wav
            item['wav_len'] = wav.shape[0]

            # -------- 3. caption = swan_caption(env+bgm+speakers) + local --------
            sc = item_.get('swan_caption', None)
            if not isinstance(sc, dict):
                extra = f"type={type(sc).__name__}"
                _print_skip(
                    "bad_swan_caption_type",
                    i_worker, n_worker,
                    item_name=item_.get('item_name', item_.get('utt_id', item_.get('id', wav_ref))),
                    extra=extra
                )
                _log_skipped_item(
                    hparams, global_stores, i_worker, n_worker,
                    reason="bad_swan_caption_type",
                    item_name=item_.get('item_name', item_.get('utt_id', item_.get('id', wav_ref))),
                    raw_item=item_,
                    extra=extra,
                )
                skip_logger.report(1, 'bad_swan_caption_type')
                continue
            if len(sc) == 0:
                _print_skip("empty_swan_caption", i_worker, n_worker)
                _log_skipped_item(
                    hparams, global_stores, i_worker, n_worker,
                    reason="empty_swan_caption",
                    item_name=item_.get('item_name', wav_ref),
                    raw_item=item_,
                )
                skip_logger.report(1, 'promptjson')
                continue

            local_raw = sc.get('mapped_local', '') or sc.get('local', '')
            if _is_effectively_empty_local(local_raw):
                _print_skip("empty_local", i_worker, n_worker)
                _log_skipped_item(
                    hparams, global_stores, i_worker, n_worker,
                    reason="empty_local",
                    item_name=item_.get('item_name', wav_ref),
                    raw_item=item_,
                )
                skip_logger.report(1, 'promptjson')
                continue

            quality_txt, quality_flag = _build_quality_caption_from_meta(item_)
            if not bool(hparams.get('use_quality_caption', True)):
                quality_txt = ""

            bgm_raw = sc.get('bgm', '')
            item['bgm_flag'] = _bgm_flag_from_raw(bgm_raw)
            item['quality_flag'] = quality_flag

            caption, global_txt, local_txt = _build_caption_from_swan_caption(item_, quality_txt=quality_txt)
            if not caption:
                sc_keys = []
                try:
                    sc_keys = [str(k) for k in list(sc.keys())[:20]]
                except Exception:
                    sc_keys = []
                extra = f"swan_caption_keys={sc_keys}"
                _print_skip("no_caption_built", i_worker, n_worker, extra=extra)
                _log_skipped_item(
                    hparams, global_stores, i_worker, n_worker,
                    reason="no_caption_built",
                    item_name=item_.get('item_name', wav_ref),
                    raw_item=item_,
                    extra=extra,
                )
                continue

            # 若 caption 里出现的 <Sx> 序号超过 speakers 列表长度，则认为标注不一致，直接跳过。
            # 仅当 speakers 字段非空时才启用该校验，避免把没有 speakers 标注的样本全部误杀。
            speakers_cnt = _count_speakers_field(sc)
            if speakers_cnt > 0:
                max_sx = _max_speaker_tag_id_in_caption(caption)
                if max_sx > speakers_cnt:
                    extra = f"max_sx={max_sx}, speakers_cnt={speakers_cnt}"
                    _print_skip("sx_id_exceeds_speakers", i_worker, n_worker, item_name=item_.get('item_name', wav_ref), extra=extra)
                    _log_skipped_item(
                        hparams, global_stores, i_worker, n_worker,
                        reason="sx_id_exceeds_speakers",
                        item_name=item_.get('item_name', wav_ref),
                        raw_item=item_,
                        extra=extra,
                    )
                    skip_logger.report(1, 'promptjson')
                    continue

            max_spk = int(hparams.get('max_speakers_in_caption', 4))
            spk_cnt = max(_count_speaker_tags_in_caption(caption), _count_speakers_field(sc))
            if spk_cnt > max_spk:
                extra = f"spk_cnt={spk_cnt}, max={max_spk}"
                _print_skip("too_many_speakers", i_worker, n_worker, item_name=item_.get('item_name', wav_ref), extra=extra)
                _log_skipped_item(
                    hparams, global_stores, i_worker, n_worker,
                    reason="too_many_speakers",
                    item_name=item_.get('item_name', wav_ref),
                    raw_item=item_,
                    extra=extra,
                )
                skip_logger.report(1, 'promptjson')
                continue

            item['caption'] = caption
            item['global'] = global_txt
            item['local'] = local_txt

            # -------- 4. text：从 <S1>... 抽取 --------
            txt = _build_text_from_caption_s1s2(caption)
            if not txt:
                _print_skip("empty_text_built_from_caption", i_worker, n_worker,)
            item['txt'] = txt
            # 只有“恰好单人且 text 中明确带 <S1>”的样本参与单人 zeroshot ref。
            # 0 人、多人、或 content 里没有 <S1> 的样本都禁用 ref。
            allow_ref = bool(spk_cnt == 1 and isinstance(txt, str) and '<S1>' in txt)
            item['disable_ref'] = not allow_ref
            item['force_ref'] = False

            # 单人 zeroshot: 始终保留 content local，避免空 local 批次。

            # -------- 5. 其他元信息 --------
            item['item_name'] = item_.get(
                'item_name',
                item_.get('utt_id', item_.get('id', wav_ref))
            )
            item['spk_name'] = item_.get(
                'spk_name',
                item_.get('speaker', item_.get('gender', item['item_name']))
            )

            if hparams.get('shuffle_spk_ids', True):
                id_map = build_speaker_shuffle_map(item['txt'])
                if id_map:
                    item['txt'] = apply_speaker_shuffle(item['txt'], id_map)
                    item['caption'] = apply_speaker_shuffle(item['caption'], id_map)
                    item['local'] = apply_speaker_shuffle(item['local'], id_map)
                    if item.get('global'):
                        item['global'] = apply_speaker_shuffle(item['global'], id_map)

            items.append(item)
            skip_logger.step(1)
        except Exception as e:
            traceback.print_exc()
            extra = f"err={str(e)}"
            _print_skip("processer_fn_jsonl_exception", i_worker, n_worker, extra=extra)
            if isinstance(item_, dict):
                item_name = item_.get('item_name', item_.get('utt_id', item_.get('id', wav_ref)))
            else:
                item_name = wav_ref
            _log_skipped_item(
                hparams, global_stores, i_worker, n_worker,
                reason="processer_fn_jsonl_exception",
                item_name=item_name,
                raw_item=item_,
                extra=extra,
            )
            skip_logger.report(1, 'promptjson')
            continue
        finally:
            if temp_wav_path is not None:
                try:
                    os.remove(temp_wav_path)
                except Exception:
                    pass

    return items
