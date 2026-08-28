import os
import random
import re
import tempfile
from datetime import datetime, timedelta
import collections
import collections.abc
import math
import numpy as np
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
from attrdictionary import AttrDict
from typing import Optional, Dict
import socket
from contextlib import closing
import yaml
from argparse import ArgumentParser
import torch.distributed as dist
import torch
import soundfile as sf
import librosa
from multiprocessing import Process, set_start_method
from utils.commons.os_utils import kill_void

#region debug-point nccl-watchdog-timeout-client
import json
import time
import traceback
from urllib import request as _dbg_urllib_request
import statistics

_DBG_SESSION_ID = "nccl-watchdog-timeout"
_DBG_ENV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".dbg", f"{_DBG_SESSION_ID}.env")
)

def _dbg_maybe_load_env():
    if os.environ.get("DEBUG_SERVER_URL"):
        return
    try:
        with open(_DBG_ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = (line or "").strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = (k or "").strip()
                v = (v or "").strip()
                if k and v and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        return

def _dbg_event(event: str, **fields):
    _dbg_maybe_load_env()
    url = os.environ.get("DEBUG_SERVER_URL")
    if not url:
        return
    payload = {
        "ts": time.time(),
        "sessionId": os.environ.get("DEBUG_SESSION_ID", _DBG_SESSION_ID),
        "runId": os.environ.get("DEBUG_RUN_ID", "pre"),
        "event": event,
        "fields": fields,
    }
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = _dbg_urllib_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        _dbg_urllib_request.urlopen(req, timeout=0.5).read()
    except Exception:
        return
#endregion debug-point nccl-watchdog-timeout-client

try:
    import pyloudnorm as pyln
except Exception:
    pyln = None
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint, torch_load_dist
from utils.commons.hparams import set_hparams, hparams
from modules.asr.sensevoice.sensevoice_api import build_asr_model, run_asr_model
from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin, SemanticLMBuildModelMixin, build_vae
from utils.commons.upload_tos_utils import send_file_to_tos
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text import is_chinese
from utils.text.cosyvoice2_tokenizer import CosyVoice2Tokenizer
from tasks.tts.task_utils.prompttts_task_utils import build_audio_mask_from_ids
from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import build_spk_mask_from_text_tokens, _get_sx_token_patterns
from tasks.tts.dataset_utils.swan_caption_fastdataset import (
    QUALITY_FLAG_HIGH,
    QUALITY_FLAG_LOW,
    QUALITY_FLAG_NORMAL,
    QUALITY_FLAG_UNKNOWN,
    build_quality_caption_from_flag,
    build_quality_caption_from_meta,
)

def _read_model_config_yaml(dit_ckpt: str) -> Dict:
    """
    Read model-side `config.yaml` next to the checkpoint/dir, without mutating global `hparams`.
    Used by post-processing (HTML) so it matches the model's actual switches.
    """
    if not isinstance(dit_ckpt, str) or not dit_ckpt:
        return {}
    config_path = (
        os.path.join(os.path.dirname(dit_ckpt), 'config.yaml')
        if dit_ckpt.endswith(('.ckpt', '.pt', '.pth', '.safetensors'))
        else os.path.join(dit_ckpt, 'config.yaml')
    )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}

def _safe_fname(x: str) -> str:
    """尽量生成跨平台安全文件名（保留中文也可以；这里主要干掉分隔符和空白）"""
    if x is None:
        return "none"
    x = str(x)
    x = x.replace("/", "_").replace("\\", "_").replace(os.sep, "_")
    x = re.sub(r"\s+", "_", x).strip("_")
    return x or "none"

def make_out_wav_name(bench: str, group: str, group_idx: int) -> str:
    bench = _safe_fname(bench)
    group = _safe_fname(group)
    return f"{bench}_{group}_{group_idx:04d}.wav"  # 0000 起，想 0001 起就 +1

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

def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        # 不用 SO_REUSEADDR，避免“看起来可用但其实被占用”的假象
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False

def normalize_wav_to_target_loudness(
    wav,
    sr: int = 24000,
    target_lufs: float = -23.0,
):
    if wav is None:
        return wav

    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()

    wav = np.asarray(wav)
    if wav.size == 0:
        return wav

    if wav.ndim == 2 and 1 in wav.shape:
        wav = wav.reshape(-1)
    elif wav.ndim != 1:
        wav = wav.reshape(-1)

    wav_f = wav.astype(np.float64, copy=False)
    peak = float(np.max(np.abs(wav_f))) if wav_f.size else 0.0
    if (not np.isfinite(peak)) or peak < 1e-8:
        return wav.astype(np.float32, copy=False)

    wav_n = None

    if pyln is not None:
        try:
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(wav_f)
            if np.isfinite(loudness):
                wav_n = pyln.normalize.loudness(wav_f, loudness, float(target_lufs))
        except Exception:
            wav_n = None

    if wav_n is None:
        eps = 1e-12
        rms = float(np.sqrt(np.mean(np.square(wav_f))) + eps)
        cur_db = 20.0 * float(np.log10(rms + eps))
        gain_db = float(target_lufs) - cur_db
        gain = float(np.power(10.0, gain_db / 20.0))
        wav_n = wav_f * gain

    peak_n = float(np.max(np.abs(wav_n))) if wav_n.size else 0.0
    if peak_n >= 1.0 and np.isfinite(peak_n) and peak_n > 0:
        wav_n = wav_n / peak_n * 0.95

    return wav_n.astype(np.float32)


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

def find_available_port(base_port: int = 10521, host: str = "127.0.0.1", max_search: int = 2000) -> int:
    """
    按 base, base+1, base-1, base+2, base-2 ... 搜索可用端口。
    max_search 表示最多尝试多少个偏移步（2000 => 最大探测到 base±2000）
    """
    if base_port < 1 or base_port > 65535:
        raise ValueError(f"Invalid base_port={base_port}")

    # offsets: 0, +1, -1, +2, -2, ...
    for k in range(0, max_search + 1):
        if k == 0:
            candidates = [base_port]
        else:
            candidates = []
            p1 = base_port + k
            p2 = base_port - k
            if 1 <= p1 <= 65535:
                candidates.append(p1)
            if 1 <= p2 <= 65535:
                candidates.append(p2) 

        for p in candidates:
            if _is_port_free(p, host=host):
                return p

    raise RuntimeError(f"No free port found around {base_port} within ±{max_search}")

def _fmt_mb(x: int) -> str:
    return f"{x / 1024**2:,.1f} MB"

def _sync_cuda(device=None):
    """Synchronize CUDA for accurate timing (no-op on CPU)."""
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
        return
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(idx)

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

# ===== target 时长自动估计 =====
_AUDIO_TAG_RE = re.compile(r"<\s*Audio\s*>", flags=re.IGNORECASE)
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_BGM_TAG_RE = re.compile(r"<\s*BGM\s*>(.*?)</\s*BGM\s*>", flags=re.IGNORECASE | re.DOTALL)
_BGM_VALUE_PREFIX_RE = re.compile(r"^\s*(?:BGM|背景音乐)\s*[:：]\s*", flags=re.IGNORECASE)
_QUALITY_PREFIX_RE = re.compile(r"\bQuality\s*:", flags=re.IGNORECASE)
_NO_BGM_VALUES = {"无", "无背景音乐", "none", "no", "null", "0"}
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# For length estimation: count JA/KO scripts as "character units" similar to CJK.
_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
_KATAKANA_RE = re.compile(r"[\u30a0-\u30ff\uff66-\uff9d]")  # includes halfwidth katakana
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_DEFAULT_CPM = 220.0
_LATIN_CPM_RATIO = 0.5


def _extract_bgm_flag_from_caption(caption: Optional[str]) -> int:
    if not isinstance(caption, str):
        return 0
    m = _BGM_TAG_RE.search(caption)
    if m is None:
        return 0

    bgm = (m.group(1) or "").strip()
    bgm = _BGM_VALUE_PREFIX_RE.sub("", bgm).strip()
    if not bgm:
        return 0

    if bgm.lower() in _NO_BGM_VALUES:
        return 0
    return 1


def _normalize_quality_flag(flag) -> Optional[int]:
    if flag is None:
        return None
    if torch.is_tensor(flag):
        if flag.numel() == 0:
            return None
        flag = flag.view(-1)[0].item()
    if isinstance(flag, str):
        s = flag.strip().lower()
        if not s:
            return None
        mapping = {
            "0": QUALITY_FLAG_LOW,
            "1": QUALITY_FLAG_NORMAL,
            "2": QUALITY_FLAG_HIGH,
            "3": QUALITY_FLAG_UNKNOWN,
            "low": QUALITY_FLAG_LOW,
            "low_quality": QUALITY_FLAG_LOW,
            "normal": QUALITY_FLAG_NORMAL,
            "normal_quality": QUALITY_FLAG_NORMAL,
            "standard": QUALITY_FLAG_NORMAL,
            "high": QUALITY_FLAG_HIGH,
            "high_quality": QUALITY_FLAG_HIGH,
            "unknown": QUALITY_FLAG_UNKNOWN,
        }
        if s in mapping:
            return mapping[s]
    try:
        flag = int(flag)
    except (TypeError, ValueError):
        return None
    if QUALITY_FLAG_LOW <= flag <= QUALITY_FLAG_UNKNOWN:
        return flag
    return None


def _extract_quality_flag_from_caption(caption: Optional[str]) -> Optional[int]:
    if not isinstance(caption, str):
        return None
    caption = _norm_spaces_caption(caption)
    if not caption or _QUALITY_PREFIX_RE.search(caption) is None:
        return None
    lowered = caption.lower()
    if "high quality" in lowered or "high-quality recording" in lowered:
        return QUALITY_FLAG_HIGH
    if "normal quality" in lowered or "standard-quality recording" in lowered:
        return QUALITY_FLAG_NORMAL
    if "low quality" in lowered or "low-quality recording" in lowered:
        return QUALITY_FLAG_LOW
    return None


def _resolve_quality_text_and_flag(
    sample: Optional[Dict] = None,
    explicit_quality_flag=None,
    caption: Optional[str] = None,
    default_to_high: bool = True,
):
    explicit_flag = _normalize_quality_flag(explicit_quality_flag)
    if explicit_flag is not None and explicit_flag != QUALITY_FLAG_UNKNOWN:
        return build_quality_caption_from_flag(explicit_flag)

    if isinstance(sample, dict):
        quality_txt, meta_flag = build_quality_caption_from_meta(sample)
        if quality_txt:
            return quality_txt, meta_flag

        sample_caption = sample.get('caption', None)
        parsed_flag = _extract_quality_flag_from_caption(sample_caption)
        if parsed_flag is None:
            joined = []
            for key in ('global', 'local'):
                value = sample.get(key, None)
                if value is None:
                    continue
                value = _norm_spaces_caption(str(value))
                if value:
                    joined.append(value)
            parsed_flag = _extract_quality_flag_from_caption(" ".join(joined).strip())
        if parsed_flag is not None:
            return build_quality_caption_from_flag(parsed_flag)

        sample_flag = _normalize_quality_flag(sample.get('quality_flag', None))
        if sample_flag is not None and sample_flag != QUALITY_FLAG_UNKNOWN:
            return build_quality_caption_from_flag(sample_flag)

    parsed_flag = _extract_quality_flag_from_caption(caption)
    if parsed_flag is not None:
        return build_quality_caption_from_flag(parsed_flag)

    if default_to_high:
        return build_quality_caption_from_flag(QUALITY_FLAG_HIGH)

    return "", explicit_flag


def _ensure_quality_caption(caption: Optional[str], quality_txt: str) -> str:
    if not isinstance(caption, str):
        caption = ""
    caption = _norm_spaces_caption(caption)
    if not caption:
        return caption
    quality_txt = _norm_spaces_caption(quality_txt)
    if not quality_txt:
        return caption
    if _QUALITY_PREFIX_RE.search(caption):
        return caption

    content_idx = caption.find("Content:")
    if content_idx >= 0:
        head = caption[:content_idx].rstrip()
        tail = caption[content_idx:].lstrip()
        if head:
            return f"{head} {quality_txt} {tail}"
        return f"{quality_txt} {tail}"

    return caption


def _ensure_high_quality_caption(caption: Optional[str]) -> str:
    quality_txt, _ = build_quality_caption_from_flag(QUALITY_FLAG_HIGH)
    return _ensure_quality_caption(caption, quality_txt)


def _compose_caption_from_fields(sample: Dict, quality_txt: str = "") -> str:
    caption = sample.get('caption', '')
    if caption is None:
        caption = ''
    caption = _norm_spaces_caption(str(caption))
    if caption:
        return _ensure_quality_caption(caption, quality_txt)

    parts = []
    for key in ('global', 'local'):
        part = sample.get(key, '')
        if part is None:
            continue
        part = _norm_spaces_caption(str(part))
        if not part:
            continue
        parts.append(part)

    return _ensure_quality_caption(" ".join(parts).strip(), quality_txt)

def _count_speech_units_for_len(
    s: Optional[str],
    latin_cpm_ratio: float = _LATIN_CPM_RATIO,
) -> float:
    """
    用于估时长的“单位数”：
    - 汉字/日文假名/韩文按“字符单位”计数
    - 拉丁词/数字按“词”计数，并按更慢语速折算成中文基准单位
    会去掉 <S1>...</S1> / <S2>... 等标签以及其他 <...> 标签
    """
    if not isinstance(s, str):
        return 0.0
    s = s.strip()
    if not s:
        return 0.0

    s = _ANY_TAG_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return 0.0

    cjk = len(_CJK_RE.findall(s))
    ja = len(_HIRAGANA_RE.findall(s)) + len(_KATAKANA_RE.findall(s))
    ko = len(_HANGUL_RE.findall(s))
    latin_words = len(_WORD_RE.findall(s))
    latin_weight = 1.0 / max(float(latin_cpm_ratio), 1e-6)
    return float(cjk + ja + ko) + float(latin_words) * float(latin_weight)

def _estimate_target_infer_length(
    target_text: str,
    caption: Optional[str],
    ref_audio: Optional[np.ndarray],
    ref_text: Optional[str],
    sr: int = 24000,
    default_cpm: float = _DEFAULT_CPM,
    min_seconds: float = 2.0,
    rate: float = 1.0,
) -> float:
    """
    返回“新生成段”的时长（秒），不含 ref 段。
    规则：
    - 有 target_text：若有 ref，则用 ref 语速替代；rate>1 表示按更快语速估时长
    - caption 有 <Audio>：额外 +2s
    - 无 text 且有 ref_audio：5s
    """

    tgt_units = _count_speech_units_for_len(target_text)

    # 没有 text：如果有 audio(ref) => 2s；否则也给个 2s 兜底（防止生成 0 帧）
    min_seconds = max(float(min_seconds), 0.0)

    if tgt_units <= 0:
        base = min_seconds
        return float(base)

    extra = 1.0 if (isinstance(caption, str) and _AUDIO_TAG_RE.search(caption)) else 0.0
    rate = max(float(rate), 1e-6)

    if ref_audio is not None and isinstance(ref_audio, np.ndarray) and ref_audio.size > 0:
        ref_units = _count_speech_units_for_len(ref_text)
        ref_dur = float(ref_audio.shape[0]) / float(sr)
        if ref_units > 0 and ref_dur > 1e-3:
            units_per_sec = ref_units / ref_dur
            base = tgt_units / max(units_per_sec, 1e-6)
        else:
            base = (tgt_units / default_cpm) * 60.0
        base = base / rate
    else:
        base = (tgt_units / default_cpm) * 60.0

    # 稳定性保护：太短/太长都裁一下
    final = base + extra
    final = float(np.clip(final, min_seconds, 120.0))
    return final


def merge_model_weights(model, new_ckpt_path, ignore_module=[], weight=0.5):
    """
    Args:
        model: 已经加载了原始权重的 torch.nn.Module
        new_ckpt_path: 新的 checkpoint 文件路径 (state_dict)
        weight: 融合比例，merged = weight * old + (1 - weight) * new
    """
    # 加载新的 ckpt
    if os.path.isfile(new_ckpt_path):
        base_dir = os.path.dirname(new_ckpt_path)
        ckpt_path = new_ckpt_path
        new_state_dict = torch_load_dist(new_ckpt_path, map_location='cpu', mmap=None)
    else:
        base_dir = new_ckpt_path
        new_state_dict, ckpt_path = get_last_checkpoint(new_ckpt_path)
    # new_state_dict = torch.load(new_ckpt_path, map_location="cpu")
    print(f'merge model from {ckpt_path} with weight {1 - weight}')
    # 拿到旧模型参数
    old_state_dict = model.state_dict()

    merged_state_dict = {}
    new_state_dict = new_state_dict['dit']
    for k, old_param in old_state_dict.items():
        if k in new_state_dict and old_param.shape == new_state_dict[k].shape and not any(ign in k for ign in ignore_module):
            new_param = new_state_dict[k]
            merged_state_dict[k] = weight * old_param + (1 - weight) * new_param
        else:
            # 如果没有对应权重，就保留旧的
            merged_state_dict[k] = old_param

    # 加载融合后的权重
    model.load_state_dict(merged_state_dict)

    return model

def gen_audio_html(group_infos: Dict[str, list], output_fp: Optional[str] = None,
                   title_name=None, extra_desc=None):
    """
    group_infos: { group_name: [info1, info2, ...], ... }
    每个 info 内至少包含：
      - 'tos_url'
      - 可选 'prompt_tos_url'
      - 可选 'caption'
    """
    if output_fp is None:
        output_fp = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name

    num_per_row = 5

    with open(output_fp, 'w', encoding='utf-8') as f:
        print('<html lang="en">', file=f)
        print('<head>', file=f)
        print('<meta charset="UTF-8">', file=f)
        print('<meta name="viewport" content="width=device-width, initial-scale=1.0">', file=f)
        if title_name is not None:
            print(f'<title>{title_name}</title>', file=f)
        else:
            print('<title>Audio Samples</title>', file=f)

        print('<style>', file=f)
        print(r'''
            body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
            .container { max-width: 1280px; margin: 0 auto; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 32px; }
            h1 { font-size: 2em; margin-bottom: 0.2em; }
            h2.group-title { font-size: 1.4em; margin: 1.2em 0 0.4em 0; }
            p.description { color: #666; margin-bottom: 1.5em; }
            td { padding: 10px; border: 2px solid DodgerBlue; vertical-align: top; text-align: center; }
            audio { width: 100%; }
            .audio-block { margin-bottom: 10px; text-align: left; }
            .audio-title { font-size: 13px; color: #333; margin: 0 0 4px 0; }
            .desc { margin-top: 10px; white-space: pre-wrap; text-align: left; font-size: 14px; background: #f8f8f8; padding: 8px; border-radius: 5px; }
        ''', file=f)
        print('</style>', file=f)
        print('</head>', file=f)
        print('<body>', file=f)
        if title_name is not None:
            print(f'  <h1>{title_name}</h1>', file=f)
        if extra_desc is not None:
            print(f'  <p class="description">{extra_desc}</p>', file=f)
        print('<div class="container">', file=f)

        # 按组依次写
        for group_name, infos in group_infos.items():
            if not infos:
                continue

            print(f'<h2 class="group-title">{group_name}</h2>', file=f)
            print('<table>', file=f)

            total = len(infos)
            rows = (total + num_per_row - 1) // num_per_row

            for row in range(rows):
                print('<tr>', file=f)
                for col in range(num_per_row):
                    j = row * num_per_row + col
                    if j >= total:
                        print('<td></td>', file=f)
                        continue

                    info = infos[j]

                    tos_url = info.get('tos_url', '')
                    prompt_tos_url = info.get('prompt_tos_url', '')
                    prompt_tos_urls = info.get('prompt_tos_urls', None)

                    caption = info.get('caption', '')
                    if caption is None:
                        caption = ''
                    caption = str(caption)

                    text = info.get('text', '')
                    if text is None:
                        text = ''
                    text = str(text)

                    print('<td>', file=f)

                    # Prompt audio(s) - 多人场景分行显示
                    if prompt_tos_urls and isinstance(prompt_tos_urls, (list, tuple)):
                        for si, p_url in enumerate(prompt_tos_urls):
                            if p_url:
                                print('<div class="audio-block">', file=f)
                                print(f'<div class="audio-title">Prompt S{si + 1}</div>', file=f)
                                print(f'<audio controls preload="none">', file=f)
                                print(f'  <source src="{p_url}" type="audio/wav">', file=f)
                                print('  Your browser does not support the audio element.', file=f)
                                print('</audio>', file=f)
                                print('</div>', file=f)
                    elif prompt_tos_url:
                        print('<div class="audio-block">', file=f)
                        print('<div class="audio-title">Prompt</div>', file=f)
                        print(f'<audio controls preload="none">', file=f)
                        print(f'  <source src="{prompt_tos_url}" type="audio/wav">', file=f)
                        print('  Your browser does not support the audio element.', file=f)
                        print('</audio>', file=f)
                        print('</div>', file=f)

                    # Output audio
                    print('<div class="audio-block">', file=f)
                    print('<div class="audio-title">Output</div>', file=f)
                    print(f'<audio controls preload="none">', file=f)
                    print(f'  <source src="{tos_url}" type="audio/wav">', file=f)
                    print('  Your browser does not support the audio element.', file=f)
                    print('</audio>', file=f)
                    print('</div>', file=f)

                    print('<div class="desc">', file=f)
                    safe_text = text.replace("<", "&lt;").replace(">", "&gt;")
                    safe_caption = caption.replace("<", "&lt;").replace(">", "&gt;")
                    print(f'text: {safe_text}\n', file=f)
                    print(f'caption: {safe_caption}\n', file=f)
                    print('</div>', file=f)

                    print('</td>', file=f)
                print('</tr>', file=f)

            print('</table>', file=f)

        print('</div>', file=f)
        print('</body>', file=f)
        print('</html>', file=f)

    return output_fp

def upload_tos_html(yml=None, out_path=None, title_name=None, extra_desc=None, use_quality_caption: Optional[bool] = None):
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = f'prompttts/{run_tag}'

    from collections import defaultdict
    group_infos = defaultdict(list)

    with open(yml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        samples_cfg = cfg['samples']

    # 将分组格式 flatten，保证顺序与推理时一致
    flat_samples = flatten_grouped_samples(samples_cfg)

    bucket = os.environ.get("TOS_BUCKET") or "humanaigc-ads-data"

    bench = cfg.get("bench", "bench")
    for _, sample in enumerate(flat_samples):
        group = sample.get("__group__", "default")
        group_idx = int(sample.get("__group_idx__", 0))

        wav_name = make_out_wav_name(bench, group, group_idx)
        wav = os.path.join(out_path, wav_name)
        if not os.path.exists(wav):
            continue

        # 1) 上传 output wav
        tos_url = send_file_to_tos(wav, sub_dir=sub_dir, bucket=bucket)
        print("out tos_url: ", tos_url)

        info = dict(sample)
        info['tos_url'] = tos_url

        # 2) 如果有 prompt_audio，也上传
        prompt_audio = sample.get('prompt_audio', None)
        if prompt_audio:
            try:
                if os.path.exists(prompt_audio):
                    prompt_tos_url = send_file_to_tos(prompt_audio, sub_dir=sub_dir, bucket=bucket)
                    print("prompt tos_url: ", prompt_tos_url)
                    info['prompt_tos_url'] = prompt_tos_url
                else:
                    print(f"[WARN] prompt_audio not found: {prompt_audio}")
            except Exception as e:
                print(f"[WARN] upload prompt_audio failed: {prompt_audio}, err={e}")

        # 3) 如果有 prompt_audios（多人 ref wav），逐个上传
        prompt_audios = sample.get('prompt_audios', None)
        if prompt_audios and isinstance(prompt_audios, (list, tuple)):
            prompt_tos_urls = []
            for pi, pa in enumerate(prompt_audios):
                try:
                    if os.path.exists(pa):
                        pa_tos_url = send_file_to_tos(pa, sub_dir=sub_dir, bucket=bucket)
                        print(f"prompt_audios[{pi}] tos_url: {pa_tos_url}")
                        prompt_tos_urls.append(pa_tos_url)
                    else:
                        print(f"[WARN] prompt_audios[{pi}] not found: {pa}")
                        prompt_tos_urls.append('')
                except Exception as e:
                    print(f"[WARN] upload prompt_audios[{pi}] failed: {pa}, err={e}")
                    prompt_tos_urls.append('')
            if prompt_tos_urls:
                info['prompt_tos_urls'] = prompt_tos_urls

        # IMPORTANT:
        # HTML caption should match inference behavior. Do NOT inject "Quality:" unless the model
        # was actually configured with use_quality_caption=true.
        if use_quality_caption is None:
            use_quality_caption = bool(hparams.get("use_quality_caption", False))
        if use_quality_caption:
            quality_txt, _ = _resolve_quality_text_and_flag(info, default_to_high=True)
        else:
            quality_txt = ""

        # 3) 如果没有 caption，但有 global/local，则自动拼一个 caption
        if 'caption' not in info or not str(info.get('caption', '')).strip():
            info['caption'] = _compose_caption_from_fields(info, quality_txt=quality_txt)

        text = info.get('text', None)
        if text is None:
            cap_for_extract = info.get('caption', '')
            if not isinstance(cap_for_extract, str):
                cap_for_extract = str(cap_for_extract)
            cap_for_extract = _norm_spaces_caption(cap_for_extract)
            text = extract_s_text(cap_for_extract)

        if not isinstance(text, str):
            text = str(text)
        text = _norm_spaces_caption(text)
        text = raw_text_process_s1s2_tagged(text)
        if text is None:
            text = ""
        info['text'] = text

        group_name = info.get('__group__', 'default')
        group_infos[group_name].append(info)

    html_path = gen_audio_html(group_infos, title_name=title_name, extra_desc=extra_desc)
    print(f"生成的HTML文件路径：{html_path}")
    html_tos = send_file_to_tos(html_path, sub_dir=sub_dir, bucket=bucket)
    print(f"生成的HTML TOS：{html_tos}")
    return html_tos
    
class ScriptSpeechInfer(DiTBuildModelMixin, SemanticLMBuildModelMixin):
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

    def build_model(self, dit_ckpt,
                    vae_ckpt=None,
                    merge_ckpt=None, merge_weight=0.5,):
        self.asr_model = None

        # 建议：每次 build 前清一下峰值统计，方便看 max_alloc
        if torch.cuda.is_available():
            idx = torch.device(self.device).index if torch.device(self.device).index is not None else torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(idx)

        snap = log_cuda_mem("enter build_model()", self.device)

        # ====== hparams & config ======
        config_path = (
            os.path.join(os.path.dirname(dit_ckpt), 'config.yaml')
            if dit_ckpt.endswith(('.ckpt', '.pt', '.pth', '.safetensors'))
            else os.path.join(dit_ckpt, 'config.yaml')
        )
        set_hparams(config=config_path, print_hparams=False, global_hparams=True)
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
        self.use_quality_caption = bool(hparams.get('use_quality_caption', False))
        print(
            f"[INFO] use_caption={self.use_caption}, "
            f"use_quality_caption={self.use_quality_caption}, "
            f"model_size={hparams.get('model_size', 'base')}"
        )

        if self.use_caption:
            if hparams.get('model_size', 'base') == 'seedance_7b':
                self.build_sd_text_encoder(hparams['text'])
                print_module_mem("sd_text_encoder (CPU)", self.sd_text_encoder)
                snap = log_cuda_mem("after build_sd_text_encoder() (CPU)", self.device, prev=snap)

                self.sd_text_encoder.eval()
                self.sd_text_encoder.to(self.device, dtype=self.precision)
                print_module_mem("sd_text_encoder (GPU)", self.sd_text_encoder)
                snap = log_cuda_mem("after sd_text_encoder.to(device)", self.device, prev=snap)

            elif 'goku' in hparams.get('model_size', 'base'):
                self.build_goku_text_encoder(hparams)
                print_module_mem("goku_text_encoder (CPU)", self.goku_text_encoder)
                snap = log_cuda_mem("after build_goku_text_encoder() (CPU)", self.device, prev=snap)

                self.goku_text_encoder.eval()
                self.goku_text_encoder.to(self.device, dtype=self.precision)
                print_module_mem("goku_text_encoder (GPU)", self.goku_text_encoder)
                snap = log_cuda_mem("after goku_text_encoder.to(device)", self.device, prev=snap)
        else:
            self.sd_text_encoder = None
            self.goku_text_encoder = None

        log_cuda_mem("leave build_model()", self.device, prev=snap)


    def run_goku_text_encoder(self, captions: list):
        inputs = self.goku_tokenizer(
            captions,
            padding=True,  # Dynamic / longest
            truncation=True,
            max_length=hparams['text_max_token_length'],
            return_tensors="pt",
        )

        input_ids, attention_masks = inputs.input_ids.cuda(), inputs.attention_mask.cuda()
        encoder_hidden_states = self.goku_text_encoder(
            input_ids,
            return_dict=False,
            attention_mask=attention_masks,
        )[0]  # [B, T, C]

        # 新的多类掩码（0/1/2）
        if hparams.get('use_caption_text_mark', False):
            caption_text_mark = build_audio_mask_from_ids(
                input_ids=input_ids,
                attention_mask=attention_masks,
                tokenizer=self.goku_tokenizer,
            )  
        else:
            caption_text_mark = None

        return encoder_hidden_states, caption_text_mark, attention_masks

    @torch.no_grad()
    def forward(self,
                text,
                ref_audio=None,
                ref_audios=None,
                ref_text=None,
                ref_texts=None,
                prompt=None,
                quality_flag=None,
                cfg_w=None,
                negative_prompt=None,
                infer_length=5.0,
                min_infer_length=None,
                start_time=0.2,
                end_time=0.2,
                num_step=100,
                timestep_annealing_w=(1.0,0.0,1.0),
                use_amo_sampler=False,
                use_sway=False,
                cpm=_DEFAULT_CPM,
                rate: float = 1.0,
                return_profile: bool = False,):
        """
        infer_length 语义：
          - 无 ref_audio：infer_length = 纯生成时长（秒）
          - 有 ref_audio：infer_length = 目标“新生成”时长（秒），
            内部会自动做 tgt_len = ref_lat_len + gen_lat_len
        """
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)
        speech = len(text) > 0
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']
        if min_infer_length is None:
            min_infer_length = hparams.get('min_infer_length', 3.0)
        min_infer_length = max(float(min_infer_length), 0.0)

        # ====== 参考音频 & 参考文本 ======
        multi_spk = ref_audios is not None and len(ref_audios) > 0

        if multi_spk:
            max_prompt_sec = 10.0
            sr = 24000
            sil_sec = 0.2
            sil_len = int(max(0.0, float(sil_sec)) * sr)

            ref_wav_parts = []
            ref_text_tagged_parts = []

            for spk_idx, spk_audio in enumerate(ref_audios):
                spk_tag = f"<S{spk_idx + 1}>"
                spk_close_tag = f"</S{spk_idx + 1}>"

                spk_audio_np = np.asarray(spk_audio, dtype=np.float32)
                if spk_audio_np.ndim > 1:
                    spk_audio_np = spk_audio_np[:, 0]

                per_spk_max_sec = max_prompt_sec / len(ref_audios)
                per_spk_max_len = int(per_spk_max_sec * sr)
                if sil_len > 0:
                    keep_len = max(0, per_spk_max_len - sil_len)
                    spk_audio_np = np.concatenate(
                        [spk_audio_np[:keep_len], np.zeros(sil_len, dtype=np.float32)],
                        axis=0,
                    )
                else:
                    spk_audio_np = spk_audio_np[:per_spk_max_len]

                ref_wav_parts.append(spk_audio_np)

                spk_ref_text = None
                if ref_texts is not None and spk_idx < len(ref_texts):
                    spk_ref_text = ref_texts[spk_idx]

                if spk_ref_text is None:
                    if getattr(self, "asr_model", None) is None:
                        self.asr_model = build_asr_model(self.device)
                    wav_16k = librosa.resample(
                        spk_audio_np.astype(np.float32, copy=False),
                        orig_sr=24000,
                        target_sr=16000
                    )
                    asr_out = run_asr_model([wav_16k], self.asr_model, with_segments=False)
                    if isinstance(asr_out, (list, tuple)) and len(asr_out) > 0:
                        asr_item = asr_out[0] or {}
                    else:
                        asr_item = asr_out or {}
                    spk_ref_text = asr_item.get('text_normed') or asr_item.get('text') or ''
                    if spk_ref_text and not spk_ref_text.endswith(('.', '。', '!', '！', '?', '？')):
                        spk_ref_text = spk_ref_text + '.'
                    print(f'asr_result S{spk_idx + 1}:', asr_item)

                if spk_ref_text and spk_ref_text.strip():
                    ref_text_tagged_parts.append(f"{spk_tag}{spk_ref_text}{spk_close_tag}")

            concat_audio = np.concatenate(ref_wav_parts, axis=0)
            concat_audio = np.ascontiguousarray(concat_audio, dtype=np.float32)
            ref_wav = torch.from_numpy(concat_audio)[None, :].to(self.device)

            ref_wav_lens = torch.LongTensor(
                [ref_wav.shape[1] // fm_wav * fm_wav]
            ).to(self.device)
            ref_wav = ref_wav[:, :ref_wav_lens[0]]

            ref_text = "".join(ref_text_tagged_parts) if ref_text_tagged_parts else None
            ref_audio = concat_audio

        elif ref_audio is not None:
            # 例如最多用 120 秒 prompt
            max_prompt_sec = 10.0
            sr = 24000
            sil_sec = 0.2
            sil_len = int(max(0.0, float(sil_sec)) * sr)
            max_len = int(max_prompt_sec * sr)

            ref_audio_np = np.asarray(ref_audio, dtype=np.float32)
            if ref_audio_np.ndim > 1:
                ref_audio_np = ref_audio_np[:, 0]
            if sil_len > 0:
                keep_len = max(0, max_len - sil_len)
                ref_audio_np = np.concatenate(
                    [ref_audio_np[:keep_len], np.zeros(sil_len, dtype=np.float32)],
                    axis=0,
                )
            else:
                ref_audio_np = ref_audio_np[:max_len]
            ref_audio = ref_audio_np

            ref_audio_np = np.ascontiguousarray(ref_audio, dtype=np.float32)
            ref_wav = torch.from_numpy(ref_audio_np)[None, :].to(self.device)

            ref_wav_lens = torch.LongTensor(
                [ref_wav.shape[1] // fm_wav * fm_wav]
            ).to(self.device)
            ref_wav = ref_wav[:, :ref_wav_lens[0]]

            # 自动 ASR 得到 ref_text（跟原来一样）
            if ref_text is None:
                if getattr(self, "asr_model", None) is None:
                    self.asr_model = build_asr_model(self.device)
                wav_16k = librosa.resample(
                    ref_audio.astype(np.float32, copy=False),
                    orig_sr=24000,
                    target_sr=16000
                )
                asr_out = run_asr_model([wav_16k], self.asr_model, with_segments=False)
                if isinstance(asr_out, (list, tuple)) and len(asr_out) > 0:
                    asr_item = asr_out[0] or {}
                else:
                    asr_item = asr_out or {}

                print('asr_result', asr_item)
                ref_text = asr_item.get('text_normed') or asr_item.get('text') or ''
                if ref_text and not ref_text.endswith(('.', '。', '!', '！', '?', '？')):
                    ref_text = ref_text + '.'

        else:
            ref_text = None
            ref_wav = None
        estimated_infer_length = _estimate_target_infer_length(
            target_text=text,
            caption=prompt,
            ref_audio=ref_audio,
            ref_text=ref_text,
            sr=24000,
            default_cpm=cpm,
            min_seconds=min_infer_length,
            rate=rate,
        )
        if infer_length is not None and float(infer_length) > 0:
            infer_length = max(float(infer_length), min_infer_length)
        else:
            infer_length = estimated_infer_length

        # ====== 计算新生成的 latent 长度 ======
        gen_lat_len = int(
            float(infer_length) * 24000 / self.hp_vae['hop_size'] / self.hp_vae['vae_stride']
        )
        gen_lat_len = max(gen_lat_len, 0)

        # ====== 准备唯一的 caption 字符串 ======
        if ref_wav is not None:
            # Keep ref inference aligned with training: ref caption path should
            # not consume quality caption text.
            quality_txt = ""
            caption = prompt
            bgm_flag_val = 2
            quality_flag_val = None
        else:
            quality_txt, quality_flag_val = _resolve_quality_text_and_flag(
                explicit_quality_flag=quality_flag,
                caption=prompt,
                default_to_high=True,
            )
            caption = _ensure_quality_caption(prompt, quality_txt) if getattr(self, "use_quality_caption", False) else prompt
            bgm_flag_val = _extract_bgm_flag_from_caption(caption)

        use_caption = getattr(self, "use_caption", False) and (caption is not None)
        caption_emb = None
        caption_lens = None
        caption_text_mark = None

        if use_caption and 'goku' in self.config.model_size:
            if negative_prompt is None:
                negative_prompt = (
                    "Environment: { 嘈杂远场环境，底噪和混响明显，人声不够贴近，<Audio>电流声、喷麦、碰麦、按键声、突发爆音</Audio>}."
                    "<BGM>BGM: { 背景音乐或环境声过强，掩蔽人声，未做 ducking }.</BGM>"
                    "Speaker: { 咬字含混，音高不稳，语气机械 }."
                    "Quality: { 低信噪比，爆麦削波，发闷发糊，明显伪影，音量不稳定 }."
                    "Content: { <Audio>电流声、喷麦、碰麦、按键声、突发爆音</Audio>，停连异常，重音错位，局部吞字和噪声干扰 }."
                )

            # 一次性编码，动态 padding 会对齐到同一 T
            all_embs, all_mark, all_att = self.run_goku_text_encoder([caption, negative_prompt])
            # all_embs: [2, T, C], all_att: [2, T]

            pos_text_embs = all_embs[0:1] * all_att[0:1][..., None]
            neg_text_embs = all_embs[1:2] * all_att[1:2][..., None]
            pos_lens = all_att[0:1].sum(-1)
            neg_lens = all_att[1:2].sum(-1)

            caption_emb = torch.cat([
                pos_text_embs,
                neg_text_embs,
                neg_text_embs,
            ], dim=0)

            caption_lens = torch.cat([
                pos_lens,
                neg_lens,
                neg_lens,
            ], dim=0).long()

            if all_mark is not None:
                pos_mark = all_mark[0:1]
                neg_mark = all_mark[1:2]
                caption_text_mark = torch.cat([
                    pos_mark,
                    neg_mark,
                    neg_mark,
                ], dim=0)

        # ====== 推断 latent_dim ======
        latent_dim = None
        if hasattr(self.vae, "latent_dim"):
            latent_dim = getattr(self.vae, "latent_dim")
        if latent_dim is None and hasattr(self.dit, "hp") and hasattr(self.dit.hp, "in_channels"):
            latent_dim = self.dit.hp.in_channels
        if latent_dim is None:
            latent_dim = 32  # 兜底

        # ====== VAE 编码参考音频，构造 lat_ctx / ctx_mask ======
        if ref_wav is not None:
            with torch.inference_mode():
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    lat_ctx_ref = self.vae.encode_latent(ref_wav)  # [1, L_ref, C]
                latent_dim = lat_ctx_ref.size(-1)

                ref_lat_len = lat_ctx_ref.size(1)
                tgt_len = ref_lat_len + gen_lat_len

                ctx_mask_ref = torch.ones_like(lat_ctx_ref[:, :, 0:1])

                lat = torch.nn.functional.pad(
                    lat_ctx_ref, (0, 0, 0, tgt_len - ref_lat_len),
                    mode='constant', value=0
                )
                ctx_mask = torch.nn.functional.pad(
                    ctx_mask_ref, (0, 0, 0, tgt_len - ref_lat_len),
                    mode='constant', value=0
                )
                if ref_text is not None and isinstance(ref_text, str) and ref_text.strip():
                    if '<S1>' in ref_text:
                        ref_text_tagged = ref_text
                    else:
                        ref_text_tagged = f"<S1>{ref_text}</S1>"
                    full_text = ref_text_tagged + (text or "")
                else:
                    full_text = text or ""
        else:
            tgt_len = gen_lat_len
            lat_ctx_ref = torch.zeros(1, 0, latent_dim).to(self.device)
            lat = torch.zeros(1, tgt_len, latent_dim).to(self.device)
            ctx_mask = torch.zeros(1, tgt_len, 1).to(self.device)
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
        if quality_flag_val is not None and hasattr(self.dit, 'hp') and bool(getattr(self.dit.hp, 'use_quality_flag', False)):
            if batch_size % 3 == 0:
                b0 = batch_size // 3
                base = torch.full(
                    (b0,),
                    int(quality_flag_val),
                    dtype=torch.long,
                    device=self.device,
                )
                inputs['quality_flag'] = torch.cat([
                    base,
                    torch.full_like(base, 3),
                    torch.full_like(base, 3),
                ], dim=0)
            else:
                inputs['quality_flag'] = torch.full(
                    (batch_size,),
                    int(quality_flag_val),
                    dtype=torch.long,
                    device=self.device,
                )

        # ====== 调 Diffusion.inference 生成 latent，再用 VAE decode 回波形 ======
        # RTF/profile only counts DiT forward (diffusion inference) + VAE decode.
        dit_time_s = None
        vae_decode_time_s = None
        with torch.autocast(device_type='cuda', dtype=self.precision):
            _sync_cuda(self.device)
            t0 = time.perf_counter()
            x = self.dit.inference(
                inputs,
                timesteps=num_step,
                seq_cfg_w=cfg_w,
                timestep_annealing_w=timestep_annealing_w,
                use_amo_sampler=use_amo_sampler,
                use_sway=use_sway,
            )
            _sync_cuda(self.device)
            dit_time_s = float(time.perf_counter() - t0)

            # 把前面 ref 的部分替换为真实 prompt latent
            if lat_ctx_ref.shape[1] > 0:
                x[:, :lat_ctx_ref.shape[1]] = lat_ctx_ref

            hop_size = self.hp_vae['hop_size']
            vae_stride = self.hp_vae['vae_stride']

            ref_lat = lat_ctx_ref.size(1) if lat_ctx_ref is not None else 0
            gen_lat = gen_lat_len

            # 0.5 秒 overlap -> latent 帧数
            overlap_sec = 0.5
            overlap_lat = int(overlap_sec * 24000 / hop_size / vae_stride)

            # 只解码 [ref_lat - overlap_lat, ref_lat + gen_lat]
            start_lat = max(0, ref_lat - overlap_lat)
            end_lat = ref_lat + gen_lat
            x_dec = x[:, start_lat:end_lat]   # [1, L_dec, C]

            # 解码（计时只包含 VAE.decode）
            _sync_cuda(self.device)
            t1 = time.perf_counter()
            with torch.autocast(device_type='cuda', dtype=self.precision):
                wav_dec = self.vae.decode(x_dec)[0, 0].to(torch.float32)
            _sync_cuda(self.device)
            vae_decode_time_s = float(time.perf_counter() - t1)

            # 丢掉 overlap 对应的 wav（以及 ref 部分）
            drop_lat = ref_lat - start_lat               # <= overlap_lat
            drop_wav = drop_lat * vae_stride * hop_size  # samples
            wav_pred = wav_dec[drop_wav:]                # 现在基本就是 “gen wav”

            # clip 防止溢出
            if wav_pred.abs().max() > 1:
                print('Wav amplitude exceed 1, clip it.')
                wav_pred = wav_pred / (wav_pred.abs().max())

            wav_pred = wav_pred.cpu().numpy()

            # For RTF, use generated segment length before adding extra silence.
            gen_dur_s = float(wav_pred.shape[0]) / 24000.0 if wav_pred is not None else 0.0
            wav_pred = splice_silence(wav_pred, sr=24000, sil_sec=0.2, mode="both")

        if not return_profile:
            return wav_pred

        fwd_decode_s = None
        rtf = None
        dit_rtf = None
        vae_decode_rtf = None
        if dit_time_s is not None and vae_decode_time_s is not None:
            fwd_decode_s = float(dit_time_s + vae_decode_time_s)
            if gen_dur_s > 1e-6:
                rtf = float(fwd_decode_s / gen_dur_s)
                dit_rtf = float(dit_time_s / gen_dur_s)
                vae_decode_rtf = float(vae_decode_time_s / gen_dur_s)
        profile = {
            "dit_time_s": dit_time_s,
            "vae_decode_time_s": vae_decode_time_s,
            "fwd_decode_time_s": fwd_decode_s,
            "gen_dur_s": gen_dur_s,
            "rtf": rtf,
            "dit_rtf": dit_rtf,
            "vae_decode_rtf": vae_decode_rtf,
        }
        return wav_pred, profile


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


def _norm_spaces_caption(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u3000", " ")
    s = _SPACE_RE.sub(" ", s)
    return s.strip()

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

    # Keep special tokens like "<|sp|>" intact. Bench texts frequently end with "<|sp|>".
    # If we blindly append punctuation at the very end, we may produce "<|sp|>。", which
    # breaks downstream tokenization. Put punctuation before trailing "<|sp|>" tokens.
    sp_tok = "<|sp|>"
    tail = ""
    base = txt.rstrip()
    while base.endswith(sp_tok):
        base = base[:-len(sp_tok)].rstrip()
        tail = sp_tok + tail

    if base and base[-1] not in '.,?!;。，？！；、':
        base = base + ('。' if is_chinese(base) else '.')

    txt = base + tail

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


def worker(rank, world_size, args, cfg, out_path, master_port):

    try:
        _dbg_event(
            "worker_start",
            rank=rank,
            world_size=world_size,
            pid=os.getpid(),
            master_port=master_port,
        )

        device = f'cuda:{rank}'
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(rank)
        print(f"[CUDA:{rank}] name={props.name}, total={_fmt_mb(props.total_memory)}")
        log_cuda_mem("after set_device()", device)


        # ====== init process group（保持你原来的逻辑）======
        if world_size > 1:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = str(master_port)
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['LOCAL_RANK'] = str(rank)

            from utils.commons import trainer
            trainer.LOCAL_RANK = rank

            dist.init_process_group(
                backend='nccl',
                rank=rank,
                world_size=world_size,
                device_id=torch.device(rank),
                timeout=timedelta(seconds=3000)
            )
            _dbg_event("dist_inited", rank=rank, backend="nccl")

        dit_ckpt = args.dit_ckpt
        _dbg_event("model_build_start", rank=rank, dit_ckpt=str(dit_ckpt))
        infer_ins = ScriptSpeechInfer(
            device,
            dit_ckpt=dit_ckpt,
            vae_ckpt=args.vae_ckpt,
            merge_ckpt=args.merge_ckpt,
            merge_weight=args.merge_weight,
            use_sa_front=cfg.get('use_sa_front', False),
        )
        _dbg_event("model_build_done", rank=rank)
        os.makedirs(out_path, exist_ok=True)
        negative_prompt = cfg.get('negative_prompt', None)

        raw_samples = cfg.get('samples', {}) or {}
        # 新格式：dict[group_name -> list[sample]]
        # 这里 flatten 一下，内部仍按 list 跑推理
        samples = flatten_grouped_samples(raw_samples)
        N = len(samples)

        # 每个 rank 固定跑这么多轮：保证所有 rank forward 次数一致
        num_iters = int(math.ceil(N / float(world_size))) if N > 0 else 0
        _dbg_event("loop_plan", rank=rank, N=N, num_iters=num_iters)

        for t in range(num_iters):
            idx = t * world_size + rank
            is_dummy = idx >= N

            if is_dummy:
                # dummy 占位：仍然调用 forward 以参与 collective
                sample = {"caption": "", "prompt_audio": None}
            else:
                sample = samples[idx]

            print(f"[Rank {rank}] Iter {t}/{num_iters-1} | idx={idx} | dummy={is_dummy}")
            _dbg_event("iter_start", rank=rank, t=t, idx=idx, dummy=is_dummy)

            # ====== 构造 text / caption ======
            quality_txt, resolved_quality_flag = _resolve_quality_text_and_flag(sample, default_to_high=True)
            # 没有 caption/global/local 字段时不构造 caption，跳过 caption encoder
            has_caption_fields = any(
                sample.get(k) for k in ('caption', 'global', 'local')
            )
            if not has_caption_fields:
                # IMPORTANT:
                # When running with multi-GPU and a sharded caption encoder (e.g. FSDP),
                # all ranks must execute the same collective sequence. If some ranks
                # skip caption encoding (caption=None) while others do it, NCCL will
                # deadlock on collectives like allgather.
                # Use an empty string to still go through caption encoder when enabled.
                caption = "" if (world_size > 1 and getattr(infer_ins, "use_caption", False)) else None
            elif bool(hparams.get('use_quality_caption', False)):
                caption = _compose_caption_from_fields(sample, quality_txt=quality_txt)
            else:
                caption = _compose_caption_from_fields(sample, quality_txt="")

            text = sample.get('text', None)
            if text is None:
                text = extract_s_text(caption)
            if not isinstance(text, str):
                text = str(text)
            text = _norm_spaces_caption(text)

            text = raw_text_process_s1s2_tagged(text)
            if text is None:
                text = ""

            print(f"[Rank {rank}] text: {text}")
            print(f"[Rank {rank}] caption: {caption}")

            # dummy 的 seed 也给一个固定值即可
            set_seed((len(text) + idx) if not is_dummy else (100000 + rank * 1000 + t))
            # set_seed(1234)

            # prompt audio（ref wav）
            audios = None
            if (not is_dummy) and sample.get('prompt_audios'):
                audios = []
                for p in sample['prompt_audios']:
                    a, _ = librosa.load(p, sr=24000)
                    a = normalize_wav_to_target_loudness(a, sr=24000, target_lufs=-23.0)
                    audios.append(a)
            elif (not is_dummy) and sample.get('prompt_audio'):
                audio_path = sample['prompt_audio']
                audio, _ = librosa.load(audio_path, sr=24000)
                audio = normalize_wav_to_target_loudness(audio, sr=24000, target_lufs=-23.0)
            else:
                audio = None

            # dummy 可以把 num_step 降到 1，减少无意义计算（仍会走一次 forward 参与 collective）
            num_step = cfg.get('num_step', 100)
            if is_dummy:
                num_step = 1

            t0 = time.time()
            _dbg_event(
                "forward_start",
                rank=rank,
                t=t,
                idx=idx,
                dummy=is_dummy,
                num_step=num_step,
                text_len=len(text or ""),
                has_ref=(audio is not None) or (audios is not None),
            )
            wav, profile = infer_ins.forward(
                text,
                ref_audio=audio if audios is None else None,
                ref_audios=audios,
                prompt=caption,
                quality_flag=resolved_quality_flag,
                cfg_w=cfg.get('cfg_w', None),
                infer_length=cfg.get('infer_length', 0),
                min_infer_length=cfg.get('min_infer_length', None),
                num_step=num_step,
                start_time=cfg.get('vad_len', 0.2),
                end_time=cfg.get('vad_len', 0.2),
                negative_prompt=negative_prompt,
                timestep_annealing_w=cfg.get('timestep_annealing_w', (1.0, 0.0, 1.0)),
                use_amo_sampler=cfg.get('use_amo_sampler', False),
                use_sway=cfg.get('use_sway', False),
                cpm=cfg.get('cpm', _DEFAULT_CPM),
                rate=cfg.get('rate', 1.0),
                return_profile=True,
            )
            _dbg_event(
                "forward_done",
                rank=rank,
                t=t,
                idx=idx,
                dummy=is_dummy,
                dt_s=(time.time() - t0),
                wav_len=(int(len(wav)) if wav is not None else None),
            )

            # Persist per-sample RTF stats (per-rank file to avoid write races).
            if not is_dummy:
                try:
                    rtf_fp = os.path.join(out_path, f"rtf_rank{rank}.jsonl")
                    rec = {
                        "rank": rank,
                        "t": t,
                        "idx": idx,
                        "group": sample.get("__group__", "default"),
                        "group_idx": int(sample.get("__group_idx__", 0)),
                        "num_step": int(num_step),
                    }
                    if isinstance(profile, dict):
                        rec.update(profile)
                    with open(rtf_fp, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"[WARN] write rtf jsonl failed: {e}")

            # dummy 不落盘
            if (not is_dummy) and wav is not None:
                bench = cfg.get("bench", "bench")
                group = sample.get("__group__", "default")
                group_idx = int(sample.get("__group_idx__", 0))

                out_name = make_out_wav_name(bench, group, group_idx)
                out_file = os.path.join(out_path, out_name)

                print(f"[Rank {rank}] save wav at {out_file}")
                _dbg_event("save_start", rank=rank, t=t, idx=idx, out_file=str(out_file))
                wav = normalize_wav_to_target_loudness(
                    wav,
                    sr=24000,
                    target_lufs=float(cfg.get('target_lufs', -23.0)),
                )
                sf.write(out_file, wav, 24000, "PCM_16")
                _dbg_event("save_done", rank=rank, t=t, idx=idx, out_file=str(out_file))

        # ====== 确保所有 rank 一起结束，防止有人提前退出导致别人 collective 卡住 ======
        if world_size > 1 and dist.is_initialized():
            _dbg_event("barrier_enter", rank=rank)
            dist.barrier()
            _dbg_event("barrier_exit", rank=rank)
            dist.destroy_process_group()
            _dbg_event("dist_destroyed", rank=rank)

        print(f"[Rank {rank}] Finished all samples (with dummy padding if needed).")
        _dbg_event("worker_done", rank=rank)
    except BaseException as e:
        _dbg_event(
            "worker_exception",
            rank=rank,
            err=repr(e),
            tb=traceback.format_exc()[-8000:],
        )
        raise


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
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument("--dit_ckpt", help="Path to model", type=str,
                        default='checkpoints/250622_scriptspeech_dit_singlespk_01')
    parser.add_argument("--merge_ckpt", help="Path to merge model", type=str)
    parser.add_argument("--merge_weight", help="Weight to merge model", type=float)
    parser.add_argument("--vae_ckpt", help="Path to VAE ckpt", type=str)
    args = parser.parse_args()
    # 读取 config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_path = f'{cfg["out_path"]}/{os.path.basename(args.dit_ckpt)}_{cfg["bench"]}/{cfg.get("cfg_w", "noCFG")}_timew{cfg.get("timestep_annealing_w", (1.0, 0.0, 1.0))}_amo{cfg.get("use_amo_sampler", False)}_sway{cfg.get("use_sway", False)}_step{cfg.get("num_step", 100)}_cpm{cfg.get("cpm", 300.0)}_seedlength'

    base_port = int(cfg.get("master_port", 10521))
    master_port = find_available_port(base_port, host="127.0.0.1", max_search=2000)
    print(f"[PORT] MASTER_PORT selected: {master_port} (base={base_port})")

    # 启动多进程，每个进程绑定一张 GPU
    processes = []
    gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    for rank in range(gpus):
        p = Process(target=worker, args=(rank, gpus, args, cfg, out_path, master_port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All ranks finished. 可以做后处理或上传结果")
    # upload_tos_html(out_path)  # 可选，上传结果
    # Aggregate RTF stats (forward+decode only) across ranks.
    rtf_vals = []
    fwd_decode_vals = []
    gen_dur_vals = []
    dit_rtf_vals = []
    vae_decode_rtf_vals = []
    try:
        for rank in range(gpus):
            rtf_fp = os.path.join(out_path, f"rtf_rank{rank}.jsonl")
            if not os.path.exists(rtf_fp):
                continue
            with open(rtf_fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    rtf = rec.get("rtf", None)
                    if isinstance(rtf, (int, float)) and math.isfinite(float(rtf)):
                        rtf_vals.append(float(rtf))
                    fd = rec.get("fwd_decode_time_s", None)
                    if isinstance(fd, (int, float)) and math.isfinite(float(fd)):
                        fwd_decode_vals.append(float(fd))
                    gd = rec.get("gen_dur_s", None)
                    if isinstance(gd, (int, float)) and math.isfinite(float(gd)):
                        gen_dur_vals.append(float(gd))
                    dr = rec.get("dit_rtf", None)
                    if isinstance(dr, (int, float)) and math.isfinite(float(dr)):
                        dit_rtf_vals.append(float(dr))
                    vr = rec.get("vae_decode_rtf", None)
                    if isinstance(vr, (int, float)) and math.isfinite(float(vr)):
                        vae_decode_rtf_vals.append(float(vr))
    except Exception as e:
        print(f"[WARN] aggregate rtf failed: {e}")

    rtf_line = ""
    if rtf_vals:
        rtf_avg = statistics.mean(rtf_vals)
        rtf_line = f" | RTF(avg, DiT forward+VAE decode only): {rtf_avg:.3f} (N={len(rtf_vals)})"
        if dit_rtf_vals:
            dit_rtf_avg = statistics.mean(dit_rtf_vals)
            rtf_line += f" | RTF(avg, model forward only): {dit_rtf_avg:.3f} (N={len(dit_rtf_vals)})"
        if vae_decode_rtf_vals:
            vae_decode_rtf_avg = statistics.mean(vae_decode_rtf_vals)
            rtf_line += f" | RTF(avg, VAE decode only): {vae_decode_rtf_avg:.3f} (N={len(vae_decode_rtf_vals)})"

    desc = (
        f"Inference setting: cfg weight: {cfg.get('cfg_w', None)}, timestep annealing weight: {cfg.get('timestep_annealing_w', (1.0, 0.0, 1.0))}, use amo sampler: {cfg.get('use_amo_sampler', False)}, use sway: {cfg.get('use_sway', False)}, "
        f"inference step: {cfg.get('num_step', 100)}, cpm: {cfg.get('cpm', _DEFAULT_CPM)}, rate: {cfg.get('rate', 1.0)}, seed: length."
        f"{rtf_line}"
    )

    from pathlib import Path

    p = Path(out_path)
    # 倒数两级：上上层/上层（模型目录 + 当前设置目录）
    title_name = "/".join(p.parts[-2:]) if len(p.parts) >= 2 else p.name
    
    model_cfg = _read_model_config_yaml(args.dit_ckpt)
    model_use_quality_caption = bool(model_cfg.get("use_quality_caption", False))
    upload_tos_html(
        yml=args.config,
        out_path=out_path,
        title_name=title_name,
        extra_desc=desc,
        use_quality_caption=model_use_quality_caption,
    )
