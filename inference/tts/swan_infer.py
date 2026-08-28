import os
import random
import re
import tempfile
from datetime import datetime, timedelta
import collections
import collections.abc
import math
from pathlib import Path
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
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint, torch_load_dist, get_all_ckpts
from utils.commons.hparams import set_hparams, hparams
from modules.asr.sensevoice.sensevoice_api import build_asr_model, run_asr_model
from modules.tts.swanaudio.build_model_utils import DiTBuildModelMixin, build_vae
from utils.commons.upload_tos_utils import send_file_to_tos
from tasks.tts.task_utils.prompttts_task_utils import build_audio_mask_from_ids
from utils.text.split_text import get_word_list
from utils.text.cosyvoice2_tokenizer import CosyVoice2Tokenizer
from tasks.tts.dataset_utils.swan_caption_fastdataset import (
    build_spk_mask_from_text_tokens,
    _get_sx_token_patterns,
    build_quality_caption_from_flag,
    QUALITY_FLAG_HIGH,
)

try:
    import pyloudnorm as pyln
except Exception:
    pyln = None
from tasks.tts.dataset_utils.base_fastdataset_v2 import simple_text_process

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

def _parse_ckpt_step_from_path(ckpt_path: Optional[str]) -> Optional[int]:
    if not isinstance(ckpt_path, str) or not ckpt_path:
        return None
    pattern = r'.*steps_(\d+)(?:\.ckpt|_backbone\.ckpt)'
    m = re.findall(pattern, ckpt_path)
    if not m:
        return None
    try:
        return int(m[0])
    except Exception:
        return None

def _resolve_ckpt_paths_for_report(ckpt_arg: Optional[str]):
    info = {
        "arg": ckpt_arg,
        "primary_path": None,
        "primary_step": None,
        "latest_path": None,
        "latest_step": None,
    }
    if ckpt_arg is None:
        return info
    ckpt_arg = str(ckpt_arg).strip()
    if not ckpt_arg:
        return info

    ckpt_arg = os.path.expanduser(ckpt_arg)

    if os.path.isfile(ckpt_arg):
        p = os.path.realpath(ckpt_arg)
        info["primary_path"] = p
        info["primary_step"] = _parse_ckpt_step_from_path(p)
        info["latest_path"] = p
        info["latest_step"] = info["primary_step"]
        return info

    if os.path.isdir(ckpt_arg):
        base_dir = os.path.realpath(ckpt_arg)

        model_only_last = os.path.join(base_dir, "model_only_last.ckpt")
        if os.path.exists(model_only_last):
            info["primary_path"] = os.path.realpath(model_only_last)
            info["primary_step"] = _parse_ckpt_step_from_path(info["primary_path"])
        else:
            ckpts = get_all_ckpts(base_dir, steps=None)
            if ckpts:
                info["primary_path"] = os.path.realpath(ckpts[0])
                info["primary_step"] = _parse_ckpt_step_from_path(info["primary_path"])
            else:
                info["primary_path"] = base_dir
                info["primary_step"] = None

        ckpts = get_all_ckpts(base_dir, steps=None)
        if ckpts:
            info["latest_path"] = os.path.realpath(ckpts[0])
            info["latest_step"] = _parse_ckpt_step_from_path(info["latest_path"])

        return info

    # 既不是文件也不是目录：照样展示一下用户传入的值（并尝试解析 steps）
    p = os.path.realpath(ckpt_arg)
    info["primary_path"] = p
    info["primary_step"] = _parse_ckpt_step_from_path(p)
    return info

def _ckpt_report_line(name: str, ckpt_arg: Optional[str]) -> str:
    info = _resolve_ckpt_paths_for_report(ckpt_arg)
    if info["primary_path"] is None:
        return f"{name}: <none>"

    primary = info["primary_path"]
    primary_step = info["primary_step"]
    if primary_step is None:
        primary_part = primary
    else:
        primary_part = f"{primary} (steps={primary_step})"

    latest = info["latest_path"]
    if latest and os.path.realpath(latest) != os.path.realpath(primary):
        latest_step = info["latest_step"]
        if latest_step is None:
            latest_part = latest
        else:
            latest_part = f"{latest} (steps={latest_step})"
        return f"{name}: {primary_part}; latest={latest_part}"

    return f"{name}: {primary_part}"

def _infer_exp_name_from_ckpt_arg(ckpt_arg: Optional[str]) -> str:
    if ckpt_arg is None:
        return "exp"
    p = os.path.expanduser(str(ckpt_arg)).strip()
    if not p:
        return "exp"

    if os.path.isfile(p):
        return os.path.basename(os.path.dirname(os.path.realpath(p))) or "exp"
    if os.path.isdir(p):
        return os.path.basename(os.path.realpath(p)) or "exp"

    # 既不是文件也不是目录：尽量从字符串末尾取名
    p2 = p.rstrip("/").rstrip("\\")
    return os.path.basename(p2) or "exp"

def _choose_step_for_dir(ckpt_arg: Optional[str]) -> Optional[int]:
    info = _resolve_ckpt_paths_for_report(ckpt_arg)
    return info.get("latest_step", None) or info.get("primary_step", None)

def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        # 不用 SO_REUSEADDR，避免“看起来可用但其实被占用”的假象
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False

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

# ===== Loudness 归一化(到 target LUFS) =====
def normalize_wav_to_target_loudness(wav, sr: int = 24000, target_lufs: float = -23.0):
    if wav is None:
        return wav
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav)
    if wav.size == 0:
        return wav
    if wav.ndim != 1:
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
        gain = float(np.power(10.0, (float(target_lufs) - 20.0 * np.log10(rms + eps)) / 20.0))
        wav_n = wav_f * gain

    peak_n = float(np.max(np.abs(wav_n))) if wav_n.size else 0.0
    if peak_n >= 1.0 and np.isfinite(peak_n) and peak_n > 0:
        wav_n = wav_n / peak_n * 0.95
    return wav_n.astype(np.float32)


# ===== target 时长自动估计 =====
_AUDIO_TAG_RE = re.compile(r"<\s*Audio\s*>", flags=re.IGNORECASE)
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
_KATAKANA_RE = re.compile(r"[\u30a0-\u30ff\uff66-\uff9d]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_DEFAULT_CPM = 220.0
_LATIN_CPM_RATIO = 0.5

# ===== BGM / Quality flag \u89e3\u6790 =====
_BGM_TAG_RE = re.compile(r"<\s*BGM\s*>(.*?)</\s*BGM\s*>", flags=re.IGNORECASE | re.DOTALL)
_BGM_VALUE_PREFIX_RE = re.compile(r"^\s*(?:BGM|\u80cc\u666f\u97f3\u4e50)\s*[:\uff1a]\s*", flags=re.IGNORECASE)
_NO_BGM_VALUES = {"\u65e0", "\u65e0\u80cc\u666f\u97f3\u4e50", "none", "no", "null", "0"}
_QUALITY_PREFIX_RE = re.compile(r"\bQuality\s*:", flags=re.IGNORECASE)


def _extract_bgm_flag_from_caption(caption: Optional[str]) -> int:
    """\u4ece caption \u7684 <BGM>...</BGM> \u62bd\u53d6:\u6709\u5185\u5bb9 BGM=1,\u65e0/\u7a7a\u6807\u7b7e=0"""
    if not isinstance(caption, str):
        return 0
    m = _BGM_TAG_RE.search(caption)
    if m is None:
        return 0
    bgm = _BGM_VALUE_PREFIX_RE.sub("", (m.group(1) or "").strip()).strip()
    if not bgm or bgm.lower() in _NO_BGM_VALUES:
        return 0
    return 1


def _ensure_quality_caption(caption: Optional[str], quality_txt: str) -> str:
    """\u628a quality_txt \u6ce8\u5165\u5230 caption \u7684 Content: \u6bb5\u4e4b\u524d,\u5df2\u5b58\u5728 Quality: \u5219\u4e0d\u52a8\u3002"""
    if not isinstance(caption, str):
        caption = ""
    caption = _norm_spaces_caption(caption)
    if not caption:
        return caption
    quality_txt = _norm_spaces_caption(quality_txt or "")
    if not quality_txt or _QUALITY_PREFIX_RE.search(caption):
        return caption
    idx = caption.find("Content:")
    if idx < 0:
        return caption
    head, tail = caption[:idx].rstrip(), caption[idx:].lstrip()
    return f"{head} {quality_txt} {tail}" if head else f"{quality_txt} {tail}"


def _count_speech_units_for_len(
    s: Optional[str],
    latin_cpm_ratio: float = _LATIN_CPM_RATIO,
) -> float:
    """
    用于估时长的"单位数":
    - 中文/日文(平假/片假)/韩文按"字符"计 1 单位
    - 拉丁词/数字按"词"计数,再按 latin_cpm_ratio 折算成 CJK 等价单位
      (默认 0.5,即 1 英文词 ≈ 2 中文字 在朗读时长上)
    会去掉 <S1>...</S1> / <Audio> 等所有 <...> 标签
    """
    if not isinstance(s, str):
        return 0.0
    s = _ANY_TAG_RE.sub("", s.strip())
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
    min_seconds: float = 5.0,
    rate: float = 1.0,
) -> float:
    """
    返回"新生成段"的时长(秒),不含 ref 段。
    - 有 ref:用 ref 语速换算 (units_per_sec),按目标单位反推时长
    - 无 ref:用 default_cpm 估算
    - rate>1 表示更快语速 → 时长按 rate 缩短
    - caption 含 <Audio>:额外 +2s
    """
    tgt_units = _count_speech_units_for_len(target_text)
    if tgt_units <= 0:
        return float(max(min_seconds, 0.0))

    extra = 2.0 if (isinstance(caption, str) and _AUDIO_TAG_RE.search(caption)) else 0.0
    rate = max(float(rate), 1e-6)

    if ref_audio is not None and isinstance(ref_audio, np.ndarray) and ref_audio.size > 0:
        ref_units = _count_speech_units_for_len(ref_text)
        ref_dur = float(ref_audio.shape[0]) / float(sr)
        if ref_units > 0 and ref_dur > 1e-3:
            base = tgt_units / max(ref_units / ref_dur, 1e-6)
        else:
            base = (tgt_units / default_cpm) * 60.0
        base = base / rate
    else:
        base = (tgt_units / default_cpm) * 60.0 / rate

    return float(np.clip(base + extra, min_seconds, 120.0))


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
            p.description {
                color: #666;
                margin-bottom: 1.5em;
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                word-break: break-word;
            }
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
            safe_extra_desc = str(extra_desc)
            safe_extra_desc = (
                safe_extra_desc.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            print(f'  <p class="description">{safe_extra_desc}</p>', file=f)
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

                    caption = info.get('caption', '')
                    if caption is None:
                        caption = ''
                    caption = str(caption)

                    text = info.get('text', '')
                    if text is None:
                        text = ''
                    text = str(text)

                    print('<td>', file=f)

                    # Prompt audio
                    if prompt_tos_url:
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

def upload_tos_html(
    yml=None,
    out_path=None,
    title_name=None,
    extra_desc=None,
    output_fp: Optional[str] = None,
    run_tag: Optional[str] = None,
    delete_local_wavs: bool = False,
):
    if run_tag is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = f'prompttts/{run_tag}'

    from collections import defaultdict
    group_infos = defaultdict(list)

    with open(yml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        samples_cfg = cfg['samples']

    # 将分组格式 flatten，保证顺序与推理时一致
    flat_samples = flatten_grouped_samples(samples_cfg)

    cluster = os.environ.get('CLUSTER', '').lower()
    if cluster == 'va':
        bucket = 'sa-ag-sg-research-sg'
    else:
        bucket = 'humanaigc-ads'

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

        if delete_local_wavs:
            try:
                os.remove(wav)
            except Exception as e:
                print(f"[WARN] remove local wav failed: {wav}, err={e}")

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

        # 3) 如果没有 caption，但有 global/local，则自动拼一个 caption
        if 'caption' not in info:
            g = info.get('global', '')
            l = info.get('local', '')
            parts = []
            if isinstance(g, str) and g.strip():
                parts.append(f"Global: {g}")
            if isinstance(l, str) and l.strip():
                parts.append(f"Local: {l}")
            info['caption'] = "\n".join(parts) if parts else ""

        # 4) 确保有 text：优先 sample['text']，否则从 caption 里提取（与推理侧一致）
        if 'text' not in info or info.get('text') is None:
            cap_for_extract = info.get('caption', '')
            if not isinstance(cap_for_extract, str):
                cap_for_extract = str(cap_for_extract)
            cap_for_extract = _norm_spaces_caption(cap_for_extract)

            t = extract_s_text(cap_for_extract)  # 兼容旧逻辑：从 <S1>/<S2> 提取
            if not isinstance(t, str):
                t = str(t)
            t = _norm_spaces_caption(t)

            t = raw_text_process_s1s2_tagged(t)
            if t is None:
                t = ""
            info['text'] = t


        group_name = info.get('__group__', 'default')
        group_infos[group_name].append(info)

    if output_fp is not None:
        os.makedirs(os.path.dirname(output_fp), exist_ok=True)

    html_path = gen_audio_html(group_infos, output_fp=output_fp, title_name=title_name, extra_desc=extra_desc)
    print(f"生成的HTML文件路径：{html_path}")
    html_tos = send_file_to_tos(html_path, sub_dir=sub_dir, bucket=bucket)
    print(f"生成的HTML TOS：{html_tos}")
    return html_tos
    
class SwanInfer(DiTBuildModelMixin):
    def __init__(self, device, dit_ckpt,
                 vae_ckpt=None,
                 merge_ckpt=None, merge_weight=0.5,
                 g2p_model: str = 'qwen'):
        self.device = device
        self.precision = torch.bfloat16
        self.build_model(
            dit_ckpt,
            vae_ckpt=vae_ckpt,
            merge_ckpt=merge_ckpt,
            merge_weight=merge_weight,
        )

    def _pad_wav_for_vae(self, wav: np.ndarray) -> torch.Tensor:
        """
        对齐 SwanBaseInfer.preprocess 的 wav padding：
        1) 按 win_size 做 pad，使得 len(wav) % win_size == win_size - 1（当余数 < win_size-1 时补）
        2) 再 pad sr//2 的尾巴（0.5s）
        返回: [1, T] 的 torch Tensor，dtype=self.precision，device=self.device
        """
        if wav is None:
            return None

        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1:
            wav = wav.reshape(-1)

        ws = int(hparams.get("win_size", 0) or 0)
        if ws > 0:
            r = len(wav) % ws
            if r < ws - 1:
                pad = (ws - 1 - r)
                if pad > 0:
                    wav = np.pad(wav, (0, pad), mode="constant", constant_values=0.0)

        # Swan: pad 0.5s tail
        tail = int(self.sr // 2)
        if tail > 0:
            wav = np.pad(wav, (0, tail), mode="constant", constant_values=0.0)

        return torch.tensor(wav, dtype=self.precision, device=self.device)[None, :]


    def _vae_encode_latent(self, wav_1xT: torch.Tensor) -> torch.Tensor:
        """
        兼容 encode_latent 可能返回 Tensor 或 (Tensor, other) 的两种实现。
        返回: latent [1, L, C]
        """
        out = self.vae.encode_latent(wav_1xT)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


    def _vae_decode_to_wav(self, lat_1xLxC: torch.Tensor) -> torch.Tensor:
        """
        兼容 decode 可能返回:
        - Tensor [B, 1, T]
        - Tensor [B, T]
        - (Tensor, other)
        返回: wav [T]，float32
        """
        out = self.vae.decode(lat_1xLxC)
        if isinstance(out, (tuple, list)):
            out = out[0]

        if out.ndim == 3:
            wav = out[0, 0]
        elif out.ndim == 2:
            wav = out[0]
        else:
            raise RuntimeError(f"Unexpected vae.decode output shape: {tuple(out.shape)}")

        return wav.to(torch.float32)

    def _asr_ref(self, ref_np: np.ndarray) -> str:
        """对 ref 音频做 ASR 并补尾标点。"""
        if self.asr_model is None:
            self.asr_model = build_asr_model(self.device)
        wav_16k = librosa.resample(ref_np.astype(np.float32), orig_sr=self.sr, target_sr=16000)
        asr_out = run_asr_model([wav_16k], self.asr_model, with_segments=False)
        asr_item = (asr_out[0] if isinstance(asr_out, (list, tuple)) and asr_out else asr_out) or {}
        print('asr_result', asr_item)
        text = asr_item.get('text_normed') or asr_item.get('text') or ''
        if text and not text.endswith(('.', '。', '!', '！', '?', '？')):
            text = text + '.'
        return text

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
        if hparams.get("use_spk_mask", False):
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
        else:
            spk_mask = None

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
        if dit_ckpt.endswith('.ckpt'):
            set_hparams(config=os.path.join(Path(dit_ckpt).parent, 'config.yaml'),
                    print_hparams=False, global_hparams=True)
        else:
            set_hparams(config=os.path.join(dit_ckpt, 'config.yaml'),
                    print_hparams=False, global_hparams=True)
        hparams["exp_name"] = 'infer'
        self.config = AttrDict(hparams)

        snap = log_cuda_mem("after set_hparams()", self.device, prev=snap)

        # ====== VAE & audio tokenizer ======
        self.vae, self.hp_vae = build_vae(
            hparams.get('vae_ckpt'), 
            hparams.get('vae_latent_mean', None), hparams.get('vae_latent_std', None),
            hparams.get('latent_norm_mode', 'global'),
            attn_implementation='flash_attention_2',
        )
        self.dit_text_tokenizer, self.dit_vocab_size = self.build_dit_text_tokenizer()
        print_module_mem("VAE (on CPU)", self.vae)
        snap = log_cuda_mem("after build_vae() (still CPU)", self.device, prev=snap)

        # 搬到 GPU
        self.vae.to(self.device)
        print_module_mem("VAE (on GPU)", self.vae)
        snap = log_cuda_mem("after vae.to(device)", self.device, prev=snap)

        # ====== 对齐 Swan：从 VAE 读取 sr / hop_length ======
        self.sr = int(getattr(self.vae, "sample_rate", 24000))

        hop_len = getattr(self.vae, "hop_length", None)
        if hop_len is None:
            hop_size = int(self.hp_vae.get("hop_size", hparams.get("hop_size", 960)))
            vae_stride = int(self.hp_vae.get("vae_stride", 1))
            hop_len = hop_size * vae_stride
        self.hop_length = int(hop_len)

        self.hop_size = int(self.hp_vae.get("hop_size", hparams.get("hop_size", 960)))
        self.vae_stride = int(self.hp_vae.get("vae_stride", max(1, self.hop_length // max(self.hop_size, 1))))
        # Swan: fm = hop_length // hop_size
        self.fm = int(self.hop_length // max(self.hop_size, 1))

        print(f"[INFO] VAE io: sr={self.sr}, hop_length={self.hop_length}, hop_size={self.hop_size}, vae_stride={self.vae_stride}, fm={self.fm}")

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
        print(f"[INFO] use_caption={self.use_caption}, use_quality_caption={self.use_quality_caption}, model_size={hparams.get('model_size', 'base')}")

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
                ref_text=None,
                ref_audios=None,      # 多说话人:list[np.ndarray] 时会拼接为 <S1>...<S2>... 的 ref
                ref_texts=None,       # 多说话人对应的 ref 文本(可为 None,逐条 ASR)
                prompt=None,          # caption
                cfg_w=None,
                negative_prompt=None,
                infer_length=5.0,     # 仍保留接口:无 ref 时用秒;有 ref 时主要走 token 比例法
                start_time=0.2,
                end_time=0.2,
                num_step=100,
                timestep_annealing_w=(1.0, 0.0, 1.0),
                use_amo_sampler=False,
                use_sway=False,
                cpm=_DEFAULT_CPM,
                rate: float = 1.0,    # 语速倍率:>1 输出语速更快(目标段更短)
                bgm_flag=None,        # None → unknown(2)
                quality_flag=None,    # None → unknown(3)
                ):
        """
        对齐 Swan 的 VAE / sr / hop_length 行为：
        - sr 从 self.sr 取（VAE sample_rate）
        - hop_length 从 self.hop_length 取（VAE hop_length）
        - ref_audio -> VAE encode_latent 得到 prompt latent
        - 有 ref 时 target_size 用 tokenizer.encode 长度比例估算（Swan 同款）
        - decode 后丢掉 prompt 部分 wav：ref_lat_len * hop_length
        """
        import torch.nn.functional as F

        speech = isinstance(text, str) and (len(text) > 0)

        # ====== 参考音频 & 参考文本 ======
        # 多说话人:拼接每个 ref + 0.2s 静音,组装 <S1>...<S2>... 的 tagged ref_text
        # 单 ref:走原 ref_audio 路径
        multi_spk = ref_audios is not None and len(ref_audios) > 0
        max_prompt_sec = 15.0
        sil_sec = 0.2
        sil_len = int(sil_sec * self.sr)

        if multi_spk:
            per_spk_max_len = int(max_prompt_sec * self.sr / max(len(ref_audios), 1))
            audio_parts = []
            text_parts = []
            for spk_idx, spk_audio in enumerate(ref_audios):
                spk_np = np.asarray(spk_audio, dtype=np.float32)
                if spk_np.ndim > 1:
                    spk_np = spk_np[:, 0]
                keep = max(0, per_spk_max_len - sil_len)
                spk_np = np.concatenate([spk_np[:keep], np.zeros(sil_len, dtype=np.float32)], axis=0)
                spk_np = normalize_wav_to_target_loudness(spk_np, sr=self.sr, target_lufs=-23.0)
                audio_parts.append(spk_np)

                spk_text = ref_texts[spk_idx] if (ref_texts is not None and spk_idx < len(ref_texts)) else None
                if spk_text is None:
                    spk_text = self._asr_ref(spk_np)
                if spk_text and spk_text.strip():
                    text_parts.append(f"<S{spk_idx + 1}>{spk_text}</S{spk_idx + 1}>")

            ref_audio = np.ascontiguousarray(np.concatenate(audio_parts, axis=0), dtype=np.float32)
            ref_text = "".join(text_parts) if text_parts else None
        elif ref_audio is not None:
            ref_audio = np.asarray(ref_audio, dtype=np.float32)
            if ref_audio.ndim > 1:
                ref_audio = ref_audio[:, 0]
            ref_audio = ref_audio[:int(max_prompt_sec * self.sr)]
            ref_audio = normalize_wav_to_target_loudness(ref_audio, sr=self.sr, target_lufs=-23.0)
            if ref_text is None:
                ref_text = self._asr_ref(ref_audio)

        use_ref = ref_audio is not None
        ref_wav = self._pad_wav_for_vae(ref_audio) if use_ref else None
        vae_latent = None
        ref_lat_len = 0

        if not use_ref:
            ref_text = None

        print(f'| task_type: {1 if use_ref else 2}, CFG: {cfg_w}, multi_spk: {multi_spk}, rate: {rate}')

        # ====== caption / goku caption encoder ======
        caption = prompt
        # 无 ref 时,按 quality_flag(默认 HIGH)注入 Quality: 段到 caption,与训练 use_quality_caption 一致
        if getattr(self, 'use_quality_caption', False) and not use_ref:
            qf = QUALITY_FLAG_HIGH if quality_flag is None else int(quality_flag)
            quality_txt, _ = build_quality_caption_from_flag(qf)
            caption = _ensure_quality_caption(caption, quality_txt)
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

            all_embs, all_mark, all_att = self.run_goku_text_encoder([caption, negative_prompt])

            pos_text_embs = all_embs[0:1] * all_att[0:1][..., None]
            neg_text_embs = all_embs[1:2] * all_att[1:2][..., None]
            pos_lens = all_att[0:1].sum(-1)
            neg_lens = all_att[1:2].sum(-1)

            caption_emb = torch.cat([pos_text_embs, neg_text_embs, neg_text_embs], dim=0)
            caption_lens = torch.cat([pos_lens, neg_lens, neg_lens], dim=0).long()

            if all_mark is not None:
                pos_mark = all_mark[0:1]
                neg_mark = all_mark[1:2]
                caption_text_mark = torch.cat([pos_mark, neg_mark, neg_mark], dim=0)

        # ====== 先 VAE encode prompt latent ======
        latent_dim = None
        if hasattr(self.vae, "latent_dim"):
            latent_dim = getattr(self.vae, "latent_dim")

        if use_ref:
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=self.precision):
                vae_latent = self._vae_encode_latent(ref_wav)  # [1, L_ref, C]
            if latent_dim is None:
                latent_dim = vae_latent.size(-1)
            ref_lat_len = int(vae_latent.size(1))
        else:
            if latent_dim is None and hasattr(self.dit, "hp") and hasattr(self.dit.hp, "in_channels"):
                latent_dim = int(self.dit.hp.in_channels)
            if latent_dim is None:
                latent_dim = 32

        # ====== 组装 full_text ======
        if use_ref and ref_text is not None:
            full_text = ref_text + text
        else:
            full_text = text

        # ====== target_size(latent length): 有 ref 用 token 比例,无 ref 用秒估算;rate 调速 ======
        rate_safe = max(float(rate), 1e-6)
        if use_ref and (ref_text is not None) and (ref_lat_len > 0):
            try:
                ref_tok_len = len(self.dit_text_tokenizer.encode(ref_text))
                full_tok_len = len(self.dit_text_tokenizer.encode(full_text))
            except Exception:
                ref_tok_len = full_tok_len = 0

            if ref_tok_len > 0 and full_tok_len > 0:
                gen_lat_len = round((full_tok_len - ref_tok_len) / ref_tok_len * ref_lat_len / rate_safe)
            else:
                sec = _estimate_target_infer_length(
                    target_text=text, caption=caption,
                    ref_audio=np.asarray(ref_audio), ref_text=ref_text,
                    sr=self.sr, default_cpm=cpm, rate=rate,
                )
                gen_lat_len = int(sec * self.sr / max(self.hop_length, 1))
            target_size = ref_lat_len + max(gen_lat_len, 1)
            print(f"| target_size(latent)={target_size}, ref_lat_len={ref_lat_len}")
        else:
            sec = _estimate_target_infer_length(
                target_text=text, caption=caption,
                ref_audio=None, ref_text=None,
                sr=self.sr, default_cpm=cpm, rate=rate,
            )
            target_size = int(np.clip(int(sec * self.sr / max(self.hop_length, 1)), 1, 999999))
            print(f"| target_size(latent)={target_size} (no ref), sec={sec:.2f}")

        # ====== 构造 lat / ctx_mask（对齐 Swan）======
        if use_ref:
            ctx_mask = torch.ones_like(vae_latent[:, :, 0:1])
            lat = F.pad(vae_latent, (0, 0, 0, target_size - ref_lat_len), mode='constant', value=0)
            ctx_mask = F.pad(ctx_mask, (0, 0, 0, target_size - ctx_mask.size(1)), mode='constant', value=0)
        else:
            lat = torch.zeros(1, target_size, latent_dim, device=self.device, dtype=self.precision)
            ctx_mask = torch.zeros(1, target_size, 1, device=self.device, dtype=self.precision)

        # ====== 文本 token（你的 spk_mask 逻辑保留）======
        txt_tokens, txt_mask, txt_lens, spk_mask = self._tokenize_dit_text(full_text)

        # ====== VAD mask：把 25fps 改成 sr/hop_length ======
        vad_mask = torch.zeros_like(lat[:, :, :1])
        if not self.config.get('drop_vad', False) and speech:
            fps = float(self.sr) / float(max(self.hop_length, 1))  # latent frames per sec
            st = int(start_time * fps)
            ed = int(end_time * fps)
            if ed <= 0:
                vad_mask[:, st:] = 1.0
            else:
                vad_mask[:, st:-ed] = 1.0

        # 3 路 CFG：复制 VAD
        vad_mask = torch.cat([vad_mask] * 3, dim=0)

        # ====== 文本 CFG ======
        txt_tokens[~txt_mask] = self.cfg_mask_text_token
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

        # ====== latent / ctx_mask 也复制 3 路 ======
        lat = torch.cat([lat, torch.zeros_like(lat), torch.zeros_like(lat)], dim=0)
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
            'tgt_len': torch.full((batch_size,), target_size, dtype=torch.long, device=self.device),
        }

        # bgm_flag / quality_flag: 与 3 路 CFG 对齐 → [cond, uncond, uncond]
        # 默认值:
        #   - 有 ref:bgm/quality 都给 unknown(ref 已携带这些信号,避免重复条件)
        #   - 无 ref:bgm 从 caption 抽,quality 默认 HIGH(2)
        hp = getattr(self.dit, 'hp', None)
        bsz1 = batch_size // 3
        if hp is not None and bool(getattr(hp, 'use_bgm_flag', False)):
            if bgm_flag is None:
                bgm_val = 2 if use_ref else _extract_bgm_flag_from_caption(prompt)
            else:
                bgm_val = int(bgm_flag)
            cond = torch.full((bsz1,), bgm_val, dtype=torch.long, device=self.device)
            uncond = torch.full((bsz1,), 2, dtype=torch.long, device=self.device)
            inputs['bgm_flag'] = torch.cat([cond, uncond, uncond], dim=0)
        if hp is not None and bool(getattr(hp, 'use_quality_flag', False)):
            if quality_flag is None:
                qual_val = 3 if use_ref else 2  # 无 ref 默认 HIGH
            else:
                qual_val = int(quality_flag)
            cond = torch.full((bsz1,), qual_val, dtype=torch.long, device=self.device)
            uncond = torch.full((bsz1,), 3, dtype=torch.long, device=self.device)
            inputs['quality_flag'] = torch.cat([cond, uncond, uncond], dim=0)

        # ====== DiT inference ======
        with torch.autocast(device_type='cuda', dtype=self.precision):
            x = self.dit.inference(
                inputs,
                timesteps=num_step,
                seq_cfg_w=cfg_w,
                timestep_annealing_w=timestep_annealing_w,
                use_amo_sampler=use_amo_sampler,
                use_sway=use_sway
            )

        # 只解码第一路（正向条件），减少显存
        x0 = x[0:1]

        # 把 prompt latent 覆盖回去（对齐 Swan）
        if use_ref and vae_latent is not None and ref_lat_len > 0:
            x0[:, :ref_lat_len] = vae_latent

        # ====== VAE decode ======
        with torch.autocast(device_type='cuda', dtype=self.precision):
            wav_dec = self._vae_decode_to_wav(x0)  # [T] float32

        # ====== 丢掉 prompt 部分 wav（对齐 Swan：ref_lat_len * hop_length）======
        if use_ref and ref_lat_len > 0:
            drop_wav = int(ref_lat_len * self.hop_length)
            wav_pred = wav_dec[drop_wav:]
        else:
            wav_pred = wav_dec

        if wav_pred.abs().max() > 1:
            print('Wav amplitude exceed 1, clip it.')
            wav_pred = wav_pred / (wav_pred.abs().max())

        wav_pred = wav_pred.cpu().numpy()
        wav_pred = normalize_wav_to_target_loudness(wav_pred, sr=self.sr, target_lufs=-23.0)
        wav_pred = splice_silence(wav_pred, sr=self.sr, sil_sec=0.2, mode="both")

        return wav_pred



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

def raw_text_process_s1s2_tagged(txt, wav=None, wav_len=None, check_len=False):
    """
    对含 <S1>/<S2> 的 text 做 simple_text_process 等价清洗，
    只处理标签内部文本，保留标签结构与顺序。
    """
    if not isinstance(txt, str) or not txt.strip():
        return ""

    # 没标签就走普通清洗
    if _S1S2_TEXT_RE.search(txt) is None:
        return simple_text_process(txt, wav=wav, wav_len=wav_len)

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
            plain_norm = simple_text_process(plain) or ""
            try:
                if len(get_word_list(plain_norm)) > wav_len // hparams['hop_size'] // 4:
                    return None
            except Exception:
                pass

    # 分段清洗并重建
    def _repl(m):
        tag = m.group(1).upper()
        inner = m.group(2)
        inner_proc = simple_text_process(inner)
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

def _encode_tag_pattern(tokenizer, s: str):
    ids = tokenizer.encode(s)
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    return list(map(int, ids))

def _find_pattern_starts(arr: np.ndarray, pat: np.ndarray):
    n = arr.shape[0]
    m = pat.shape[0]
    if m == 0 or n < m:
        return np.empty((0,), dtype=np.int64)
    if m == 1:
        return np.flatnonzero(arr == pat[0]).astype(np.int64)
    try:
        win = np.lib.stride_tricks.sliding_window_view(arr, m)
    except Exception:
        shape = (n - m + 1, m)
        strides = (arr.strides[0], arr.strides[0])
        win = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    eq = (win == pat)
    return np.flatnonzero(eq.all(axis=1)).astype(np.int64)


def worker(rank, world_size, args, cfg, out_path, master_port):

    device = f'cuda:{rank}'
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(rank)
    print(f"[CUDA:{rank}] name={props.name}, total={_fmt_mb(props.total_memory)}")
    log_cuda_mem("after set_device()", device)

    # ====== init process group ======
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

    dit_ckpt = args.dit_ckpt
    infer_ins = SwanInfer(
        device,
        dit_ckpt=dit_ckpt,
        vae_ckpt=args.vae_ckpt,
        merge_ckpt=args.merge_ckpt,
        merge_weight=args.merge_weight,
    )

    os.makedirs(out_path, exist_ok=True)
    negative_prompt = cfg.get('negative_prompt', None)

    raw_samples = cfg.get('samples', {}) or {}
    samples = flatten_grouped_samples(raw_samples)
    N = len(samples)

    num_iters = int(math.ceil(N / float(world_size))) if N > 0 else 0

    for t in range(num_iters):
        idx = t * world_size + rank
        is_dummy = idx >= N

        if is_dummy:
            sample = {"caption": "", "prompt_audio": None}
        else:
            sample = samples[idx]

        print(f"[Rank {rank}] Iter {t}/{num_iters-1} | idx={idx} | dummy={is_dummy}")

        caption = sample.get('caption', '')
        if not isinstance(caption, str):
            caption = str(caption)
        caption = _norm_spaces_caption(caption)

        # 新增：优先用 sample['text']，否则回退到从 caption 提取
        text = sample.get('text', None)
        if text is None:
            text = extract_s_text(caption)   # 兼容旧逻辑
        if not isinstance(text, str):
            text = str(text)
        text = _norm_spaces_caption(text)

        # 这函数本身也兼容“无 <S1>/<S2> 标签”的纯文本，会自动走普通清洗
        text = raw_text_process_s1s2_tagged(text)
        if text is None:
            text = ""

        print(f"[Rank {rank}] text: {text}")
        print(f"[Rank {rank}] caption: {caption}")

        if text is None:
            text = ""

        print(f"[Rank {rank}] text: {text}")
        print(f"[Rank {rank}] caption: {caption}")

        set_seed((len(text) + idx) if not is_dummy else (100000 + rank * 1000 + t))

        # ====== prompt audio:sr 对齐到 VAE sample_rate;支持单 ref(prompt_audio)与多说话人(prompt_audios) ======
        audio = None
        audios = None
        if not is_dummy:
            prompt_audios = sample.get('prompt_audios', None)
            if prompt_audios and isinstance(prompt_audios, (list, tuple)):
                audios = []
                for p in prompt_audios:
                    a, _ = librosa.load(p, sr=infer_ins.sr)
                    audios.append(a)
            elif sample.get('prompt_audio'):
                audio, _ = librosa.load(sample['prompt_audio'], sr=infer_ins.sr)

        num_step = cfg.get('num_step', 100)
        if is_dummy:
            num_step = 1

        wav = infer_ins.forward(
            text,
            ref_audio=audio,
            ref_audios=audios,
            ref_texts=sample.get('ref_texts', None),
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
            cpm=cfg.get('cpm', _DEFAULT_CPM),
            rate=cfg.get('rate', 1.0),
            bgm_flag=sample.get('bgm_flag', None),
            quality_flag=sample.get('quality_flag', None),
        )

        if (not is_dummy) and wav is not None:
            bench = cfg.get("bench", "bench")
            group = sample.get("__group__", "default")
            group_idx = int(sample.get("__group_idx__", 0))

            out_name = make_out_wav_name(bench, group, group_idx)
            out_file = os.path.join(out_path, out_name)

            print(f"[Rank {rank}] save wav at {out_file}")
            # ====== 保存 sr 也用 infer_ins.sr ======
            sf.write(out_file, wav, int(infer_ins.sr), "PCM_16")

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    print(f"[Rank {rank}] Finished all samples (with dummy padding if needed).")


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
                        default='checkpoints/250622_Swan_dit_singlespk_01')
    parser.add_argument("--merge_ckpt", help="Path to merge model", type=str)
    parser.add_argument("--merge_weight", help="Weight to merge model", type=float)
    parser.add_argument("--vae_ckpt", help="Path to VAE ckpt", type=str)
    args = parser.parse_args()
    # 读取 config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bench = cfg.get("bench", "bench")
    exp_name = _infer_exp_name_from_ckpt_arg(args.dit_ckpt)

    dit_step_for_dir = _choose_step_for_dir(args.dit_ckpt)
    step_dirname = f"steps_{dit_step_for_dir}" if dit_step_for_dir is not None else "steps_unknown"

    setting_dir = (
        f'{cfg.get("cfg_w", "noCFG")}'
        f'_timew{cfg.get("timestep_annealing_w", (1.0, 0.0, 1.0))}'
        f'_amo{cfg.get("use_amo_sampler", False)}'
        f'_sway{cfg.get("use_sway", False)}'
        f'_step{cfg.get("num_step", 100)}'
        f'_cpm{cfg.get("cpm", _DEFAULT_CPM)}'
        f'_rate{cfg.get("rate", 1.0)}'
        f'_seedlength'
    )

    step_root = os.path.join(cfg["out_path"], exp_name, step_dirname)
    os.makedirs(step_root, exist_ok=True)

    out_path = os.path.join(step_root, bench, setting_dir)

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

    desc_lines = [
        (
            f"Inference setting: cfg weight: {cfg.get('cfg_w', None)}, "
            f"timestep annealing weight: {cfg.get('timestep_annealing_w', (1.0, 0.0, 1.0))}, "
            f"use amo sampler: {cfg.get('use_amo_sampler', False)}, "
            f"use sway: {cfg.get('use_sway', False)}, "
            f"inference step: {cfg.get('num_step', 100)}, "
            f"cpm: {cfg.get('cpm', _DEFAULT_CPM)}, rate: {cfg.get('rate', 1.0)}, seed: length."
        ),
        _ckpt_report_line("DiT", args.dit_ckpt),
        _ckpt_report_line("VAE", args.vae_ckpt),
    ]

    if args.merge_ckpt:
        merge_line = _ckpt_report_line("Merge", args.merge_ckpt)
        if args.merge_weight is not None:
            merge_line = f"{merge_line} (merge_weight={args.merge_weight})"
        desc_lines.append(merge_line)

    desc = "\n".join(desc_lines)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    title_name = "/".join([exp_name, step_dirname, bench, setting_dir])

    keep_local_wavs = bool(cfg.get("keep_local_wavs", True))
    local_html_fp = os.path.join(step_root, f"report_{_safe_fname(bench)}_{run_tag}.html")

    upload_tos_html(
        yml=args.config,
        out_path=out_path,
        title_name=title_name,
        extra_desc=desc,
        output_fp=local_html_fp,
        run_tag=run_tag,
        delete_local_wavs=(not keep_local_wavs),
    )