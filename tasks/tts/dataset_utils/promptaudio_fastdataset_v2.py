import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import random
import re
import traceback

import torch
import torchaudio
import numpy as np
import torch.utils
import torch.utils.data

from utils.commons.hparams import hparams
from utils.commons.base_shm_dataset import get_from_global_stores
from utils.commons.dataset_utils import  SkipLogger
from utils.audio.vad import run_vad_trim
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text import is_chinese

from tasks.tts.dataset_utils.tts_fastdataset_v2 import BaseTTSShmDataset

DEBUG = False

# 没有 ph/tone 时统一使用 PAD ID（与 collater 的 pad 值保持一致，默认 0）
# PH_PAD_ID = int(hparams.get('ph_pad_id', 0))
# TONE_PAD_ID = int(hparams.get('tone_pad_id', 0))

# ======== precompiled regex patterns ========
_SPACE_RE = re.compile(r'\s+')

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

def _find_pattern_starts(arr: np.ndarray, pat: np.ndarray):
    """
    在 1D int array `arr` 中找到 pattern `pat` 的所有起点下标（vectorized）。
    返回 np.ndarray[int64]，升序。
    """
    n = arr.shape[0]
    m = pat.shape[0]
    if m == 0 or n < m:
        return np.empty((0,), dtype=np.int64)

    if m == 1:
        return np.flatnonzero(arr == pat[0]).astype(np.int64)

    # sliding window view（优先用 numpy 自带；不行就用 stride trick）
    try:
        win = np.lib.stride_tricks.sliding_window_view(arr, m)
    except Exception:
        shape = (n - m + 1, m)
        strides = (arr.strides[0], arr.strides[0])
        win = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    eq = (win == pat)
    starts = np.flatnonzero(eq.all(axis=1)).astype(np.int64)
    return starts

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

def _norm_spaces_caption(s: str) -> str:
    if not isinstance(s, str):
        return ''
    # 行尾统一
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    # 把所有连续空白压成一个空格（行为与原来 re.sub(r'\s+', ' ', s) 一致）
    s = _SPACE_RE.sub(' ', s)
    return s.strip()


def _normalize_text_field_caption(val):
    if isinstance(val, list):
        try:
            val = ''.join(map(str, val))
        except Exception:
            val = ' '.join(map(str, val))
    return _norm_spaces_caption(val) if isinstance(val, str) else ''


def augment_text_with_pinyin_s1s2_safe(text: str, hparams):
    """
    只对 <S1>/<S2> 标签内部做 pinyin 混入，保留标签结构。
    若文本不含 S1/S2 标签，则等价于原 augment。
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


def _build_caption_from_subjects_narration(meta, return_global_local: bool = False):
    """从 new_caption / subjects / narration 构建 caption:
    给subjects添加"<SPK>"，并且把 <I></I> 标签替换为 <Audio></Audio>。
    当 return_global_local=True 时，同时返回归一化后的 subjects、narration，
    方便外面挂到 global/local 字段上。
    """
    new_cap = meta.get('new_caption')
    subjects = ''
    narration = ''
    if isinstance(new_cap, dict):
        subjects = _normalize_text_field_caption(new_cap.get('subjects', ''))
        narration = _normalize_text_field_caption(new_cap.get('narration', ''))

    if not subjects:
        subjects = _normalize_text_field_caption(meta.get('subjects', ''))
    if not narration:
        narration = _normalize_text_field_caption(meta.get('narration', ''))

    parts = []
    if subjects:
        subjects = f"<SPK>{subjects}</SPK>"
        parts.append(f"{subjects}")
    if narration:
        parts.append(f"{narration}")
    caption = ' '.join(parts)

    if not caption:
        # 兼容旧用法
        if return_global_local:
            return '', subjects, narration
        return ''

    # 把 <I>...</I> 标签替换为 <Audio>...</Audio>（和原逻辑完全一致）
    caption = _I_OPEN_TAG_RE.sub('<Audio>', caption)
    caption = _I_CLOSE_TAG_RE.sub('</Audio>', caption)

    if return_global_local:
        # subjects / narration 已做过空白归一化，原样返回
        return caption, subjects, narration
    return caption

def valid_item_kv(item, k):
    return k in item and item[k] is not None

def merge_A2B(A2B, B_lens):
    token_lens_cumsum = np.cumsum([0] + B_lens[:-1])
    token_lens_cumsum = torch.LongTensor(token_lens_cumsum)
    for i in range(len(B_lens)):
        A2B[i] = A2B[i] + token_lens_cumsum[i]
    A2B = torch.cat(A2B, 0)
    return A2B

def raw_text_process(txt, wav=None, wav_len=None, check_len=True):
    txt = txt.strip()
    if txt.startswith('sil '):
        txt = txt[4:]
    txt = txt.replace(' sil ', ' ')
    txt = txt.replace(' ,', ',').replace(',,', ',').replace(' ，', '，').replace('， ', '，')
    txt = txt.replace(' .', '.').replace(' 。', '。').replace('。 ', '。').replace('。 ', '。')
    txt = txt.replace(' ?', '?').replace(' ？', '？').replace('？ ', '？').replace('？ ', '？')
    txt = txt.replace(' !', '!').replace(' ！', '！').replace('！ ', '！').replace('！ ', '！')
    txt = txt.replace(' ;', ',').replace(' ；', '，').replace('； ', '，').replace('； ', '，').replace(';', ',').replace('；', '，')
    txt = txt.replace(' :', ',').replace(' ：', '，').replace('： ', '，').replace(':', ',').replace('：', '，')
    txt = txt.replace(' 、', '，').replace('、 ', '，').replace('、', '，')
    txt = txt.replace('"', '').replace('“', '').replace('”', '')
    txt = txt.replace('- ', ' ')
    txt = txt.replace('+', ' ')
    txt = txt.replace('，。', '。').replace('。，', '。')
    txt = txt.replace(':。', '。').replace('：。', '。')
    txt = txt.replace('……', '，')
    txt = remove_spaces_between_chinese(txt)
    if txt and txt[-1] not in '.,?!;。，？！；、':
        if is_chinese(txt):
            txt = txt + '。'
        else:
            txt = txt + '. '

    if wav is not None:
        wav_len = wav.shape[0]

    # 仅当 check_len=True 才做长度过滤
    if check_len and wav_len is not None and len(get_word_list(txt)) > wav_len // hparams['hop_size'] // 4:
        _print_skip(
            reason="text_exceeds_latent_capacity",
            extra=f"words={len(get_word_list(txt))}, latent={wav_len // hparams['hop_size'] // 4}"
        )
        return None

    return txt

def raw_text_process_s1s2_tagged(txt, wav=None, wav_len=None):
    """
    对含 <S1>/<S2> 的 text 做 raw_text_process 等价清洗，
    但只处理标签内部文本，保留标签结构与顺序。
    """
    if not isinstance(txt, str) or not txt.strip():
        return ""

    # 没标签就走原逻辑
    if _S1S2_TEXT_RE.search(txt) is None:
        return raw_text_process(txt, wav=wav, wav_len=wav_len)

    if wav is not None:
        wav_len = wav.shape[0]

    # --- 1) 可选：做一次“总字数”长度检查（不重复对每段检查） ---
    if wav_len is not None:
        # 把所有 S1/S2 内部文本拼成 plain 用于容量判断
        plain_parts = []
        for m in _S1S2_TEXT_RE.finditer(txt):
            inner = _norm_spaces_caption(m.group(2))
            if inner:
                plain_parts.append(inner)
        plain = " ".join(plain_parts)

        if plain:
            # 用同一套清洗规则做一次“近似等价”的长度评估
            plain_norm = raw_text_process(plain, wav=None, wav_len=None, check_len=False) or ""
            if len(get_word_list(plain_norm)) > wav_len // hparams['hop_size'] // 4:
                _print_skip(
                    reason="tagged_text_exceeds_latent_capacity",
                    extra=f"words={len(get_word_list(plain_norm))}, latent={wav_len // hparams['hop_size'] // 4}"
                )
                return None

    # --- 2) 分段清洗并重建 ---
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


class PromptAudioShmDataset(BaseTTSShmDataset):

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

        if hparams.get('add_vad_mask', False):
            from utils.audio.vad import get_vad_model
            vad_model = get_from_global_stores(
                'vad_model', global_stores,
                lambda: get_vad_model()
            )

        cosyvoice2_text_tokenizer = None
        if hparams.get('use_cosyvoice2_text_tokenizer', False):
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
                # if item_tgt.get('ctx_wav') is None:
                #     max_sec = float(hparams.get('random_fade_pad_max_sec', 1.0))
                #     if max_sec > 0 and random.random() < hparams.get('random_fade_pad_prob', 0.8):
                #         item_tgt['wav'], pre_pad, post_pad = add_random_fade_in_out_padding(
                #             item_tgt['wav'], sr=sr, fm_wav=fm_wav, max_sec=max_sec
                #         )

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

            # text：caption 数据集可以直接用 processer_fn 写好的 txt，其它数据仍走 raw_text_process
            if item_tgt.get('use_raw_txt_as_text', False):
                # 如果 raw txt 自带 S1/S2，就用 tag-safe 版本
                txt = raw_text_process_s1s2_tagged(item_tgt['txt'], wav_len=item_tgt['wav_len'])
            else:
                txt = raw_text_process(item_tgt['txt'], wav_len=item_tgt['wav_len'])

            if txt is None:
                txt = ''
            item_tgt['text'] = txt

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

            if hparams.get('add_vad_mask', False):
                try:
                    vad_start, vad_end = run_vad_trim(item_tgt['wav'], hparams['audio_sample_rate'], vad_model)
                    vm = hparams['hop_size'] * hparams['vae_stride']
                    wav_len = item_tgt['wav'].shape[0]
                    vad_mask = np.zeros((wav_len // vm))
                    vad_mask[int(vad_start * hparams['audio_sample_rate'] // vm): int(
                        vad_end * hparams['audio_sample_rate'] // vm)] = 1
                    item_tgt['vad_mask'] = vad_mask  # 直接是 lat 的 shape
                except Exception as e:
                    item_tgt['vad_mask'] = None
                    _print_skip(
                        "vad_mask_failed",
                        i_worker, n_worker,
                        item_name=item_tgt.get('item_name', ''),
                        extra=f"err={str(e)}"
                    )
            else:
                item_tgt['vad_mask'] = None

            item_tgt['len'] = mel_len // int(hparams.get('vae_stride', 4))
            yield item_tgt
            skip_logger.step(1)

def processer_fn_promptaudio(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    """基于 subjects / narration 构造 caption 和 text，且不依赖 mel2ph / dur。
    最终每个 item 至少包含: wav/wav_len, phone, tone, txt, caption, item_name, spk_name。
    如果缺失 phone / tone，则分别回退为 SIL_TOKEN_ID 和 0。

    这里额外做两件事：
      1）先把 wav 对齐到 frames_multiple * hop_size 的整数倍，再记录 wav_len；
      2）用裁剪后的 wav_len 计算 mel_len 和 latent_len = mel_len // int(hparams.get('vae_stride', 4))，
         若 phone_encoded 长度 > latent_len，则直接丢掉该 sample。
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

            # 优先用 vocal（如果有的话）
            if 'vocal' in item_ and item_['vocal'] is not None:
                wav = torch.as_tensor(item_['vocal'], dtype=torch.float32)
                org_sr = int(item_.get('vocal_sr', sr))
            elif 'wav' in item_ and item_['wav'] is not None:
                wav = torch.as_tensor(item_['wav'], dtype=torch.float32)
                org_sr = int(item_.get('sr', sr))
            else:
                _print_skip("missing_wav_or_vocal", i_worker, n_worker, item_name=item_.get('item_name'))
                continue

            # 多通道 -> 单通道（假设 [C, T] 或 [T]，与原 np.mean(axis=0) 语义一致）
            if wav.dim() > 1:
                wav = wav.mean(dim=0)

            # 重采样到训练采样率（使用 torchaudio）
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

            # ========= 2. phone / tone + ph vs latent 过滤 =========
            # phone_encoded = item_.get('phone_encoded')

            # # 用裁剪后的 wav_len 计算 mel_len / latent_len，先过滤 ph 太长的样本
            # if hparams.get('load_wav', True) and phone_encoded is not None and len(phone_encoded) > 0:
            #     mel_len = item['wav_len'] // hop_size
            #     latent_len = mel_len // int(hparams.get('vae_stride', 4))
            #     if len(phone_encoded) > latent_len:
            #         _print_skip("phone_longer_than_latent", i_worker, n_worker,
            #                     item_name=item_.get('item_name'),
            #                     extra=f"ph={len(phone_encoded)}, latent={latent_len}")
            #         continue

            # # 缺失 phone -> 用 PAD
            # if phone_encoded is None or len(phone_encoded) == 0:
            #     phone = torch.LongTensor([PH_PAD_ID])
            #     _print_skip("fallback_pad_phone", i_worker, n_worker,
            #                 item_name=item_.get('item_name'),
            #                 extra=f"pad_id={PH_PAD_ID}")
            # else:
            #     phone = torch.LongTensor(phone_encoded)

            # tone_encoded = item_.get('tone_encoded')
            # # 缺失 tone -> 用 PAD，长度与 phone 对齐
            # if tone_encoded is None or len(tone_encoded) == 0:
            #     tone = torch.full((phone.shape[0],), TONE_PAD_ID, dtype=torch.long)
            #     _print_skip("fallback_pad_tone", i_worker, n_worker,
            #                 item_name=item_.get('item_name'),
            #                 extra=f"len={phone.shape[0]}, pad_id={TONE_PAD_ID}")
            # else:
            #     tone = torch.LongTensor(tone_encoded)
            #     # 长度不一致时做对齐（pad 用 PAD）
            #     if tone.shape[0] != phone.shape[0]:
            #         if tone.shape[0] < phone.shape[0]:
            #             pad_len = phone.shape[0] - tone.shape[0]
            #             tone = torch.cat(
            #                 [tone, torch.full((pad_len,), TONE_PAD_ID, dtype=torch.long)],
            #                 dim=0
            #             )
            #             _print_skip("tone_padded_to_match_phone", i_worker, n_worker,
            #                         item_name=item_.get('item_name'),
            #                         extra=f"pad={pad_len}, pad_id={TONE_PAD_ID}")
            #         else:
            #             _print_skip("tone_truncated_to_match_phone", i_worker, n_worker,
            #                         item_name=item_.get('item_name'),
            #                         extra=f"tone_len={tone.shape[0]} -> {phone.shape[0]}")
            #             tone = tone[:phone.shape[0]]

            # item['phone'] = phone
            # item['tone'] = tone


            caption, subj_txt, narr_txt = _build_caption_from_subjects_narration(
                item_, return_global_local=hparams.get('use_global', False)
            )
            if not caption:
                _print_skip("no_caption_built", i_worker, n_worker, item_name=item_.get('item_name'))
                continue

            # 如果 caption 里既没有 <S1>~<S4>，也没有 <Audio> / <I>，直接丢弃
            has_s_tag = _S1S2_TEXT_RE.search(caption) is not None
            has_audio_tag = ('<Audio>' in caption) or (_I_OPEN_TAG_RE.search(caption) is not None)
            if not has_s_tag and not has_audio_tag:
                _print_skip(
                    "caption_without_S_or_Audio",
                    i_worker, n_worker,
                    item_name=item_.get('item_name'),
                    extra=f"caption={caption[:80]}..."
                )
                continue

            item['caption'] = caption
            item['txt'] = _build_text_from_caption_s1s2(caption)  # <S1-4>... 拼起来，没有就 ""

            # 新增：subjects / narration 分别挂到 global / local
            item['global'] = subj_txt   # subjects
            item['local'] = narr_txt    # narration


            # ========= 4. 其他元信息 =========
            item['item_name'] = item_.get('item_name', '')
            item['spk_name'] = item_.get('spk_name', item['item_name'])
            # 告诉 _process_item 不再跑 raw_text_process，直接用 txt
            item['use_raw_txt_as_text'] = True

            items.append(item)
        except Exception as e:
            traceback.print_exc()
            _print_skip("processer_fn_promptaudio_exception", i_worker, n_worker, item_name=item_.get('item_name'), extra=f"err={str(e)}")
            continue

    return items

def processer_fn_promptjson(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    """
    读取 jsonl 样本，按 subjects/narration 组合 caption，
    从 caption 中抽取 <S1>...</S1> 作为 text（若没有则用 ）。
    若存在 start/end 字段，则只从 wav 中读取对应时间段，而不是整段读完再切。

      1）先把 wav 对齐到 frames_multiple * hop_size 的整数倍，再记录 wav_len；
      2）用裁剪后的 wav_len 计算 mel_len 和 latent_len = mel_len // int(hparams.get('vae_stride', 4))，
         若 phone_encoded 长度 > latent_len，则直接丢掉该 sample。
    """
    fm = hparams['frames_multiple']
    hop_size = hparams['hop_size']
    fm_wav = fm * hop_size
    sr = hparams['audio_sample_rate']

    items = []
    for item_ in raw_item:
        try:
            # -------- 1. 读 wav，必要时只按 start/end 读一小段 --------
            if 'flag' in item_ and not item_['flag']:
                _print_skip("flag_not_true", i_worker, n_worker, item_name=item_.get('item_name', item_.get('id')))
                continue

            wav_path = item_.get('wav_path')
            if not wav_path:
                _print_skip("missing_wav_path", i_worker, n_worker, item_name=item_.get('item_name', item_.get('id')))
                continue

            # 有 start/end 的情况：仅解码对应时间段
            if 'start' in item_ and 'end' in item_:
                start = max(float(item_['start']), 0.0)
                end = float(item_['end'])
                if end <= start:
                    _print_skip("invalid_time_range", i_worker, n_worker,
                                item_name=item_.get('item_name', wav_path),
                                extra=f"start={start}, end={end}")
                    continue

                # 先拿到原始采样率和总帧数（不会把整段音频都解码到内存）
                info = torchaudio.info(wav_path)
                org_sr = int(info.sample_rate)

                # 注意：frame_offset / num_frames 都是按“原始采样率”的帧来算
                s_idx = int(start * org_sr)
                e_idx = int(end * org_sr)
                if e_idx <= s_idx:
                    _print_skip("time_slice_empty", i_worker, n_worker,
                                item_name=item_.get('item_name', wav_path),
                                extra=f"s_idx={s_idx}, e_idx={e_idx}")
                    continue
                num_frames = e_idx - s_idx

                # 只解码 [start, end) 这一小段
                wav, org_sr = torchaudio.load(
                    wav_path,
                    frame_offset=s_idx,
                    num_frames=num_frames,
                )
                wav = wav.to(torch.float32)

            else:
                # 无 start/end：整段加载（与原实现一致）
                try:
                    wav, org_sr = torchaudio.load(wav_path)
                    wav = wav.to(torch.float32)
                except Exception as e:
                    _print_skip("torchaudio_load_failed", i_worker, n_worker,
                                item_name=item_.get('item_name', wav_path),
                                extra=f"path={wav_path}, err={str(e)}")
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
                    _print_skip("resample_failed", i_worker, n_worker,
                                item_name=item_.get('item_name', wav_path),
                                extra=f"from={org_sr} to={sr}, err={str(e)}")
                    skip_logger.report(1, 'promptjson')
                    continue

            # 现在 wav: [1, T]
            wav = wav[0]  # -> [T]

            # ★ 对齐到 frames_multiple * hop_size 的整数倍（和 audioset 一致）
            if wav.numel() > 0 and fm_wav > 0:
                wav = wav[: (wav.shape[0] // fm_wav) * fm_wav]
            if wav.numel() == 0:
                _print_skip("empty_wav_after_alignment", i_worker, n_worker,
                            item_name=item_.get('item_name', wav_path))
                continue

            wav = wav.contiguous()

            item = {}
            item['wav'] = wav
            item['wav_len'] = wav.shape[0]

            # -------- 3. caption = subjects + narration --------
            meta_for_caption = dict(item_)
            if 'subjects' not in meta_for_caption and 'subject' in meta_for_caption:
                meta_for_caption['subjects'] = meta_for_caption['subject']

            caption, subj_txt, narr_txt = _build_caption_from_subjects_narration(
                meta_for_caption, return_global_local=hparams.get('use_global', False)
            )
            if not caption:
                _print_skip("no_caption_built", i_worker, n_worker, item_name=item_.get('item_name', wav_path))
                continue

            has_s_tag = _S1S2_TEXT_RE.search(caption) is not None
            has_audio_tag = ('<Audio>' in caption) or (_I_OPEN_TAG_RE.search(caption) is not None)
            if not has_s_tag and not has_audio_tag:
                _print_skip(
                    "caption_without_S_or_Audio",
                    i_worker, n_worker,
                    item_name=item_.get('item_name', wav_path),
                    extra=f"caption={caption[:80]}..."
                )
                continue

            item['caption'] = caption
            item['global'] = subj_txt
            item['local'] = narr_txt

            # -------- 4. text：从 <S1>... 抽取 --------
            txt = _build_text_from_caption_s1s2(caption)
            if not txt:
                _print_skip("empty_text_built_from_caption", i_worker, n_worker,
                            item_name=item_.get('item_name', wav_path))
            item['txt'] = txt

            # -------- 5. 其他元信息 --------
            item['item_name'] = item_.get(
                'item_name',
                item_.get('utt_id', item_.get('id', wav_path))
            )
            item['spk_name'] = item_.get(
                'spk_name',
                item_.get('speaker', item_.get('gender', item['item_name']))
            )
            item['use_raw_txt_as_text'] = True

            items.append(item)
            skip_logger.step(1)
        except Exception as e:
            traceback.print_exc()
            _print_skip("processer_fn_promptjson_exception", i_worker, n_worker,
                        item_name=item_.get('item_name', item_.get('id', '')),
                        extra=f"err={str(e)}")
            skip_logger.report(1, 'promptjson')
            continue

    return items

def processer_fn_robust_mega3(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    fm_wav = hparams['frames_multiple'] * hparams['hop_size']
    
    items = []
    for item_ in raw_item:
        try:
            item = {}

            item['wav'] = torch.FloatTensor(item_['wav'])
            item['wav'] = item['wav'][:len(item['wav']) // fm_wav * fm_wav]
            item['ctx_wav'] = torch.FloatTensor(item_['ref_wav'])
            item['ctx_wav'] = item['ctx_wav'][:len(item['ctx_wav']) // fm_wav * fm_wav]
            item['wav'] = torch.cat(item['ctx_wav'], item['wav'])
            item['wav_len'] = item['wav'].shape[0]
            ctx_mask = torch.zeros((item['wav'].shape[0] // hparams['hop_size'], 1))
            ctx_mask[:item['ctx_wav'].shape[0] // hparams['hop_size']] = 1.0
            item['ctx_mask'] = ctx_mask[::hparams['vae_stride']]
            
            item['item_name'] = item_['item_name']
            ds_name = item_['ds_name']
            item['spk_name'] = f"{ds_name}#{item_['spk']}"
            item['skip_merge_same_spk'] = True
            item['caption'] = f"<S1>{item_['ref_text']}{item_['text']}</S1>"
            item['txt'] = item['caption']
            item['use_raw_txt_as_text'] = True
            items.append(item)
        except:
            continue
    return items