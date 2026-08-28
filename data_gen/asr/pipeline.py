import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langdetect import detect as classify_language, LangDetectException
import torch

from utils.text import isPUNC
from utils.text.split_text import get_word_list

_SP_TOKEN = "<|sp|>"

# 通用：speaker tag 的默认过滤规则（speaker_tagged=True 时启用）
_DEFAULT_SPEAKER_TAG_IGNORE_PATTERNS = [r"</?\s*S\d+\s*>"]

# 用于把 sp_token 当成“整体 token”的占位符候选（优先选这些，极少出现在正常文本里）
_SP_PLACEHOLDER_CANDIDATES = ("＿", "＾", "｀", "＃", "＄", "％", "＆", "＠", "～", "＼")
_EN_CONTRACTION_SUFFIXES = {"s", "t", "m", "d", "ll", "re", "ve"}
_EN_APOSTROPHES = {"'", "’"}


def _looks_like_en_word(token: str) -> bool:
    return bool(re.search(r"[A-Za-z]", token or ""))


def _merge_english_contractions_with_str_after(
    words: List[str],
    punc_after: List[str],
) -> Tuple[List[str], List[str]]:
    merged_words: List[str] = []
    merged_after: List[str] = []
    i = 0
    while i < len(words):
        if (
            i + 2 < len(words)
            and words[i + 1] in _EN_APOSTROPHES
            and _looks_like_en_word(words[i])
            and words[i + 2].lower() in _EN_CONTRACTION_SUFFIXES
        ):
            merged_words.append(f"{words[i]}{words[i + 1]}{words[i + 2]}")
            merged_after.append(punc_after[i + 2])
            i += 3
            continue
        if (
            i + 1 < len(words)
            and punc_after[i] in _EN_APOSTROPHES
            and _looks_like_en_word(words[i])
            and words[i + 1].lower() in _EN_CONTRACTION_SUFFIXES
        ):
            merged_words.append(f"{words[i]}{punc_after[i]}{words[i + 1]}")
            merged_after.append(punc_after[i + 1])
            i += 2
            continue
        merged_words.append(words[i])
        merged_after.append(punc_after[i])
        i += 1
    return merged_words, merged_after


def _merge_english_contractions_with_token_after(
    words: List[str],
    after: List[List[str]],
) -> Tuple[List[str], List[List[str]]]:
    merged_words: List[str] = []
    merged_after: List[List[str]] = []
    i = 0
    while i < len(words):
        if (
            i + 2 < len(words)
            and words[i + 1] in _EN_APOSTROPHES
            and _looks_like_en_word(words[i])
            and words[i + 2].lower() in _EN_CONTRACTION_SUFFIXES
        ):
            merged_words.append(f"{words[i]}{words[i + 1]}{words[i + 2]}")
            merged_after.append(after[i + 2])
            i += 3
            continue
        if (
            i + 1 < len(words)
            and len(after[i]) == 1
            and after[i][0] in _EN_APOSTROPHES
            and _looks_like_en_word(words[i])
            and words[i + 1].lower() in _EN_CONTRACTION_SUFFIXES
        ):
            merged_words.append(f"{words[i]}{after[i][0]}{words[i + 1]}")
            merged_after.append(after[i + 1])
            i += 2
            continue
        merged_words.append(words[i])
        merged_after.append(after[i])
        i += 1
    return merged_words, merged_after


def _build_special_token_ignore_regex(
    *,
    ignore_literals: Optional[Sequence[str]],
    ignore_patterns: Optional[Sequence[str]],
) -> Optional[re.Pattern]:
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
    pieces: List[Dict[str, str]] = []
    last = 0
    for m in ignore_regex.finditer(text):
        if m.start() > last:
            pieces.append({"kind": "text", "content": text[last:m.start()]})
        pieces.append({"kind": "ignore", "content": m.group(0)})
        last = m.end()
    if last < len(text):
        pieces.append({"kind": "text", "content": text[last:]})
    return pieces


def _pick_sp_token_placeholder(text: str) -> str:
    for ch in _SP_PLACEHOLDER_CANDIDATES:
        if ch and (ch not in text):
            return ch
    # 兜底：私有区字符，几乎不可能出现在正常数据里
    fallback = "\uE000"
    if fallback in text:
        raise ValueError("Cannot find a safe placeholder for sp_token in text.")
    return fallback


def _tokenize_for_boundary(text: str, *, sp_token: str = _SP_TOKEN) -> List[str]:
    """
    用于“按 word 边界恢复 special tokens”的 tokenizer：

    - 目标：尽量复用 get_word_list 的行为
    - 但：把 sp_token（默认 <|sp|>）当成一个整体 token，而不是被 get_word_list 拆成 '<' '|' 'sp' ...
    """
    if not text:
        return []

    placeholder = _pick_sp_token_placeholder(text) if (sp_token and sp_token in text) else None
    if placeholder is not None:
        text = text.replace(sp_token, placeholder)

    raw_tokens = get_word_list(text)

    if placeholder is None:
        return raw_tokens

    tokens: List[str] = []
    for tok in raw_tokens:
        if placeholder not in tok:
            tokens.append(tok)
            continue

        parts = tok.split(placeholder)
        for i, part in enumerate(parts):
            if part:
                tokens.append(part)
            if i != len(parts) - 1:
                tokens.append(sp_token)

    return tokens


def _extract_words_and_after_tokens_for_boundary(
    text: str,
    *,
    sp_token: str = _SP_TOKEN,
) -> Tuple[List[str], List[str], List[List[str]]]:
    """
    把文本拆成：leading_tokens + words + after_tokens（token 粒度保留标点和 <|sp|>）。
    """
    tokens = _tokenize_for_boundary(text, sp_token=sp_token)

    leading: List[str] = []
    words: List[str] = []
    after: List[List[str]] = []

    for tok in tokens:
        if tok == sp_token or isPUNC(tok):
            if after:
                after[-1].append(tok)
            else:
                leading.append(tok)
            continue

        words.append(tok)
        after.append([])

    words, after = _merge_english_contractions_with_token_after(words, after)
    return leading, words, after


def _rebuild_text_from_words_and_after(
    words: Sequence[str],
    after: Sequence[Sequence[str]],
    *,
    lang: str,
    sp_token: str = _SP_TOKEN,
    prefix_tokens: Optional[Sequence[str]] = None,
) -> str:
    tokens: List[str] = []
    if prefix_tokens:
        tokens.extend([t for t in prefix_tokens if t])

    for w, aft in zip(words, after):
        if w:
            tokens.append(w)
        for t in aft:
            if t:
                tokens.append(t)

    return _join_tokens(tokens, lang, sp_token=sp_token)


def _restore_text_from_pieces(
    *,
    punct_text: str,
    pieces: List[Dict[str, str]],
    lang: str,
    sp_token: str = _SP_TOKEN,
    restore_ignored: bool = True,
) -> str:
    """
    把 ignore pieces（例如 <S1>, </S1>）按 word 边界插回到 punct_text。
    restore_ignored=False 时表示“过滤掉就不恢复”。
    """
    leading_punct, punct_words, punct_after = _extract_words_and_after_tokens_for_boundary(punct_text, sp_token=sp_token)

    out_parts: List[str] = []
    w_pos = 0
    pending_prefix = list(leading_punct)

    for p in pieces:
        kind = p.get("kind")
        content = p.get("content", "")

        if kind == "ignore":
            if restore_ignored:
                out_parts.append(content)
            continue

        _, piece_words, _ = _extract_words_and_after_tokens_for_boundary(content, sp_token=sp_token)
        cnt = len(piece_words)

        if cnt <= 0:
            out_parts.append(content)
            continue

        out_parts.append(
            _rebuild_text_from_words_and_after(
                punct_words[w_pos : w_pos + cnt],
                punct_after[w_pos : w_pos + cnt],
                lang=lang,
                sp_token=sp_token,
                prefix_tokens=pending_prefix,
            )
        )
        pending_prefix = []
        w_pos += cnt

    if w_pos < len(punct_words) or pending_prefix:
        out_parts.append(
            _rebuild_text_from_words_and_after(
                punct_words[w_pos:],
                punct_after[w_pos:],
                lang=lang,
                sp_token=sp_token,
                prefix_tokens=pending_prefix,
            )
        )

    return "".join(out_parts)

def _norm_lang(lang: str) -> str:
    lang = (lang or "").strip().lower()
    if lang in {"en", "english"}:
        return "en"
    if lang in {"zh", "chinese", "cn"}:
        return "zh"
    if "en" in lang and "zh" not in lang:
        return "en"
    if "zh" in lang:
        return "zh"
    return lang or "en"


def _normalize_token_for_match(token: str, lang: str) -> str:
    token = (token or "").strip()
    if not token:
        return ""

    if lang == "en":
        token = token.lower()
        token = re.sub(r"[^a-z0-9]+", "", token)
        return token

    return token


def _extract_words_and_punc_after(asr_text: str) -> Tuple[List[str], List[str]]:
    tokens = get_word_list(asr_text)

    words: List[str] = []
    punc_after: List[str] = []

    i = 0
    while i < len(tokens):
        t = tokens[i]

        if isPUNC(t):
            if len(punc_after) > 0:
                punc_after[-1] += t
            i += 1
            continue

        words.append(t)
        punc_after.append("")
        i += 1

        while i < len(tokens) and isPUNC(tokens[i]):
            punc_after[-1] += tokens[i]
            i += 1

    return words, punc_after


def _format_alignment_word_group(
    words: Sequence[str],
    punc_after: Sequence[str],
    start: int,
    end: int,
    *,
    lang: str,
) -> str:
    """
    Rebuild the original text-side token group that corresponds to one aligner token.

    The grouping decision is made from normalized text, but the emitted word should
    preserve the user's text as much as possible, e.g. can + ' + t -> can't.
    """
    if start == end:
        return words[start]

    lang = _norm_lang(lang)
    pieces: List[str] = []
    prev_norm = ""

    for idx in range(start, end + 1):
        word = words[idx]
        word_norm = _normalize_token_for_match(word, lang)

        if pieces and lang == "en" and not punc_after[idx - 1]:
            # If two real English word pieces were separated by whitespace in the
            # source text, get_word_list has already discarded that whitespace.
            # Reinsert a space unless an empty-normalized connector such as "'" is
            # between them.
            if prev_norm and word_norm:
                pieces.append(" ")

        pieces.append(word)
        if idx < end and punc_after[idx]:
            pieces.append(punc_after[idx])

        prev_norm = word_norm

    return "".join(pieces)


def _find_alignment_group_end(
    words: Sequence[str],
    start: int,
    target_norm: str,
    *,
    lang: str,
) -> Optional[int]:
    if not target_norm:
        return None

    accum = ""
    for end in range(start, len(words)):
        piece_norm = _normalize_token_for_match(words[end], lang)
        accum += piece_norm

        if accum == target_norm:
            return end

        # Empty-normalized connector tokens, e.g. apostrophes, are allowed to sit
        # inside a group without advancing accum.
        if accum and not target_norm.startswith(accum):
            return None

    return None


def _group_words_by_aligned_tokens(
    words: Sequence[str],
    punc_after: Sequence[str],
    aligned_words: Sequence[str],
    *,
    lang: str,
) -> Optional[Tuple[List[str], List[str]]]:
    """
    Convert text-side tokens to aligner token boundaries.

    This is intentionally driven by the aligner output rather than by English
    contraction heuristics. For example, if the aligner returns one token
    "can't", text tokens can + ' + t are grouped into one emitted token.
    """
    if not words or not aligned_words:
        return None

    grouped_words: List[str] = []
    grouped_punc_after: List[str] = []
    word_pos = 0

    for aligned_word in aligned_words:
        target_norm = _normalize_token_for_match(aligned_word, lang)
        end = _find_alignment_group_end(words, word_pos, target_norm, lang=lang)
        if end is None:
            return None

        grouped_words.append(
            _format_alignment_word_group(
                words,
                punc_after,
                word_pos,
                end,
                lang=lang,
            )
        )
        grouped_punc_after.append(punc_after[end])
        word_pos = end + 1

    if word_pos != len(words):
        return None

    return grouped_words, grouped_punc_after


def _choose_pause_punct(
    pause_s: float,
    lang: str,
    *,
    sp_min_s: float,
    comma_min_s: float,
    period_min_s: float,
    sp_token: str = _SP_TOKEN,
    enable_sp_token: bool = True,
) -> str:
    if pause_s is None:
        return ""

    pause_s = float(pause_s)

    # 如果不插入 sp：那么 sp 阈值及以下的停顿都忽略。
    # 等价于：pause < comma_min_s 时一律不加任何符号。
    if not enable_sp_token:
        if pause_s < comma_min_s:
            return ""

    if pause_s < sp_min_s:
        return ""

    if pause_s < comma_min_s:
        return sp_token

    if pause_s < period_min_s:
        return "，" if lang == "zh" else ","

    return "。" if lang == "zh" else "."


def _get_audio_duration_s(audio) -> Union[float, None]:
    """
    尽力拿到音频时长（秒）。

    支持两类最常见输入：
    1) 本地 wav/flac 等路径（str 且 os.path.isfile 为 True）
    2) (audio_array, sr) 形式的二元组/列表（len==2）

    如果拿不到就返回 None（此时句尾会走保底策略）。
    """
    if audio is None:
        return None

    if isinstance(audio, str):
        if not os.path.isfile(audio):
            return None
        try:
            import soundfile as sf
            return float(sf.info(audio).duration)
        except Exception:
            return None

    if isinstance(audio, (tuple, list)) and len(audio) == 2:
        arr, sr = audio
        try:
            sr_f = float(sr)
            if sr_f <= 0:
                return None
        except Exception:
            return None

        try:
            if hasattr(arr, "shape"):
                shape = getattr(arr, "shape")
                if shape is None:
                    n = len(arr)
                else:
                    if len(shape) == 0:
                        return None
                    if len(shape) == 1:
                        n = int(shape[0])
                    else:
                        n = int(max(shape[0], shape[1]))
            else:
                n = len(arr)
            return float(n) / sr_f
        except Exception:
            return None

    return None


def _is_batched_wavs_and_lens_audio(audio) -> bool:
    """
    判断 audio 是否为 (wavs, wav_lens) 的 batch 输入：
    - wavs: torch.Tensor
    - wav_lens: torch.Tensor / list / tuple / np.ndarray
    """
    if not (isinstance(audio, (tuple, list)) and len(audio) == 2):
        return False
    wavs, wav_lens = audio
    if not torch.is_tensor(wavs):
        return False
    if isinstance(wav_lens, (int, float, np.integer, np.floating)):
        return False
    return torch.is_tensor(wav_lens) or isinstance(wav_lens, (list, tuple, np.ndarray))


def _batched_audio_duration_list_s(audio, *, sr: int = 16000) -> Optional[List[float]]:
    """
    给 (wavs, wav_lens) 形式的 batch 音频预计算每条样本时长（秒）。
    注意：如果 wav_lens 在 CUDA 上，这里会发生一次 D2H（同步/拷贝）把 lens 拿回 CPU。
    """
    if not _is_batched_wavs_and_lens_audio(audio):
        return None

    _, wav_lens = audio
    try:
        sr_f = float(sr)
        if sr_f <= 0:
            return None

        if torch.is_tensor(wav_lens):
            wav_lens_cpu = wav_lens.detach().to("cpu").reshape(-1).long()
        elif isinstance(wav_lens, np.ndarray):
            wav_lens_cpu = torch.from_numpy(wav_lens).reshape(-1).long()
        else:
            wav_lens_cpu = torch.tensor(list(wav_lens), dtype=torch.long).reshape(-1)

        return (wav_lens_cpu.float() / sr_f).tolist()
    except Exception:
        return None
    

def _join_tokens(tokens: Sequence[str], lang: str, *, sp_token: str = _SP_TOKEN) -> str:
    lang = _norm_lang(lang)

    if lang == "zh":
        out: List[str] = []
        for tok in tokens:
            if tok:
                out.append(tok)
        return "".join(out)

    out: List[str] = []
    for i, tok in enumerate(tokens):
        if not tok:
            continue

        nxt = tokens[i + 1] if i + 1 < len(tokens) else None

        if tok == sp_token:
            if len(out) > 0 and not out[-1].endswith(" "):
                out.append(" ")
            out.append(tok)
            if nxt and (not isPUNC(nxt)):
                out.append(" ")
            continue

        if isPUNC(tok):
            if len(out) > 0 and out[-1].endswith(" "):
                out[-1] = out[-1].rstrip(" ")
            out.append(tok)
            if nxt and (nxt != sp_token) and (not isPUNC(nxt)):
                out.append(" ")
            continue

        if len(out) > 0 and (not out[-1].endswith(" ")):
            prev = out[-1]
            if prev and (not isPUNC(prev)) and prev != sp_token:
                out.append(" ")
        out.append(tok)

    return "".join(out).strip()


def _prefer_asr_punct_when_pause(
    existing: str,
    lang: str,
    insert_p: str,
    *,
    sp_token: str = _SP_TOKEN,
) -> str:
    """
    当 alignment/pause 已经判断“这里要插标点”(insert_p 非空) 后：

    - insert_p == <|sp|>：强制返回 <|sp|>（短停顿优先）
    - 否则如果 ASR 原本就有标点 existing：优先返回 existing（做一点中英文符号归一化）
    - 否则：回退用 insert_p（也就是按 pause bucket 推出来的逗号/句号）
    """
    lang = _norm_lang(lang)
    existing = (existing or "").strip()

    if not insert_p:
        return ""

    if insert_p == sp_token:
        return sp_token

    if not existing:
        return insert_p

    if lang == "zh":
        existing = existing.replace(",", "，").replace(".", "。").replace("?", "？").replace("!", "！")
        if "..." in existing and "…" not in existing:
            existing = existing.replace("...", "…")
        return existing

    existing = existing.replace("，", ",").replace("。", ".").replace("？", "?").replace("！", "!")
    if "…" in existing and "..." not in existing:
        existing = existing.replace("…", "...")
    if "、" in existing:
        existing = existing.replace("、", ",")
    return existing


def punctuate_by_alignment_pause(
    *,
    asr_text: str,
    align_items: Sequence,
    lang: str,
    sp_token: str = _SP_TOKEN,
    enable_sp_token: bool = True,
    sp_min_s: float = 0.08,
    comma_min_s: float = 0.18,
    period_min_s: float = 0.45,
    keep_asr_punc_without_pause: bool = False,
    keep_final_asr_punc: bool = True,
    audio_duration_s: Union[float, None] = None,
) -> str:
    lang = _norm_lang(lang)
    if not asr_text:
        return asr_text

    raw_words, raw_punc_after = _extract_words_and_punc_after(asr_text)
    aligned_words = [getattr(it, "text", "") for it in align_items]
    grouped = _group_words_by_aligned_tokens(
        raw_words,
        raw_punc_after,
        aligned_words,
        lang=lang,
    )
    if grouped is not None:
        words, punc_after = grouped
    else:
        # Fallback keeps the previous forgiving behavior for imperfect aligner
        # output, but still avoids silently dropping unmatched text.
        words, punc_after = _merge_english_contractions_with_str_after(raw_words, raw_punc_after)

    if len(words) == len(aligned_words):
        idx_map = list(range(len(words)))
    else:
        idx_map = []
        w_i = 0
        for aw in aligned_words:
            aw_n = _normalize_token_for_match(aw, lang)
            while w_i < len(words) and _normalize_token_for_match(words[w_i], lang) != aw_n:
                w_i += 1
            if w_i >= len(words):
                break
            idx_map.append(w_i)
            w_i += 1

        if len(idx_map) != len(aligned_words):
            n = min(len(words), len(aligned_words))
            idx_map = list(range(n))
            align_items = list(align_items)[:n]

    out_tokens: List[str] = []
    last_emitted_word_idx = -1
    for i, it in enumerate(align_items):
        w_idx = idx_map[i]
        if w_idx > last_emitted_word_idx + 1:
            for missing_idx in range(last_emitted_word_idx + 1, w_idx):
                out_tokens.append(words[missing_idx])
                if punc_after[missing_idx]:
                    out_tokens.append(punc_after[missing_idx])

        out_tokens.append(words[w_idx])
        last_emitted_word_idx = w_idx

        existing = punc_after[w_idx]

        # ====== 句尾逻辑：用尾静音时长决定 ======
        if i + 1 >= len(align_items):
            if audio_duration_s is not None:
                last_end = float(getattr(it, "end_time", 0.0))
                tail_silence_s = float(audio_duration_s) - last_end
                if tail_silence_s < 0:
                    tail_silence_s = 0.0

                # 把“尾静音”当成最后一个 pause
                insert_p = _choose_pause_punct(
                    tail_silence_s,
                    lang,
                    sp_min_s=sp_min_s,
                    comma_min_s=comma_min_s,
                    period_min_s=period_min_s,
                    sp_token=sp_token,
                    enable_sp_token=enable_sp_token,
                )

                # 是否有句尾标点由 insert_p 是否为空决定；
                # 具体符号尽量沿用 ASR 的 existing（除非 insert_p 是 <|sp|>）
                if insert_p:
                    out_tokens.append(
                        _prefer_asr_punct_when_pause(
                            existing,
                            lang,
                            insert_p,
                            sp_token=sp_token,
                        )
                    )
            else:
                # 拿不到音频时长：保底（不影响你“完全按尾静音”的主路径）
                if keep_final_asr_punc and existing:
                    out_tokens.append(existing)
            continue
        # ====== 句尾逻辑结束 ======

        pause_s = float(getattr(align_items[i + 1], "start_time", 0.0)) - float(getattr(it, "end_time", 0.0))
        insert_p = _choose_pause_punct(
            pause_s,
            lang,
            sp_min_s=sp_min_s,
            comma_min_s=comma_min_s,
            period_min_s=period_min_s,
            sp_token=sp_token,
            enable_sp_token=enable_sp_token,
        )

        if keep_asr_punc_without_pause:
            if existing:
                out_tokens.append(existing)
                continue
            if insert_p:
                out_tokens.append(insert_p)
            continue

        if insert_p:
            out_tokens.append(
                _prefer_asr_punct_when_pause(
                    existing,
                    lang,
                    insert_p,
                    sp_token=sp_token,
                )
            )

    for missing_idx in range(last_emitted_word_idx + 1, len(words)):
        out_tokens.append(words[missing_idx])
        if punc_after[missing_idx]:
            out_tokens.append(punc_after[missing_idx])

    return _join_tokens(out_tokens, lang, sp_token=sp_token)


class ASRPipeline:
    def __init__(
        self,
        asr_backend: str = 'sensevoice',
        punc_backend: str = 'qwen3aligner',
        device='cuda:0',
        aligner_ckpt="checkpoints/260206_aligner_2tower",
        precision: torch.dtype = torch.bfloat16,
        attn_implementation='flash_attention_2',
        *,
        sp_token: str = _SP_TOKEN,
        enable_sp_token: bool = True,
        sp_min_s=0.08,
        comma_min_s=0.18,
        period_min_s=0.45,
        special_token_ignore_literals: Optional[List[str]] = None,
        special_token_ignore_patterns: Optional[List[str]] = None,
        restore_special_tokens: bool = True,
    ):
        self.asr_backend = asr_backend
        self.punc_backend = punc_backend
        self.sp_token = sp_token
        self.enable_sp_token = bool(enable_sp_token)

        self.sp_min_s = sp_min_s
        self.comma_min_s = comma_min_s
        self.period_min_s = period_min_s

        self.special_token_ignore_literals = list(special_token_ignore_literals or [])
        self.special_token_ignore_patterns = list(special_token_ignore_patterns or [])
        self.restore_special_tokens = bool(restore_special_tokens)

        if asr_backend == 'sensevoice':
            from data_gen.qwen.sensevoice import build_asr_model
            self.asr_model = build_asr_model(
                device=device,
                # use_punc=punc_backend != 'qwen3aligner'
            )

        if punc_backend == 'qwen3aligner':
            from data_gen.qwen.qwen3aligner import Qwen3AlignerInfer
            self.aligner = Qwen3AlignerInfer(device=device, precision=precision, attn_implementation=attn_implementation)
        elif punc_backend in ['aligner_mdm', 'aligner_2tower', 'aligner_2tower_v4', 'aligner_2tower_v5', 'aligner_2tower_v6']:
            from inference.asr.aligner_infer import ForcedAlignerInfer
            self.aligner = ForcedAlignerInfer(aligner_ckpt, device=device, precision=precision, attn_implementation=attn_implementation)

    def process(
        self,
        audio,
        text=None,
        langs=None,
        *,
        special_token_ignore_literals: Optional[List[str]] = None,
        special_token_ignore_patterns: Optional[List[str]] = None,
        restore_special_tokens: Optional[bool] = None,
    ):
        """
        audio: 
            1. if text is None
                a. wav_path or list of wav_paths
                b. np array or list of np arrays
            2. if text is not None
                a. wav_path or list of wav_paths
                b. np array or list of np arrays
                c. if punc_backend == aligner_2tower_vxxx, then torch tensors with devices and sample rate == 16000: (wavs, wav_lens)
        text: text or list of texts
        """
        if self.asr_backend == 'sensevoice':

            is_batched_audio = _is_batched_wavs_and_lens_audio(audio)

            if not isinstance(audio, list) and not is_batched_audio:
                audio = [audio]

            if text is None:

                from data_gen.qwen.sensevoice import run_asr_model

                asr_results = run_asr_model(audio, self.asr_model)

                texts = [s['text_normed'] for s in asr_results]
                langs = []
                for asr_result in asr_results:
                    langs_ = []
                    for s in asr_result['segments']:
                        langs_.append(s['lang'])
                    lang = Counter(langs_).most_common(1)[0][0]
                    langs.append(lang)

                out = {
                    "asr_results": asr_results,
                    "texts": texts,
                    "langs": langs,
                }

            else:

                if isinstance(text, list):
                    texts = text
                elif isinstance(text, str):
                    texts = [text]

                ignore_literals_eff = (
                    special_token_ignore_literals
                    if special_token_ignore_literals is not None
                    else self.special_token_ignore_literals
                )
                ignore_patterns_eff = (
                    special_token_ignore_patterns
                    if special_token_ignore_patterns is not None
                    else self.special_token_ignore_patterns
                )
                restore_eff = self.restore_special_tokens if restore_special_tokens is None else bool(restore_special_tokens)

                ignore_regex = _build_special_token_ignore_regex(
                    ignore_literals=ignore_literals_eff,
                    ignore_patterns=ignore_patterns_eff,
                )

                # 为每条样本准备：原始文本、stripped 文本、以及 pieces（用于恢复）
                pieces_list: List[Optional[List[Dict[str, str]]]] = []
                texts_stripped: List[str] = []
                for t in texts:
                    if ignore_regex is None:
                        pieces_list.append(None)
                        texts_stripped.append(t)
                    else:
                        pieces = _split_by_ignore_regex(t, ignore_regex)
                        pieces_list.append(pieces)
                        texts_stripped.append("".join(p["content"] for p in pieces if p["kind"] == "text"))

                if langs is None:
                    # 语言检测用 stripped，更稳（避免 <S1> 影响 langdetect）
                    langs = []
                    for t in texts_stripped:
                        try:
                            lang = classify_language(t)
                        except LangDetectException:
                            lang = 'zh'
                        langs.append(lang)
                elif isinstance(langs, str):
                    langs = [langs] * len(texts)
                else:
                    assert isinstance(langs, list) and len(langs) == len(texts), "langs must have the same length as texts"

                out = {
                    "asr_results": None,
                    "texts": texts,
                    "langs": langs,
                }
                if ignore_regex is not None:
                    out["texts_stripped"] = texts_stripped

            if self.punc_backend in ['qwen3aligner', 'aligner_mdm', 'aligner_2tower', 'aligner_2tower_v4', 'aligner_2tower_v5', 'aligner_2tower_v6']:

                if self.punc_backend == 'qwen3aligner':
                    langs_for_aligner = []
                    for lang in langs:
                        lang_norm = _norm_lang(lang)
                        if lang_norm == 'ko':
                            langs_for_aligner.append('korean')
                        elif lang_norm == 'ja':
                            langs_for_aligner.append('japanese')
                        elif lang_norm == 'en':
                            langs_for_aligner.append('english')
                        else:
                            langs_for_aligner.append('chinese')

                    # 对齐用 stripped 文本（如果没过滤则 texts_stripped==texts）
                    texts_for_align = texts if text is None else (out.get("texts_stripped") or out["texts"])
                    align_results = self.aligner.process(audio, texts_for_align, langs_for_aligner)

                elif self.punc_backend in ['aligner_mdm', 'aligner_2tower', 'aligner_2tower_v4', 'aligner_2tower_v5', 'aligner_2tower_v6']:

                    texts_for_align = texts if text is None else (out.get("texts_stripped") or out["texts"])
                    align_results = self.aligner.align(audio, texts_for_align)

                pause_punct_texts_stripped = []
                pause_punct_texts = []

                sr_for_duration = 16000
                try:
                    if hasattr(self.aligner, "hp") and isinstance(self.aligner.hp, dict):
                        sr_for_duration = int(self.aligner.hp.get("audio_sample_rate", 16000))
                except Exception:
                    sr_for_duration = 16000

                audio_duration_s_list = _batched_audio_duration_list_s(audio, sr=sr_for_duration)

                sp_min_s = self.sp_min_s
                comma_min_s = self.comma_min_s
                period_min_s = self.period_min_s

                if text is None:

                    if audio_duration_s_list is None:
                        for a, t, lang, align_result in zip(audio, texts, langs, align_results):
                            pause_punct_texts.append(
                                punctuate_by_alignment_pause(
                                    asr_text=t,
                                    align_items=list(align_result),
                                    lang=lang,
                                    sp_token=self.sp_token,
                                    enable_sp_token=self.enable_sp_token,
                                    sp_min_s=sp_min_s,
                                    comma_min_s=comma_min_s,
                                    period_min_s=period_min_s,
                                    keep_asr_punc_without_pause=False,
                                    keep_final_asr_punc=True,
                                    audio_duration_s=_get_audio_duration_s(a),
                                )
                            )
                    else:
                        for i, (t, lang, align_result) in enumerate(zip(texts, langs, align_results)):
                            dur_i = audio_duration_s_list[i] if i < len(audio_duration_s_list) else None
                            pause_punct_texts.append(
                                punctuate_by_alignment_pause(
                                    asr_text=t,
                                    align_items=list(align_result),
                                    lang=lang,
                                    sp_token=self.sp_token,
                                    enable_sp_token=self.enable_sp_token,
                                    sp_min_s=sp_min_s,
                                    comma_min_s=comma_min_s,
                                    period_min_s=period_min_s,
                                    keep_asr_punc_without_pause=False,
                                    keep_final_asr_punc=True,
                                    audio_duration_s=dur_i,
                                )
                            )

                    out["pause_punct_texts"] = pause_punct_texts

                else:
                    if audio_duration_s_list is None:
                        for a, t_orig, t_stripped, lang, align_result, pieces in zip(
                            audio, out["texts"], texts_for_align, langs, align_results, pieces_list
                        ):
                            pp_stripped = punctuate_by_alignment_pause(
                                asr_text=t_stripped,
                                align_items=list(align_result),
                                lang=lang,
                                sp_token=self.sp_token,
                                enable_sp_token=self.enable_sp_token,
                                sp_min_s=sp_min_s,
                                comma_min_s=comma_min_s,
                                period_min_s=period_min_s,
                                keep_asr_punc_without_pause=False,
                                keep_final_asr_punc=True,
                                audio_duration_s=_get_audio_duration_s(a),
                            )
                            pause_punct_texts_stripped.append(pp_stripped)

                            if pieces is not None:
                                pause_punct_texts.append(
                                    _restore_text_from_pieces(
                                        punct_text=pp_stripped,
                                        pieces=pieces,
                                        lang=lang,
                                        sp_token=self.sp_token,
                                        restore_ignored=restore_eff,
                                    )
                                )
                            else:
                                pause_punct_texts.append(pp_stripped)
                    else:
                        for i, (t_orig, t_stripped, lang, align_result, pieces) in enumerate(
                            zip(out["texts"], texts_for_align, langs, align_results, pieces_list)
                        ):
                            dur_i = audio_duration_s_list[i] if i < len(audio_duration_s_list) else None
                            pp_stripped = punctuate_by_alignment_pause(
                                asr_text=t_stripped,
                                align_items=list(align_result),
                                lang=lang,
                                sp_token=self.sp_token,
                                enable_sp_token=self.enable_sp_token,
                                sp_min_s=sp_min_s,
                                comma_min_s=comma_min_s,
                                period_min_s=period_min_s,
                                keep_asr_punc_without_pause=False,
                                keep_final_asr_punc=True,
                                audio_duration_s=dur_i,
                            )
                            pause_punct_texts_stripped.append(pp_stripped)

                            if pieces is not None:
                                pause_punct_texts.append(
                                    _restore_text_from_pieces(
                                        punct_text=pp_stripped,
                                        pieces=pieces,
                                        lang=lang,
                                        sp_token=self.sp_token,
                                        restore_ignored=restore_eff,
                                    )
                                )
                            else:
                                pause_punct_texts.append(pp_stripped)

                    out["pause_punct_texts"] = pause_punct_texts
                    if ignore_regex is not None:
                        out["pause_punct_texts_stripped"] = pause_punct_texts_stripped

                out["align_results"] = [
                    [(item.start_time, item.end_time, item.text, item.conf) for item in align_result]
                    for align_result in align_results
                ]

            return out

        raise ValueError(f"Unsupported asr_backend={self.asr_backend}")


if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')

    # aligner = ASRPipeline(
    #     device='cuda',
    #     # punc_backend='aligner_2tower_v4',
    #     # aligner_ckpt='checkpoints/260225_aligner_2tower_v4'
    #     # punc_backend='aligner_2tower_v5',
    #     # aligner_ckpt='checkpoints/260225_aligner_2tower_v5',
    #     # punc_backend='aligner_2tower_v6',
    #     # aligner_ckpt='checkpoints/260304_aligner_2tower_v6',
    #     punc_backend='aligner_mdm',
    #     aligner_ckpt='checkpoints/260311_aligner_mdm',
    #     # attn_implementation='sdpa',
    #     # precision=torch.float32,
    #     sp_min_s=0.08,
    #     comma_min_s=0.18,
    #     period_min_s=0.45,
    # )
    # results = aligner.process(
    #     # audio="user/temp/test_dl/0.wav",
    #     # audio="/mnt/bn/sa-ag-data/liruiqi/data/speech/sa_data_v1/SA_AmericanMaleDB1/AmericanMaleDB1_ENNews_100005.wav"
    #     audio="infer_out/tts_dialogue/infer_once/260108_prompt_base_dialogue/step84000/0-我家这餐馆，菜味儿正宗分量也足，怎么饭点.wav"
    # )

    # print(f"{results = }")
    # print("pause_punct_texts:", results.get("pause_punct_texts"))


    # # profile
    # from utils.profile.cuda_memory import take_gpu_memory_snapshot, diff_gpu_memory_snapshots
    # from tqdm import tqdm
    # import time

    # snap_init = take_gpu_memory_snapshot("start", devices=[0])
    # snap_init.log()

    # aligner = ASRPipeline(
    #     device='cuda',
    #     # punc_backend='aligner_2tower_v4',
    #     # aligner_ckpt='checkpoints/260225_aligner_2tower_v4'
    #     punc_backend='aligner_2tower_v5',
    #     aligner_ckpt='checkpoints/260225_aligner_2tower_v5'
    # )

    # T = 1000
    # time_start = time.time()

    # snap_start = take_gpu_memory_snapshot("start", devices=[0])
    # snap_start.log()
    # diff = diff_gpu_memory_snapshots(snap_init, snap_start)
    # diff.log()
    # snap_prev = snap_start

    # for i in tqdm(range(T)):
    #     results = aligner.process(
    #         audio=["infer_out/tts_dialogue/infer_once/260108_prompt_base_dialogue/step84000/0-我家这餐馆，菜味儿正宗分量也足，怎么饭点.wav"] * 100,
    #         text=["<S1>我家这餐馆，菜味儿正宗分量也足，怎么饭点总坐不满啊？ </S1><S2>你试过做抖音团购套餐吗？ </S2><S1>抖音团购？哎呦，那是不是得花钱推广，还怕没人买亏了本啊？ </S1><S2>不用花推广费，套餐定价灵活还能吸引新客。 </S2><S1>真不用花钱？这靠谱吗？ </S1><S2>现在点击视频下方链接，就能零元上架抖音团购套餐，官方还会给本地流量推荐，帮你把附近想吃的人都引过来。像双人餐、家庭餐都能做，还能设置到店核销，不怕跑单。 </S2><S1>这听着行啊，那团购套餐咋设计啊？ </S1><S2>平台有现成的套餐模板，你按自家招牌菜搭配就行，还能自动生成图文海报，就算不会做宣传，用户刷到直接就能下单，用完都说好还会带朋友来。 </S2><S1>那我现在就去弄这个团购套餐！ </S1><S2>你们做餐饮的也赶紧点下方链接，上架抖音团购，让店里天天满座！</S2>"] * 100
    #     )
    #     snap_step = take_gpu_memory_snapshot(f"step_{i}", devices=[0])
    #     snap_step.log()
    #     diff = diff_gpu_memory_snapshots(snap_prev, snap_step)
    #     diff.log()
    #     snap_prev = snap_step
    # time_end = time.time()
    # elapsed = time_end - time_start
    # print(f"elapsed: {elapsed:.3f}s, average {elapsed/T:.3f}s per step")




    aligner = ASRPipeline(
        device='cuda',
    )
    results = aligner.process(
        audio="/mnt/bn/lq-ads-aigc/share/dataset/speech_edit_260421/megatts_datasets/audible_v1/3cecd4bca0034b131c3ba9cd0b1862fbc925790d/wav_orig.wav",
        text="I can't press against some other part of the airlock.",
        langs='english',
    )

    print(f"{results = }")
