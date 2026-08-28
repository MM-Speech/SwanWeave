import re
import re
import re
import torch

def _encode_pattern_ids(tokenizer, text: str, device):
    """
    将固定标签文本编码为 token id 序列（不加特殊符号），返回 torch.LongTensor [L].
    这一步只用于获取“模式”，后续逻辑完全在 input_ids 张量上。
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if isinstance(ids[0], list):
        ids = ids[0]
    return torch.tensor(ids, dtype=torch.long, device=device)
def _find_pattern_starts(seq: torch.LongTensor, pattern: torch.LongTensor) -> torch.LongTensor:

    """
    在一维序列 seq 中查找子序列 pattern 的所有起始位置。
    纯张量实现：unfold + 全维比较。
    返回：torch.LongTensor [K]，元素为起始 index。
    """
    T = seq.shape[0]
    L = pattern.shape[0]
    if L == 0 or T < L:
        return torch.empty(0, dtype=torch.long, device=seq.device)
    windows = seq.unfold(dimension=0, size=L, step=1)        # [T-L+1, L]
    matches = (windows == pattern).all(dim=1)                 # [T-L+1]
    starts = torch.nonzero(matches, as_tuple=False).squeeze(1)
    return starts

def _zero_ranges_between_pairs(mask_row: torch.Tensor,
                               starts: torch.LongTensor,
                               ends: torch.LongTensor,
                               end_len: int):
    """
    将 mask_row 的范围 [start, end+end_len) 置 0。
    贪心配对：每个 start 配最近的、在其之后的 end。若某个 start 找不到 end，则忽略。
    """
    if starts.numel() == 0 or ends.numel() == 0:
        return
    s_list = starts.tolist()
    e_list = ends.tolist()
    e_idx = 0
    for s in s_list:
        while e_idx < len(e_list) and e_list[e_idx] <= s:
            e_idx += 1
        if e_idx >= len(e_list):
            break
        e = e_list[e_idx]
        e_idx += 1
        mask_row[s: e + end_len] = 0  # 含闭合标签本身

def build_dialogue_mask_from_ids(input_ids: torch.LongTensor,
                                 attention_mask: torch.LongTensor,
                                 tokenizer) -> torch.LongTensor:
    """
    依据 input_ids 中的标签子序列，构造“台词 token 掩码”（台词=1，caption=0）。
    - 完全在张量上进行子序列匹配；不使用 offset；不做文本解码。
    - tokenizer 仅用于获取标签的 id 序列（自适应不同 tokenizer）。
    输入：
      - input_ids: [B, T] LongTensor
      - attention_mask: [B, T] LongTensor
      - tokenizer: 当前 tokenizer（Qwen 可用 fast/slow 均可）
    输出：
      - dialogue_mask: [B, T] LongTensor（0/1），与 attention_mask 同形状
    """
    device = input_ids.device

    # 1) 获取标签的 id 序列（单 token 或多 token 都可）
    GP_OPEN  = _encode_pattern_ids(tokenizer, "<GPROMPT>", device)
    GP_CLOSE = _encode_pattern_ids(tokenizer, "</GPROMPT>", device)
    TAG_OPEN  = _encode_pattern_ids(tokenizer, "<TAG>", device)
    TAG_CLOSE = _encode_pattern_ids(tokenizer, "</TAG>", device)

    B, T = input_ids.shape
    dialogue_mask = torch.ones_like(attention_mask)

    # 2) 逐样本匹配并置零
    for b in range(B):
        seq = input_ids[b]      # [T]
        mask_row = dialogue_mask[b]

        gp_starts = _find_pattern_starts(seq, GP_OPEN)
        gp_ends   = _find_pattern_starts(seq, GP_CLOSE)
        tag_starts = _find_pattern_starts(seq, TAG_OPEN)
        tag_ends   = _find_pattern_starts(seq, TAG_CLOSE)

        _zero_ranges_between_pairs(mask_row, gp_starts, gp_ends, end_len=GP_CLOSE.numel())
        _zero_ranges_between_pairs(mask_row, tag_starts, tag_ends, end_len=TAG_CLOSE.numel())

    # 3) 去除 padding（padding 位置强制 0）
    dialogue_mask = dialogue_mask * attention_mask
    return dialogue_mask

import torch

def build_audio_mask_from_ids(input_ids: torch.LongTensor,
                              attention_mask: torch.LongTensor,
                              tokenizer) -> torch.LongTensor:
    """
    基于 input_ids 中的标签子序列构造“多类 token 掩码”：
      - <S1>...</S1>、<S2>...</S2>、<S3>...</S3>、<S4>...</S4> 内为 1
      - <Audio>...</Audio> 内为 2（覆盖前面的 1）
      - <BGM>...</BGM> 内为 3（覆盖前面的 1/2）
      - 其它/标签本身/未匹配/padding 为 0

    纯张量级子序列匹配，不解码，不依赖 offset。
    输入：
      - input_ids: [B, T] LongTensor
      - attention_mask: [B, T] LongTensor
      - tokenizer: 当前 tokenizer
    输出：
      - dialogue_mask: [B, T] LongTensor（取值 0/1/2/3），与 attention_mask 同形状
    """
    device = input_ids.device
    B, T = input_ids.shape

    # ------------------------------------------------------------------
    # 1) pattern 编码
    # ------------------------------------------------------------------
    S1_OPEN   = _encode_pattern_ids(tokenizer, "<S1>",     device)
    S1_CLOSE  = _encode_pattern_ids(tokenizer, "</S1>",    device)
    S2_OPEN   = _encode_pattern_ids(tokenizer, "<S2>",     device)
    S2_CLOSE  = _encode_pattern_ids(tokenizer, "</S2>",    device)
    S3_OPEN   = _encode_pattern_ids(tokenizer, "<S3>",     device)
    S3_CLOSE  = _encode_pattern_ids(tokenizer, "</S3>",    device)
    S4_OPEN   = _encode_pattern_ids(tokenizer, "<S4>",     device)
    S4_CLOSE  = _encode_pattern_ids(tokenizer, "</S4>",    device)

    AU_OPEN   = _encode_pattern_ids(tokenizer, "<Audio>",  device)
    AU_CLOSE  = _encode_pattern_ids(tokenizer, "</Audio>", device)

    BGM_OPEN  = _encode_pattern_ids(tokenizer, "<BGM>",    device)
    BGM_CLOSE = _encode_pattern_ids(tokenizer, "</BGM>",   device)
    ENV_OPEN  = _encode_pattern_ids(tokenizer, "<ENV>",    device)
    ENV_CLOSE = _encode_pattern_ids(tokenizer, "</ENV>",   device)

    # ------------------------------------------------------------------
    # 2) batch 级子序列匹配：返回 [B, T] bool，表示 pattern 的起点
    # ------------------------------------------------------------------
    def pattern_starts_batch(ids: torch.LongTensor,
                             pattern: torch.LongTensor) -> torch.BoolTensor:
        """
        ids:     [B, T]
        pattern: [L]
        返回:     [B, T] bool，True 表示该位置是 pattern 的起始 token
        """
        B_, T_ = ids.shape
        L = int(pattern.numel())
        if L == 0:
            return torch.zeros(B_, T_, dtype=torch.bool, device=ids.device)
        if L == 1:
            return ids == pattern[0]
        if T_ < L:
            return torch.zeros(B_, T_, dtype=torch.bool, device=ids.device)

        windows = ids.unfold(dimension=1, size=L, step=1)      # [B, T-L+1, L]
        eq = (windows == pattern.view(1, 1, L))                # broadcast
        starts = eq.all(dim=-1)                                # [B, T-L+1]

        pad = torch.zeros(B_, L - 1, dtype=torch.bool, device=ids.device)
        return torch.cat([starts, pad], dim=1)                 # [B, T]

    # 统计所有 starts
    s1_open_starts  = pattern_starts_batch(input_ids, S1_OPEN)
    s1_close_starts = pattern_starts_batch(input_ids, S1_CLOSE)
    s2_open_starts  = pattern_starts_batch(input_ids, S2_OPEN)
    s2_close_starts = pattern_starts_batch(input_ids, S2_CLOSE)
    s3_open_starts  = pattern_starts_batch(input_ids, S3_OPEN)
    s3_close_starts = pattern_starts_batch(input_ids, S3_CLOSE)
    s4_open_starts  = pattern_starts_batch(input_ids, S4_OPEN)
    s4_close_starts = pattern_starts_batch(input_ids, S4_CLOSE)

    au_open_starts  = pattern_starts_batch(input_ids, AU_OPEN)
    au_close_starts = pattern_starts_batch(input_ids, AU_CLOSE)

    bgm_open_starts  = pattern_starts_batch(input_ids, BGM_OPEN)
    bgm_close_starts = pattern_starts_batch(input_ids, BGM_CLOSE)

    # ------------------------------------------------------------------
    # 3) 利用差分 + cumsum，把所有区间一次性填出来
    # ------------------------------------------------------------------
    def interval_mask_from_starts(open_starts: torch.BoolTensor,
                                  close_starts: torch.BoolTensor,
                                  open_len: int) -> torch.BoolTensor:
        """
        open_starts / close_starts: [B, T] bool，表示 pattern 起点
        open_len: 起始标签长度（token 数）

        语义：对每条序列，从所有 <open>...<close> 段中，把
              [s+open_len, e) 位置标为 True。
        """
        device_ = open_starts.device
        B_, T_ = open_starts.shape

        diff = torch.zeros(B_, T_ + 1, dtype=torch.int32, device=device_)

        # enter：在 s+open_len 位置 +1
        if open_starts.any():
            pos = open_starts.nonzero(as_tuple=False)  # [N, 2]
            b, t = pos[:, 0], pos[:, 1]
            L = t + open_len
            valid = L < T_
            if valid.any():
                b = b[valid]
                L = L[valid]
                diff[b, L] += 1

        # leave：在 e 位置 -1
        if close_starts.any():
            pos = close_starts.nonzero(as_tuple=False)  # [M, 2]
            b, e = pos[:, 0], pos[:, 1]
            e = torch.clamp(e, max=T_)                  # allow e==T_
            diff[b, e] -= 1

        inside = (diff.cumsum(dim=-1)[:, :T_] > 0)
        return inside

    # S1~S4 inside
    s1_inside = interval_mask_from_starts(s1_open_starts, s1_close_starts, open_len=int(S1_OPEN.numel()))
    s2_inside = interval_mask_from_starts(s2_open_starts, s2_close_starts, open_len=int(S2_OPEN.numel()))
    s3_inside = interval_mask_from_starts(s3_open_starts, s3_close_starts, open_len=int(S3_OPEN.numel()))
    s4_inside = interval_mask_from_starts(s4_open_starts, s4_close_starts, open_len=int(S4_OPEN.numel()))

    # Audio / BGM inside
    au_inside  = interval_mask_from_starts(au_open_starts,  au_close_starts,  open_len=int(AU_OPEN.numel()))
    bgm_inside = interval_mask_from_starts(bgm_open_starts, bgm_close_starts, open_len=int(BGM_OPEN.numel()))

    # ------------------------------------------------------------------
    # 4) 组合成最终 mask：文本 1，Audio 2，BGM 3（后写覆盖前写）
    # ------------------------------------------------------------------
    dialogue_mask = torch.zeros_like(attention_mask, dtype=torch.long)  # [B, T]

    text_inside = s1_inside | s2_inside | s3_inside | s4_inside
    dialogue_mask[text_inside] = 1
    dialogue_mask[au_inside] = 2
    dialogue_mask[bgm_inside] = 3

    # 去掉 padding
    dialogue_mask = dialogue_mask * attention_mask
    return dialogue_mask

def build_caption_mask_from_ids(input_ids: torch.LongTensor,
                               attention_mask: torch.LongTensor,
                               tokenizer) -> torch.LongTensor:
    """
    基于 input_ids 中的标签子序列构造"多类 token 掩码"：
      - padding 为 0
      - <S1>...</S1>、<S2>...</S2>、<S3>...</S3>、<S4>...</S4> 内为 1
      - <Audio>...</Audio> 内为 2
      - <BGM>...</BGM> 内为 3
      - <BGM>无背景音乐</BGM> 等明确无 BGM 的内容为 4
      - <SPK>...</SPK> 内为 5
      - <ENV>...</ENV> 内为 6
      - 其它非 padding 为 7

    覆盖优先级：ENV(6) / SPK(5) -> BGM(3/4) -> Audio(2) -> Speech(1)。
    """
    device = input_ids.device
    B, _ = input_ids.shape

    S1_OPEN   = _encode_pattern_ids(tokenizer, "<S1>",     device)
    S1_CLOSE  = _encode_pattern_ids(tokenizer, "</S1>",    device)
    S2_OPEN   = _encode_pattern_ids(tokenizer, "<S2>",     device)
    S2_CLOSE  = _encode_pattern_ids(tokenizer, "</S2>",    device)
    S3_OPEN   = _encode_pattern_ids(tokenizer, "<S3>",     device)
    S3_CLOSE  = _encode_pattern_ids(tokenizer, "</S3>",    device)
    S4_OPEN   = _encode_pattern_ids(tokenizer, "<S4>",     device)
    S4_CLOSE  = _encode_pattern_ids(tokenizer, "</S4>",    device)

    AU_OPEN   = _encode_pattern_ids(tokenizer, "<Audio>",  device)
    AU_CLOSE  = _encode_pattern_ids(tokenizer, "</Audio>", device)

    BGM_OPEN  = _encode_pattern_ids(tokenizer, "<BGM>",    device)
    BGM_CLOSE = _encode_pattern_ids(tokenizer, "</BGM>",   device)

    SPK_OPEN  = _encode_pattern_ids(tokenizer, "<SPK>",    device)
    SPK_CLOSE = _encode_pattern_ids(tokenizer, "</SPK>",   device)

    ENV_OPEN  = _encode_pattern_ids(tokenizer, "<ENV>",    device)
    ENV_CLOSE = _encode_pattern_ids(tokenizer, "</ENV>",   device)

    def pattern_starts_batch(ids: torch.LongTensor, pattern: torch.LongTensor) -> torch.BoolTensor:
        B_, T_ = ids.shape
        L = int(pattern.numel())
        if L == 0:
            return torch.zeros(B_, T_, dtype=torch.bool, device=ids.device)
        if L == 1:
            return ids == pattern[0]
        if T_ < L:
            return torch.zeros(B_, T_, dtype=torch.bool, device=ids.device)
        windows = ids.unfold(dimension=1, size=L, step=1)
        eq = (windows == pattern.view(1, 1, L))
        starts = eq.all(dim=-1)
        pad = torch.zeros(B_, L - 1, dtype=torch.bool, device=ids.device)
        return torch.cat([starts, pad], dim=1)

    def interval_mask_from_starts(open_starts: torch.BoolTensor, close_starts: torch.BoolTensor, open_len: int) -> torch.BoolTensor:
        B_, T_ = open_starts.shape
        diff = torch.zeros(B_, T_ + 1, dtype=torch.int32, device=open_starts.device)
        if open_starts.any():
            pos = open_starts.nonzero(as_tuple=False)
            b, t = pos[:, 0], pos[:, 1]
            left = t + int(open_len)
            valid = left < T_
            if valid.any():
                diff[b[valid], left[valid]] += 1
        if close_starts.any():
            pos = close_starts.nonzero(as_tuple=False)
            b, e = pos[:, 0], pos[:, 1]
            e = torch.clamp(e, max=T_)
            diff[b, e] -= 1
        return diff.cumsum(dim=-1)[:, :T_] > 0

    def span_mask_from_starts(starts: torch.BoolTensor, span_len: int) -> torch.BoolTensor:
        B_, T_ = starts.shape
        if span_len <= 0:
            return torch.zeros(B_, T_, dtype=torch.bool, device=starts.device)
        diff = torch.zeros(B_, T_ + 1, dtype=torch.int32, device=starts.device)
        if starts.any():
            pos = starts.nonzero(as_tuple=False)
            b, t = pos[:, 0], pos[:, 1]
            diff[b, t] += 1
            end = torch.clamp(t + int(span_len), max=T_)
            diff[b, end] -= 1
        return diff.cumsum(dim=-1)[:, :T_] > 0

    s1_open_starts = pattern_starts_batch(input_ids, S1_OPEN)
    s1_close_starts = pattern_starts_batch(input_ids, S1_CLOSE)
    s2_open_starts = pattern_starts_batch(input_ids, S2_OPEN)
    s2_close_starts = pattern_starts_batch(input_ids, S2_CLOSE)
    s3_open_starts = pattern_starts_batch(input_ids, S3_OPEN)
    s3_close_starts = pattern_starts_batch(input_ids, S3_CLOSE)
    s4_open_starts = pattern_starts_batch(input_ids, S4_OPEN)
    s4_close_starts = pattern_starts_batch(input_ids, S4_CLOSE)

    au_open_starts = pattern_starts_batch(input_ids, AU_OPEN)
    au_close_starts = pattern_starts_batch(input_ids, AU_CLOSE)

    bgm_open_starts = pattern_starts_batch(input_ids, BGM_OPEN)
    bgm_close_starts = pattern_starts_batch(input_ids, BGM_CLOSE)

    spk_open_starts = pattern_starts_batch(input_ids, SPK_OPEN)
    spk_close_starts = pattern_starts_batch(input_ids, SPK_CLOSE)

    env_open_starts = pattern_starts_batch(input_ids, ENV_OPEN)
    env_close_starts = pattern_starts_batch(input_ids, ENV_CLOSE)

    s1_inside = interval_mask_from_starts(s1_open_starts, s1_close_starts, open_len=int(S1_OPEN.numel()))
    s2_inside = interval_mask_from_starts(s2_open_starts, s2_close_starts, open_len=int(S2_OPEN.numel()))
    s3_inside = interval_mask_from_starts(s3_open_starts, s3_close_starts, open_len=int(S3_OPEN.numel()))
    s4_inside = interval_mask_from_starts(s4_open_starts, s4_close_starts, open_len=int(S4_OPEN.numel()))

    au_inside = interval_mask_from_starts(au_open_starts, au_close_starts, open_len=int(AU_OPEN.numel()))
    bgm_inside = interval_mask_from_starts(bgm_open_starts, bgm_close_starts, open_len=int(BGM_OPEN.numel()))
    spk_inside = interval_mask_from_starts(spk_open_starts, spk_close_starts, open_len=int(SPK_OPEN.numel()))
    env_inside = interval_mask_from_starts(env_open_starts, env_close_starts, open_len=int(ENV_OPEN.numel()))

    s1_tags = span_mask_from_starts(s1_open_starts, int(S1_OPEN.numel())) | span_mask_from_starts(s1_close_starts, int(S1_CLOSE.numel()))
    s2_tags = span_mask_from_starts(s2_open_starts, int(S2_OPEN.numel())) | span_mask_from_starts(s2_close_starts, int(S2_CLOSE.numel()))
    s3_tags = span_mask_from_starts(s3_open_starts, int(S3_OPEN.numel())) | span_mask_from_starts(s3_close_starts, int(S3_CLOSE.numel()))
    s4_tags = span_mask_from_starts(s4_open_starts, int(S4_OPEN.numel())) | span_mask_from_starts(s4_close_starts, int(S4_CLOSE.numel()))
    au_tags = span_mask_from_starts(au_open_starts, int(AU_OPEN.numel())) | span_mask_from_starts(au_close_starts, int(AU_CLOSE.numel()))
    bgm_tags = span_mask_from_starts(bgm_open_starts, int(BGM_OPEN.numel())) | span_mask_from_starts(bgm_close_starts, int(BGM_CLOSE.numel()))
    spk_tags = span_mask_from_starts(spk_open_starts, int(SPK_OPEN.numel())) | span_mask_from_starts(spk_close_starts, int(SPK_CLOSE.numel()))
    env_tags = span_mask_from_starts(env_open_starts, int(ENV_OPEN.numel())) | span_mask_from_starts(env_close_starts, int(ENV_CLOSE.numel()))

    text_mask = (s1_inside | s2_inside | s3_inside | s4_inside) | (s1_tags | s2_tags | s3_tags | s4_tags)
    au_mask = au_inside | au_tags
    bgm_mask = bgm_inside | bgm_tags
    spk_mask = spk_inside | spk_tags
    env_mask = env_inside | env_tags

    dialogue_mask = torch.zeros_like(attention_mask, dtype=torch.long)
    dialogue_mask[attention_mask.to(torch.bool)] = 7

    dialogue_mask[env_mask] = 6
    dialogue_mask[spk_mask] = 5
    dialogue_mask[bgm_mask] = 3
    dialogue_mask[au_mask] = 2
    dialogue_mask[text_mask] = 1

    if bgm_inside.any():
        bgm_disable = torch.zeros((B,), dtype=torch.bool, device=device)
        bgm_open_len = int(BGM_OPEN.numel())

        for b in range(B):
            if not bgm_inside[b].any():
                continue

            opens = torch.nonzero(bgm_open_starts[b], as_tuple=False).squeeze(1)
            closes = torch.nonzero(bgm_close_starts[b], as_tuple=False).squeeze(1)
            if opens.numel() == 0 or closes.numel() == 0:
                continue

            s = int(opens[0].item())
            closes_after = closes[closes > s]
            if closes_after.numel() == 0:
                continue
            e = int(closes_after[0].item())

            left = s + bgm_open_len
            right = e
            if left >= right:
                bgm_disable[b] = True
                continue

            content_ids = input_ids[b, left:right]
            try:
                inner = tokenizer.decode(content_ids.tolist(), skip_special_tokens=False)
            except Exception:
                continue

            inner = (inner or "").strip()
            inner_compact = re.sub(r"\s+", "", inner)
            inner_compact = re.sub(
                r"[。．\.，,！!？?、:：;；""\"'（）\(\)【】\[\]《》<>]", "", inner_compact
            )

            if (
                inner_compact == "" or inner_compact == "无" or
                inner_compact in {"无背景音乐", "无音乐", "无背景声", "无背景音", "无配乐"}
            ):
                bgm_disable[b] = True

        if bgm_disable.any():
            no_bgm_mask = bgm_mask & bgm_disable[:, None].expand_as(bgm_mask)
            dialogue_mask[no_bgm_mask] = 4

    return dialogue_mask * attention_mask