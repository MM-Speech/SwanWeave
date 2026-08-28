import random
import re
from functools import lru_cache
from typing import List, Sequence

from pypinyin import lazy_pinyin, pinyin, Style


# ======================
# 基础工具：汉字检测 & 转拼音
# ======================

# 常用 CJK 基本汉字范围
CHINESE_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')

def is_chinese_char(ch: str) -> bool:
    return bool(CHINESE_CHAR_RE.fullmatch(ch))


def char_to_pinyin(ch: str, tone3: bool = True) -> str:
    """
    单个汉字 -> 拼音字符串。
    tone3=True => li3 形式；False => li 形式。
    """
    style = Style.TONE3 if tone3 else Style.NORMAL
    # lazy_pinyin 接受字符串，这里传单字符
    pys = lazy_pinyin(ch, style=style)
    return pys[0] if pys else ch


# ======================
# 多音字检测（加缓存）
# ======================

@lru_cache(maxsize=4096)
def is_polyphonic_char(ch: str) -> bool:
    """
    判断一个汉字在 pypinyin 字典里是否有多个读音。
    注意：这是“字典层面”的多音字，具体语境中不一定都多音。
    """
    if not is_chinese_char(ch):
        return False

    # heteronym=True -> 返回所有可能读音
    pys_2d = pinyin(ch, style=Style.NORMAL, heteronym=True)
    # 扁平化并去重
    s = {py for group in pys_2d for py in group}
    return len(s) >= 2


# ======================
# 抽样工具
# ======================

def sample_ratio_pow(gamma: float = 2.5) -> float:
    """
    从 (0, 1) 采样比例 r，越小的 r 越常见：
      r = U**gamma,  U ~ Uniform(0,1), gamma > 1
    gamma 越大，越偏向小比例。
    """
    u = random.random()
    r = u ** gamma
    eps = 1e-4
    return min(1 - eps, max(eps, r))


def weighted_sample_without_replacement(
    indices: Sequence[int],
    weights: Sequence[float],
    k: int
) -> List[int]:
    """
    按给定权重在 indices 中无放回抽样 k 个。
    使用 Efraimidis-Spirakis 算法，纯 Python，无需 numpy。

    参数：
      indices: 位置列表，如 [3, 5, 6, ...]
      weights: 与 indices 对应的权重，如 [1.0, 3.0, 1.0, ...]
      k: 抽样个数（会自动截断到 [0, len(indices)]）
    """
    n = len(indices)
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(indices)

    # Efraimidis-Spirakis:
    # 对每个元素 i，生成 key_i = U_i ** (1 / weight_i),
    # 再取 key 最大的前 k 个。
    items = []
    for idx, w in zip(indices, weights):
        if w <= 0:
            continue
        u = random.random()
        key = u ** (1.0 / w)
        items.append((key, idx))

    if not items:
        return []

    # 取 key 最大的 k 个
    items.sort(key=lambda x: x[0], reverse=True)
    selected = [idx for _, idx in items[:k]]
    return selected


# ======================
# 两种增强模式的实现
# ======================

def augment_bernoulli(
    text: str,
    base_prob: float = 0.15,
    poly_weight: float = 3.0,
    tone3: bool = True,
    pinyin_tokenizer=lambda x: x
) -> str:
    """
    伯努利逐字采样：
      - 对每个汉字，以 base_prob 为基础概率；
      - 若是多音字，则概率放大 poly_weight 倍；
      - p_i = clip(base_prob * weight_i, 0, 1)；
      - 以 p_i 独立决定是否替换为拼音。
    """
    out = []

    for ch in text:
        if not is_chinese_char(ch):
            out.append(ch)
            continue

        w = poly_weight if is_polyphonic_char(ch) else 1.0
        p_i = base_prob * w
        if p_i > 1.0:
            p_i = 1.0

        if random.random() < p_i:
            out.append(pinyin_tokenizer(char_to_pinyin(ch, tone3=tone3)))
        else:
            out.append(ch)

    return ''.join(out)


def augment_ratio(
    text: str,
    gamma: float = 2.5,
    poly_weight: float = 3.0,
    tone3: bool = True,
    pinyin_tokenizer=lambda x: x
) -> str:
    """
    层级采样：
      1) 在 (0,1) 上采样一个比例 r（越小越常见）；
      2) 设该句有 N 个汉字，则 K ≈ r * N；
      3) 在 N 个汉字中，按权重无放回抽样 K 个进行替换：
         - 多音字权重 = poly_weight
         - 其他汉字权重 = 1.0
    """
    chars = list(text)
    # 所有汉字的位置
    ch_indices = [i for i, ch in enumerate(chars) if is_chinese_char(ch)]
    N = len(ch_indices)
    if N == 0:
        return text

    # 采样比例 r & 替换个数 K
    r = sample_ratio_pow(gamma=gamma)
    K = int(round(r * N))
    K = max(1, min(N, K))  # 至少 1 个，最多 N 个

    # 构造权重：多音字权重大
    weights = []
    for i in ch_indices:
        ch = chars[i]
        w = poly_weight if is_polyphonic_char(ch) else 1.0
        weights.append(w)

    # 按权重抽样 K 个位置
    selected_positions = set(
        weighted_sample_without_replacement(ch_indices, weights, K)
    )

    for i in selected_positions:
        chars[i] = pinyin_tokenizer(char_to_pinyin(chars[i], tone3=tone3))

    return ''.join(chars)


# ======================
# 总控接口：TTS 用增强逻辑
# ======================

def augment_text_with_pinyin_advanced(
    text: str,
    *,
    # 句子级：是否增强
    p_augment: float = 0.3,

    # 在“已决定增强”的前提下：
    # 使用伯努利模式的概率（小概率），
    # 使用层级采样模式的概率 = 1 - p_bernoulli_mode（大概率）
    p_bernoulli_mode: float = 0.2,

    # 伯努利模式参数
    bernoulli_base_prob: float = 0.15,
    poly_weight_bernoulli: float = 3.0,

    # 层级模式参数
    ratio_gamma: float = 2.5,
    poly_weight_ratio: float = 3.0,

    # 输出拼音形式
    tone3: bool = True,
    pinyin_tokenizer = lambda x: x
) -> str:
    """
    综合策略：
      1) 以 p_augment 决定是否对该句做拼音增强；
      2) 若增强：
         - 以 p_bernoulli_mode 选择伯努利模式（多音字概率更大）；
         - 以 1 - p_bernoulli_mode 选择层级采样模式（比例 r 偏小，多音字权重大）。

    参数建议（TTS 初始配置）：
      p_augment          ~ 0.3
      p_bernoulli_mode   ~ 0.2  (=> 6% 句子伯努利，24% 层级采样)
      bernoulli_base_prob ~ 0.15
      poly_weight_*      ~ 3.0
      ratio_gamma        ~ 2.0 ~ 3.0
      tone3              = True  (li3 形式，便于 disambiguation)
    """
    # 1) 句子级：是否增强
    if random.random() >= p_augment:
        return text  # 不增强，保持原样

    # 2) 句子级：选择模式
    if random.random() < p_bernoulli_mode:
        # 伯努利逐字采样，多音字概率更大
        return augment_bernoulli(
            text,
            base_prob=bernoulli_base_prob,
            poly_weight=poly_weight_bernoulli,
            tone3=tone3,
            pinyin_tokenizer=pinyin_tokenizer
        )
    else:
        # 层级比例采样，多音字权重大
        return augment_ratio(
            text,
            gamma=ratio_gamma,
            poly_weight=poly_weight_ratio,
            tone3=tone3,
            pinyin_tokenizer=pinyin_tokenizer
        )


def augment_text_with_pinyin_s1s2_safe(text: str, hparams):
    """
    只对 <S1>/<S2> 标签内部做 pinyin 混入，保留标签结构。
    若文本不含 S1/S2 标签，则等价于原 augment。
    """
    cfg = hparams.get('mix_text_pinyin', {}) or {}
    if not cfg.get('enable', False):
        return text

    s1s2_text_re = re.compile(
        r'<\s*(S[1-4])\s*>(.*?)</\s*\1\s*>',
        flags=re.IGNORECASE | re.DOTALL,
    )

    kwargs = dict(
        p_augment=cfg.get('enable_prob', 0.3),
        p_bernoulli_mode=0.1,
        poly_weight_bernoulli=3.0,
        ratio_gamma=3.0,
        poly_weight_ratio=3.0,
        tone3=True,
        pinyin_tokenizer=lambda x: f"<|py_{x}|>"
    )

    if s1s2_text_re.search(text) is None:
        return augment_text_with_pinyin_advanced(text, **kwargs)

    def _repl(m):
        tag = m.group(1).upper()
        inner = m.group(2)
        inner_aug = augment_text_with_pinyin_advanced(inner, **kwargs)
        return f"<{tag}>{inner_aug}</{tag}>"

    return s1s2_text_re.sub(_repl, text)


# ======================
# 简单测试
# ======================
if __name__ == "__main__":
    random.seed(1234)

    examples = [
        "我爱China的产品，还有很多用户。",
        "今天重启系统，明天重新部署。",
        "音乐播放异常，请检查日志。",
        "See openai_public.py for examples of how to construct an Encoding object.",
        "ยินดีที่ได้รู้จัก",
        "梅花（Titoni）瑞士手表男士机械表经典金色腕表生日礼物 宇宙系列 钢带表盘40MM 797 G-DB-306"
    ]

    for t in examples:
        print("原句: ", t)
        for _ in range(5):
            print("增强: ", augment_text_with_pinyin_advanced(t, pinyin_tokenizer=lambda x: f"<|{x}|>"))
        print("-" * 40)
