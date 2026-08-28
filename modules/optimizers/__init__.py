"""swan task 用的优化器构造 helper。AdamW / Muon 走对称接口。"""
from typing import Tuple

# bias / norm / 1D / 部分 Embedding 名字 → wd=0(与 swan 历史 AdamW 路径的关键字列表一致)
NO_DECAY_KEYWORDS: Tuple[str, ...] = ("bias", "norm", "text_embedder", "tone_embed", "ph_embed")
