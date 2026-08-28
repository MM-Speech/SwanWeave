"""AdamW 优化器构造,可选把 bias/norm/Embedding 类参数从 weight_decay 中排除。

接口与 `muon_aux.build_muon_aux_adam` 对称:同样的 (model, cfg, disable_flag) 入口,
返回 (optimizer, n_decay, n_no_decay)。
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

from . import NO_DECAY_KEYWORDS


def build_adamw(
    model: nn.Module,
    optimizer_cfg: Dict,
    disable_wd_on_bias_norm_embed: bool = False,
) -> Tuple[torch.optim.Optimizer, int, int]:
    """构造 AdamW(可选拆 decay/no_decay 两组)。

    disable_wd_on_bias_norm_embed=False: 全部参数共用 optimizer_cfg.weight_decay,1 组。
    disable_wd_on_bias_norm_embed=True: dim==1 或名字命中 NO_DECAY_KEYWORDS 的拆出 wd=0 组。
    返回 (optimizer, n_decay, n_no_decay)。
    """
    if not disable_wd_on_bias_norm_embed:
        n_total = sum(1 for p in model.parameters() if p.requires_grad)
        return AdamW(model.parameters(), **optimizer_cfg), n_total, 0

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 1 or any(k in name for k in NO_DECAY_KEYWORDS):
            no_decay.append(p)
        else:
            decay.append(p)

    wd = optimizer_cfg.get("weight_decay", 0.01)
    groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, **optimizer_cfg), len(decay), len(no_decay)
