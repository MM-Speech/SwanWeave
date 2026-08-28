"""Muon + AdamW 混合优化器接入(参考 DeepSeek-V3 / Kimi K2)。

Muon 只更新 transformer 内部的 2D 权重矩阵;Embedding / 输出读出层 / 1D 参数(norm/bias)
走 AdamW。两者由 `muon.MuonWithAuxAdam` 包成单个 optimizer 对象,trainer 不感知拆分。
swan attention 已含 q_norm/k_norm,等价 K2 的 QK-norm,故不需要 MuonClip。

依赖: pip install muon-optimizer
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from . import NO_DECAY_KEYWORDS

# Kimi K2 风格默认参数(语音任务建议在 yaml 里覆盖为更保守的值)
MUON_DEFAULTS: Dict = {"lr": 0.02, "momentum": 0.95, "weight_decay": 0.1}
EXTRA_ADAM_KEYWORDS: Tuple[str, ...] = ("embed", "embedder", "lm_head", "postnet")
_AUX_ADAM_ALLOWED_KEYS: Tuple[str, ...] = ("lr", "betas", "eps", "weight_decay")


def split_params_for_muon(
    model: nn.Module,
    extra_adam_keywords: Optional[Tuple[str, ...]] = None,
    disable_wd_on_bias_norm_embed: bool = False,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    """把 model 的可训练参数切成 (muon, adamw_decay, adamw_no_decay) 三组。

    Muon: 2D 权重矩阵,且不在 `nn.Embedding` 模块下,且名字不含 extra_adam_keywords。
    AdamW(其他): 1D / Embedding / norm / bias / 名字命中关键字的 2D。
    若 `disable_wd_on_bias_norm_embed=True`,把 AdamW 组里满足
    (dim==1 或名字含 NO_DECAY_KEYWORDS)的参数拆出来归到 no_decay 组(wd=0),
    与 swan 现有 AdamW 路径的 `disable_weight_decay_on_bias_and_norm_and_embed` 行为一致。
    若为 False,no_decay 组永远为空。
    """
    keywords = extra_adam_keywords or EXTRA_ADAM_KEYWORDS
    embed_param_ids = {
        id(p)
        for _, m in model.named_modules() if isinstance(m, nn.Embedding)
        for p in m.parameters(recurse=False)
    }

    muon, decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.dim() == 2
            and id(p) not in embed_param_ids
            and not any(k in name.lower() for k in keywords)
        ):
            muon.append(p)
        elif disable_wd_on_bias_norm_embed and (
            p.dim() == 1 or any(k in name for k in NO_DECAY_KEYWORDS)
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    return muon, decay, no_decay


def build_muon_aux_adam(
    model: nn.Module,
    muon_cfg: Optional[Dict] = None,
    adamw_cfg: Optional[Dict] = None,
    extra_adam_keywords: Optional[Tuple[str, ...]] = None,
    disable_wd_on_bias_norm_embed: bool = False,
) -> Tuple[torch.optim.Optimizer, int, int, int]:
    """构造 `MuonWithAuxAdam`(单一 optimizer,内部按 use_muon 分发到 Muon / AdamW)。

    - 缺省 muon_cfg 走 `MUON_DEFAULTS`
    - adamw_cfg 仅取 lr/betas/eps/weight_decay 子集,过滤 fused 等不支持的 key
    - disable_wd_on_bias_norm_embed=True 时 AdamW 组再拆 decay/no_decay,
      与 swan 现有 AdamW 路径的 disable 标志同语义

    返回 (optimizer, n_muon, n_decay, n_no_decay)。
    """
    from muon import MuonWithAuxAdam

    muon_params, decay_aux, no_decay_aux = split_params_for_muon(
        model, extra_adam_keywords, disable_wd_on_bias_norm_embed,
    )
    if not muon_params:
        raise ValueError(
            "split_params_for_muon found 0 muon params; check model / extra_adam_keywords"
        )

    aux_kwargs = {k: v for k, v in (adamw_cfg or {}).items() if k in _AUX_ADAM_ALLOWED_KEYS}
    groups = [
        {"params": muon_params, "use_muon": True, **MUON_DEFAULTS, **(muon_cfg or {})},
        {"params": decay_aux, "use_muon": False, **aux_kwargs},
    ]
    if no_decay_aux:
        groups.append({"params": no_decay_aux, "use_muon": False, **{**aux_kwargs, "weight_decay": 0.0}})

    return MuonWithAuxAdam(groups), len(muon_params), len(decay_aux), len(no_decay_aux)
