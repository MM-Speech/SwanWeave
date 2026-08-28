"""transformers 版本兼容层。集中给 modules/commons/hf/* 与 modules/tts/swanaudio/* 用。
优先 transformers 5.2 路径,缺失时退到旧路径或 no-op shim,让 4.x / 5.2+ 都能 import。
稳定 API 直接 re-export;装饰器/类型注解给 identity/typing 兜底;基类回退 nn.Module;
mask 构造类必须真有,缺则显式 NotImplementedError。
"""
from typing import Any

import torch.nn as nn

# === 稳定 API:从 4.x 起一直存在,直接 re-export ===
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

# === RoPE 工具 ===
try:
    from transformers.modeling_rope_utils import (
        ROPE_INIT_FUNCTIONS,
        dynamic_rope_update,
        rope_config_validation,
    )
except ImportError:
    ROPE_INIT_FUNCTIONS = {}

    def dynamic_rope_update(arg=None, *_args, **_kwargs):
        return arg if callable(arg) else (lambda fn: fn)

    def rope_config_validation(*_args, **_kwargs):
        return None

# === modeling_layers:GradientCheckpointingLayer + Generic* 基类 (4.50+) ===
try:
    from transformers.modeling_layers import (
        GenericForQuestionAnswering,
        GenericForSequenceClassification,
        GenericForTokenClassification,
        GradientCheckpointingLayer,
    )
except ImportError:
    GradientCheckpointingLayer = nn.Module
    GenericForQuestionAnswering = nn.Module
    GenericForSequenceClassification = nn.Module
    GenericForTokenClassification = nn.Module

# === Unpack:processing_utils → typing → typing_extensions ===
try:
    from transformers.processing_utils import Unpack
except ImportError:
    try:
        from typing import Unpack  # py3.11+
    except ImportError:
        from typing_extensions import Unpack

# === TransformersKwargs:transformers.utils → utils.generic → 空 TypedDict ===
try:
    from transformers.utils import TransformersKwargs
except ImportError:
    try:
        from transformers.utils.generic import TransformersKwargs
    except ImportError:
        try:
            from typing import TypedDict
        except ImportError:
            from typing_extensions import TypedDict

        class TransformersKwargs(TypedDict, total=False):
            pass

# === transformers.utils 装饰器:auto_docstring / can_return_tuple ===
try:
    from transformers.utils import auto_docstring
except ImportError:
    def auto_docstring(arg=None, *_args, **_kwargs):
        return arg if callable(arg) else (lambda fn: fn)

try:
    from transformers.utils import can_return_tuple
except ImportError:
    def can_return_tuple(arg=None, *_args, **_kwargs):
        return arg if callable(arg) else (lambda fn: fn)

# === transformers.utils.generic:check_model_inputs / OutputRecorder ===
try:
    from transformers.utils.generic import check_model_inputs
except ImportError:
    try:
        from transformers.utils import check_model_inputs
    except ImportError:
        def check_model_inputs(arg=None, *_args, **_kwargs):
            return arg if callable(arg) else (lambda fn: fn)

try:
    from transformers.utils.generic import OutputRecorder
except ImportError:
    class OutputRecorder:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

# === transformers.utils.deprecation:deprecate_kwarg ===
try:
    from transformers.utils.deprecation import deprecate_kwarg
except ImportError:
    def deprecate_kwarg(arg=None, *_args, **_kwargs):
        return arg if callable(arg) else (lambda fn: fn)

# === 注意力分发表 ===
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:
    ALL_ATTENTION_FUNCTIONS: dict = {}

# === Kernel hub 装饰器 ===
try:
    from transformers.integrations import use_kernel_forward_from_hub
except ImportError:
    def use_kernel_forward_from_hub(arg=None, *_args, **_kwargs):
        return arg if callable(arg) else (lambda fn: fn)

# === 因果 / 滑窗 mask:必须真有实现,缺则显式报错 ===
try:
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )
except ImportError:
    def create_causal_mask(*_args, **_kwargs):
        raise NotImplementedError("transformers.masking_utils.create_causal_mask not available")

    def create_sliding_window_causal_mask(*_args, **_kwargs):
        raise NotImplementedError("transformers.masking_utils.create_sliding_window_causal_mask not available")

# === Config 工具:layer_type_validation ===
try:
    from transformers.configuration_utils import layer_type_validation
except ImportError:
    def layer_type_validation(*_args, **_kwargs):
        return None


__all__ = [
    "ACT2FN",
    "Cache", "DynamicCache",
    "GenerationMixin",
    "BaseModelOutputWithPast", "CausalLMOutputWithPast",
    "MoeCausalLMOutputWithPast", "MoeModelOutputWithPast",
    "PreTrainedModel", "PretrainedConfig",
    "FlashAttentionKwargs", "logging",
    "ROPE_INIT_FUNCTIONS", "dynamic_rope_update", "rope_config_validation",
    "GradientCheckpointingLayer",
    "GenericForQuestionAnswering", "GenericForSequenceClassification", "GenericForTokenClassification",
    "Unpack", "TransformersKwargs",
    "auto_docstring", "can_return_tuple", "check_model_inputs",
    "OutputRecorder", "deprecate_kwarg",
    "ALL_ATTENTION_FUNCTIONS", "use_kernel_forward_from_hub",
    "create_causal_mask", "create_sliding_window_causal_mask",
    "layer_type_validation",
]
