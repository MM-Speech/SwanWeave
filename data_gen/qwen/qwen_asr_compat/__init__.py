"""Minimal local Qwen3-ASR forced-aligner compatibility layer.

This package vendors only the Qwen3 forced-alignment path needed by
``data_gen.qwen.qwen3aligner``. It intentionally avoids a ``utils.py`` module
name to prevent ambiguity with this repository's root ``utils`` package.
"""

from .audio_io import AudioLike, ensure_list, normalize_audios

__all__ = [
    "AudioLike",
    "Qwen3ForcedAligner",
    "ensure_list",
    "normalize_audios",
]


def __getattr__(name):
    if name == "Qwen3ForcedAligner":
        from .forced_aligner import Qwen3ForcedAligner

        return Qwen3ForcedAligner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
