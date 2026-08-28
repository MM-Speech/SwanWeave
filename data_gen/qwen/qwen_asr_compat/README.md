# Qwen3-ASR Forced Aligner Vendor

This directory vendors the minimal Qwen3-ASR forced-aligner code used by
`data_gen.qwen.qwen3aligner`.

Source: `QwenLM/Qwen3-ASR` main branch.
License: Apache-2.0, matching the upstream file headers.

Only the transformers forced-aligner path is included. CLI, demo, ASR wrapper,
streaming, and vLLM backend code are intentionally not vendored.

Local naming note: upstream `qwen_asr/inference/utils.py` is stored here as
`audio_io.py` to avoid ambiguity with this repository's root `utils` package.
