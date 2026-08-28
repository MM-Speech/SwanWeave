# Vevo Token Extractor

`user/vevo/` 提供一套独立的 Vevo token 推理模块，用于提取：

- `content_ids`
- `content_style_ids`

当前版本默认带有两层能力：

- 自动按 `8s` chunk 切分长音频
- 按 chunk 组织 batch 推理，再按原音频拼回结果

默认行为仍对齐 Vevo 原始链路：

- `content_ids`: HuBERT-Large layer 18 -> 32-codebook Vevo tokenizer -> `duration reduction`
- `content_style_ids`: HuBERT-Large layer 18 -> 8192-codebook tokenizer -> 不做 reduction

## Install

```bash
pip install -r user/vevo/requirements.txt
```

## Single Audio

```python
from user.vevo import build_vevo_token_model, run_vevo_token_model

model = build_vevo_token_model()
result = run_vevo_token_model(
    model,
    "input.wav",
    vector_type="both",
    batch_size=4,
)

print(result.content_ids.shape)
print(result.content_style_ids.shape)

result.to_npz("vevo_tokens.npz")
```

`build_vevo_token_model(cache_dir=...)` 和
`VevoTokenExtractor.from_pretrained(cache_dir=...)` 的下载语义如下：

- `cache_dir=None`：沿用 Hugging Face 默认 cache 机制
- `cache_dir="..."`：tokenizer 文件会直接下载到这个目录下，不再使用
  `models--.../snapshots/...` 这类多层 cache 结构

例如：

```python
model = build_vevo_token_model(cache_dir="/data/models/vevo")
```

下载完成后，你可以直接在下面这些路径找到文件：

- `/data/models/vevo/tokenizer/vq32/...`
- `/data/models/vevo/tokenizer/vq8192/...`

这里的 `batch_size` 是 chunk 级 batch 大小，不是原始音频条数。  
如果输入音频超过 `8s`，会先切 chunk，再按 chunk 分 batch 推理。

## Batch Audio

`run_vevo_token_model()` 现在同时支持单条和批量输入。

### `list[str | Path]`

```python
results = run_vevo_token_model(
    model,
    ["a.wav", "b.wav", "c.wav"],
    vector_type="content",
    batch_size=8,
)
```

返回值是 `list[VevoTokenResult]`，顺序与输入音频顺序一致。

### `list[np.ndarray | torch.Tensor]`

```python
results = run_vevo_token_model(
    model,
    [wav_a, wav_b, wav_c],
    sample_rates=[16000, 22050, 16000],
    vector_type="content_style",
    batch_size=8,
)
```

如果这些 waveform 的采样率相同，也可以把 `sample_rates` 写成单个 `int`。

### `[B, T]` ndarray / tensor

```python
results = run_vevo_token_model(
    model,
    batch_wavs,
    sample_rate=16000,
    audio_lengths=[len_a, len_b, len_c],
    vector_type="both",
    batch_size=8,
)
```

这里要求：

- `batch_wavs` 是 `[B, T]`
- `audio_lengths` 提供每条音频的真实长度
- `sample_rate` 是整个 batch 共享的采样率

如果传入二维 waveform 但没有 `audio_lengths`，会维持旧行为，按单条多声道音频处理。

## API

- `build_vevo_token_model(...) -> VevoTokenExtractor`
- `run_vevo_token_model(model, audio, ...) -> VevoTokenResult | list[VevoTokenResult]`
- `VevoTokenExtractor.from_pretrained(...)`
- `VevoTokenExtractor.extract_batch(audio, ...) -> list[VevoTokenResult]`
- `VevoTokenExtractor.extract_from_path(audio_path, ...) -> VevoTokenResult`
- `VevoTokenExtractor.extract_from_waveform(waveform, sample_rate, ...) -> VevoTokenResult`
- `VevoTokenExtractor.extract_content_ids(audio, sample_rate=None, ...) -> np.ndarray`
- `VevoTokenExtractor.extract_content_style_ids(audio, sample_rate=None, ...) -> np.ndarray`

`VevoTokenResult` 包含：

- `content_ids`
- `content_style_ids`
- `source_sample_rate`
- `vector_type`
- `content_reduced`
- `content_style_reduced`
- `token_sample_rate`

并提供 `to_npz(path)` 用于直接落盘。

## On-demand Extraction

如果你只想跑一条分支，可以直接指定 `vector_type`：

```python
content_only = run_vevo_token_model(model, "input.wav", vector_type="content")
print(content_only.content_ids.shape)
print(content_only.content_style_ids)  # None

content_style_only = run_vevo_token_model(
    model,
    "input.wav",
    vector_type="content_style",
)
print(content_style_only.content_ids)  # None
print(content_style_only.content_style_ids.shape)
```

类接口里的 `extract_content_ids()` 和 `extract_content_style_ids()` 也仍然是按需执行。

## Chunk Alignment

长音频 chunk 化时，内部规则固定为：

- chunk 核心长度是 `8s @ 16k = 128000` samples
- 非最后一个整 chunk 会额外借用后面最多 `80` 个 `16k` 采样点做 lookahead
- 最后一个 chunk 会在右侧做最小量补零，通常是 `0-5ms`
- 最终输出时会按每个 chunk 的目标 token 长度截断，所以 batch padding 和尾部补零不会泄露到最终 token 序列

`content_ids` 如果启用 reduction，会在整条音频拼回后再做一次全局 reduction，因此 chunk 边界上的重复 token 也会被一并消掉。
