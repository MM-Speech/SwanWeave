# Spatial Audio Editing — Data Generation

Scripts for generating instruction-based **spatial audio editing** data. Each sample
is a triplet of

- an **original scene**: a set of sound events (source clip, 3D position, onset/offset, gain),
- a **target scene**: the same scene after one (or several) edit operations, and
- **natural-language instructions** describing the edit,

plus **DPO preference pairs** whose rejected instruction describes a plausible but wrong edit.

The scripts produce scene descriptions (JSONL) only; they do **not** render audio. Every
event stores the relative `audio_path` of its source clip, so a downstream spatial
renderer can synthesize the original/target audio.

## Edit operations

| Type | Description |
|---|---|
| `add_event` | Add a new sound event at some position |
| `remove_event` | Remove one event |
| `replace_event` | Replace an event's sound, keeping position/timing/gain (speech ↔ audio in the mixed domain) |
| `volume_change` | Make one event 2× louder or quieter |
| `relocation` / `relocation_plane` | Move an event to a new position (aiming for ≥ 60° azimuth and ≥ 2 m distance change); `_plane` keeps it on the horizontal plane |
| `distance_change` | Move an event nearer (0–2 m) or farther (8–10 m) along the same direction |
| `angle_change` / `angle_change_plane` | Rotate an event around the listener by 60–180° at the same distance |
| `distance_motion` | *Dynamic*: the event approaches or recedes while playing |
| `angle_motion` | *Dynamic*: the event moves along an arc around the listener while playing |
| `volume_fade` | *Dynamic*: the event fades in or out |
| `extract_event` | Keep only one sound class, remove everything else |
| `room_change` | Change the room size (`small` 2×2×2 m, `medium` 6×4×6 m, `large` 21×10×21 m, `huge` 100×100×100 m) |

Multi-operation samples chain 2–4 of the above (all except `extract_event` and `room_change`
by default) on one scene.

## Setup

```bash
pip install -r requirements.txt
```

Python ≥ 3.9. `numpy`, `soundfile` and `tqdm` are required; the other packages are optional
(LLM captions, resampling, plots).

### Source data

Set `DATA_ROOT` in `data_paths.py` to your absolute dataset directory, then arrange the downloaded datasets below it. Dataset, cache and output paths come from `data_paths.py`; JSON configs retain domain, sampling, filtering and caption settings.

```
DATA_ROOT/
├── FSDKaggle2019/                         # https://zenodo.org/records/3612637
│   ├── FSDKaggle2019.audio_train_curated/
│   ├── FSDKaggle2019.audio_test/
│   └── FSDKaggle2019.meta/train_curated_post_competition.csv
├── PicoAudio/                             # https://github.com/zeyuxie29/PicoAudio
│   ├── meta_data/train.json
│   └── data/multi_event_train/*.wav
└── LibriSpeech/LibriSpeech/               # https://www.openslr.org/12  (speech_audio domain only)
    ├── SPEAKERS.TXT
    ├── train-clean-100/
    └── train-clean-360/
```

PicoAudio clips contain several events each; slice the annotated time intervals first. This performs time slicing, not source separation:

```bash
python scripts/slice_picoaudio.py
# -> DATA_ROOT/PicoAudio/sliced/*.flac + sliced_labels.csv
```

This repository does not redistribute any audio. Please follow the license terms of each
source dataset.

## Usage

Run the examples from `SwanWeave/utils/spatial_audio_editing`. Data paths are independent of the code directory when `DATA_ROOT` is absolute. Explicit command-line path overrides take precedence.

Two domains are provided:

- `configs/audio_only.json`: sound events from FSDKaggle2019 + PicoAudio
- `configs/speech_audio.json`: the same plus LibriSpeech utterances (`Female_speech` / `Male_speech`);
  every scene contains at least one speech and one non-speech event

```bash
# 1. Single-operation scenes -> DATA_ROOT/metadata/<domain>/<edit_type>.jsonl
python pipeline.py --config configs/audio_only.json
python pipeline.py --config configs/speech_audio.json --samples-per-type 1000 --num-workers 16

# Test scenes (different seed) -> DATA_ROOT/metadata/test_set/<domain>/
python pipeline.py --config configs/audio_only.json --test-set --test-size 50

# 2. Multi-operation scenes -> DATA_ROOT/metadata/<domain>_multi/
# Independent of the single-operation JSONL files; reuses the source index/cache.
python generate_multi_operations.py --config configs/audio_only.json --counts 2:200000,3:100000,4:50000

# Multi-operation test scenes
python generate_multi_operations.py --config configs/audio_only.json --test-set --counts 2:50,3:50,4:50

# 3. DPO negative scenes -> DATA_ROOT/dpo_scenes/<domain>/<split>/
python generate_dpo_negatives.py --domain audio_only
python generate_dpo_negatives.py --domain audio_only --multi
python generate_dpo_negatives.py --domain audio_only --split test
python generate_dpo_negatives.py --domain audio_only --split test --multi

# 4. (optional) Dataset statistics -> DATA_ROOT/figures/
python scripts/plot_dataset_figures.py
```

Each command reuses one main-process progress bar across its stages. Pool workers do not display their own bars.

Single- and multi-operation generation is **resumable**: re-running a command skips sample ids already present in the
output directory. Each sample is seeded from `(generation.seed, edit_type, sample index)`,
so results do not depend on the number of workers. Different test seeds do not create a raw-clip-disjoint split; that requires separately partitioned source data.

### Duration cache

Before generating, all clips are scanned once: clips that are missing, near-silent
(RMS < `generation.rms_min`) or too short are dropped, and the durations are cached in
the configured `DURATION_CACHE_FILES[domain][split]`
(`DATA_ROOT/duration_cache/<domain>/<split>.json.gz`). If the cache file exists it
is used as-is, including its list of clips per class, so **delete it after changing the
source data or `exclude_audio_classes`**.

### Captions

`api.caption_mode` selects how instructions are written:

- `"rule"` (default): 3 instructions sampled from per-edit-type templates; no network access.
- `"api"`: an LLM writes 3 instructions from a structured description of the scene and edit.
  Choose a profile with `api.active`; the key is read from `ZHIPUAI_API_KEY`,
  `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` depending on the provider. An OpenAI-compatible
  endpoint can be set with `"base_url"` in the profile.

```bash
export DEEPSEEK_API_KEY=...
# then set "caption_mode": "api" and "active": "deepseek" in the config
```

## Output format

Coordinates are in meters; the listener is at the origin facing −z
(x: left − / right +, y: down − / up +, z: front − / back +).

**Single-operation sample** (`DATA_ROOT/metadata/<domain>/<edit_type>.jsonl`, one JSON object per line):

```jsonc
{
  "sample_id": "00000042",
  "edit_type": "relocation",
  "duration": 9.3,                       // scene length (s)
  "room": "large",
  "room_dims": {"x": 21, "y": 10, "z": 21},
  "captions": ["Move the Doorbell from ... to ...", "...", "..."],
  "original_scene": [ /* events */ ],
  "target_scene":   [ /* events */ ],
  "edit_info": {"type": "relocation", "event": {...}, "old_position": {...}, "new_position": {...}},
  "original_text": "Doorbell in directly in front, nearby; ...",   // compact scene description
  "target_text": "...",
  "target_room": "small", "target_room_dims": {...}               // room_change only
}
```

Each event:

```jsonc
{
  "event_id": 0,
  "class": "Doorbell",
  "audio_path": "FSDKaggle2019.audio_train_curated/0a1b2c3d.wav",  // or pico_sliced/..., LibriSpeech/...
  "audio_duration": 4.12,
  "position": {"x": 1.2, "y": 0.0, "z": -3.4},
  "start_time": 0.85,
  "end_time": 4.97,
  "gain": 1.08,
  "domain": "audio",                     // "speech" events also have "transcript" and "speaker_id"
}
```

For dynamic edits, `target_scene` stores the **end** state; `edit_info` contains the start
and end values (`start_position`/`end_position` or `start_gain`/`end_gain`) to interpolate
over the event.

- `<edit_type>_text.jsonl`: only `sample_id`, `edit_type`, `original_text`, `target_text`.
- **Multi-operation samples** additionally have `operation_count` and `edit_types`, and
  `edit_info` is a list with one entry per step.
- **DPO negatives** (`<edit_type>_negative.jsonl`) copy the positive sample's fields and add
  `source_sample_id`, `error_mode` (e.g. `wrong_event`, `wrong_direction`, `no_operation`,
  `omit_operation`), `prompt` (= `original_text`), `chosen`, `rejected`, and the result of
  the wrong edit: `negative_scene`, `negative_text`, `negative_edit_info`.

## Repository layout

```
pipeline.py                    single-operation generation
generate_multi_operations.py   multi-operation generation
generate_dpo_negatives.py      DPO negative generation
configs/                       domain configs
data_paths.py                  shared raw data, cache, metadata and audio paths
cache/                         historical caches; not used by default
src/
  assets.py                    config / seeding / clip indexing shared by the generators
  loader.py                    FSDKaggle2019, PicoAudio and LibriSpeech indexing
  scene.py                     scene sampling and edit operations
  caption.py                   rule-based and LLM captions, scene-to-text
  writer.py                    JSONL output and resume support
scripts/
  slice_picoaudio.py           split PicoAudio multi-event clips
  plot_dataset_figures.py      dataset statistics figures
```

## License

The code in this repository is released under the [Apache License 2.0](LICENSE).
The source datasets are distributed by their authors under their own licenses:
FSDKaggle2019 is released under CC BY 4.0, but each clip keeps its Freesound license
(CC0, CC BY, CC BY-NC or Sampling+; listed in `train_curated_post_competition.csv`), so some
clips do not allow commercial use; LibriSpeech is CC BY 4.0; for PicoAudio see its repository.
Generated data that references these clips is subject to the same terms.
