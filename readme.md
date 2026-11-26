# SwanWeave: One-Stage Multi-Task Instruction-Guided 3D Spatial Audio Editing

SwanWeave supports multiple spatial audio editing tasks through natural-language instructions, including sound event manipulation and spatial motion editing.
This work edits First-Order Ambisonics (FOA), a four-channel spatial audio representation stored using ACN channel ordering (W, Y, Z, X).

## Installation

Clone the repository:

```bash
git clone https://github.com/MM-Speech/SwanWeave.git
cd SwanWeave
```

Create a Conda environment:

```bash
conda create -n swanweave python=3.11.15
conda activate swanweave

pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.23.5
```

## Model Checkpoints

Before running inference, download the required pretrained models.

Download model checkpoint: `hf download BrokenMoon/SwanWeave dit/ moe/ --local-dir checkpoints`

### Stable Audio VAE

Download the VAE checkpoint from: [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0/tree/main/vae)

Place the downloaded files under: `checkpoints/vae/`

### Text Encoder

Download the Qwen3-0.6B checkpoint from: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

Place the downloaded files under: `checkpoints/Qwen3-0.6B/`

The checkpoint directory should be organized as follows:

```text
checkpoints/
├── dit/
├── moe/
├── vae/
└── Qwen3-0.6B/
```

## Inference

SwanWeave performs spatial audio editing based on a source audio file and a natural-language editing instruction. 

** Note **: Playing FOA audio directly may not accurately convey the directions of sound sources. You can download the [IEM Plug-in Suite](https://plugins.iem.at/) and use it in Adobe Audition, or convert the audio to stereo using `utils/audiotools/foa2stereo.py`.

### Edit Examples

```bash
python inference/spatial/spat_edit_infer.py \
    --dit_ckpt checkpoints/dit \
    --src_wav assets/origin.wav \
    --caption "Add the Bicycle bell sound directly in right." \
    --out_path output/add_event.wav
```

```bash
python inference/spatial/spat_edit_infer.py \
    --dit_ckpt checkpoints/dit \
    --src_wav assets/origin.wav \
    --caption "Make the Screaming swing around from behind on the right to directly in front." \
    --out_path output/angle_motion.wav
```

## Training

SwanWeave can be trained on simulated edit pairs.

### Training Instructions

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tasks/run.py \
  --config egs/spat_audio_gen.yaml \
  --exp_name swanweave_dense_gen
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tasks/run.py \
    --config egs/spat_audio_edit.yaml \
    --exp_name swanweave_train \
    --hparams "load_ckpt=checkpoints/swanweave_dense_gen,max_sentences=8,max_tokens=2000,ds_workers=2"
```

### Preference training (DPO)

DPO starts a new experiment from a trained editor. The policy and a frozen
reference start with the same weights; only the policy is optimized. The loss
combines a preference objective with the positive target's flow-matching loss.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tasks/run.py \
    --config egs/spat_audio_edit_dpo.yaml \
    --exp_name swanweave_dpo_dense \
    --hparams "load_ckpt=checkpoints/swanweave_train,max_sentences=8,max_tokens=2000,ds_workers=2"
```

## Dataset Construction

The pipeline builds two domains: `audio_only` uses FSDKaggle2019 and PicoAudio,
while `speech_audio` additionally uses LibriSpeech. Run the following commands
from the SwanWeave repository root.

### Quick Start

After installation, extract the source datasets and set `DATA_ROOT` in
`utils/spatial_audio_editing/data_paths.py` as described below. Place the Stable
Audio VAE under `checkpoints/vae/` before running the encoding stage.

All data-construction commands are combined in [prepare_data.sh](prepare_data.sh):

```bash
bash prepare_data.sh
```

The default is a small run: both domains, train/test, 20 samples per single edit
type and 20 samples per 2/3/4-operation group, followed by DPO generation, FOA
rendering, and VAE mean/std extraction. It uses 8 CPU workers; VAE encoding uses
CUDA when available. Captions use the supplied rule-based configurations.

Increase the **training** target counts and worker count with environment variables:

```bash
NUM_WORKERS=32 SAMPLES_PER_TYPE=50000 \
MULTI_COUNTS=2:200000,3:100000,4:50000 bash prepare_data.sh
```

Test targets stay at 20 unless `TEST_SAMPLES_PER_TYPE` and `TEST_MULTI_COUNTS`
are also changed. Counts are target totals, not additional samples.

| Stage | Processing |
|---|---|
| 1 | Slice PicoAudio into FLAC clips and labels |
| 2 | Generate single/multi-operation train/test scenes |
| 3 | Generate all DPO negatives (`--max 0`) |
| 4 | Render FOA and export `gen_*`, `edit_*`, and `dpo_*` JSONL lists |
| 5 | Save VAE posterior mean/std beside the referenced FOA files |

Run only selected stages, or preview the commands without processing data:

```bash
bash prepare_data.sh --dry-run
STAGE=4 STOP_STAGE=4 NUM_WORKERS=32 bash prepare_data.sh
STAGE=4 STOP_STAGE=4 EXPORT_ONLY=1 bash prepare_data.sh
STAGE=5 CUDA_VISIBLE_DEVICES=0 bash prepare_data.sh
```

`DOMAINS="audio_only"` and `SPLITS="train"` restrict scene generation and rendering.
Stage 5 processes the available six aggregate lists. Set `PYTHON` to a Python
executable path, `VAE_DEVICE` to select an encoding device, or `OVERWRITE_VAE=1`
to regenerate posteriors after changing audio or VAE settings.
See `bash prepare_data.sh --help` for all options.

Training scene generation resumes existing IDs. The existing `--test-set` mode
rebuilds test JSONL files; stage 1 rewrites PicoAudio slices/labels and stage 3
rebuilds negative metadata. After changing generation rules, rebuild the affected
scene files as explained below. Invalid render records are excluded and recorded
in `*.errors.jsonl`; a completed render batch continues with its valid records.
Other command failures stop the script.

### 1. Prepare the source datasets

Download and extract the source datasets, then set `DATA_ROOT` in `utils/spatial_audio_editing/data_paths.py`. The data directory is independent of the repository location and holds both source data and generated outputs. The default source layout is:

```text
DATA_ROOT/
├── FSDKaggle2019/
│   ├── FSDKaggle2019.audio_train_curated/
│   └── FSDKaggle2019.meta/
│       └── train_curated_post_competition.csv
├── PicoAudio/
│   ├── data/multi_event_train/
│   └── meta_data/train.json
└── LibriSpeech/LibriSpeech/
    ├── SPEAKERS.TXT
    ├── train-clean-100/
    └── train-clean-360/
```

LibriSpeech is only needed for `speech_audio`. Keep the original source file
formats; generated PicoAudio slices and rendered FOA audio are saved as FLAC.

First, slice PicoAudio using its event timestamps.

```bash
python utils/spatial_audio_editing/scripts/slice_picoaudio.py
```

The output is `DATA_ROOT/PicoAudio/sliced/*.flac` and `DATA_ROOT/PicoAudio/sliced_labels.csv`.

### 2. Generate scene metadata

Single-operation scenes:

```bash
python utils/spatial_audio_editing/pipeline.py \
    --config utils/spatial_audio_editing/configs/audio_only.json \
    --samples-per-type 20 --num-workers 8

python utils/spatial_audio_editing/pipeline.py \
    --config utils/spatial_audio_editing/configs/speech_audio.json \
    --samples-per-type 20 --num-workers 8
```

Multi-operation scenes:

```bash
python utils/spatial_audio_editing/generate_multi_operations.py \
    --config utils/spatial_audio_editing/configs/audio_only.json \
    --counts 2:20,3:20,4:20 --num-workers 8

python utils/spatial_audio_editing/generate_multi_operations.py \
    --config utils/spatial_audio_editing/configs/speech_audio.json \
    --counts 2:20,3:20,4:20 --num-workers 8
```

Training scenes are saved under `DATA_ROOT/metadata/audio_only/`,
`speech_audio/`, `audio_only_multi/`, and `speech_audio_multi/`. The supplied
configs use template captions and do not require an LLM API key.

To generate test scenes, repeat the four commands above with `--test-set`,
keeping the explicit sample counts. Test scenes are saved under
`DATA_ROOT/metadata/test_set/`.

### 3. Generate DPO negatives

Generate negatives for both domains, using all available single- and
multi-operation training scenes:

```bash
python utils/spatial_audio_editing/generate_dpo_negatives.py \
    --split train --max 0

python utils/spatial_audio_editing/generate_dpo_negatives.py \
    --split train --multi --max 0
```

Repeat both commands with `--split test` for test data. `--max 0` processes
every positive record; omitting it limits generation to 10 records per input
file. DPO scene metadata is written to
`DATA_ROOT/dpo_scenes/<domain>/<train|test>/`.

### 4. Render FOA audio and export training lists

Acoustic settings and the default worker count are configured in
`utils/pyroom_sim/render_config.py`. Render the available scenes with:

```bash
python utils/pyroom_sim/sim_pairs_pyroom.py --num-workers 8
```

The renderer automatically exports six JSONL lists under
`DATA_ROOT/rendered_metadata/`, with absolute audio paths:

| Lists | Training task |
|---|---|
| `gen_train.jsonl` / `gen_test.jsonl` | Two generation records per edit pair: source audio with its scene caption, and target audio with its scene caption. |
| `edit_train.jsonl` / `edit_test.jsonl` | Source/target audio with one editing instruction per row; instruction variants share the same audio pair. |
| `dpo_train.jsonl` / `dpo_test.jsonl` | Source, positive, and negative audio with the correct editing instruction. |

To rebuild these lists from existing rendered metadata without rendering again:

```bash
python utils/pyroom_sim/sim_pairs_pyroom.py --export-only
```

### 5. Precompute VAE posteriors

Place the Stable Audio VAE files under `checkpoints/vae/` as described above,
then extract posteriors:

```bash
CUDA_VISIBLE_DEVICES=0 python utils/pyroom_sim/precompute_vae.py
```

Existing readable posteriors are skipped. After changing the audio, VAE weights,
or preprocessing settings, regenerate them with `--overwrite`:

```bash
CUDA_VISIBLE_DEVICES=0 python utils/pyroom_sim/precompute_vae.py --overwrite
```

Scene generation resumes existing training sample IDs. After changing scene
generation rules, rebuild the affected scene metadata before rendering and
recomputing posteriors. Multi-operation generation supports `--overwrite`;
for single-operation training scenes, remove the affected scene and text JSONL
files before regenerating them. Finally, set the literal training-list paths in
`egs/datasets/spat_gen_dataset.yaml`, `spat_edit_dataset.yaml`, and
`spat_edit_dpo_dataset.yaml` to the corresponding files under
`DATA_ROOT/rendered_metadata/`. The training order is
`gen_train → edit_train → dpo_train`.

## Citation

```bibtex
@inproceedings{lei2026one,
  title={One-Stage Multi-Task Instruction-Guided 3D Spatial Audio Editing},
  author={Lei, Ke and Wen, Chenyuhao and Zhang, Yu and Guo, Wenxiang and Pan, Changhao and Zhou, Sashuai and Li, Yongshi and Li, Ruiqi and Hu, Ruofan and Xu, Haorui and others},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  pages={20355--20369},
  year={2026}
}
```
