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
    --config egs/spat_audio_edit.yaml \
    --exp_name swanweave_train \
    --hparams "max_sentences=8,max_tokens=2000,ds_workers=2"
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

TODO TASK...

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
