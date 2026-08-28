# SwanWeave: One-Stage Multi-Task Instruction-Guided 3D Spatial Audio Editing

SwanWeave supports multiple spatial audio editing tasks through natural-language instructions, including sound event manipulation and spatial motion editing.

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

Download model checkpoint: `hf download BrokenMoon/SwanWeave dit/ --local-dir checkpoints`

### Stable Audio VAE

Download the VAE checkpoint from: [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0/tree/main/vae)

Place the downloaded files under: `checkpoints/vae/`

### Text Encoder

Download the Qwen3-0.6B checkpoint from: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

Place the downloaded files under: `checkpoints/Qwen3-0.6B/`

The checkpoint directory should be organized approximately as follows:

```text
checkpoints/
├── dit/
├── vae/
└── Qwen3-0.6B/
```

## Inference

SwanWeave performs spatial audio editing based on a source audio file and a natural-language editing instruction.

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