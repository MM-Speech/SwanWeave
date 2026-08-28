# One-Stage Multi-Task Instruction-Guided 3D Spatial Audio Editing

git clone https://github.com/MM-Speech/SwanWeave.git

cd SwanWeave

conda create -n swanweave python=3.11.15

pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.23.5



权重下载：
下载[vae](https://huggingface.co/stabilityai/stable-audio-open-1.0/tree/main/vae)，放置在checkpoints/vae

下载[text encoder](https://huggingface.co/Qwen/Qwen3-0.6B/tree/main)，放置在checkpoints/Qwen3-0.6B


运行示例：
python inference/spatial/spat_edit_infer.py --dit_ckpt checkpoints/dit --src_wav assets/origin.wav --caption "Add the Bicycle bell sound directly in right." --out_path output/add_event.wav

python inference/spatial/spat_edit_infer.py --dit_ckpt checkpoints/dit --src_wav assets/origin.wav --caption "Make the Screaming swing around from behind on the right to directly in front." --out_path output/angle_motion.wav