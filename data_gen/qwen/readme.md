funasr==1.2.9


qwen3-asr: source /mnt/bn/sa-ag-data/zhangyu.34/.bashrc; conda activate qwen3-asr

conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
pip install -U qwen-asr[vllm]
pip install tenacity silero_vad langdetect
