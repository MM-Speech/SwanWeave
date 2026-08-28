conda create -n swan -c conda-forge python=3.11 #conda配置在zshrc里有
conda activate swan

pip install -U wheel pip
pip install numpy==1.26.4
pip install torch==2.3.1 torchaudio==2.3.1 deepspeed tensorboardX accelerate nvidia-ml-py
pip install transformers==4.56.2
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" #需要代理
pip install flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install python-dotenv simplejson setproctitle attrdictionary pyarrow==15.0.0 pyphen pyloudnorm
pip install byted-dataloader -i https://bytedpypi.byted.org/simple
pip install funasr
pip install tensorboard opencv-python bytedtos tenacity
pip install loguru pyrootutils natsort 
pip install flatten_dict ffmpy rich randomname 
pip install openai-whisper pypinyin torchdiffeq orjson
pip install pedalboard pyvad silero_vad langdetect pydub
