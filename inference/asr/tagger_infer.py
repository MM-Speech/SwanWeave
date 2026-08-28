import os
import tempfile
from copy import deepcopy
import json
from pathlib import Path
import traceback

import torch
import soundfile as sf
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm

from utils.commons.os_utils import kill_void
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams
from utils.audio.transform import batch_resample
from utils.commons.dataset_utils import collate_xd, batch_by_size
from utils.nn.seq_utils import sequence_mask

# def get_mel(hparams, wav):
#     if isinstance(wav, str):
#         wav, sr = librosa.core.load(wav, sr=hparams['audio_sample_rate'])
#         ws = hparams['win_size']
#         if len(wav) % ws < ws - 1:
#             wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0)
#     h, w, m = hparams['acous_params'][-1]
#     if hparams.get('use_stft_spec', False):
#         wav2spec_dict = librosa_wav2linearspec(
#             wav,
#             fft_size=hparams['fft_size'],
#             hop_size=h,
#             win_length=hparams['win_size'],
#             num_mels=m,
#             fmin=hparams['fmin'],
#             fmax=hparams['fmax'],
#             sample_rate=hparams['audio_sample_rate'],
#             center=False)
#         mel = wav2spec_dict['linear']
#     else:
#         wav2spec_dict = librosa_wav2spec(
#             wav,
#             fft_size=hparams['fft_size'],
#             hop_size=h,
#             win_length=hparams['win_size'],
#             num_mels=m,
#             fmin=hparams['fmin'],
#             fmax=hparams['fmax'],
#             sample_rate=hparams['audio_sample_rate'],
#             center=False)
#         mel = wav2spec_dict['mel']
#     # if hparams.get('reduce_transient_noise'):
#     #     from utils.audio.noise_reduction import reduce_transient_noise
#     #     mel = reduce_transient_noise(mel)
#     wav = wav2spec_dict['wav']
#     return mel, wav

class TaggerInfer():
    def __init__(self, device, ckpt, torch_compile=False, precision=torch.bfloat16):
        self.device = device
        self.precision = precision
        self.build_model(ckpt, torch_compile)

    def build_model(self, ckpt, torch_compile=False):
        self.hparams = set_hparams(config=os.path.join(ckpt, 'config.yaml'), print_hparams=True, global_hparams=False)
        self.hparams['audio_encoder_ckpt'] = 'checkpoints/wavlm/WavLM-Large.pt'
        from modules.asr.tagger.model import build_tagger_model
        self.model = build_tagger_model(self.hparams, init_pretrained=True)
        self.resamplers = {}
        self.model.eval()
        load_ckpt(self.model, ckpt, 'model', strict=True, mmap=True)
        self.model.to(self.device)
        if torch_compile:
            self.model = torch.compile(self.model, mode='max-autotune')

    def forward_model(self, wav, wav_mask=None):
        if wav_mask is None:
            wav_mask = torch.ones_like(wav)
        if self.device == 'cuda':
            with torch.autocast(device_type='cuda', dtype=self.precision):
                model_outputs = self.model.inference(wav, wav_mask)
        else:
            model_outputs = self.model.inference(wav, wav_mask)
        age, gender, emotion = model_outputs[0]['age']['pred'], model_outputs[0]['gender']['pred'], model_outputs[0]['emotion']['pred']
        return age, gender, emotion

    @torch.no_grad()
    def forward(self, wav):
        if isinstance(wav, str):
            wav, sr = librosa.load(wav, sr=self.hparams.get('audio_sample_rate', 16000), mono=True)
        wav = torch.from_numpy(wav)[None].to(self.device)
        return self.forward_model(wav)



    def forward_model_batch(self, wavs, max_batch_duration=600, max_batch_size=100):
        wav_lengths = [len(wav) for wav in wavs]
        ordered_idxs = np.argsort(wav_lengths).tolist()
        
        batches = batch_by_size(
            ordered_idxs,
            num_tokens_fn=lambda idx: wav_lengths[idx],
            max_tokens=max_batch_duration * 16000,
            max_sentences=max_batch_size
        )
        
        results = [None] * len(wavs)
        
        for wav_idxs in batches:
            bsz = len(wav_idxs)
            
            wav_batch = collate_xd([torch.from_numpy(wavs[i]) for i in wav_idxs]).to(self.device)
            wav_mask = sequence_mask(torch.LongTensor([wav_lengths[i] for i in wav_idxs])).to(self.device)
            
            try:
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    model_outputs = self.model.inference(wav_batch, wav_mask)
                    batch_ages = [output['age'] for output in model_outputs]
                    batch_genders = [output['gender'] for output in model_outputs]
                    batch_emotions = [output['emotion'] for output in model_outputs]
            except:
                traceback.print_exc()
                continue
            
            for sample_idx, wav_idx in enumerate(wav_idxs):
                results[wav_idx] = {
                    'age': batch_ages[sample_idx]['pred'],
                    'gender': batch_genders[sample_idx]['pred'],
                    'emotion': batch_emotions[sample_idx]['pred']
                }
        return results
    
    @torch.no_grad()
    def forward_batch(self, wavs, sample_rates=16000, 
                      max_batch_duration=600, max_batch_size=100):
        wavs = batch_resample(wavs, sample_rates, tgt_sr=16000, resamplers=self.resamplers, batch_size=4, device=self.device)
        # 将wavs放到对应设备上
        results = self.forward_model_batch(wavs, max_batch_duration=max_batch_duration, max_batch_size=max_batch_size)
        return results
    
    
if __name__ == "__main__":
    import random
    if random.random() < 0.1:
        kill_void()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = '/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/checkpoints/251130_tagger_wavlm'
    tagger_infer = TaggerInfer(device, ckpt)
    
    # test audio
    audio_path = '/mnt/bn/sa-ag-data/panchanghao/data/SeniorTalk/wav_data/sentence/test/00000/Elderly0014S0046W0001.wav'
    
    age_res, gender_res, emotion_res = tagger_infer.forward(audio_path)
    print('Age:', age_res)
    print('Gender:', gender_res)
    print('Emotion:', emotion_res)
    
    # test batch
    audio_path_list = [
        "/mnt/bn/sa-ag-data/panchanghao/data/ChildMandarin/new_data/dev/005/005_5_M_L_HANZHONG_iphone_001_001.wav",
        "/mnt/bn/sa-ag-data/panchanghao/data/ChildMandarin/new_data/dev/005/005_5_M_L_HANZHONG_iphone_001_002.wav",
        "/mnt/bn/sa-ag-data/panchanghao/data/ChildMandarin/new_data/dev/005/005_5_M_L_HANZHONG_iphone_001_003.wav",
        "/mnt/bn/sa-ag-data/panchanghao/data/ChildMandarin/new_data/dev/005/005_5_M_L_HANZHONG_iphone_001_004.wav",
        "/mnt/bn/sa-ag-data/panchanghao/data/ChildMandarin/new_data/dev/005/005_5_M_L_HANZHONG_iphone_001_005.wav",
        "/mnt/bn/sa-ag-data/panchanghao/data/ChildMandarin/new_data/dev/005/005_5_M_L_HANZHONG_iphone_001_006.wav",
        "/mnt/bn/sa-ag-data/panchanghao/data/ChildMandarin/new_data/dev/005/005_5_M_L_HANZHONG_iphone_001_007.wav",
        '/mnt/bn/sa-ag-data/panchanghao/data/SeniorTalk/wav_data/sentence/test/00000/Elderly0014S0046W0001.wav',
        '/mnt/bn/sa-ag-data/panchanghao/data/SeniorTalk/wav_data/sentence/test/00001/Elderly0060S0196W0120.wav',
        '/mnt/bn/sa-ag-data/panchanghao/data/SeniorTalk/wav_data/sentence/test/00001/Elderly0060S0196W0121.wav',
        '/mnt/bn/sa-ag-data/panchanghao/data/voxbox/raw_data/casia/0000/casia_angry_0000000050.m4a',
        '/mnt/bn/sa-ag-data/panchanghao/data/voxbox/raw_data/aishell-3/0000/AISHELL-3_0000000000.m4a',
        '/mnt/bn/sa-ag-data/panchanghao/data/voxbox/raw_data/savee/0000/savee_Angry_0000000005.m4a'
    ]
    
    wavs, srs = [], []
    import time
    for wav_path in audio_path_list:
        wav, sr = librosa.load(wav_path, sr=16000, mono=True)
        wavs.append(wav)
        srs.append(sr)
        
    
    print(f"{len(wavs) = }")
    total_duration = sum([wav.shape[0] / sr for (wav, sr) in zip(wavs, srs)])
    print(f"{total_duration = }, {total_duration / 60}min")
    start = time.time()
    
    # 逐条推理
    single_infer_results = []
    for i in range(len(wavs)):
        res = tagger_infer.forward(wavs[i])
        single_infer_results.append(res)
    end = time.time()
    print(f"Single infer time: {end - start}s, RTF: {total_duration / (end - start)}")
    print("==============================Single infer results===================================")
    for i, audio_path in enumerate(audio_path_list):
        print(f'Audio: {audio_path}')
        print('  Age:', single_infer_results[i][0])
        print('  Gender:', single_infer_results[i][1])
        print('  Emotion:', single_infer_results[i][2])
    print("=====================================================================================")
    # 批量推理
    start = time.time()
    results = tagger_infer.forward_batch(wavs, sample_rates=srs, max_batch_duration=120, max_batch_size=4)
    end = time.time()
    print(f"Batch infer time: {end - start}s, RTF: {total_duration / (end - start)}")
    print("==============================Batch infer results===================================")
    for i, audio_path in enumerate(audio_path_list):
        print(f'Audio: {audio_path}')
        print('  Age:', results[i]['age'])
        print('  Gender:', results[i]['gender'])
        print('  Emotion:', results[i]['emotion'])
    print("=====================================================================================")