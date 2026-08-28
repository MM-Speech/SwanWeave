import os
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import sys
import json
import re
from typing import List, Union, Dict
import argparse
import librosa
import numpy as np
import torch
import io
import threading
import traceback
import torch.nn.functional as F
import whisper
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import subprocess
from pathlib import Path
import time

from copy import deepcopy
from langdetect import detect as classify_language, LangDetectException
from pydub import AudioSegment
import pyloudnorm as pyln
from tqdm import tqdm
# from tn.chinese.normalizer import Normalizer as ZhNormalizer
# from tn.english.normalizer import Normalizer as EnNormalizer

from utils.audio.align import mel2token_to_dur
from utils.audio.io import save_wav_bytes, to_wav_bytes, wav_bytes_to_mp3_bytes
from utils.text import is_english, YUNMU_ERHUA, SHENGMU
from utils.text.text_encoder import TokenTextEncoder
from utils.text.split_text import chunk_text_english, chunk_text_chinese, get_word_list, remove_space, remove_unprintable
from utils.text.ph_tone_convert import split_ph_timestamp, split_ph, map_phone_to_tokendict
from utils.text.ssml_utils import SSML
from utils.text.ph_alignment import align_word_phone, print_align, merge_norm_alignment
from utils.text.split_text import chunk_text_chinese, chunk_text_english, chunk_text_chinese_v2
from utils.commons.ckpt_utils import load_ckpt, get_all_ckpt_steps
from utils.commons.hparams import set_hparams, hparams
from utils.commons.meters import Timer
from utils.commons.os_utils import handle_exacption, kill_void
from utils.commons.io import print_once, json_dumps, get_wav_duration
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.commons.tensor_utils import move_to_cpu, move_to_cuda

from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixinV8, DiTBuildModelMixin
from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import build_spk_mask_from_text_tokens, _get_sx_token_patterns, augment_text_with_pinyin_s1s2_safe

DEBUG = False

_SPEAKER_TAG_BLOCK_RE = re.compile(r"<S([1-4])>(.*?)</S\1>", flags=re.IGNORECASE | re.DOTALL)

if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def convert_to_wav(wav_path):
    # Check if the file exists
    if not os.path.exists(wav_path):
        print(f"The file '{wav_path}' does not exist.")
        return

    # Check if the file already has a .wav extension
    if not wav_path.endswith(".wav"):
        # Define the output path with a .wav extension
        out_path = os.path.splitext(wav_path)[0] + ".wav"

        # Load the audio file using pydub and convert it to WAV
        audio = AudioSegment.from_file(wav_path)
        audio.export(out_path, format="wav")

        print(f"Converted '{wav_path}' to '{out_path}'")


def convert_to_wav_bytes(audio_binary):
    # Load the audio binary using pydub and convert it to WAV
    audio = AudioSegment.from_file(io.BytesIO(audio_binary))
    wav_bytes = io.BytesIO()
    audio.export(wav_bytes, format="wav")
    wav_bytes.seek(0)
    return wav_bytes

@contextmanager
def model_lock(lock):
    try:
        lock.acquire()
        yield
    finally:
        torch.cuda.synchronize()
        lock.release()
        
        
def move_to_shm(exp_name):
    os.makedirs('/dev/shm/mega_ckpt', exist_ok=True)
    shm_exp_name = f"/dev/shm/mega_ckpt/{Path(exp_name).stem}"
    if os.path.exists(shm_exp_name):
        return shm_exp_name
    subprocess.check_call(f"cp -r {exp_name} /dev/shm/mega_ckpt/", shell=True)
    return shm_exp_name


@dataclass
class MegaTTS3Output:
    wav_bytes: bytes = None
    wav: np.ndarray = None
    duration: float = None


class MegaTTS3DiTV8Infer(DiTBuildModelMixinV8):
    def __init__(
            self, 
            device=None,
            dit_exp_name='checkpoints/251204_megatts3_dit_v6_large',
            dur_exp_name='checkpoints/251128_dur_pred',
            wavvae_exp_name='checkpoints/1231_megatts3_wavvae_v3_25hz_kl001_fix4',
            max_ref_duration=20,
            use_tqdm=True,
            use_ema=False,
            precision='bf16',
            **kwargs
    ):

        if precision == 'fp16':
            self.precision = torch.float16
        elif precision == 'bf16':
            self.precision = torch.bfloat16

        self.sr = 24000
        self.fm = 8
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.use_tqdm = use_tqdm
        self.use_ema = use_ema

        self.dit_exp_name = dit_exp_name
        self.dur_exp_name = dur_exp_name
        self.wavvae_exp_name = wavvae_exp_name

        self.build_model(self.device)

        # init text normalizer
        # self.zh_normalizer = ZhNormalizer(overwrite_cache=True, remove_erhua=False, remove_interjections=False, full_to_half=False)
        # self.en_normalizer = EnNormalizer(overwrite_cache=True)
        self.ph_replace_table = {}

        # break (silence)
        self.max_silence_alive = 1.28    # 1.28s
        self.max_ref_duration = max_ref_duration
        self.chunk_num_words_zh = 60
        self.chunk_num_words_en = 130

    def build_model(self, device):
        self.device = device

        if self.dit_exp_name.endswith('.ckpt'):
            set_hparams(f'{Path(self.dit_exp_name).parent}/config.yaml', print_hparams=False)
        else:
            set_hparams(f'{self.dit_exp_name}/config.yaml', print_hparams=False)
        hparams['use_fsdp'] = False

        ''' Load DiT '''
        self._build_model()
        if self.use_ema and hparams.get('use_ema', False):
            load_ckpt(self.dit, self.dit_exp_name, 'ema_model', strict=False, mmap=True)
        else:
            load_ckpt(self.dit, self.dit_exp_name, 'dit', strict=False, mmap=True)
        self.vae.eval()
        self.vae.to(self.device, dtype=self.precision)
        self.dit.eval()
        self.dit.to(device, dtype=self.precision)
        
        ''' Load Duration Model '''
        hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
        self.dur_model = self.build_duration_predictor(hp_dur_model, init_pretrained=False)
        load_ckpt(self.dur_model, self.dur_exp_name, 'dur_predictor', strict=True, mmap=True)
        self.dur_model.eval()
        self.dur_model.to(self.device, dtype=self.precision)

        ''' ASR '''
        from modules.asr.sensevoice.sensevoice_api import build_asr_model
        self.asr_model = build_asr_model(device)

        ''' VAD '''
        from silero_vad import load_silero_vad
        self.vad_model = load_silero_vad()

        self.vae = torch.compile(self.vae, mode='max-autotune')
        self.dit = torch.compile(self.dit, mode='max-autotune')
        self.dur_model = torch.compile(self.dur_model, mode='max-autotune')
        self.lock = threading.Lock()

    def trim_silence_vad(self, wav_16k=None, speech_timestamps=None, max_duration=60, max_silence=1.28):
        if speech_timestamps is None and wav_16k is not None:
            from silero_vad import load_silero_vad, get_speech_timestamps
            wav_16k = wav_16k[:int(16000 * max_duration * 1.2)]
            speech_timestamps = get_speech_timestamps(
                wav_16k,
                self.vad_model,
                return_seconds=True,  # Return speech timestamps in seconds (default is samples)
            )
        from utils.audio.diarization_utils import merge_segments
        segments = merge_segments(speech_timestamps, max_silence=max_silence, max_voiced_duration=max_duration)
        cur_duration = 0
        for i in range(len(segments)):
            if (cur_duration + segments[i]['end'] - segments[i]['start'] > max_duration) and cur_duration > 0:
                return segments[:i]
            cur_duration += segments[i]['end'] - segments[i]['start']
        return segments

    def preprocess(self, audio_bytes, wav_path=None, topk_dur=1, **kwargs):
        wav_bytes = convert_to_wav_bytes(audio_bytes)

        device = self.device

        wav, _ = librosa.load(wav_bytes, sr=self.sr)
        ws = hparams['win_size']
        if len(wav) % ws < ws - 1:
            wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0).astype(np.float32)
        wav = np.pad(wav, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)
        wav_16k = librosa.resample(wav, orig_sr=self.sr, target_sr=16000)
        max_ref_duration = self.max_ref_duration

        segments = self.trim_silence_vad(wav_16k, max_duration=max_ref_duration)
        wav = np.concatenate([wav[int(segment['start'] * self.sr): int(segment['end'] * self.sr)] for segment in segments])
        wav_16k = np.concatenate([wav_16k[int(segment['start'] * 16000): int(segment['end'] * 16000)] for segment in segments])
        # remove gap leak
        wav = np.concatenate([wav, np.zeros(int(self.sr * 0.2))])
        wav_16k = np.concatenate([wav_16k, np.zeros(int(16000 * 0.2))])

        with torch.inference_mode():
            ''' Forward WavVAE to obtain: prompt latent '''
            wav = torch.tensor(wav, dtype=self.precision)[None].to(device)
            with torch.autocast(device_type='cuda', dtype=self.precision):
                vae_latent = self.vae.encode_latent(wav)

            ''' Duration Prompting '''
            with torch.autocast(device_type='cuda', dtype=self.precision):
                duration_ctx = self.dur_model.encode(
                    torch.from_numpy(wav_16k)[None].to(device, self.precision), 
                    wav_mask=torch.ones((1, wav_16k.shape[0]), dtype=torch.bool, device=device)
                )

            ''' ASR '''
            from modules.asr.sensevoice.sensevoice_api import run_asr_model
            text = run_asr_model([wav_16k], self.asr_model, with_segments=False)[0]['text_normed']

        ret = {
            'text_ref': text,
            'vae_latent': vae_latent.cpu(),
            'duration_ctx': move_to_cpu(duration_ctx)
        }

        return ret
    
    def preprocess_text(self, input_text: SSML, ph_replace_table=None, chunk_num_words_zh=60, chunk_num_words_en=130):
        
        # return [input_text]
        
        def _normalize_text_en(text: str):
            text_norm = common_preprocess(text)
            text_norm = self.en_normalizer.normalize(text_norm)
            if ph_replace_table is not None:
                for src, tgt in ph_replace_table['en'].items():
                    text_norm = text_norm.replace(src, tgt)
            text_norm = common_process(text_norm)
            # text_norm = common_postprocess(text_norm)
            return text_norm

        def _normalize_text_zh(text):
            text_norm = common_preprocess(text)

            from opencc import OpenCC
            jp2t_converter = OpenCC('jp2t')
            t2s_converter = OpenCC('t2s')
            text_norm = t2s_converter.convert(jp2t_converter.convert(text_norm))
            
            text_norm = self.zh_normalizer.normalize(text_norm)

            if ph_replace_table is not None:
                for src, tgt in ph_replace_table['zh'].items():
                    text_norm = text_norm.replace(src, tgt)
            text_norm = common_process(text_norm)
            # text_norm = common_postprocess(text_norm)
            return text_norm

        def common_process(text: str):
            text_norm = text
            # 处理特殊符号
            pause_punc = [
                '~', '～', ':', '$', '¥', '&', '#', '@', '^', '・', '·', '‘', '’', '“', '”', "'", "'", '"', '"',
                '（', '）', '(', ')', '【', '】', '{', '}', '「', '」', '[', ']', '<', '>', '《', '》',
                '%', '*', '|', '｜', '\\', '/', '-', '+', '_', '=',
                '²',
            ]
            text_norm = batch_replace(text_norm, pause_punc, tgt='')
            return text_norm

        def common_preprocess(text: str):
            special_symbols = [
                '&#34;'
            ]
            text_norm = batch_replace(text, special_symbols, tgt='')
            text_norm = batch_replace(text_norm, ['\n'], tgt=' ')
            return text_norm

        def common_postprocess(text: str):
            import re
            max_no_punct = 77
            errors = []
            no_punct_segments = re.split(r'[，。、；：！？,.;:!?]', text)
            for seg in no_punct_segments:
                len_seg = len(get_word_list(seg))
                if len_seg > max_no_punct:
                    errors.append(f"无效文本：存在{len_seg}个连续字符无标点（允许上限{max_no_punct}字）")
                    break
            if len(errors) > 0:
                if len(errors) > 1:
                    msg = '; '.join([f'{i+1}. {errors[i]}' for i in range(len(errors))])
                else:
                    msg = errors[0]
                raise RuntimeError(msg)
            return text

        def batch_replace(text: str, src: Union[str, List], tgt: str = ','):
            for p in src:
                text = text.replace(p, tgt)
            return text
        
        try:
            language_type = classify_language(input_text)
        except LangDetectException as err:
            handle_exacption(err, '无法检测语言，默认选择中文')
            language_type = 'zh'

        if language_type == 'en':
            # input_text = _normalize_text_en(input_text)
            text_segs = chunk_text_english(input_text, max_chars=chunk_num_words_en)
        else:
            # input_text.normalize(_normalize_text_zh)
            text_segs = chunk_text_chinese_v2(input_text, limit=chunk_num_words_zh)

        return text_segs

    def combine_audio_segments(self, segments, crossfade_duration=0.08):
        window_length = int(self.sr * crossfade_duration)
        hanning_window = np.hanning(2 * window_length)
        # Combine
        for i, segment in enumerate(segments):
            if i == 0:
                combined_audio = segment
            else:
                overlap = combined_audio[-window_length:] * hanning_window[window_length:] + segment[:window_length] * hanning_window[:window_length]
                combined_audio = np.concatenate(
                    [combined_audio[:-window_length], overlap, segment[window_length:]]
                )
        return combined_audio
        
    @torch.inference_mode()
    def process_text_seg(
        self, t_i, text, len_text_segs, profile,
        text_ref,
        vae_latent,
        duration_ctx,
        wav_pred_,
        time_step,
        seq_cfg_w, 
        timestep_annealing_w, 
        speech_rate, 
        use_amo_sampler,
    ):

        txt_tokens = self.dit_text_tokenizer(text_ref + text, padding=True, return_tensors='pt')
        txt_mask = txt_tokens['attention_mask'].bool().to(self.device)
        txt_tokens = txt_tokens['input_ids'].to(self.device)

        target_size = len(self.dit_text_tokenizer.encode(text_ref + text)) / len(self.dit_text_tokenizer.encode(text_ref)) * vae_latent.size(1)
        target_size = round(target_size)

        # ''' Duration Prediction '''
        # with Timer('Duration Prediction', enable=profile):
        #     with model_lock(self.lock), torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
        #         dur_pred = self.dur_model.decode(
        #             txt_tokens, txt_mask, 
        #             audio_feat=duration_ctx[0], audio_mask=duration_ctx[1]
        #         )

        # target_size = dur_pred.long().item() // self.hp_vae['vae_stride']

        # print(f"{vae_latent.shape = }")
        # print(f"{target_size = }")

        speech_rate = max(min(speech_rate, 2), 0.5)
        if speech_rate != 1:
            ref_size = int(vae_latent.shape[1])
            target_size = round(ref_size + (target_size - ref_size) / speech_rate)

        if target_size <= vae_latent.size(1):
            new_target_size = len(self.dit_text_tokenizer.encode(text_ref + text)) / len(self.dit_text_tokenizer.encode(text_ref)) * vae_latent.size(1)
            new_target_size = round(new_target_size)
            print(f"target_size <= vae_latent.size(1), target_size = {target_size}, vae_latent.size(1) = {vae_latent.size(1)}, text = {text}. Choose target_size = {new_target_size}")
            target_size = new_target_size

        ''' DiT target speech generation '''
        with Timer('DiT target speech generation', enable=profile):
            
            ctx_mask = torch.ones_like(vae_latent[:, :, 0:1])
            lat = F.pad(vae_latent, (0, 0, 0, target_size - vae_latent.size(1)), mode='constant', value=0)
            ctx_mask = F.pad(ctx_mask, (0, 0, 0, target_size - ctx_mask.size(1)), mode='constant', value=0)
        
            txt_tokens[~txt_mask] = self.cfg_mask_text_token
        
            txt_tokens = torch.cat([
                txt_tokens,
                txt_tokens,
                torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device)
            ], dim=0)
            txt_mask = torch.cat([txt_mask] * 3, dim=0)

            lat = torch.cat([
                lat,
                torch.zeros_like(lat),
                torch.zeros_like(lat)
            ], dim=0)
            ctx_mask = torch.cat([ctx_mask] * 3, dim=0)

            inputs = {
                "lat_ctx": lat * ctx_mask,
                "ctx_mask": ctx_mask,
                "txt_tokens": txt_tokens,
                'txt_mask': txt_mask,
                'tgt_len': torch.LongTensor([target_size] * 3).to(self.device)
            }

            with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=self.precision):
                x = self.dit.inference(
                    inputs, timesteps=time_step, 
                    seq_cfg_w=seq_cfg_w, 
                    timestep_annealing_w=timestep_annealing_w,
                )

        # WavVAE decode
        with Timer('WavVAE decode', enable=profile):
            x[:, :vae_latent.size(1)] = vae_latent
            with model_lock(self.lock):
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    wav_pred = self.vae.decode(x)[0,0].to(torch.float32)

        with Timer('Post-processing', enable=profile):
            hop_size = self.hp_vae['hop_size']
            vae_stride = self.hp_vae['vae_stride']
            wav_pred = wav_pred[vae_latent.size(1)*vae_stride*hop_size:]
            if wav_pred.abs().max() > 1:
                print('Wav amplitude exceed 1, clip it.')
                wav_pred = wav_pred / (wav_pred.abs().max())

            wav_pred = wav_pred.cpu().numpy()
            wav_pred_[t_i] = wav_pred
            print(f"{t_i = } {wav_pred.shape = } {text = }")

    def forward(
            self, resource_context, input_text, time_step, seq_cfg_w=(1.4, 3), 
            speech_rate=1, timestep_annealing_w=(1.0, 0.0, 1.0), use_amo_sampler=False, 
            return_format='wav', custom_ph_table=None, num_parallel_workers=5, 
            chunk_num_words_zh=60, chunk_num_words_en=130, **kwargs
    ):
        
        device = self.device

        profile = os.environ.get('MEGA_PROFILE', 'false').strip().lower() == 'true'

        with torch.inference_mode():
            ''' Generating '''
            # input_text = remove_space(input_text)
            # remove blank special symbols
            input_text = ''.join(c for c in input_text if c.isprintable())

            ''' preprocess text '''
            with Timer('preprocess text', enable=profile):
                ph_replace_table = deepcopy(self.ph_replace_table)
                if custom_ph_table is not None:
                    ph_replace_table.update(custom_ph_table)
                text_segs = self.preprocess_text(input_text, ph_replace_table, chunk_num_words_zh, chunk_num_words_en)

            # print(f'{text_segs = }')

            len_text_segs = len(text_segs)
            wav_pred_ = [None] * len_text_segs

            with ThreadPoolExecutor(max_workers=num_parallel_workers) as executor:
                futs = []
                for t_i, text in enumerate(text_segs):
                    future = executor.submit(
                                self.process_text_seg, 
                                *(t_i, text, len_text_segs, profile), 
                                **{
                                    "text_ref": resource_context['text_ref'],
                                    "vae_latent": resource_context['vae_latent'].detach().clone().to(device),
                                    "duration_ctx": move_to_cuda(resource_context['duration_ctx'], device=device),
                                    "wav_pred_": wav_pred_,
                                    "time_step": time_step,
                                    "seq_cfg_w": seq_cfg_w,
                                    "timestep_annealing_w": timestep_annealing_w,
                                    "speech_rate": speech_rate,
                                    "use_amo_sampler": use_amo_sampler,
                                })
                    futs.append(future)
                
                results = [f.result() for f in futs]

            wav_pred_ = [s for s in wav_pred_ if s is not None]

            # normalize loudness
            if len(wav_pred_) > 1:
                silent_speech = False
                meter = pyln.Meter(self.sr)  # create BS.1770 meter
                j = 0
                while j < len(wav_pred_):
                    try:
                        loudness_1 = meter.integrated_loudness(wav_pred_[0].astype(float))
                        break
                    except:
                        pass
                    j += 1
                else:
                    silent_speech = True
                if not silent_speech:
                    for i in range(j+1, len(wav_pred_)):
                        wav_pred__ = wav_pred_[i]
                        loudness_pred = meter.integrated_loudness(wav_pred__)
                        wav_pred__ = pyln.normalize.loudness(wav_pred__, loudness_pred, loudness_1)
                        if np.abs(wav_pred__).max() >= 1:
                            wav_pred__ = wav_pred__ / np.abs(wav_pred__).max() * 0.95
                        wav_pred_[i] = wav_pred__

            wav_pred_ = self.combine_audio_segments(wav_pred_)

            wav_bytes = to_wav_bytes(wav_pred_.astype(float), self.sr)
            if return_format == 'mp3':
                wav_bytes = wav_bytes_to_mp3_bytes(wav_bytes)

            output = MegaTTS3Output(
                wav_bytes=wav_bytes,
                duration=wav_pred_.shape[-1] / self.sr,
            )

            return output


class MegaTTS3DiTV8MInfer(DiTBuildModelMixin, MegaTTS3DiTV8Infer):
    def build_model(self, device):
        self.device = device

        if self.dit_exp_name.endswith('.ckpt'):
            set_hparams(f'{Path(self.dit_exp_name).parent}/config.yaml', print_hparams=False)
        else:
            set_hparams(f'{self.dit_exp_name}/config.yaml', print_hparams=False)
        hparams['use_fsdp'] = False

        ''' Load DiT '''
        self._build_model()
        if self.use_ema and hparams.get('use_ema', False):
            load_ckpt(self.dit, self.dit_exp_name, 'ema_model', strict=False, mmap=True)
        else:
            load_ckpt(self.dit, self.dit_exp_name, 'dit', strict=False, mmap=True)
        self.vae.eval()
        self.vae.to(self.device, dtype=self.precision)
        self.dit.eval()
        self.dit.to(device, dtype=self.precision)
        
        ''' ASR '''
        from modules.asr.sensevoice.sensevoice_api import build_asr_model
        self.asr_model = build_asr_model(device)

        ''' VAD '''
        from silero_vad import load_silero_vad
        self.vad_model = load_silero_vad()

        self.vae = torch.compile(self.vae, mode='max-autotune')
        self.dit = torch.compile(self.dit, mode='max-autotune')
        self.lock = threading.Lock()

    @staticmethod
    def parse_tagged_speaker_text(text: str, allowed_speaker_ids=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("speaker tagged text must be a non-empty string")

        if allowed_speaker_ids is not None:
            allowed_speaker_ids = {str(sid) for sid in allowed_speaker_ids}

        turns = []
        cursor = 0
        for match in _SPEAKER_TAG_BLOCK_RE.finditer(text):
            prefix = text[cursor:match.start()]
            if prefix.strip():
                raise ValueError(f"invalid speaker tagged text: unexpected content outside tags: {prefix!r}")

            speaker_id = match.group(1)
            if allowed_speaker_ids is not None and speaker_id not in allowed_speaker_ids:
                raise ValueError(f"speaker <S{speaker_id}> is not allowed here")

            inner_text = match.group(2).strip()
            if not inner_text:
                raise ValueError(f"speaker <S{speaker_id}> contains empty text")

            turns.append((speaker_id, inner_text))
            cursor = match.end()

        suffix = text[cursor:]
        if suffix.strip():
            raise ValueError(f"invalid speaker tagged text: unexpected trailing content: {suffix!r}")
        if len(turns) <= 0:
            raise ValueError("speaker tagged text must contain at least one <Sx>...</Sx> block")
        return turns

    @staticmethod
    def normalize_single_speaker_text(text: str, speaker_id='1'):
        if text is None:
            return None
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            return ""

        if _SPEAKER_TAG_BLOCK_RE.search(text) is None:
            return text

        turns = MegaTTS3DiTV8MInfer.parse_tagged_speaker_text(text, allowed_speaker_ids=[speaker_id])
        if len(turns) != 1:
            raise ValueError("single-speaker text can contain at most one speaker block")
        tagged_speaker_id, normalized = turns[0]
        if tagged_speaker_id != str(speaker_id):
            raise ValueError(f"single-speaker text must use <S{speaker_id}>...</S{speaker_id}> when tagged")
        return normalized

    def _load_reference_audio(self, audio_path):
        wav, _ = librosa.load(audio_path, sr=self.sr)
        ws = hparams['win_size']
        if len(wav) % ws < ws - 1:
            wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0).astype(np.float32)
        wav = np.pad(wav, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)
        wav_16k = librosa.resample(wav, orig_sr=self.sr, target_sr=16000)
        return wav.astype(np.float32), wav_16k.astype(np.float32)

    def _trim_reference_audio(self, wav, wav_16k):
        max_ref_duration = self.max_ref_duration
        segments = self.trim_silence_vad(wav_16k, max_duration=max_ref_duration)
        if len(segments) > 0:
            wav = np.concatenate([wav[int(segment['start'] * self.sr): int(segment['end'] * self.sr)] for segment in segments])
            wav_16k = np.concatenate([wav_16k[int(segment['start'] * 16000): int(segment['end'] * 16000)] for segment in segments])
        # remove gap leak
        wav = np.concatenate([wav, np.zeros(int(self.sr * 0.2), dtype=np.float32)])
        wav_16k = np.concatenate([wav_16k, np.zeros(int(16000 * 0.2), dtype=np.float32)])
        return wav.astype(np.float32), wav_16k.astype(np.float32)

    def _run_reference_asr(self, wav_16k):
        from modules.asr.sensevoice.sensevoice_api import run_asr_model
        return run_asr_model([wav_16k], self.asr_model, with_segments=False)[0]['text_normed']

    def _encode_reference_latent(self, wav_tensors):
        if len(wav_tensors) <= 0:
            raise ValueError("at least one reference audio is required")
        wav = torch.cat(wav_tensors, dim=1)
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=self.precision):
                vae_latent = self.vae.encode_latent(wav)
        return vae_latent.cpu()

    def preprocess_refs(self, refs: List[Dict]):
        device = self.device

        texts = []
        wavs = []
        resolved_refs = []

        for ref in refs:
            speaker_id = str(ref['speaker_id'])
            audio_path = ref['audio']
            wav, wav_16k = self._load_reference_audio(audio_path)
            wav, wav_16k = self._trim_reference_audio(wav, wav_16k)

            ref_text = self.normalize_single_speaker_text(ref.get('ref_text'), speaker_id=speaker_id)
            ref_text_source = 'provided'
            if ref_text is None or ref_text == "":
                ref_text = self._run_reference_asr(wav_16k)
                ref_text_source = 'asr'

            wav_tensor = torch.tensor(wav, dtype=self.precision)[None].to(device)
            wavs.append(wav_tensor)

            texts.append(f"<S{speaker_id}>{ref_text}</S{speaker_id}>")
            resolved_refs.append({
                'speaker_id': speaker_id,
                'audio': audio_path,
                'ref_text': ref_text,
                'text_source': ref_text_source,
            })

        ret = {
            'text_ref': ''.join(texts),
            'vae_latent': self._encode_reference_latent(wavs),
            'resolved_references': resolved_refs,
            'reference_mode': 'multi_refs' if len(resolved_refs) > 1 else 'single',
        }
        return ret

    def preprocess_dialogue_ref(self, audio_path: str, ref_text: str, num_speakers: int):
        allowed_speaker_ids = [str(i) for i in range(1, int(num_speakers) + 1)]
        turns = self.parse_tagged_speaker_text(ref_text, allowed_speaker_ids=allowed_speaker_ids)
        wav, wav_16k = self._load_reference_audio(audio_path)
        wav, wav_16k = self._trim_reference_audio(wav, wav_16k)
        del wav_16k

        wav_tensor = torch.tensor(wav, dtype=self.precision)[None].to(self.device)
        normalized_ref_text = ''.join(f"<S{speaker_id}>{text_}</S{speaker_id}>" for speaker_id, text_ in turns)

        return {
            'text_ref': normalized_ref_text,
            'vae_latent': self._encode_reference_latent([wav_tensor]),
            'resolved_references': {
                'audio': audio_path,
                'ref_text': normalized_ref_text,
                'num_speakers': int(num_speakers),
                'turns': [{'speaker_id': speaker_id, 'text': text_} for speaker_id, text_ in turns],
            },
            'reference_mode': 'dialogue_ref',
        }

    def preprocess(self, audio_paths: List):
        refs = [{'speaker_id': str(spk_id), 'audio': audio_path, 'ref_text': None} for spk_id, audio_path in audio_paths]
        ret = self.preprocess_refs(refs)
        return {
            'text_ref': ret['text_ref'],
            'vae_latent': ret['vae_latent'],
        }

    def forward(
            self, resource_context, input_text, time_step, seq_cfg_w=(1.4, 3), 
            speech_rate=1, timestep_annealing_w=(1.0, 0.0, 1.0), use_amo_sampler=False, 
            return_format='wav', custom_ph_table=None, num_parallel_workers=5, 
            chunk_num_words_zh=60, chunk_num_words_en=130, **kwargs
    ):
        
        device = self.device

        profile = os.environ.get('MEGA_PROFILE', 'false').strip().lower() == 'true'

        with torch.inference_mode():
            ''' Generating '''
            text = ''
            for spk_id, text_ in input_text:
                text = text + f"<S{spk_id}>{text_}</S{spk_id}>"

            text_ref = resource_context['text_ref']
            vae_latent = resource_context['vae_latent'].detach().clone().to(device)

            txt_tokens = self.dit_text_tokenizer(text_ref + text, padding=True, return_tensors='pt')
            txt_mask = txt_tokens['attention_mask'].bool().to(self.device)
            txt_tokens = txt_tokens['input_ids'].to(self.device)

            sx_patterns = _get_sx_token_patterns(self.dit_text_tokenizer)
            spk_mask = build_spk_mask_from_text_tokens(txt_tokens[0], sx_patterns)[None, ...]
            assert spk_mask.shape == txt_tokens.shape, (spk_mask.shape, txt_tokens.shape)

            target_size = len(self.dit_text_tokenizer.encode(text_ref + text)) / len(self.dit_text_tokenizer.encode(text_ref)) * vae_latent.size(1)
            target_size = round(target_size)

            speech_rate = max(min(speech_rate, 2), 0.5)
            if speech_rate != 1:
                ref_size = int(vae_latent.shape[1])
                target_size = round(ref_size + (target_size - ref_size) / speech_rate)

            if target_size <= vae_latent.size(1):
                new_target_size = len(self.dit_text_tokenizer.encode(text_ref + text)) / len(self.dit_text_tokenizer.encode(text_ref)) * vae_latent.size(1)
                new_target_size = round(new_target_size)
                print(f"target_size <= vae_latent.size(1), target_size = {target_size}, vae_latent.size(1) = {vae_latent.size(1)}, text = {text}. Choose target_size = {new_target_size}")
                target_size = new_target_size

            ''' DiT target speech generation '''
            with Timer('DiT target speech generation', enable=profile):
                
                ctx_mask = torch.ones_like(vae_latent[:, :, 0:1])
                lat = F.pad(vae_latent, (0, 0, 0, target_size - vae_latent.size(1)), mode='constant', value=0)
                ctx_mask = F.pad(ctx_mask, (0, 0, 0, target_size - ctx_mask.size(1)), mode='constant', value=0)
            
                txt_tokens[~txt_mask] = self.cfg_mask_text_token
            
                txt_tokens = torch.cat([
                    txt_tokens,
                    txt_tokens,
                    torch.full(txt_tokens.size(), self.cfg_mask_text_token, device=self.device)
                ], dim=0)
                txt_mask = torch.cat([txt_mask] * 3, dim=0)

                lat = torch.cat([
                    lat,
                    torch.zeros_like(lat),
                    torch.zeros_like(lat)
                ], dim=0)
                ctx_mask = torch.cat([ctx_mask] * 3, dim=0)

                spk_mask = torch.cat([spk_mask] * 3, dim=0).to(self.device)

                inputs = {
                    "lat_ctx": lat * ctx_mask,
                    "ctx_mask": ctx_mask,
                    "txt_tokens": txt_tokens,
                    'txt_mask': txt_mask,
                    'txt_lens': txt_mask.sum(1),
                    'spk_mask': spk_mask,
                    'tgt_len': torch.LongTensor([target_size] * 3).to(self.device)
                }

                with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=self.precision):
                    x = self.dit.inference(
                        inputs, timesteps=time_step, 
                        seq_cfg_w=seq_cfg_w, 
                        timestep_annealing_w=timestep_annealing_w,
                    )

            # WavVAE decode
            with Timer('WavVAE decode', enable=profile):
                x[:, :vae_latent.size(1)] = vae_latent
                with model_lock(self.lock):
                    with torch.autocast(device_type='cuda', dtype=self.precision):
                        wav_pred = self.vae.decode(x)[0,0].to(torch.float32)

            with Timer('Post-processing', enable=profile):
                hop_size = self.hp_vae['hop_size']
                vae_stride = self.hp_vae['vae_stride']
                wav_pred = wav_pred[vae_latent.size(1)*vae_stride*hop_size:]
                if wav_pred.abs().max() > 1:
                    print('Wav amplitude exceed 1, clip it.')
                    wav_pred = wav_pred / (wav_pred.abs().max())

                wav_pred = wav_pred.cpu().numpy()
                print(f"{wav_pred.shape = } {text = }")

            wav_bytes = to_wav_bytes(wav_pred.astype(float), self.sr)
            if return_format == 'mp3':
                wav_bytes = wav_bytes_to_mp3_bytes(wav_bytes)

            output = MegaTTS3Output(
                wav_bytes=wav_bytes,
                duration=wav_pred.shape[-1] / self.sr,
            )

            return output


if __name__ == '__main__':
    from utils.text.cosyvoice2_tokenizer import get_tokenizer
    dit_text_tokenizer = get_tokenizer(multilingual=True, num_languages=100)
    dit_vocab_size = dit_text_tokenizer.encoding.n_vocab

    dur_exp_name = 'checkpoints/251210_durpred_ditv8'
    hp_dur_model = hp_dur_model = set_hparams(f'{dur_exp_name}/config.yaml', global_hparams=False)

    from modules.tts.scriptspeech.dit_ph_v8 import ModelArgs, TotalDurationPredictor
    config = ModelArgs()
    config.vocab_size = dit_vocab_size
    config.encoder_dim = 768
    dur_predictor = TotalDurationPredictor(config, init_pretrained=False)
    
    dur_predictor.eval()
    dur_predictor.to('cuda', dtype=torch.bfloat16)

    wav, _ = librosa.load('user/prompts/mega_eval_prompt0731/0_vocal.wav', sr=16000)
    wav = torch.from_numpy(wav)[None].bfloat16().to('cuda')
    wav_mask = torch.ones_like(wav).bool()

    # txt_tokens = self.dit_text_tokenizer(text_ref + text, padding=True, return_tensors='pt')
