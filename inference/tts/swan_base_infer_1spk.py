import os
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import sys
import json
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
from utils.text.ph_alignment import align_word_phone, print_align, merge_norm_alignment
from utils.text.split_text import chunk_text_chinese, chunk_text_english, chunk_text_chinese_v2
from utils.commons.ckpt_utils import load_ckpt, get_all_ckpt_steps
from utils.commons.hparams import set_hparams, hparams
from utils.commons.meters import Timer
from utils.commons.os_utils import handle_exacption, kill_void
from utils.commons.io import print_once, json_dumps, get_wav_duration
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.commons.tensor_utils import move_to_cpu, move_to_cuda

from modules.tts.swanaudio.build_model_utils import DiTBuildModelMixin
from tasks.tts.dataset_utils.swan_caption_fastdataset import build_spk_mask_from_text_tokens, _get_sx_token_patterns, augment_text_with_pinyin_s1s2_safe

DEBUG = False

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


class SwanBaseInfer(DiTBuildModelMixin):
    def __init__(
            self, 
            device=None,
            dit_exp_name='checkpoints/251204_megatts3_dit_v6_large',
            wavvae_exp_name='checkpoints/260130_vae_v9_48k_25hz_z96/model_ckpt_steps_100000_model_gen.ckpt',
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

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.use_tqdm = use_tqdm
        self.use_ema = use_ema

        self.dit_exp_name = dit_exp_name
        self.wavvae_exp_name = wavvae_exp_name

        self.build_model(self.device)

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
        hparams['vae_ckpt'] = self.wavvae_exp_name

        ''' Load DiT '''
        self._build_model(attn_implementation='flash_attention_2')
        if self.use_ema and hparams.get('use_ema', False):
            load_ckpt(self.dit, self.dit_exp_name, 'ema_model', strict=False, mmap=True)
        else:
            load_ckpt(self.dit, self.dit_exp_name, 'dit', strict=False, mmap=True)
        self.vae.eval()
        self.vae.to(self.device, dtype=self.precision)
        self.dit.eval()
        self.dit.to(device, dtype=self.precision)
        self.sr = self.vae.sample_rate
        self.hop_length = self.vae.hop_length
        self.fm = self.hop_length // hparams['hop_size']
        
        ''' ASR '''
        from modules.asr.sensevoice.sensevoice_api import build_asr_model
        self.asr_model = build_asr_model(device)

        ''' VAD '''
        from silero_vad import load_silero_vad
        self.vad_model = load_silero_vad()

        # self.vae = torch.compile(self.vae, mode='max-autotune')
        # self.dit = torch.compile(self.dit, mode='max-autotune')
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

    def preprocess(self, audio_path):

        device = self.device

        wav, _ = librosa.load(audio_path, sr=self.sr)
        ws = hparams['win_size']
        if len(wav) % ws < ws - 1:
            wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0).astype(np.float32)
        wav = np.pad(wav, (0, self.sr // 2), mode='constant', constant_values=0.0).astype(np.float32)
        wav_16k = librosa.resample(wav, orig_sr=self.sr, target_sr=16000)
        max_ref_duration = self.max_ref_duration

        segments = self.trim_silence_vad(wav_16k, max_duration=max_ref_duration)
        wav = np.concatenate([wav[int(segment['start'] * self.sr): int(segment['end'] * self.sr)] for segment in segments])
        wav_16k = np.concatenate([wav_16k[int(segment['start'] * 16000): int(segment['end'] * 16000)] for segment in segments])
        # remove gap leak
        wav = np.concatenate([wav, np.zeros(int(self.sr * 0.2))])
        wav_16k = np.concatenate([wav_16k, np.zeros(int(16000 * 0.2))])

        wav = torch.tensor(wav, dtype=self.precision)[None].to(device)

        from modules.asr.sensevoice.sensevoice_api import run_asr_model
        text = run_asr_model([wav_16k], self.asr_model, with_segments=False)[0]['text_normed']

        with torch.inference_mode():
            ''' Forward WavVAE to obtain: prompt latent '''
            with torch.autocast(device_type='cuda', dtype=self.precision):
                vae_latent, _ = self.vae.encode_latent(wav)

        ret = {
            'text_ref': text,
            'vae_latent': vae_latent.cpu(),
        }

        return ret

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
            text = input_text

            text_ref = resource_context['text_ref']
            vae_latent = resource_context['vae_latent'].detach().clone().to(device)

            txt_tokens = self.dit_text_tokenizer(text_ref + text, padding=True, return_tensors='pt')
            txt_mask = txt_tokens['attention_mask'].bool().to(self.device)
            txt_tokens = txt_tokens['input_ids'].to(self.device)

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

                inputs = {
                    "lat_ctx": lat * ctx_mask,
                    "ctx_mask": ctx_mask,
                    "txt_tokens": txt_tokens,
                    'txt_mask': txt_mask,
                    'txt_lens': txt_mask.sum(1),
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
                        wav_pred = self.vae.decode(x)[0][0,0].to(torch.float32)

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
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')

    dit_exp_name = '/mnt/bn/sa-ag-data/liruiqi/code/ScriptSpeech/checkpoints/260303_swan_base'
    # dit_exp_name = 'checkpoints/260131_swan_base'

    cls_kwargs = dict(
        dit_exp_name=dit_exp_name,
        use_ema=True,
        max_ref_duration=20,
    )

    infer_kwargs = dict(
        time_step=50, 
        seq_cfg_w=(1.5, 3), 
        speech_rate=1.0, 
        timestep_annealing_w=(0.6, 0.6, 1.0)
    )

    infer_ins = SwanBaseInfer(device=f'cuda:0', **cls_kwargs)

    if True:
        # wav_path = '/mnt/bn/sa-ag-data/liruiqi/code/MegaTTS3-inference-vc/infer_out/timbre_prompt/情绪音色供给4.17/活力姐姐-MEGA_87x6TAJSqzN.wav'
        wav_path = 'user/prompts/女性_中年_娓娓道来-QM_0002+5_0131.wav'
        # wav_path = 'user/prompts/mega_eval_prompt0731/9_vocal.wav'
        # wav_path = 'prompts/2025-07-27_211109_509_vocal.wav'
        print(f"| Start processing {wav_path}")
        resource_context = infer_ins.preprocess(wav_path)
        # torch.save(resource_context, 'infer_out/resource_fanxian.pt')
    else:
        resource_context = torch.load('/mnt/bn/sa-ag-data/liruiqi/code/MegaTTS3-inference/infer_out/resource_fanxian.pt')

    test_text = [
        # 'I think our not-so-normal son might be going through some classic teenage rebellion.',
        # '我认为我们不那么正常的儿子可能正在经历一些典型的青少年叛逆。',
        # 'tracted by women with big chests. Um you know I might be looking at them or something but I I I',
        # '被漂亮的女性吸引。呃，你知道我可能正在看她们或者什么，但是我我我',
        # '老妹儿还是那句话，我说你行，你就行，不行也得行。就老妹儿手里拿这个断货王升级版第二代小蜜丸，你一定要去体验体验一次，让你回味无穷。体验一次，你肯定会回来找我回购的，肯定会回来感谢我的。咱经常出门出差，交朋会友的这个老妹儿，真心推荐给你啊。你别担心老妹已经替你试过了。视频左下角安排了同款。抓紧去拼手速'
        # '你好，请问您为交友软件充过钱吗？当然了，想和美女聊天不就是要充钱吗？但是我现在是真没钱充会员了。那你可以试试这个软件啊，上线就送七天会员。真的吗？还有这种好事？当然是真的了，只要你是通过本条视频下方链接下载的，就会给你匹配同城的单身美女，可以语音。可以，视频还能看到电话号码和位置呢，你要是不信的话，就点击下方链接下载试试吧！赶紧点击下方链接下载吧！'
        # '都官方直発',
        # '梅花（Titoni）瑞士手表男士机械表经典金色腕表生日礼物 宇宙系列 钢带表盘40MM 797 G-DB-306',
        # '这个抹布你千万别嫌它便宜，我怕你用完舍不得丢，因为实在是太好用了&#34;不沾油厨房抹布 &#34;厨房清洁 ...更多好用好用好用好用好用好用',
        # '入手闪铸AD5X太值了！它打印速度超快，600mm/s的速度，加速度达20000mm/s²，大大提升效率。支持四色打印，多色TPU柔性线材也能驾驭，色彩超绚丽。自带全自动调平校准，一键启动超省心。全金属CoreXY框架，稳定又精准，新手也能轻松上手，太好用啦！ '
        # '三分靠长相，七分靠打扮，女人的气质是打扮出来的。衣服要是穿的对，显得年轻好几岁。就这款超好看的蚂蚁腰防晒衬衣，穿在身上既优雅又有气质，还特别的舒服得劲，走到哪都是焦点。新品上市，厂家为了快速出一波店铺的销量，给咱姐妹们炸一波福利。今天下单还包邮送到家。姐妹们放心带回家试穿，喜欢满意就留下，不喜欢不满意，您直接退回来。精选轻薄透气面料，触感丝滑，穿上自带清凉感，仿佛给肌肤敷上冰膜，拒绝夏日闷热黏腻。版型设计独具匠心，巧妙运用公主线剪裁，勾勒出盈盈一握的蚂蚁腰，轻松展现女性曼妙身姿，谁穿谁是 “腰精”。短款衣长适配多种下装，搭配高腰裤或裙子，瞬间优化身材比例，大长腿既视感轻松拿捏。宽松的版型，胖瘦姐妹都能穿，简约的同时又不失精致。无论是日常出街、办公室久坐，还是户外游玩，这件防晒衬衣都能让你在防晒的同时，时尚感拉满，成为夏日里的靓丽风景线！不挑年龄不挑身材，能从30岁穿到五六十岁。关键价格不贵，还包邮送到家。但是福利数量不多了，喜欢的姐妹下方小黄车赶紧拼手速了。',
        # '这可是国际救援中心认证的"神器"！不用插电，晒两三个小时就能亮40个小时，露营、停电应急都能用，角度亮度随便调，还有爆闪求救模式！想省钱又图安心的，赶紧点我头像进橱窗，多备几个准没错！'
        # '家人们，来给评评理！上门女婿被老婆当众扇巴掌赶出门，人家愣是没吵没闹，抽根烟就给500公里外的亲弟弟打电话：&#34;弟，快来接我回家！&#34;这事儿，搁谁身上能忍？\n \n挨打这位李哥，打完电话扭头回屋，一声不吭就收拾行李。衣柜里的衣服哗啦往包里塞，摸着脸上火辣辣的巴掌印，对着镜子苦笑——这段婚姻，算是彻底凉透了。\n \n说起来，李哥和老婆可是实打实的校园爱情。大学四年腻歪得不行，毕业却卡在了&#34;去留&#34;难题上。老婆哭着说：&#34;我家就我一个闺女没嫁，总不能留爸妈孤苦伶仃吧？&#34;李哥纠结得不行，给亲弟弟打电话倒苦水。弟弟一拍胸脯：&#34;哥你放心！家里有我呢，爸妈绝对照顾得妥妥当当！&#34;\n \n就这么着，李哥留在老婆的城市，成了上门女婿。刚开始小两口日子过得挺甜蜜，李哥脾气温和又顾家，妥妥的&#34;模范丈夫&#34;，家里渐渐形成女主外男主内的模式。可谁能想到，岳父越看他越不顺眼！\n \n尤其是李哥事业走下坡路那阵儿，岳父逢年过节就冷嘲热讽：&#34;天天在家待着，吃我女儿软饭！&#34;起初李哥还能忍，后来连曾经心疼他的老婆，也跟着岳父阴阳怪气。有次岳父当着全家面羞辱他，李哥刚回怼两句，老婆抬手就是一巴掌！\n \n这巴掌下去，李哥彻底清醒了。任凭老婆哭着道歉、拽着衣角求原谅，他头也不回拉着行李箱就走。关上门那刻，后视镜里老婆的身影越来越小，李哥心里五味杂陈——十几年感情，终究抵不过一巴掌的伤害。\n \n话说回来，家人们可得留个心眼！现在电费贵，天气还总出幺蛾子，强烈建议囤几个太阳能足球灯！这可是国际救援中心认证的&#34;神器&#34;！不用插电，晒两三个小时就能亮40个小时，露营、停电应急都能用，角度亮度随便调，还有爆闪求救模式！想省钱又图安心的，赶紧点我头像进橱窗，多备几个准没错！',
        # '我的老朋友，你真的很不简单,这段时间悄悄看着你，发现你跟人相处时，总是带着春风般的亲切劲儿，让人打心眼里觉得温暖。你对生活那股子热乎劲儿，谁看了都受感染。每天乐呵呵的，好像心里揣着小太阳，跟你待在一起，连心情都跟着亮堂起来。这种积极的日子态度，可不就是咱们身边的正能量担嘛.你这人的特点就是实在，有啥说啥，从不藏着掖着。就像一块干干净净的璞玉，看着朴实，却透着真诚的光。现在这世道多复杂啊，但你始终活得明白 ，该咋过就咋过，这份清醒劲太难得啦，有时候我忍不住琢磨。你平时没少读书吧，不然咋活得这么通透，肯定也经历过不少风风雨雨，不然面对事儿时，咋能这么不慌不忙呢，你这人缘儿为啥这么好？看看你做的事儿就知道了，别人对你好一分你对别人好五分；宁可自己吃点亏，也不愿欠人情。遇到困难从不喊苦喊累，咬咬牙就挺过去了；日子过得不攀比、不将就，独立又有骨气，这些优点搁谁身上不招人夸，跟你说句心里话，像你这么重情重义、踏实实过日子的人，福气肯定在后头呢。相信以后你的日子只会越来越顺溜，最后还想跟你念叨两句：咱们这辈子就像跑马拉松，如今已经跑了大半程啦，别总闷头往前冲，偶尔放慢脚步，看看路边的花花草草，感受感受生活里的小确幸，才不辜负这一路的风景呀\n',
        # '梅花（Titoni）瑞士手表男士机械表经典金色腕表生日礼物 宇宙系列 钢带表盘40MM 797 G-DB-306',
        # '浪琴（LONGINES）瑞士手表 名匠系列月相腕表 机械皮带男表L29194783 银色麦粒饰纹42.0 mm',
        # '浪琴（LONGINES）瑞士手表 嘉岚系列 石英钢带女表 L42091917 白色哑光24.0 mm',
        # '百丽（Belle）简约一字拖女商场同款编织休闲拖鞋A8Z1DBT4预售 米白-平跟款 37 (235mm)',
        # 'Keep动感单车 家用智能健身器材AI调阻自发电款C2 Lite 20斤飞轮组 白',
        # '蓝立哆 （Elydo）电动升降桌电脑桌办公书桌双电机站立式工作台学习桌H2 H2白色桌腿+苏丹象牙白色桌面 25mm加厚1.2*0.6m',
        # '追剧到凌晨眼睛干涩模糊？叶黄素冰敷眼罩 yyds！叶黄素深入养护，冰感缓解疲劳，360° 透气设计，敷完眼睛明亮清晰，速抢！',
        # '蓝立哆电动升降桌双电机电脑电竞桌书桌学习桌站立式工作台写字桌 H2e Pro平椭圆白腿+白色桌面 25mm加厚环保生态板1.2*0.6m',
        # '家里有猫的注意啦\n麦富迪猫罐头来咯\n它成幼猫通用\n是猫咪都爱的零食湿粮\n这款罐头采用浓汤三种肉配方\n还添加了鱼油\n营养丰富又美味\n能满足猫咪日常营养所需\n让猫咪吃得开心 主人更放心\n如果你还在为给猫咪选零食发愁\n那就别犹豫了\n赶紧给你家宝贝安排上吧',
        # '还在为视频播放烦恼不已吗？\n别再纠结\n开通我们的升级服务\n专业团队技术加持\n就像一场及时雨\n提升你的视频品质\n让你的内容无处不在\n解锁自由发布\n点击即可拥有',
        # '蔻驰（COACH）奢侈品TOTE女包单肩拉链托特包4455【品牌授权】520送女友礼物',
        # '我真庆幸买了西门子极净魔盒Plus洗碗机！16套大容量，再多餐具都能装下。微米蒸汽洗，55°C蒸汽提前软化顽渍，搭配46000pa超强水压，啥污渍都能洗净。超五星消杀认证，除菌率高达 99.9999%。智能抽湿烘干3.0，餐具干干爽爽。操作简单，还有智能语音控制，用着超省心，强烈推荐！ ',
        # '今天给大家分享一款超赞的耐克NIKE缓震篮球鞋，型号为乔丹AJ1 JORDAN1运动鞋553558 - 145紫白42.5。这双鞋的第一个卖点，是经典设计，怎么搭配都很潮。鞋面采用皮革和合成材质组合，结实又耐用。后跟内嵌式Air缓震技术，缓震效果一流。硬质橡胶外底，抓地力强，多种地面都能轻松驾驭。我穿着它打球、日常出行都很舒服，长时间走路脚也不累，真心推荐给大家！ ',
        # '家人们！手速快的先赚翻了，金刚网绣花防蚊门帘直接炸到8块9毛9还包邮！这价格我都怀疑厂家贴钱做活动！\n \n瞅瞅这配置——中间加宽磁条&#34;啪嗒&#34;一下自动吸合，比普通门帘关得严实，魔术贴往门框一贴就能用，租房搬家拆下来还能接着用！网眼细到蚊子苍蝇钻不进，牛津布包边使劲扯都不变形，木门铁门都能装！\n \n两百单库存眼瞅着要见底，这价格平时连半片网都买不到！刷到这条就是大数据在喊你薅羊毛！想要夏天不被蚊子追着咬的，赶紧戳小黄车，手慢真的要拍大腿！',
        # '结合全网对【宝路华（BULOVA）熊猫盘手表男石英表计时码表阿波罗飞行员登月同款赠原装带98K112 98K112】这款商品做一个70字左右的种草分享。呈现出来的只要文案部分，不要乱七八糟给我发一堆，不需要AI写作助手来提供，你直接给我生成出来就可以',
        # '<speak>假期"还不知道"带孩子去哪儿的，一定要来青山上遇东坡画剧展，馆内几十个裸眼3D打卡点，直接把课本里的内容演活了，历史知识秒变视觉盛宴，让孩子感受宋式风韵，体验点茶、挂画、制香、插画，还有桌游和趣味拼图可以玩哦，关键是亲子票一大一小仅需59.9，<phoneme ph="zhen3 de3" alphabet="py">真的</phoneme>超划算，链接就在左下角，刷到赶紧囤~</speak>',
        # '都现在了，谁还傻乎乎的刷题啊，高分通过的考生早把知识点浓缩成了这300个必考点，近5年法考考来考去，翻不出这几个命门，你花几百小时刷题，不如直接啃透这套学霸笔记，别看它只有这薄薄的一套，里面是大咖讲师根据官方教材总结成的核心考点，近五年卷子90%考点对标，零基础也能懂得底层破题思路，案例分析一眼看穿命题陷阱，搭配4节直播课，老师手把手拆解考点逻辑，不到一杯奶茶钱，印刷成本都不够，下单激活还有3000＋智能题库，赶紧点进直播间看看吧，晚了可真没有了',
        # '<speak>给所有花大钱买熊胆粉的朋友们道个歉对不起我们的活动来的太迟了为了广大干友们都能用得起这次啊我们厂家亲自下场送福利免费咨询福利带回家我们这款熊胆粉呢成分就三个字熊胆粉不加辅料不混杂其他成分现在点开视频下方链接不下载不分享不转发客服直接对接<phoneme alphabet="py" ph="gong1 chang3 zhi2 fa1">工厂直发</phoneme>让中老年朋友都能用得上机会可一定要把握住一大盒熊胆粉不是07的也不是体验装是一大盒正装还不赶紧点击视频下方链接去看看无论你有多忙啊我都建议你现在就打开下方获取这个优惠</speak>',
        # '<speak>我们要搬去香港了————————这种苹果床就30一套了————————户外帐篷18————————阳台茶桌就15了————————花架6块————————垃圾桶3块————————风扇就全部10块了————————这边还有些衣帽架全身镜啥的————————都是全新的————————有需要的进我直播间<phoneme alphabet="py" ph="ba5">吧</phoneme>————————</speak>',
        # '<speak>假期"还不知道"带孩子去哪儿的，一定要来青山上遇东坡画剧展<sub alias="兄弟们">xdm</sub>，<break time="5s"></break>馆内几十个裸眼3D打卡点，直接把课本里的内容演活了，历史知识秒变视觉盛宴，让孩子感受宋式风韵，体验点茶、挂画、制香、插画，还有桌游和趣味拼图可以玩哦，关键是亲子票一大一小仅需59.9，<phoneme ph="zhen3 de3" alphabet="py">真的</phoneme>超划算，链接就在左下角，刷到赶紧囤~</speak>',
        # '<speak>这一粒小钙片富含330mg 高含量钙,<break time="5s"></break></speak>',
        # '<speak> <break time="1s"></break></speak>',
        # '<speak><break time="5s"></break>算了吧，哪有这么好的事！</speak>',
        # '<speak><break time="1s"></break> <break time="1s"></break>两种<break time="1s"></break>方式来帮你训练</speak>',
        # '九十九元不是发两袋也不是发四袋是真的六袋家好这次活动真的是拼了买过的朋友也趁着活动多囤点没买过的朋友也要趁这个活动入手简直不要太划算了拿回家不管怎么做都不心疼链接就放在直播间千万不要错过啦'
        '<S1>一九五五到一九八零年出生的您注意了！您的二零二五新款超声波洗菜盆专属优惠，已送达！别划走！这份福利正等着您签收！如果这次您再错过，优惠名额将转给他人！但现在，它依然属于您！点击下方链接，立即锁定您的优惠！名额已发送至您的手机。今天不参与，资格即失效！无论多忙，只需几秒就能抢到手！无需转发！无需下载！看到即得！所见即所得！现在参与，最快明天到货！名额告急，错过后悔莫及！赶紧点击下方链接，抢占最后名额！</S1>',
        # '想赚钱吗，点击下方链接，这款真的能赚钱的软件，你快来试试吧',
        # '家人们！我必须给大家分享一下这款宠物鱼油胶囊！如果你家的猫狗掉毛严重！到处都是毛！那你一定得试试这个！它富含高浓度的 omega3和各种维生素，口感很好！可以直接喂食！也可以拌在猫粮或喜欢的零食里！这是专为猫狗设计的！每一瓶都有80 粒！只需要一杯奶茶的钱！现在厂家直接发货！性价比超高！赶紧下单吧！',
        # '家人们，我必须给大家分享一下这款宠物鱼油胶囊。如果你家的猫狗掉毛严重，到处都是毛，那你一定得试试这个。它富含高浓度的 omega 三和各种维生素，口感很好，可以直接喂食，也可以拌在猫粮或喜欢的零食里。这是专为猫狗设计的，每一瓶都有八十粒，只需要一杯奶茶的钱。现在厂家直接发货，性价比超高，赶紧下单吧。',
        # '家人们，我必须给大家分享一下这款宠物鱼油胶囊，如果你家的猫狗掉毛严重，到处都是毛，那你一定得试试这个。',
        # '<|py_jia2|><|py_ren4|><|py_men3|>，<|py_wo4|>必须给大家分享一下这款宠物鱼油胶囊，如果你家的猫狗掉毛严重，到处都是毛，那你一定得试试这个。',
        # '家人们！我必须给 family 分享一下这款宠物鱼油胶囊！如果你家的猫狗掉毛严重！到处都是毛！那你一定得试试这个！它富含高浓度的 raining days 和各种维生素，口感很好！可以直接喂食！也可以拌在猫粮或喜欢的零食里！这是专为猫狗设计的！每一瓶都有八十粒！只需要一杯奶茶的钱！现在厂家直接发货！性价比超高！赶紧下单吧！',
    ]

    if dit_exp_name.endswith('.ckpt'):
        ckpt_name = Path(dit_exp_name).stem
        if "steps_" in ckpt_name:
            step_part = ckpt_name.split("steps_")[-1]
            step = step_part.split("_")[0]
        else:
            step = ckpt_name.split("_")[-1]
        dit_exp_name = str(Path(dit_exp_name).parent)
        out_path = f'infer_out/tts/infer_once/{Path(dit_exp_name).stem}/step{step}'
    else:
        out_path = f'infer_out/tts/infer_once/{Path(dit_exp_name).stem}/step{get_all_ckpt_steps(dit_exp_name)[-1]}'
    
    print(f"cls_kwargs = {json_dumps(cls_kwargs)}")
    print(f"infer_kwargs = {json_dumps(infer_kwargs)}")

    kill_void()
    pbar = tqdm(total=len(test_text))
    total_time_elapsed = 0
    total_duration_processed = 0
    for i, input_text in enumerate(test_text):
        print('='*100)
        print(f'| Generating text [{i}]: {input_text}')
        start_time = time.time()
        output = infer_ins.forward(
            resource_context, input_text, **infer_kwargs
        )
        total_time_elapsed += time.time() - start_time

        wav_bytes = output.wav_bytes
        # input_text = SSML(input_text).text_str
        os.makedirs(out_path, exist_ok=True)
        save_wav_bytes(wav_bytes, f'{out_path}/{input_text[:20]}.wav')
        gen_duration = get_wav_duration(f'{out_path}/{input_text[:20]}.wav')
        total_duration_processed += gen_duration

        print(f'| Generation of text [{i}] succeed')
        pbar.update(1)
        
    print(f"{total_time_elapsed = }, {total_duration_processed = }, RTF = {total_time_elapsed / total_duration_processed}")


# CUDA_VISIBLE_DEVICES=5 python user/inference/swan_base_infer_1spk.py
