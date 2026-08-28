import os
import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import sys
import io
import re
import json
import glob
import math
import time
import random
import argparse
import threading
import traceback
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import List, Union, Dict, Any, Tuple, Optional
from utils.commons.tensor_utils import move_to_cpu, move_to_cuda

import numpy as np
import librosa
import torch
import torch.nn.functional as F
import soundfile as sf
import pyloudnorm as pyln
import whisper
from tqdm import tqdm
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from langdetect import detect as classify_language, LangDetectException

from utils.audio.align import mel2token_to_dur
from utils.audio.io import save_wav_bytes, to_wav_bytes, wav_bytes_to_mp3_bytes
from utils.text import is_english, YUNMU_ERHUA, SHENGMU
from utils.text.text_encoder import TokenTextEncoder
from utils.text.split_text import chunk_text_english, chunk_text_chinese, get_word_list, remove_space, remove_unprintable
from utils.text.ph_tone_convert import split_ph_timestamp, split_ph, map_phone_to_tokendict
from utils.text.ssml_utils import SSML
from utils.text.ph_alignment import align_word_phone, print_align, merge_norm_alignment
from utils.commons.ckpt_utils import load_ckpt,load_ckpt2
from utils.commons.hparams import set_hparams, hparams
from utils.commons.meters import Timer
from utils.commons.os_utils import handle_exacption, kill_void
from utils.commons.io import print_once
from utils.commons.import_utils import import_module_bystr, get_class_from_module
from utils.nn.ema import restore_ema

from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixinV2
import inspect  # 新增：用于安全探测 dur_model.inference 的签名

# ===== numpy>=1.24 compatibility (np.float/np.int removed) =====
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "str"):
    np.str = str
# ==============================================================


# ===== 参数/环境统一 =====
DEBUG = False
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from contextlib import contextmanager
import sys
import io

@contextmanager
def suppress_output(enabled: bool = True):
    """
    静默 stdout/stderr（用于屏蔽 preprocess/forward 内部大量 print）
    """
    if not enabled:
        yield None
        return

    old_out, old_err = sys.stdout, sys.stderr
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        sys.stdout, sys.stderr = buf_out, buf_err
        yield (buf_out, buf_err)
    finally:
        sys.stdout, sys.stderr = old_out, old_err

@contextmanager
def model_lock(lock: threading.Lock):
    """统一：锁 + 退出时 cuda sync（对 compile/autocast 下更稳）"""
    lock.acquire()
    try:
        yield
    finally:
        # try:
        #     if torch.cuda.is_available():
        #         torch.cuda.synchronize()
        # except Exception:
        #     pass
        lock.release()

def move_to_shm(exp_name: str):
    """把 ckpt 目录复制到 /dev/shm 加速加载（可选开关）"""
    os.makedirs('/dev/shm/mega_ckpt', exist_ok=True)
    shm_exp_name = f"/dev/shm/mega_ckpt/{Path(exp_name).stem}"
    if os.path.exists(shm_exp_name):
        return shm_exp_name
    import subprocess
    subprocess.check_call(f"cp -r {exp_name} /dev/shm/mega_ckpt/", shell=True)
    return shm_exp_name

# =========================================
# Utils
# =========================================

def convert_to_wav_bytes(audio_binary: bytes) -> io.BytesIO:
    """任意格式→WAV（内存字节流）"""
    audio = AudioSegment.from_file(io.BytesIO(audio_binary))
    wav_bytes = io.BytesIO()
    audio.export(wav_bytes, format="wav")
    wav_bytes.seek(0)
    return wav_bytes

@dataclass
class MegaTTS3Output:
    wav_bytes: bytes = None
    wav: np.ndarray = None
    words_timestamps: Dict[str, List] = None
    words_timestamps_post: Dict[str, List] = None
    duration: float = None
    ph_pred: List[str] = None
    tone_pred: List[str] = None


# =========================================
# Inference Class
# =========================================

class MegaTTS3DiTInfer(DiTBuildModelMixinV2):
    def __init__(
        self,
        device=None,
        dit_exp_name=None,
        dur_exp_name=None,
        frontend_exp_name=None,
        wavvae_exp_name='checkpoints/1231_megatts3_wavvae_v3_25hz_kl001_fix4',
        use_old_aligner=True,
        use_old_dur=True,
        max_ref_duration=40,
        use_tqdm=True,
        use_ema=False,
        precision='bf16',          # 'bf16' or 'fp16'
        compile_models=False,      # 对齐第二份：可选 torch.compile
        use_shm_ckpt=False,        # 对齐第二份：可选 move_to_shm
        chunk_num_words_zh=60,
        chunk_num_words_en=130,
        **kwargs
    ):
        self.sr = 24000
        self.fm = 8

        # ---- precision ----
        if precision == 'fp16':
            self.precision = torch.float16
        else:
            self.precision = torch.bfloat16

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_tqdm = use_tqdm
        self.use_ema = use_ema
        self.compile_models = compile_models
        self.use_shm_ckpt = use_shm_ckpt

        self.chunk_num_words_zh = int(chunk_num_words_zh)
        self.chunk_num_words_en = int(chunk_num_words_en)

        self.use_old_aligner = use_old_aligner
        self.use_old_dur = use_old_dur

        # ---- ckpt path 统一：兼容 .ckpt / 目录；兼容 old 默认 ----
        self.dit_exp_name = dit_exp_name
        self.dur_exp_name = 'checkpoints/megatts3_wavdit/duration_lm' if use_old_dur else dur_exp_name
        self.frontend_exp_name = 'checkpoints/megatts3_wavdit/aligner_lm' if use_old_aligner else frontend_exp_name
        self.wavvae_exp_name = wavvae_exp_name

        self.max_ref_duration = max_ref_duration
        self.max_silence_alive = 1.28
        self.lock = threading.Lock()

        # 可选：把目录 ckpt move 到 shm（注意：.ckpt 文件不做）
        if self.use_shm_ckpt:
            if self.dit_exp_name and (not str(self.dit_exp_name).endswith('.ckpt')):
                self.dit_exp_name = move_to_shm(self.dit_exp_name)
            if (not self.use_old_dur) and self.dur_exp_name and (not str(self.dur_exp_name).endswith('.ckpt')):
                self.dur_exp_name = move_to_shm(self.dur_exp_name)
            if (not self.use_old_aligner) and self.frontend_exp_name and (not str(self.frontend_exp_name).endswith('.ckpt')):
                self.frontend_exp_name = move_to_shm(self.frontend_exp_name)
            if self.wavvae_exp_name and (not str(self.wavvae_exp_name).endswith('.ckpt')):
                self.wavvae_exp_name = move_to_shm(self.wavvae_exp_name)

        self.build_model(self.device)

    def build_dur_model(self):
        self.length_regulator = LengthRegulator()
        if self.use_old_dur:
            from modules.tts.ar_dur.ar_dur_predictor import ARDurPredictor
            hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
            hp_dur_model['frames_multiple'] = hparams['frames_multiple']
            self.dur_model = ARDurPredictor(
                hp_dur_model, hp_dur_model['dur_txt_hs'], hp_dur_model['dur_model_hidden_size'],
                hp_dur_model['dur_model_layers'], len(self.token_encoder),
                hp_dur_model['dur_code_size'],
                use_rot_embed=hp_dur_model.get('use_rot_embed', False), 
                precision=self.precision
                )
            load_ckpt(self.dur_model, f'{self.dur_exp_name}', 'dur_model')
            self.dur_model.eval()
            self.dur_model.to(self.device, dtype=self.precision)
            self.dur_model.precision = self.precision
        else:
            if 'lm' in (dur_exp_name_eles := [n.lower() for n in self.dur_exp_name.split('_')]):
                if 'seq2seq' in dur_exp_name_eles:
                    self.dur_model_type = 'lm_seq2seq'
                else:
                    self.dur_model_type = 'lm'
            elif 'dit' in dur_exp_name_eles:
                self.dur_model_type = 'dit'
            if self.dur_model_type in ['lm', 'lm_seq2seq']:
                from modules.tts.ar_dur.dur_lm import build_dur_model
                hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
                self.dur_model = build_dur_model(hp_dur_model, vocab_size=810, padding_idx=797)
                self.dur_model.hparams = {}
                self.dur_model.eval()
                load_ckpt(self.dur_model, self.dur_exp_name, 'model', mmap=True)
                self.dur_model.to(self.device)
            elif self.dur_model_type == 'dit':
                from modules.tts.scriptspeech.dit_dur import build_dur_model
                hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
                self.dur_model = build_dur_model(hp_dur_model)
                self.dur_model.hparams = {}
                self.dur_model.eval()
                load_ckpt(self.dur_model, self.dur_exp_name, 'model', strict=True, mmap=True)
                self.dur_model.to(self.device)
                
            if hasattr(self.dur_model, 'module'):  # 兼容 DDP/FSDP
                setattr(self.dur_model.module, 'precision', self.precision)
            setattr(self.dur_model, 'precision', self.precision)

    def _count_effective_chunks_from_raw(
        self,
        raw_text: str,
        ph_replace_table: dict,
        use_sa_frontend: bool,
        speech_rate: float
    ) -> int:
        """
        复刻 forward 前半段的解析/normalize/chunk/（可选）SA frontend 过滤逻辑，
        返回该 raw_text 最终会产生多少个有效 chunk（即 chunk_items 的数量）！
        目的是：用它来确定 prefix_text 对应的是前多少个 chunk，从而用预测 duration 精准裁音频！
        """
        raw_text = ''.join(c for c in (raw_text or "") if c.isprintable())
        if not raw_text.strip():
            return 0

        # 禁止 <S..>（与 forward 一致）
        if re.search(r"<\s*/?\s*S\s*\d+\s*>", raw_text, flags=re.IGNORECASE):
            raise RuntimeError("检测到 <S1>...</S1> 格式：本版本只支持 <SPK>1</SPK>！")

        # 解析 <SPK>
        spk_segs = self._parse_dialogue_segments(raw_text)  # [(sid, content)]
        if len(spk_segs) == 0:
            return 0

        clean_text_total = re.sub(
            r"<\s*SPK\s*>\s*\d+\s*<\s*/\s*SPK\s*>",
            "",
            raw_text,
            flags=re.IGNORECASE
        )

        ssml_root = SSML(clean_text_total)
        ssml_root.rate = float(speech_rate)

        cnt = 0
        for sid, seg_content in spk_segs:
            sub_ssml = SSML(seg_content)
            sub_ssml.rate = ssml_root.rate
            sub_chunks = self.preprocess_text(sub_ssml, ph_replace_table, use_sa_frontend)

            for ch in sub_chunks:
                if use_sa_frontend:
                    from modules.tts.frontend_lm.sa_frontend import call_sa_frontend
                    sa_ret = call_sa_frontend(ch.sa_ssml_str, debug=0)
                    if sa_ret is None:
                        assert False, "SA 前端返回 None，可能是配置/模型有问题！"
                    cnt += 1
                else:
                    # 非 SA 前端：forward 里不会跳过 chunk
                    cnt += 1

        return cnt

    def dereverb_wpe_mono(
        self,
        wav: np.ndarray,
        sr: int,
        n_fft: int = 1024,
        hop: int = 256,
        taps: int = 10,
        delay: int = 3,
        iterations: int = 3,
        rms_match: bool = True,
    ):
        """
        单通道 WPE 去混响：
        - 输入 wav: float32/float64, shape [T]
        - 返回: 去混响后的 wav, shape [T]
        """
        if wav is None or len(wav) < sr * 1.0:  # 太短的 ref（<1s）WPE 容易副作用大，直接跳过
            return wav

        try:
            from nara_wpe.wpe import wpe
        except Exception as e:
            print_once(f"| nara_wpe not available, skip dereverb: {e}")
            return wav

        x = wav.astype(np.float32)

        # 记录原始能量，避免去混响后整体电平塌掉影响后续
        if rms_match:
            rms0 = float(np.sqrt(np.mean(x * x) + 1e-8))

        # STFT: (F, T) complex
        Y = librosa.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann", center=True)
        # WPE 期望 (F, D, T)，单通道 D=1
        Y = Y[:, None, :]

        # WPE: 输出同形状 (F, D, T)
        Z = wpe(Y, taps=taps, delay=delay, iterations=iterations)

        # ISTFT 回时域
        y = librosa.istft(Z[:, 0, :], hop_length=hop, win_length=n_fft, window="hann", length=len(x)).astype(np.float32)

        if rms_match:
            rms1 = float(np.sqrt(np.mean(y * y) + 1e-8))
            y = y * (rms0 / rms1)

        # 轻微限幅，防爆
        y = np.clip(y, -1.0, 1.0)
        return y


    def _dur_predict_onepass_with_spk(
        self,
        chunk_items,
        resource_context,
        dur_disturb: float = 0.1,
        normalize_dur: bool = True
    ):
        """
        一步推完整目标段（2spk 兼容）：
        - 按出现顺序拼接所有 chunk 的 ph/tone
        - 构造与之等长的 spk_ids（phone-level）
        - 先用全局 ref 段 prefill，再一次性 inference
        - （可选）按说话人 × {静音, 非静音}在 log 域对齐到对应 ref 统计
        返回：{chunk_idx: Tensor[1, L_chunk]}，与逐段接口一致

        注意：本版本 caption 只使用 <SPK>sid</SPK>，不支持 <S1>..</S1>
        """
        import torch
        device = self.device
        compute_dtype = self.precision

        # ---------- 1) 汇总目标段 ph/tone + 对应 spk_ids ----------
        ph_all, tone_all, spk_ids_all = [], [], []
        lens = []
        for it in chunk_items:
            sid = int(it['sid'])
            ph  = it['ph'].to(device)
            tone= it['tone'].to(device)
            L   = ph.shape[1]
            lens.append(L)
            ph_all.append(ph)
            tone_all.append(tone)
            spk_ids_all.append(torch.full((1, L), sid, dtype=torch.long, device=device))

        ph_all        = torch.cat(ph_all,        dim=1)  # [1, T_all]
        tone_all      = torch.cat(tone_all,      dim=1)  # [1, T_all]
        spk_ids_all   = torch.cat(spk_ids_all,   dim=1)  # [1, T_all]

        # ---------- 2) 参考前缀 prefill（用全局 ref 段） ----------
        ref_tokens = map_phone_to_tokendict(
            {'txt_token': resource_context['ph_ref'].to(device),
            'tone':      resource_context['tone_ref'].to(device)},
            pad_bos_eos=False
        )
        with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=compute_dtype):
            start_pos = self.dur_model.prefill(ref_tokens, resource_context['dur_ref'].to(device))

        # ---------- 3) 文本条件（若是 seq2seq） ----------
        needs_caption = (getattr(self, "dur_model_type", "lm") == "lm_seq2seq")
        caption_embs = None
        if needs_caption:
            # ref_raw 现在应为：<SPK>1</SPK>...<SPK>2</SPK>...
            ref_raw = resource_context.get('text_ref_raw', '')
            # 目标段也用 <SPK>sid</SPK> 前缀式拼接
            tgt_raw = ''.join([f"<SPK>{int(it['sid'])}</SPK>{it['ch'].text_str}" for it in chunk_items])
            caption_text = ref_raw + tgt_raw

            cap_inputs = self.caption_tokenizer([caption_text], padding=True, return_tensors="pt")
            cap_ids, cap_am = cap_inputs.input_ids.to(device), cap_inputs.attention_mask.to(device)
            with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=compute_dtype):
                caption_embs = self.caption_encoder(cap_ids, return_dict=False, attention_mask=cap_am)[0]
            caption_embs = caption_embs * cap_am[..., None]

        # ---------- 4) 一次性 inference ----------
        modeling_type = getattr(getattr(self.dur_model, "config", None), "modeling_type", None)
        merged = map_phone_to_tokendict({'txt_token': ph_all, 'tone': tone_all}, pad_bos_eos=False)

        infer_kwargs = dict(
            txt_tokens=merged,
            start_pos=start_pos,
            condition=caption_embs if needs_caption else None,
            temperature=dur_disturb,
            topk=5,
            use_tqdm=self.use_tqdm
        )
        if modeling_type in ['ar', 'ar_cond_durtok', 'ar_dur']:
            infer_kwargs['dur_tokens'] = resource_context['dur_ref'].to(device)

        try:
            with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=compute_dtype):
                dur_pred_all = self.dur_model.inference(**infer_kwargs, spk_ids=spk_ids_all)
        except TypeError:
            merged['spk_ids'] = spk_ids_all
            infer_kwargs['txt_tokens'] = merged
            with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=compute_dtype):
                dur_pred_all = self.dur_model.inference(**infer_kwargs)

        dur_pred_all = dur_pred_all.to(torch.int)  # [1, T_all]

        # ---------- 5) （可选）按 spk 分组做 log-域静/非静音对齐 ----------
        if normalize_dur and dur_pred_all.shape[1] > 10:
            try:
                sil_ph_list = self.ling_dict['phone'].sil_phonemes()
            except Exception:
                sil_ph_list = []
            sil_ids = []
            for p in sil_ph_list:
                try:
                    sil_ids.append(self.ling_dict['phone'].encode(p)[0])
                except Exception:
                    pass
            sil_ids = list(set(sil_ids))

            sil_mask_pred_all = torch.zeros_like(ph_all, dtype=torch.bool)
            for sid_ in sil_ids:
                sil_mask_pred_all |= (ph_all == sid_)

            z_pred_all = torch.log1p(dur_pred_all.float())

            ref_by_spk = resource_context.get('ref_by_spk', {}) or {}
            present_sids = torch.unique(spk_ids_all[spk_ids_all > 0]).tolist()

            for sid in present_sids:
                ref_pack = ref_by_spk.get(int(sid))
                if not ref_pack:
                    continue
                ph_ref_spk  = ref_pack['ph_ref'].to(device)
                dur_ref_spk = ref_pack['dur_ref']
                if dur_ref_spk.dim() == 1:
                    dur_ref_spk = dur_ref_spk[None]
                dur_ref_spk = dur_ref_spk.to(device)

                sil_mask_ref = torch.zeros_like(ph_ref_spk, dtype=torch.bool)
                for sid_ in sil_ids:
                    sil_mask_ref |= (ph_ref_spk == sid_)

                z_ref_spk = torch.log1p(dur_ref_spk.float())

                sid_mask = (spk_ids_all == sid)
                if sid_mask.sum() == 0:
                    continue
                pred_sil_sid  = sil_mask_pred_all & sid_mask
                pred_non_sid  = (~sil_mask_pred_all) & sid_mask
                ref_sil_sid   = sil_mask_ref
                ref_non_sid   = ~sil_mask_ref

                if pred_sil_sid.any() and ref_sil_sid.any():
                    diff_sil = z_ref_spk[ref_sil_sid].mean() - z_pred_all[pred_sil_sid].mean()
                    z_pred_all[pred_sil_sid] += diff_sil

                if pred_non_sid.any() and ref_non_sid.any():
                    diff_non = z_ref_spk[ref_non_sid].mean() - z_pred_all[pred_non_sid].mean()
                    z_pred_all[pred_non_sid] += diff_non

            dur_pred_all = torch.expm1(z_pred_all).clamp_min(0.0)
            d_floor = torch.floor(dur_pred_all)
            frac    = (dur_pred_all - d_floor).clamp(0, 1)
            dur_pred_all = (d_floor + torch.bernoulli(frac)).to(torch.int)

        # ---------- 6) 切分回各 chunk ----------
        results = {}
        off = 0
        for ci, L in enumerate(lens):
            results[ci] = dur_pred_all[:, off:off + L]
            off += L
        return results

    def build_frontend_model(self):
        if self.use_old_aligner:
            from modules.tts.frontend_lm.whisper.whisper_small import Whisper
            self.aligner_lm = Whisper()
            load_ckpt(self.aligner_lm, f'{self.frontend_exp_name}', 'model')
            self.aligner_lm.eval()
            self.aligner_lm.to(self.device, dtype=self.precision)
            self.kv_cache = None
            self.hooks = None
        else:
            from modules.asr.scriptasr.build_model_utils import build_asr_model
            if self.frontend_exp_name.endswith('ckpt'):
                aligner_hparams = set_hparams(f'{Path(self.frontend_exp_name).parent}/config.yaml', global_hparams=False)
            else:
                aligner_hparams = set_hparams(f'{self.frontend_exp_name}/config.yaml', global_hparams=False)
            self.aligner_lm = build_asr_model(aligner_hparams, init_pretrained=False, vocab_size=6800, padding_idx=797)
            self.aligner_lm.eval()
            load_ckpt(self.aligner_lm, self.frontend_exp_name, 'model', strict=True, mmap=True)
            self.aligner_lm.to(self.device)

    def build_model(self, device):
        self.device = device

        # ---- set_hparams：兼容 .ckpt / 目录 ----
        if str(self.dit_exp_name).endswith('.ckpt'):
            set_hparams(f'{Path(self.dit_exp_name).parent}/config.yaml', print_hparams=False)
        else:
            set_hparams(f'{self.dit_exp_name}/config.yaml', print_hparams=False)
        hparams['use_fsdp'] = False

        # 字典
        ling_dict = json.load(open('egs/tts/megatts3_dict.json'))
        self.ling_dict = {k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov='<UNK>') for k in ['phone', 'tone']}
        self.token_encoder = token_encoder = self.ling_dict['phone']

        # 与单人版保持一致 —— 初始化发音替换表
        self.ph_replace_table = {'en': {}, 'zh': {}}

        # Dur
        self.build_dur_model()

        # DiT/WavVAE
        self._build_model()
        if self.use_ema and hparams.get('use_ema', False):
            load_ckpt(self.dit, f'{self.dit_exp_name}', 'ema_model', strict=False, mmap=True)
        else:
            load_ckpt(self.dit, f'{self.dit_exp_name}', 'dit', strict=False, mmap=True)
        # load_ckpt2(self.dit,  f'{self.dit_exp_name}', 'dit', strict=False, mmap=True, ckpt_path2='checkpoints/260113_megatts3_dit_dialogue_prompt/model_ckpt_steps_8000.ckpt',ckpt2_ratio=0)
        self.vae.eval(); self.vae.to(self.device, dtype=self.precision)
        self.dit.eval(); self.dit.to(self.device, dtype=self.precision)
        self.cfg_mask_token_phone = 302 - 1
        self.cfg_mask_token_tone  = 32  - 1
        if hparams.get('use_caption_encoder', False):
            self.caption_encoder.to(self.device, dtype=self.precision)

        # Frontend & ASR
        self.build_frontend_model()
        from modules.asr.sensevoice.sensevoice_api import build_asr_model
        self.asr_model = build_asr_model(self.device)

        # VAD
        from silero_vad import load_silero_vad
        self.vad_model = load_silero_vad()

        if self.compile_models:
            try:
                self.vae = torch.compile(self.vae, mode='max-autotune')
                self.dit = torch.compile(self.dit, mode='max-autotune')
                self.aligner_lm = torch.compile(self.aligner_lm, mode='max-autotune')
                self.dur_model = torch.compile(self.dur_model, mode='max-autotune')
            except Exception as e:
                print_once(f'| torch.compile failed, fallback to eager: {e}')

    # ---------------- Text normalize & helpers ----------------

    def preprocess_text(self, input_text: SSML, ph_replace_table=None, use_sa_frontend=False,
                        chunk_num_words_zh=None, chunk_num_words_en=None):
        chunk_num_words_zh = self.chunk_num_words_zh if chunk_num_words_zh is None else int(chunk_num_words_zh)
        chunk_num_words_en = self.chunk_num_words_en if chunk_num_words_en is None else int(chunk_num_words_en)
        def batch_replace(text: str, src: Union[str, List], tgt: str = ','):
            for p in src:
                text = text.replace(p, tgt)
            return text

        def _normalize_text_en(text: str):
            text_norm = common_preprocess(text)
            if not use_sa_frontend:
                text_norm = self.en_normalizer.normalize(text_norm)
            if ph_replace_table is not None:
                for src, tgt in ph_replace_table['en'].items():
                    text_norm = text_norm.replace(src, tgt)
            text_norm = common_process(text_norm)
            return text_norm

        def _normalize_text_zh(text):
            text_norm = common_preprocess(text)
            if not use_sa_frontend:
                from opencc import OpenCC
                jp2t_converter = OpenCC('jp2t')
                t2s_converter = OpenCC('t2s')
                text_norm = t2s_converter.convert(jp2t_converter.convert(text_norm))
                text_norm = self.zh_normalizer.normalize(text_norm)
            if ph_replace_table is not None:
                for src, tgt in ph_replace_table['zh'].items():
                    text_norm = text_norm.replace(src, tgt)
            text_norm = common_process(text_norm)
            return text_norm

        def common_process(text: str):
            text_norm = text
            if not use_sa_frontend:
                pause_punc = [
                    '~', '～', ':', '$', '¥', '&', '#', '@', '^', '・', '·', '‘', '’', '', '', "'", "'", '"', '"',
                    '（', '）', '(', ')', '', '', '{', '}', '「', '」', '[', ']', '<', '>', '《', '》',
                    '%', '*', '|', '｜', '\\', '/', '-', '+', '_', '=',
                    '²',
                ]
                text_norm = batch_replace(text_norm, pause_punc, tgt='')
            return text_norm

        def common_preprocess(text: str):
            special_symbols = ['&#34;']
            if use_sa_frontend:
                special_symbols.extend(['"'])
            text_norm = batch_replace(text, special_symbols, tgt='')
            text_norm = batch_replace(text_norm, ['\n'], tgt=' ')
            return text_norm

        input_text.apply_sub()
        try:
            language_type = classify_language(input_text.text_str)
        except LangDetectException as err:
            print_once('无法检测语言，默认选择中文')
            language_type = 'zh'

        if language_type == 'en':
            input_text.normalize(_normalize_text_en)
            text_segs = SSML.chunk_text_with_breaks(input_text, limit=chunk_num_words_en, language_type='en', debug=False)
        else:
            input_text.normalize(_normalize_text_zh)
            text_segs = SSML.chunk_text_with_breaks(input_text, limit=chunk_num_words_zh, language_type='zh', debug=False)

        return text_segs

    def refine_ph_tone(self, text: SSML, ph_pred: torch.Tensor, tone_pred: torch.Tensor):
        ph_tokens = ph_pred.squeeze().cpu().numpy()
        tone_tokens = tone_pred.squeeze().cpu().numpy()
        ph_tokens = self.ling_dict['phone'].decode(ph_tokens).split(' ')
        tone_tokens = self.ling_dict['tone'].decode(tone_tokens).split(' ')

        # 处理儿化等
        ph_tokens_, tone_tokens_ = [], []
        for p_i, p in enumerate(ph_tokens):
            if (p_i > 0 and p == "C0er" and ph_tokens[p_i - 1] in SHENGMU) or (p in YUNMU_ERHUA):
                ph_tokens_.append(p[:-1]); tone_tokens_.append(tone_tokens[p_i])
                ph_tokens_.append("C0er"); tone_tokens_.append('5')
            else:
                ph_tokens_.append(p); tone_tokens_.append(tone_tokens[p_i])
        ph_tokens, tone_tokens = ph_tokens_, tone_tokens_

        text_, ph_tokens, ph2word = align_word_phone(text.text_str, ph_tokens)
        ph2word = [p-1 for p in ph2word]

        ph_tokens, tone_tokens, ph2word = SSML.replace_ph_tone(text, ph_tokens, tone_tokens, ph2word)

        ph_tokens = self.ling_dict['phone'].encode(' '.join(ph_tokens))
        ph_pred = torch.LongTensor(ph_tokens)[None].to(ph_pred)
        tone_tokens = self.ling_dict['tone'].encode(' '.join(tone_tokens))
        tone_pred = torch.LongTensor(tone_tokens)[None].to(tone_pred)
        return ph_pred, tone_pred, ph2word

    def add_breaks(self, text: SSML, ph_pred: torch.Tensor, tone_pred: torch.Tensor, dur_pred: torch.Tensor,
                   ph2word: List, break_token=145, break_tone=3):
        ph_tokens = ph_pred.squeeze().cpu().numpy().tolist()
        tone_tokens = tone_pred.squeeze().cpu().numpy().tolist()
        dur_tokens = dur_pred.squeeze().cpu().numpy().tolist()
        ph_tokens, tone_tokens, ph2word, dur_tokens = SSML.add_breaks(
            text, ph_tokens, tone_tokens, ph2word, dur_tokens, break_token, break_tone, 0.01
        )
        ph_pred = torch.Tensor(ph_tokens)[None].to(ph_pred)
        tone_pred = torch.Tensor(tone_tokens)[None].to(tone_pred)
        dur_pred = torch.Tensor(dur_tokens)[None].to(dur_pred)
        return ph_pred, tone_pred, dur_pred, ph2word

    def make_word_timestamps(self, text: SSML, dur_pred: np.ndarray, ph2word: List):
        dur_timestep = 0.01
        offsets = [0] + np.cumsum(dur_pred).tolist()
        words_to_get = get_word_list(text.text_str)
        ph2word = ph2word + [-3]
        words, timestamps = [], []
        ph_start_idx = 0
        for ph_end_idx in range(1, len(ph2word)):
            if ph2word[ph_end_idx] != ph2word[ph_start_idx]:
                if ph2word[ph_start_idx] >= 0:
                    words.append(words_to_get[ph2word[ph_start_idx]])
                    timestamps.append([offsets[ph_start_idx] * dur_timestep, offsets[ph_end_idx] * dur_timestep])
                ph_start_idx = ph_end_idx

        text_merged, text_norm_merged, text_idx_merged, text_norm_idx_merged = merge_norm_alignment(
            text.origin.text_str, words, debug=False
        )

        words_merged, timestamps_merged = [], []
        word_idx = 0
        for merge_idx in range(len(text_merged)):
            if isinstance(text_merged[merge_idx], list):
                word_merged, timestamp_merged = [], []
                for i in range(len(text_merged[merge_idx])):
                    if len(word_merged) > 0 and is_english(word_merged[-1]) and is_english(text_merged[merge_idx][i]):
                        word_merged.append(' ')
                    word_merged.append(text_merged[merge_idx][i])
                for i in range(len(text_norm_merged[merge_idx])):
                    timestamp_merged.append(timestamps[word_idx]); word_idx += 1
                words_merged.append(''.join(word_merged))
                if len(timestamp_merged) > 0:
                    timestamps_merged.append([timestamp_merged[0][0], timestamp_merged[-1][-1]])
                else:
                    if len(timestamps_merged) <= 0:
                        timestamps_merged.append([0.0, 0.0])
                    else:
                        timestamps_merged.append([timestamps_merged[-1][-1], timestamps_merged[-1][-1]])
            else:
                words_merged.append(text_merged[merge_idx])
                timestamps_merged.append(timestamps[word_idx]); word_idx += 1

        return {'words': words_merged, 'timestamps': timestamps_merged}

    def chunk_wavs_vad(self, wav_16k=None, speech_timestamps=None,
                        chunk_duration=10, max_duration=60,
                        vad_thresholds=(0.50, 0.35, 0.25),
                        min_speech_ms=150, min_silence_ms=100):
        """
        并行安全版：
        - Silero VAD 是有状态的，同一实例跨线程使用会发生竞态；
        - 这里对 get_speech_timestamps 加锁，并在每次调用前 reset_states()！
        """
        if speech_timestamps is None and wav_16k is not None:
            from silero_vad import get_speech_timestamps
            # 限制输入长度，和原逻辑一致
            wav_16k = wav_16k[: int(16000 * max_duration * 1.2)]

            for thr in vad_thresholds:
                try:
                    # 关键：加锁 + reset，避免多线程破坏内部 RNN 状态
                    with self.lock:
                        if hasattr(self.vad_model, "reset_states"):
                            self.vad_model.reset_states()
                        st = get_speech_timestamps(
                            wav_16k, self.vad_model, return_seconds=True,
                            threshold=thr,
                            min_speech_duration_ms=min_speech_ms,
                            min_silence_duration_ms=min_silence_ms
                        )
                except Exception as e:
                    # 极端情况下兜底为整段（不影响后续逻辑）
                    print_once(f'| VAD failed with threshold={thr}: {e}; fallback to full clip.')
                    total_sec = (len(wav_16k) / 16000.0)
                    st = [{'start': 0.0, 'end': min(total_sec, float(max_duration))}]

                if len(st) > 0:
                    speech_timestamps = st
                    print_once(f'| VAD detected {len(st)} segments with threshold={thr}')
                    break
            else:
                speech_timestamps = []

        # 后处理与切块逻辑保持不变
        start = max(0.0, float(speech_timestamps[0]['start'])) if len(speech_timestamps) else 0.0
        end   = min(float(speech_timestamps[-1]['end']), float(max_duration)) if len(speech_timestamps) else float(max_duration)
        if end <= start:
            total_dur = (len(wav_16k) / 16000.0) if wav_16k is not None else float(max_duration)
            end = max(start + 0.1, min(total_dur, max_duration))

        offs, cur = [], start
        while cur < end - 1e-6:
            nxt = min(end, cur + float(chunk_duration))
            offs.append((cur, nxt))
            if nxt == cur:
                break
            cur = nxt
        return offs

    def _silence_by_speech_timestamps(
        self,
        wav: np.ndarray,
        sr: int,
        speech_timestamps: List[Dict[str, float]],
        fade_ms: float = 10.0,
    ) -> np.ndarray:
        """
        将 VAD 判为 non-speech 的区域直接置 0（保留 speech 区域）。
        speech_timestamps: [{'start':sec,'end':sec}, ...]
        """
        if wav is None or wav.size == 0:
            return wav
        if not speech_timestamps:
            # VAD 没结果：保守起见不改
            return wav

        T = int(wav.shape[0])
        mask = np.zeros(T, dtype=np.float32)

        fade = int(round(sr * (fade_ms / 1000.0)))
        fade = max(0, fade)

        for seg in speech_timestamps:
            s0 = int(round(float(seg["start"]) * sr))
            s1 = int(round(float(seg["end"]) * sr))
            s0 = max(0, min(s0, T))
            s1 = max(0, min(s1, T))
            if s1 <= s0:
                continue

            mask[s0:s1] = 1.0

            # 边界淡入淡出，避免爆音/click
            if fade > 0:
                # fade-in
                a1 = min(s0 + fade, s1)
                if a1 > s0:
                    mask[s0:a1] = np.maximum(
                        mask[s0:a1],
                        np.linspace(0.0, 1.0, a1 - s0, endpoint=False, dtype=np.float32)
                    )
                # fade-out
                b0 = max(s1 - fade, s0)
                if s1 > b0:
                    mask[b0:s1] = np.maximum(
                        mask[b0:s1],
                        np.linspace(1.0, 0.0, s1 - b0, endpoint=False, dtype=np.float32)
                    )

        return (wav.astype(np.float32) * mask).astype(np.float32)

    def _boundaries_from_vad_silence(
        self,
        speech_timestamps: List[Dict[str, float]],
        total_samples: int,
        sr: int,
        min_gap_ms: float = 300.0,
        cut_mode: str = "mid",   # "mid" | "sil_start" | "sil_end"
    ) -> np.ndarray:
        """
        根据 VAD speech 段之间的静音间隙切段：
        - gap >= min_gap_ms 才认为是“段落边界”
        - cut_mode:
            - "mid":     切在静音中点（最推荐，边界一定落在静音里）
            - "sil_start":切在静音开始（=上一段 speech end）
            - "sil_end": 切在静音结束（=下一段 speech start）
        返回: int64 sample 索引边界，含 0 和 total_samples
        """
        if total_samples <= 0:
            return np.asarray([0], dtype=np.int64)

        if not speech_timestamps:
            return np.asarray([0, total_samples], dtype=np.int64)

        # 按 start 排序（保险）
        st = sorted(
            [{"start": float(x["start"]), "end": float(x["end"])} for x in speech_timestamps],
            key=lambda x: x["start"]
        )

        min_gap = float(min_gap_ms) / 1000.0
        boundaries = [0]

        # 段落边界来自 speech_i.end 与 speech_{i+1}.start 之间的静音
        for i in range(len(st) - 1):
            a = st[i]["end"]
            b = st[i + 1]["start"]
            gap = b - a
            if gap < min_gap:
                continue

            if cut_mode == "sil_start":
                cut_t = a
            elif cut_mode == "sil_end":
                cut_t = b
            else:
                cut_t = 0.5 * (a + b)  # mid

            cut_samp = int(round(cut_t * sr))
            cut_samp = max(0, min(cut_samp, total_samples))
            boundaries.append(cut_samp)

        boundaries.append(total_samples)

        # 单调 & 去重 & 至少递增 1 sample，避免空段
        boundaries = np.asarray(boundaries, dtype=np.int64)
        boundaries = np.clip(boundaries, 0, total_samples)
        boundaries = np.unique(boundaries)
        boundaries.sort()

        if boundaries[0] != 0:
            boundaries = np.insert(boundaries, 0, 0)
        if boundaries[-1] != total_samples:
            boundaries = np.append(boundaries, total_samples)

        # 避免相等导致空段
        for i in range(1, len(boundaries)):
            if boundaries[i] <= boundaries[i - 1]:
                boundaries[i] = min(total_samples, boundaries[i - 1] + 1)
        boundaries[-1] = total_samples
        return boundaries


    def _get_vad_speech_timestamps(
        self,
        wav_24k: np.ndarray,
        vad_thresholds=(0.50, 0.35, 0.25),
        min_speech_ms=150,
        min_silence_ms=100,
    ) -> List[Dict[str, float]]:
        """
        对生成的 wav（24k）跑 silero VAD，返回 speech 段列表（单位：秒）。
        线程安全：加锁 + reset_states。
        """
        if wav_24k is None or wav_24k.size < int(self.sr * 0.3):
            return []

        try:
            from silero_vad import get_speech_timestamps
        except Exception as e:
            print_once(f"| silero_vad not available: {e}")
            return []

        # silero 期望 16k
        wav_16k = librosa.resample(wav_24k.astype(np.float32), orig_sr=self.sr, target_sr=16000)
        wav_16k = wav_16k.astype(np.float32)

        for thr in vad_thresholds:
            try:
                with self.lock:
                    if hasattr(self.vad_model, "reset_states"):
                        self.vad_model.reset_states()
                    st = get_speech_timestamps(
                        wav_16k, self.vad_model,
                        return_seconds=True,
                        threshold=float(thr),
                        min_speech_duration_ms=int(min_speech_ms),
                        min_silence_duration_ms=int(min_silence_ms),
                    )
            except Exception as e:
                print_once(f"| VAD failed thr={thr}: {e}")
                st = []

            if st:
                return st

        return []


    def _snap_boundaries_to_vad_silence(
            self,
            wav_24k: np.ndarray,
            boundaries: np.ndarray,          # int64, sample indices @24k
            search_ms: float = 250.0,
            vad_thresholds=(0.50, 0.35, 0.25),
            min_speech_ms=150,
            min_silence_ms=100,
            fallback_energy_min: bool = True,
            speech_timestamps: Optional[List[Dict[str, float]]] = None,  # <<< 新增：外部传入，避免重复跑VAD
        ) -> np.ndarray:

        """
        将 duration 计算出来的边界吸附到最近的 VAD 静音点（non-speech 区间）。
        - 只调整内部边界（不动 0 和末尾）。
        - 只在 ±search_ms 内找到静音点才吸附，否则保持原边界。
        - 若 VAD 不可用/无静音区间：可选用能量最小点兜底（fallback_energy_min）。
        """
        if wav_24k is None or wav_24k.size == 0:
            return boundaries
        if boundaries is None or len(boundaries) <= 2:
            return boundaries

        total = int(wav_24k.shape[0])
        boundaries = boundaries.astype(np.int64).copy()
        boundaries[0] = 0
        boundaries[-1] = total
        boundaries = np.clip(boundaries, 0, total)
        boundaries = np.maximum.accumulate(boundaries)

        # 复用外部 VAD 结果；若没传才内部跑一次（兼容旧调用）
        if speech_timestamps is not None:
            st = speech_timestamps
        else:
            st = self._get_vad_speech_timestamps(
                wav_24k,
                vad_thresholds=vad_thresholds,
                min_speech_ms=min_speech_ms,
                min_silence_ms=min_silence_ms,
            )


        # --- 构建 silence 区间（sample @24k）---
        sil_intervals = []
        if st:
            # st: [{'start':s, 'end':e}, ...] seconds
            prev = 0.0
            dur_sec = total / float(self.sr)
            for seg in st:
                s0 = max(0.0, float(seg["start"]))
                s1 = min(dur_sec, float(seg["end"]))
                if s0 > prev + 1e-4:
                    sil_intervals.append((int(round(prev * self.sr)), int(round(s0 * self.sr))))
                prev = max(prev, s1)
            if prev < dur_sec - 1e-4:
                sil_intervals.append((int(round(prev * self.sr)), total))
        else:
            # 没检测到 speech：等价于全静音
            sil_intervals = [(0, total)]

        search = int(round(search_ms / 1000.0 * self.sr))

        def _nearest_point_in_silence(b: int) -> Tuple[int, int]:
            """返回 (best_point, best_dist)."""
            best_p = b
            best_d = 10**18
            for a, c in sil_intervals:
                if c <= a:
                    continue
                # clamp 到 [a, c]
                p = b
                if b < a:
                    p = a
                elif b > c:
                    p = c
                d = abs(p - b)
                if d < best_d:
                    best_d = d
                    best_p = p
            return best_p, best_d

        # 可选：能量最小点兜底（在 search window 内找最安静的 frame）
        def _energy_min_in_window(b: int) -> int:
            # 20ms hop 的短时 RMS
            win = int(round(0.02 * self.sr))
            hop = int(round(0.01 * self.sr))
            l = max(0, b - search)
            r = min(total, b + search)
            x = wav_24k[l:r].astype(np.float32)
            if x.size < win:
                return b
            # frame RMS
            nfrm = 1 + (x.size - win) // hop
            rms = []
            for i in range(nfrm):
                s = i * hop
                frm = x[s:s+win]
                rms.append(float(np.mean(frm * frm)))
            k = int(np.argmin(rms))
            return l + k * hop + win // 2

        # --- 吸附内部边界 ---
        for i in range(1, len(boundaries) - 1):
            b = int(boundaries[i])
            p, d = _nearest_point_in_silence(b)
            if d <= search:
                boundaries[i] = p
            elif fallback_energy_min:
                boundaries[i] = _energy_min_in_window(b)

        # 保证单调 & 不越界
        boundaries[0] = 0
        boundaries[-1] = total
        boundaries = np.clip(boundaries, 0, total)
        boundaries = np.maximum.accumulate(boundaries)

        # 避免重复边界导致大量空段（至少递增 1 sample）
        for i in range(1, len(boundaries)):
            if boundaries[i] <= boundaries[i-1]:
                boundaries[i] = min(total, boundaries[i-1] + 1)
        boundaries[-1] = total
        return boundaries


    def _print_target_ph_durations_seconds(self,
                                        ph_seq_target: torch.Tensor,   # [1, N_tgt]
                                        tone_seq_target: torch.Tensor, # [1, N_tgt]
                                        dur_seq_target: torch.Tensor,  # [1, N_tgt]（单位：0.01s）
                                        tag: str = "TARGET"):
        """
        仅打印 target 段的逐 phone 时长（单位：秒），包含 start/end（相对 target 起点，0s 开始），并同步打印 tone！
        """
        ph_seq = ph_seq_target.squeeze(0).detach().cpu()
        tone_seq = tone_seq_target.squeeze(0).detach().cpu()
        dur_seq = dur_seq_target.squeeze(0).detach().cpu()

        L = int(min(ph_seq.numel(), tone_seq.numel(), dur_seq.numel()))
        ph_seq = ph_seq[:L]
        tone_seq = tone_seq[:L]
        dur_seq = dur_seq[:L]

        ph_list = self.ling_dict['phone'].decode(ph_seq.numpy()).split(' ')
        tone_list = self.ling_dict['tone'].decode(tone_seq.numpy()).split(' ')
        if len(ph_list) != L:
            ph_list = ph_list[:L]
        if len(tone_list) != L:
            tone_list = tone_list[:L]

        starts_cs = np.cumsum([0] + dur_seq.numpy().tolist()[:-1]).tolist()
        total_sec = float(dur_seq.sum().item()) / 100.0
        print(f"[PH/TIME][{tag}] total_phones={L}, total_dur={total_sec:.3f}s")
        print(f"[PH/TIME][{tag}] {'idx':>4} | {'ph':>10} | {'tone':>4} | {'dur(s)':>8} | {'start(s)':>10} | {'end(s)':>10}")

        for i in range(L):
            dur_s = float(dur_seq[i].item()) / 100.0
            start_s = float(starts_cs[i]) / 100.0
            end_s = start_s + dur_s
            print(f"[PH/TIME][{tag}] {i:4d} | {ph_list[i]:>10} | {tone_list[i]:>4} | {dur_s:8.3f} | {start_s:10.3f} | {end_s:10.3f}")


    def _print_dit_inputs_debug(self, tag: str, caption_str: str, prompt_text: str, text_inputs=None, max_ids: int = 64):
        """
        打印送入 DiT 的 caption 与 text（统一为 <S{sid}>...</S{sid}> 串）：
        - 原始字符串
        - 可见 token 数（去掉 pad）
        - 前 max_ids 个 token id
        - 反解码文本（不跳过 special tokens）
        通过设置环境变量 MEGA_PRINT_DIT_TEXT=0 可关闭打印！
        """

        # --- Caption ---
        try:
            cap_inputs = self.caption_tokenizer([caption_str], padding=True, return_tensors="pt")
            cap_ids  = cap_inputs.input_ids[0]
            cap_mask = cap_inputs.attention_mask[0].bool()
            cap_vis  = cap_ids[cap_mask]
            cap_head = cap_vis[:max_ids].tolist()
            print(f"[DiT INPUT][{tag}] caption='{caption_str}'")
            print(f"[DiT INPUT][{tag}] caption_token_len={cap_vis.numel()}  ids(head {len(cap_head)}): {cap_head}{'...' if cap_vis.numel()>max_ids else ''}")
            try:
                cap_dec = self.caption_tokenizer.decode(cap_vis, skip_special_tokens=False)
                print(f"[DiT INPUT][{tag}] caption_decoded='{cap_dec}'")
            except Exception as e:
                print(f"[DiT INPUT][{tag}] caption_decode_error: {e}")
        except Exception as e:
            print(f"[DiT INPUT][{tag}] caption_tokenize_error: {e}")

        # --- Text / Prompt ---
        try:
            if text_inputs is None:
                txt_inputs = self.dit_text_tokenizer(prompt_text, padding=True, return_tensors="pt")
            else:
                txt_inputs = text_inputs  # 允许直接传 GPU 张量
            ids  = txt_inputs["input_ids"][0].detach().to("cpu")
            mask = txt_inputs["attention_mask"][0].detach().to("cpu").bool()
            vis  = ids[mask]
            head = vis[:max_ids].tolist()

            print(f"[DiT INPUT][{tag}] text='{prompt_text}'")
            print(f"[DiT INPUT][{tag}] text_token_len={vis.numel()}  ids(head {len(head)}): {head}{'...' if vis.numel()>max_ids else ''}")
            try:
                dec = self.dit_text_tokenizer.decode(vis, skip_special_tokens=False)
                print(f"[DiT INPUT][{tag}] text_decoded='{dec}'")
            except Exception as e:
                print(f"[DiT INPUT][{tag}] text_decode_error: {e}")
        except Exception as e:
            print(f"[DiT INPUT][{tag}] text_tokenize_error: {e}")

    def _parse_dialogue_segments(self, raw_text: str) -> List[Tuple[int, str]]:
        """
        只解析一种格式：
            <SPK>1</SPK>...<SPK>2</SPK>...<SPK>1</SPK>...
        """

        segs: List[Tuple[int, str]] = []

        pat_spk = re.compile(r"<\s*SPK\s*>\s*(\d+)\s*<\s*/\s*SPK\s*>", re.IGNORECASE)

        parts = pat_spk.split(raw_text)
        # parts 形如： [preamble, sid1, text1, sid2, text2, ...]

        # preamble（在第一个标签前的裸文本），若非空则归为 sid=1
        preamble = (parts[0] or "").strip()
        if preamble:
            segs.append((1, parts[0]))

        for i in range(1, len(parts), 2):
            sid = int(parts[i])

            text_i = parts[i + 1] if (i + 1) < len(parts) else ""
            if text_i is None:
                text_i = ""
            if text_i.strip():
                segs.append((sid, text_i))

        if not segs:
            assert False,f'no segs'
        return segs

    def preprocess(self, audio_bytes: Union[bytes, List[bytes], Tuple[bytes, bytes]],
                wav_path: Optional[str]=None, topk_dur=1, ref_texts: Optional[List[str]]=None, **kwargs):
        """
        audio_bytes: bytes 或 [bytes, bytes]（两位说话人的参考音频）
        ref_texts:   None 或 [str, str]（两人的 clean 文本，不含标签；若不给则用 ASR）

        返回 resource_context 增加：
        - text_ref_clean: 无标签的参考文本（拼接）
        - text_ref_raw:   '<SPK>1</SPK>ref1<SPK>2</SPK>ref2'（单人则 '<SPK>1</SPK>ref'）
        - text_ref_by_spk:{sid: clean_ref_text_for_that_speaker}
        - spk_ids_ref:    [1, Tph_ref] 1-based phone-level 说话人 id
        - ref_by_spk:     {sid: {'ph_ref','tone_ref','dur_ref'}}
        - （当 use_old_dur=True 时）dur_prefill_by_spk: {sid: {'incremental_state','ctx_dur_tokens','last_pos'}}
        """
        def _convert_to_wav_and_16k(ab):
            wav_bytes = convert_to_wav_bytes(ab)
            wav_24k, _ = librosa.core.load(wav_bytes, sr=self.sr)
            ws = hparams['win_size']
            if len(wav_24k) % ws < ws - 1:
                wav_24k = np.pad(
                    wav_24k,
                    (0, ws - 1 - (len(wav_24k) % ws)),
                    mode='constant',
                    constant_values=0.0
                ).astype(np.float32)
            wav_24k = np.pad(wav_24k, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)
            wav_16k = librosa.resample(wav_24k, orig_sr=self.sr, target_sr=16000)
            return wav_24k, wav_16k

        @torch.no_grad()
        def _process_alignment(alignment_tokens, prompt_max_frame):
            ph_ref, tone_ref, dur_ref, _ = split_ph_timestamp(deepcopy(alignment_tokens))
            ph_ref = torch.Tensor(ph_ref)[None].to(self.device)
            tone_ref = torch.Tensor(tone_ref)[None].to(self.device)

            if dur_ref.sum() < prompt_max_frame:
                dur_ref[-1] += prompt_max_frame - dur_ref.sum()
            elif dur_ref.sum() > prompt_max_frame:
                len_diff = dur_ref.sum() - prompt_max_frame
                while len_diff > 0:
                    for i in range(len(dur_ref)):
                        dur_ref[i] -= 1; len_diff -= 1
                        if len_diff == 0: break
                    if len_diff == 0: break

            mel2ph_ref = self.length_regulator(torch.LongTensor(dur_ref)[None].to(self.device)).to(self.device)
            mel2ph_ref = mel2ph_ref[:, :mel2ph_ref.size(1)//self.fm*self.fm]
            dur_ref = mel2token_to_dur(mel2ph_ref)
            return ph_ref, tone_ref, dur_ref, mel2ph_ref

        @torch.no_grad()
        def _align_one_ref(wav_24k, wav_16k):
            chunk_wav_offsets = self.chunk_wavs_vad(wav_16k, chunk_duration=10, max_duration=self.max_ref_duration)
            print(f"Detected {len(chunk_wav_offsets)} speech segments in one reference audio.")
            s0, s1 = int(chunk_wav_offsets[0][0]*self.sr), int(chunk_wav_offsets[-1][-1]*self.sr)
            wav_24k_ = wav_24k[s0:s1]
            s0k, s1k = int(chunk_wav_offsets[0][0]*16000), int(chunk_wav_offsets[-1][-1]*16000)
            wav_16k_ = wav_16k[s0k:s1k]
            
            # === dereverb on trimmed ref ===
            # if getattr(self, "dereverb_ref", True):
            #     wav_24k_ = self.dereverb_wpe_mono(
            #         wav_24k_, sr=self.sr,
            #         n_fft=1024, hop=256,
            #         taps=getattr(self, "wpe_taps", 10),
            #         delay=getattr(self, "wpe_delay", 3),
            #         iterations=getattr(self, "wpe_iters", 3),
            #         rms_match=True,
            #     )
            #     # 重新得到 16k（后面 VAD 已用过，不需要再跑一次 VAD；但 aligner/asr 用更新后的更好）
            #     wav_16k_ = librosa.resample(wav_24k_, orig_sr=self.sr, target_sr=16000)


            if self.use_old_aligner:
                fm = 160 * 8
                ph_lst, tone_lst, dur_lst, m2p_lst = [], [], [], []
                base = chunk_wav_offsets[0][0]
                for (chunk_start, chunk_end) in chunk_wav_offsets:
                    rel0 = max(0.0, chunk_start - base)
                    rel1 = max(rel0, chunk_end - base)
                    c0 = int((rel0 * 16000) // fm * fm)
                    c1 = int((rel1 * 16000) // fm * fm)
                    wav_16k_chunk = wav_16k_[c0:c1]

                    with model_lock(self.lock):
                        mel = torch.tensor(whisper.log_mel_spectrogram(wav_16k_chunk).T, dtype=self.precision).to(self.device)[None].transpose(1,2)
                        prompt_max_frame = mel.size(2) // self.fm * self.fm
                        mel = mel[:, :, :prompt_max_frame]
                        token = torch.LongTensor([[798]]).to(self.device)
                        audio_features = self.aligner_lm.embed_audio(mel)
                        for _ in tqdm(range(1024)) if self.use_tqdm else range(1024):
                            logits = self.aligner_lm.logits(token, audio_features, None)
                            token_pred = torch.argmax(F.softmax(logits[:, -1], dim=-1), 1)[None]
                            token = torch.cat([token, token_pred], dim=1)
                            if token_pred[0] == 799: break
                        alignment_tokens = token[0, 1:-1].detach().to("cpu")

                    ph_i, tone_i, dur_i, m2p_i = _process_alignment(alignment_tokens, prompt_max_frame)
                    ph_lst.append(ph_i); tone_lst.append(tone_i); dur_lst.append(dur_i); m2p_lst.append(m2p_i)

                ph_ref = torch.cat(ph_lst, dim=1)
                tone_ref = torch.cat(tone_lst, dim=1)
                dur_ref  = torch.cat(dur_lst, dim=1) if dur_lst[0].dim()==2 else torch.cat([d if d.dim()==2 else d[None] for d in dur_lst], dim=1)
                mel2ph_ref = torch.cat(m2p_lst, dim=1)

            else:
                with model_lock(self.lock):
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        whisper_wav = torch.from_numpy(wav_16k_)[None].to(self.device, non_blocking=True)
                        whisper_wav = whisper_wav[:, :whisper_wav.shape[-1] // 1280 * 1280]
                        prompt_max_frame = whisper_wav.shape[-1] // 160 // 8 * 8
                        token = torch.LongTensor([798])[None, :].to(self.device)
                        token = self.aligner_lm.inference(
                            whisper_wav, token, topk=1, temperature=0.7,
                            max_new_tokens=16384, eos_idx=799, use_tqdm=False
                        )
                    alignment_tokens = token[0].detach().to("cpu")
                ph_ref, tone_ref, dur_ref, mel2ph_ref = _process_alignment(alignment_tokens, prompt_max_frame)

            from modules.asr.sensevoice.sensevoice_api import run_asr_model

            # ===== reference 上 VAD，把 non-speech 直接置 0（裁剪之后，ASR 之前）=====
            wav_16k_asr = wav_16k_

            # 复用你已有的线程安全 VAD（它内部会把 24k resample 到 16k）
            vad_st = self._get_vad_speech_timestamps(
                wav_24k_,  # 注意：这里用裁剪+去混响后的 24k 参考
                vad_thresholds=(0.50, 0.35, 0.25),
                min_speech_ms=150,
                min_silence_ms=100,
            )
            if vad_st:
                wav_16k_asr = self._silence_by_speech_timestamps(
                    wav_16k_asr.astype(np.float32),
                    sr=16000,
                    speech_timestamps=vad_st,
                    fade_ms=10.0,
                )
                    
            wav_24k_ = self._silence_by_speech_timestamps(
                wav_24k_.astype(np.float32),
                sr=self.sr,
                speech_timestamps=vad_st,
                fade_ms=10.0,
            )

            with model_lock(self.lock):
                text_ref = run_asr_model([wav_16k_asr], self.asr_model, with_segments=False)[0]['text_normed']


            return {
                'wav_24k': wav_24k_,
                'ph_ref': ph_ref, 'tone_ref': tone_ref,
                'dur_ref': dur_ref, 'mel2ph_ref': mel2ph_ref,
                'text_ref': text_ref,
            }

        # ========= 单/双参考入口 =========
        if isinstance(audio_bytes, (list, tuple)):
            assert len(audio_bytes) == 2, "双说话人请传入两个 bytes！"
            w24_1, w16_1 = _convert_to_wav_and_16k(audio_bytes[0])
            w24_2, w16_2 = _convert_to_wav_and_16k(audio_bytes[1])

            ret1 = _align_one_ref(w24_1, w16_1)
            ret2 = _align_one_ref(w24_2, w16_2)

            ph1, tn1, dr1, m2p1 = ret1['ph_ref'], ret1['tone_ref'], ret1['dur_ref'], ret1['mel2ph_ref']
            ph2, tn2, dr2, m2p2 = ret2['ph_ref'], ret2['tone_ref'], ret2['dur_ref'], ret2['mel2ph_ref']
            if dr1.dim() == 1: dr1 = dr1[None]
            if dr2.dim() == 1: dr2 = dr2[None]

            T1 = ph1.shape[1]
            m2p2_off = m2p2 + (m2p2 > 0).long() * T1

            ph_ref   = torch.cat([ph1,   ph2],   dim=1)
            tone_ref = torch.cat([tn1,   tn2],   dim=1)
            dur_ref  = torch.cat([dr1,   dr2],   dim=1)
            mel2ph_ref = torch.cat([m2p1, m2p2_off], dim=1)

            # 文本（拼接版 + 逐说话人版）
            if isinstance(ref_texts, (list, tuple)) and len(ref_texts) == 2:
                text_ref_by_spk = {1: ref_texts[0], 2: ref_texts[1]}
                text_ref_clean  = ref_texts[0] + ref_texts[1]
                text_ref_raw    = f"<SPK>1</SPK>{ref_texts[0]}<SPK>2</SPK>{ref_texts[1]}"
            else:
                text_ref_by_spk = {1: ret1['text_ref'], 2: ret2['text_ref']}
                text_ref_clean  = ret1['text_ref'] + ret2['text_ref']
                text_ref_raw    = f"<SPK>1</SPK>{ret1['text_ref']}<SPK>2</SPK>{ret2['text_ref']}"

            wav_24k_full = np.concatenate([ret1['wav_24k'], ret2['wav_24k']])
            with model_lock(self.lock):
                wav = torch.tensor(wav_24k_full, dtype=self.precision, device=self.device)[None]
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    vae_latent = self.vae.encode_latent(wav)
            vae_latent = vae_latent[:, :mel2ph_ref.size(1)//4]

            T2 = ph2.shape[1]
            spk_ids_ref = torch.cat([
                torch.full((1, T1), 1, dtype=torch.long),
                torch.full((1, T2), 2, dtype=torch.long)
            ], dim=1)

            ref_by_spk = {
                1: {'ph_ref': ph1.cpu(), 'tone_ref': tn1.cpu(), 'dur_ref': (dr1 if dr1.dim()==2 else dr1[None]).cpu()},
                2: {'ph_ref': ph2.cpu(), 'tone_ref': tn2.cpu(), 'dur_ref': (dr2 if dr2.dim()==2 else dr2[None]).cpu()},
            }

        else:
            wav_bytes = convert_to_wav_bytes(audio_bytes)
            w24, _ = librosa.core.load(wav_bytes, sr=self.sr)
            ws = hparams['win_size']
            if len(w24) % ws < ws - 1:
                w24 = np.pad(w24, (0, ws - 1 - (len(w24) % ws)), mode='constant', constant_values=0.0).astype(np.float32)
            w24 = np.pad(w24, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)
            w16 = librosa.resample(w24, orig_sr=self.sr, target_sr=16000)

            ret = _align_one_ref(w24, w16)
            ph_ref, tone_ref, dur_ref, mel2ph_ref = ret['ph_ref'], ret['tone_ref'], ret['dur_ref'], ret['mel2ph_ref']
            if dur_ref.dim() == 1: dur_ref = dur_ref[None]
            text_ref_clean  = ret['text_ref']
            text_ref_raw    = f"<SPK>1</SPK>{ret['text_ref']}"
            text_ref_by_spk = {1: ret['text_ref']}

            if topk_dur > 1: self.dur_model.hparams["infer_top_k"] = topk_dur
            else:            self.dur_model.hparams["infer_top_k"] = None

            with model_lock(self.lock):
                wav = torch.tensor(w24, dtype=self.precision, device=self.device)[None]
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    vae_latent = self.vae.encode_latent(wav)
            vae_latent = vae_latent[:, :mel2ph_ref.size(1)//4]
            spk_ids_ref = torch.ones((1, ph_ref.shape[1]), dtype=torch.long)

            ref_by_spk = {
                1: {'ph_ref': ph_ref.cpu(), 'tone_ref': tone_ref.cpu(), 'dur_ref': (dur_ref if dur_ref.dim()==2 else dur_ref[None]).cpu()}
            }

        # ========= Duration Prompting =========
        if self.use_old_dur:
            dur_tokens_2d_ = mel2token_to_dur(mel2ph_ref, ph_ref.shape[1]).clamp(
                max=self.hp_dur_model.get('dur_code_size', self.hp_dur_model.get('dur_max_value', 128)) - 1) + 1
            ctx_dur_tokens = dur_tokens_2d_.clone().flatten(0, 1).to(self.device)
            txt_tokens_flat_ = ph_ref.flatten(0, 1)
            ctx_dur_tokens = ctx_dur_tokens[txt_tokens_flat_ > 0][None]
            last_dur_pos_prompt = ctx_dur_tokens.shape[1]
            dur_spk_pos_ids_flat = torch.arange(0, last_dur_pos_prompt, device=mel2ph_ref.device)[None, :].long()
            with model_lock(self.lock):
                _, incremental_state_dur_prompt = self.dur_model.infer(
                    ph_ref, {'tone': tone_ref}, None, None, None,
                    ctx_vqcodes=ctx_dur_tokens, spk_pos_ids_flat=dur_spk_pos_ids_flat, return_state=True)

            dur_prefill_by_spk = self._build_old_dur_prefill_by_speaker(ref_by_spk)

            ret_dur = {
                'incremental_state_dur_prompt': incremental_state_dur_prompt,
                'ctx_dur_tokens': ctx_dur_tokens,
                'last_dur_pos_prompt': last_dur_pos_prompt,
                'dur_prefill_by_spk': dur_prefill_by_spk,
            }
        else:
            merged_ph_tokens = map_phone_to_tokendict({'txt_token': ph_ref, 'tone': tone_ref}, pad_bos_eos=False)
            with model_lock(self.lock):
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    dur_start_pos = self.dur_model.prefill(merged_ph_tokens, dur_ref.to(self.device))
            ret_dur = {'dur_start_pos': dur_start_pos}

        return {
            'text_ref_clean': text_ref_clean,
            'text_ref_raw':   text_ref_raw,
            'text_ref_by_spk': text_ref_by_spk,
            'ph_ref': ph_ref.cpu(),
            'tone_ref': tone_ref.cpu(),
            'dur_ref': dur_ref.cpu(),
            'mel2ph_ref': mel2ph_ref.cpu(),
            'vae_latent': vae_latent.cpu(),
            'spk_ids_ref': spk_ids_ref.cpu(),
            'ref_by_spk': ref_by_spk,
            **ret_dur
        }


    def _dur_predict_per_chunk_by_speaker(self, chunk_items, resource_context, dur_disturb=0.1, normalize_dur=True):
        """
        多说话人 & 逐段时长预测（for 新 dur_lm / dur_lm_seq2seq）：
        - 先按说话人 prefill（仅用该说话人参考 ph/tone/dur）；
        - 再对该说话人的所有 chunk 逐个 decode；
        - 关键修复：每个 chunk 的 caption 使用该说话人的参考文本 + 该 chunk 文本，
        而不是全局拼接的 text_ref_clean！

        Args:
            chunk_items: List[dict]，来自 forward 前的切分结果，每项含：
                {'sid': int, 'ch': SSML片段, 'ph': Tensor[1,L], 'tone': Tensor[1,L], 'ph2word': Optional}
            resource_context: dict，来自 preprocess(...)
                需要包含：
                - 'ref_by_spk': {sid: {'ph_ref','tone_ref','dur_ref'}}
                - 'text_ref_by_spk': {sid: str}   # 本函数将优先使用
                - （兼容兜底）'text_ref_clean' 或 'text_ref'
            dur_disturb: float
            normalize_dur: bool
        Returns:
            {chunk_idx: Tensor[1, L_chunk]}  # 每个 chunk 的离散时长序列（单位：code 0..K-1）
        """
        import torch
        from collections import defaultdict

        device = self.device
        compute_dtype = self.precision

        ref_by_spk = resource_context.get('ref_by_spk')
        if not ref_by_spk:
            raise RuntimeError("严格模式：未找到任何参考（ref_by_spk 为空）！")

        # 该说话人的参考文本（优先）
        text_ref_by_spk = resource_context.get('text_ref_by_spk', {}) or {}
        # 兼容老资源：若缺失则退化到全局文本
        global_text_ref = (
            resource_context.get('text_ref_clean')
            or resource_context.get('text_ref')
            or ""
        )

        # 按出现顺序分组
        groups = defaultdict(list)  # sid -> [(ci, item)]
        for ci, it in enumerate(chunk_items):
            groups[int(it['sid'])].append((ci, it))

        dur_model_type = getattr(self, "dur_model_type", "lm")
        modeling_type  = getattr(getattr(self.dur_model, "config", None), "modeling_type", None)

        results = {}

        def _encode_caption(text_str: str) -> torch.Tensor:
            """编码单个 caption（逐 chunk 独立 caption）"""
            inputs = self.caption_tokenizer([text_str], padding=True, return_tensors="pt")
            ids = inputs.input_ids.to(device)
            am  = inputs.attention_mask.to(device)
            with torch.autocast(device_type='cuda', dtype=compute_dtype):
                embs = self.caption_encoder(ids, return_dict=False, attention_mask=am)[0]
            embs = embs * am[..., None]
            return embs.to(dtype=compute_dtype)

        # 构建静音掩码（给 normalize_dur 用）
        def _build_sil_masks(ph_pred: torch.Tensor, ph_ref: torch.Tensor):
            try:
                sil_ph_list = self.ling_dict['phone'].sil_phonemes()
            except Exception:
                sil_ph_list = []
            sil_ids = []
            for sp in sil_ph_list:
                try:
                    sil_ids.append(self.ling_dict['phone'].encode(sp)[0])
                except Exception:
                    pass
            extra_sil_ids = [145, 148, 153, 166, 163, 165]  # 你自己认为属于“停顿/静音”的 phone token
            sil_ids = list(set(sil_ids + extra_sil_ids))

            sil_mask_pred = torch.zeros_like(ph_pred, dtype=torch.long)
            sil_mask_ref  = torch.zeros_like(ph_ref,  dtype=torch.long)
            for sid_ in sil_ids:
                sil_mask_pred[ph_pred == sid_] = 1
                sil_mask_ref[ ph_ref  == sid_] = 1
            return sil_mask_pred, sil_mask_ref

        # === 关键：按说话人循环，prefill 后立刻解该说话人的所有 chunk ===
        for sid, items in groups.items():
            sid = int(sid)
            ref_pack = ref_by_spk[sid]
            ph_ref_spk   = ref_pack['ph_ref'].to(device)
            tone_ref_spk = ref_pack['tone_ref'].to(device)
            dur_ref_spk  = ref_pack['dur_ref']
            if dur_ref_spk.dim() == 1:
                dur_ref_spk = dur_ref_spk[None]
            dur_ref_spk = dur_ref_spk.to(device)

            # 仅用该说话人的参考做 prefill
            ref_tokens = map_phone_to_tokendict(
                {'txt_token': ph_ref_spk, 'tone': tone_ref_spk},
                pad_bos_eos=False
            )
            with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=compute_dtype):
                start_pos = self.dur_model.prefill(ref_tokens, dur_ref_spk)

            # 逐段推理该说话人的每个 chunk
            for (ci, it) in items:
                ph   = it['ph'].to(device)
                tone = it['tone'].to(device)
                merged = map_phone_to_tokendict({'txt_token': ph, 'tone': tone}, pad_bos_eos=False)

                if dur_model_type == 'lm':
                    # 纯 LM 无 caption
                    with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=compute_dtype):
                        dur_pred = self.dur_model.inference(
                            txt_tokens=merged,
                            start_pos=start_pos,
                            temperature=dur_disturb,
                            use_tqdm=self.use_tqdm
                        )

                elif dur_model_type == 'lm_seq2seq':
                    # —— 修复点：逐 chunk caption = 该说话人的参考文本 + 当前 chunk 文本 ——
                    local_ref_txt = text_ref_by_spk.get(sid, global_text_ref)
                    caption_text  = f"{local_ref_txt}{it['ch'].text_str}"
                    caption_embs  = _encode_caption(caption_text)

                    infer_kwargs = dict(
                        txt_tokens=merged,
                        condition=caption_embs,
                        start_pos=start_pos,
                        temperature=dur_disturb,
                        topk=5,
                        use_tqdm=self.use_tqdm
                    )
                    if modeling_type in ['ar', 'ar_cond_durtok', 'ar_dur']:
                        infer_kwargs['dur_tokens'] = dur_ref_spk

                    with model_lock(self.lock), torch.autocast(device_type='cuda', dtype=compute_dtype):
                        dur_pred = self.dur_model.inference(**infer_kwargs)

                else:
                    raise NotImplementedError(f"dur_model_type={dur_model_type} 暂不支持该逐段推理！")

                # ===== 可选：normalize_dur —— 用参考段的静/非静音均值，拉回语速 =====
                if normalize_dur and dur_pred.shape[1] > 10:
                    sil_mask_pred, sil_mask_ref = _build_sil_masks(ph, ph_ref_spk)
                    z_dur_pred = torch.log1p(dur_pred.float())
                    z_dur_ref  = torch.log1p(dur_ref_spk.float())

                    # 静音部分对齐
                    if sil_mask_pred.sum() > 0 and sil_mask_ref.sum() > 0:
                        diff_sil = z_dur_ref[sil_mask_ref == 1].mean() - z_dur_pred[sil_mask_pred == 1].mean()
                        z_dur_pred[sil_mask_pred == 1] += diff_sil

                    # 非静音部分对齐
                    non_pred = (sil_mask_pred != 1)
                    non_ref  = (sil_mask_ref  != 1)
                    if non_pred.sum() > 0 and non_ref.sum() > 0:
                        diff_non = z_dur_ref[non_ref].mean() - z_dur_pred[non_pred].mean()
                        z_dur_pred[non_pred] += diff_non

                    # 还原为整数
                    dur_pred = torch.expm1(z_dur_pred).clamp_min(0)
                    d_floor  = torch.floor(dur_pred)
                    frac     = (dur_pred - d_floor).clamp(0, 1)
                    dur_pred = (d_floor + torch.bernoulli(frac)).to(torch.int)

                results[ci] = dur_pred  # Tensor[1, L_chunk]

        return results

    def _dur_predict_old_ar_grouped_by_speaker(
        self,
        chunk_items,
        resource_context,
        dur_disturb: float = 0.1,
        mode: str = "reset",
        normalize_dur: bool = False,
        disturb_in_predictor: bool = True,   # NEW
    ):
        """
        严格对齐单人：每一句单独预测 duration的旧 ARDurPredictor 路线（use_old_dur=True）：

        - 按 chunk_items 的顺序逐句处理（chunk_items 就是你按 <SPK>x</SPK> 拆出来的 n 句/turn）！
        - 每一句：
            1) 取该句 sid 对应的 ref prefill（dur_prefill_by_spk[sid]）；
            2) incremental_state 每句都从 ref prefill 复制到 GPU（不滚动）；
            3) first_decoder_inp 永远使用 ref 的最后一个 ctx dur token（1..K）；
            4) spk_pos_ids_flat 每句都从 ref 的 last_pos 开始（不跨句累加）；
            5) infer 得到该句 dur（1..K），转成 0..K-1 返回！
        - 这样保证：把同一句单独拿去跑单人脚本得到的 duration，与这里完全一致！

        mode:
        - "reset" / "single"：严格单人逐句独立（默认）
        - "rolling"：保留原先跨句滚动状态（不保证与单人逐句一致）

        返回:
            {chunk_idx: Tensor[1, L_chunk]}  # 0..K-1 的离散 dur code
        """
        import torch

        device = self.device
        hp = getattr(self, "hp_dur_model", {})
        dur_max = hp.get("dur_code_size", hp.get("dur_max_value", 128)) - 1

        # 目标 device（兼容 device="cuda:x" / torch.device）
        try:
            dev_idx = int(str(device).split(":")[1])
            target_device = torch.device(f"cuda:{dev_idx}")
        except Exception:
            target_device = torch.device(device if isinstance(device, str) else device)

        # per-speaker prefill（preprocess() 里 _build_old_dur_prefill_by_speaker 构建）
        prefill_by_spk = resource_context.get("dur_prefill_by_spk", {}) or {}

        # 兜底：若缺 sid 的 prefill，则退化为全局 prompt（兼容单参考或资源不全）
        global_prefill = None
        if ("incremental_state_dur_prompt" in resource_context) and ("ctx_dur_tokens" in resource_context):
            global_prefill = {
                "incremental_state": resource_context["incremental_state_dur_prompt"],
                "ctx_dur_tokens": resource_context["ctx_dur_tokens"],
                "last_pos": int(resource_context.get("last_dur_pos_prompt", resource_context["ctx_dur_tokens"].shape[1])),
            }

        # 准备静音音素 id（给 normalize_dur 用）
        try:
            sil_ph_list = self.ling_dict["phone"].sil_phonemes()
        except Exception:
            sil_ph_list = []
        sil_ids = []
        for sp in sil_ph_list:
            try:
                sil_ids.append(self.ling_dict["phone"].encode(sp)[0])
            except Exception:
                pass
        sil_ids = list(set(sil_ids))

        results = {}

        # rolling 模式下，需要为每个 sid 维护滚动状态与 last_token
        rolling_state = {}  # sid -> (inc_state_cpu, last_token_1k)
        rolling_lastpos = {}  # sid -> last_pos (会跨句累加)

        for ci, it in enumerate(chunk_items):
            sid = int(it["sid"])

            # 取该说话人的 ref prefill（三件套）
            pref = prefill_by_spk.get(sid, None)
            if pref is None:
                if global_prefill is None:
                    raise RuntimeError(f"严格模式：缺少 sid={sid} 的 dur_prefill_by_spk，且无全局 prompt 兜底！")
                pref = global_prefill

            inc_state_cpu = pref["incremental_state"]          # CPU 端树状状态
            ctx_dur_tokens_cpu = pref["ctx_dur_tokens"]        # [1, Lctx]，值域 1..K
            pref_last_pos = int(pref["last_pos"])              # ref ctx token 长度
            last_token_1k_pref = ctx_dur_tokens_cpu[:, -1:]    # [1,1]，值域 1..K

            # 该句输入
            ph_pred = it["ph"].to(target_device)
            tone_pred = it["tone"].to(target_device)
            txt_len = int(ph_pred.shape[1])

            # ===== 位置编码：关键差异点 =====
            # 单人每句独立时：每句都从 ref 的 last_pos 开始，不跨句累加
            if mode.lower() in ["rolling"]:
                # rolling：跨句累计 last_pos
                if sid not in rolling_lastpos:
                    rolling_lastpos[sid] = pref_last_pos
                start_pos = rolling_lastpos[sid]
                spk_pos_ids_flat = torch.arange(start_pos, start_pos + txt_len, device=target_device)[None, :].long()
                rolling_lastpos[sid] = start_pos + txt_len
            else:
                # reset/single：严格与单人每句独立一致
                spk_pos_ids_flat = torch.arange(pref_last_pos, pref_last_pos + txt_len, device=target_device)[None, :].long()

            # ===== 状态与 first token：关键差异点 =====
            if mode.lower() == "rolling":
                if sid not in rolling_state:
                    rolling_state[sid] = (inc_state_cpu, last_token_1k_pref)
                inc_cpu_in, last_tok_in = rolling_state[sid]
                inc_state_in = self._clone_tensor_tree_to(inc_cpu_in, target_device)
                first_decoder_inp = last_tok_in.to(target_device)
            else:
                # reset/single：每句都从 ref prefill 重置开始
                inc_state_in = self._clone_tensor_tree_to(inc_state_cpu, target_device)
                first_decoder_inp = last_token_1k_pref.to(target_device)

            # ===== 推理（旧 ARDurPredictor）=====
            with model_lock(self.lock):
                ret = self.dur_model.infer(
                    ph_pred,
                    {"tone": tone_pred},
                    None, None, None,
                    incremental_state=inc_state_in,
                    first_decoder_inp=first_decoder_inp,   # 1..K
                    spk_pos_ids_flat=spk_pos_ids_flat,
                    use_tqdm=False,
                    return_state=(mode.lower() == "rolling")
                )

            if mode.lower() == "rolling":
                # 需要拿回滚动状态
                if not (isinstance(ret, tuple) and len(ret) == 2):
                    raise RuntimeError("ARDurPredictor.infer(return_state=True) 应返回 (pred, state)")
                dur_pred_1k, inc_state_out = ret
                # 更新 rolling_state：把 GPU 状态收回 CPU 树
                last_tok_out = dur_pred_1k[:, -1:].detach()
                inc_state_cpu_out = self._clone_tensor_tree_to(inc_state_out, torch.device("cpu"))
                rolling_state[sid] = (inc_state_cpu_out, last_tok_out)
            else:
                # reset/single：不需要滚动状态
                dur_pred_1k = ret[0] if isinstance(ret, tuple) else ret

            # 1..K -> 0..K-1
            dur_pred = (dur_pred_1k - 1).to(torch.int)

            # ===== 可选：normalize_dur====
            if normalize_dur:
                ref_pack = (resource_context.get("ref_by_spk", {}) or {}).get(sid, None)
                if ref_pack is not None and dur_pred.shape[1] > 10:
                    ph_ref_spk = ref_pack["ph_ref"].to(target_device)
                    dur_ref_spk = ref_pack["dur_ref"]
                    if dur_ref_spk.dim() == 1:
                        dur_ref_spk = dur_ref_spk[None]
                    dur_ref_spk = dur_ref_spk.to(target_device)

                    sil_mask_pred = torch.zeros_like(ph_pred, dtype=torch.long)
                    sil_mask_ref = torch.zeros_like(ph_ref_spk, dtype=torch.long)
                    for sil_id in sil_ids:
                        sil_mask_pred[ph_pred == sil_id] = 1
                        sil_mask_ref[ph_ref_spk == sil_id] = 1

                    z_pred = torch.log1p(dur_pred.float())
                    z_ref = torch.log1p(dur_ref_spk.float())

                    # 静音对齐
                    if sil_mask_pred.sum() > 0 and sil_mask_ref.sum() > 0:
                        diff_sil = z_ref[sil_mask_ref == 1].mean() - z_pred[sil_mask_pred == 1].mean()
                        z_pred[sil_mask_pred == 1] += diff_sil
                    # 非静音对齐
                    non_pred = (sil_mask_pred != 1)
                    non_ref = (sil_mask_ref != 1)
                    if non_pred.sum() > 0 and non_ref.sum() > 0:
                        diff_non = z_ref[non_ref].mean() - z_pred[non_pred].mean()
                        z_pred[non_pred] += diff_non

                    dur_pred = torch.expm1(z_pred).clamp_min(0)
                    d_floor = torch.floor(dur_pred)
                    frac = (dur_pred - d_floor).clamp(0, 1)
                    dur_pred = (d_floor + torch.bernoulli(frac)).to(torch.int)

            dur_pred = dur_pred.clamp(0, dur_max)
            results[ci] = dur_pred

        return results

    def forward(self, resource_context, input_text, time_step,
                w_all=None, w_txt=None, w_cap=None, w_ref=None, w_spk=None,
                seq_cfg_w: Optional[List[float]] = None,
                speech_rate=1, timestep_annealing_w=(1.0, 0.0, 1.0),
                return_timestamp=True, timestamp_postprocess=False,
                return_format='wav', custom_ph_table=None, dur_disturb=0.1, dur_alpha=1.0,
                num_parallel_workers=5, use_sa_frontend=True,
                normalize_dur: bool = True,
                use_amo_sampler: bool = False,
                prefer_onepass_dur: bool = True,
                trim_prefix_text: Optional[str] = None,   # ===== 传入 prefix_text，用预测 duration 裁掉 prefix 音频 =====
                **kwargs):
        """
        仅支持 <SPK>sid</SPK> 标注
        """
        device = self.device
        profile = os.environ.get('MEGA_PROFILE', 'false').strip().lower() == 'true'

        with torch.inference_mode():
            # 0) 清洗输入
            input_text = ''.join(c for c in input_text if c.isprintable())
            if not input_text.strip():
                raise RuntimeError('输入为空，输入不合法')

            # 防止读错
            import re

            repl = {
                '血': '谑',
                '抹布': '妈布',
                '教学': '叫学',
                '关税': '关睡',
                '蹭蹭': '噌噌',
                '扎针': '渣针','扎手': '渣手','扎心': '渣心',
                '恐怖片':'恐怖篇','动作片':'动作篇','喜剧片':'喜剧篇','爱情片':'爱情篇',
                '挑花眼': '聎花眼',
                '劈叉': '痞岔',
                '一对一': '【依对依】',
                '防带出': '【防带出】',
                '脂': '知',
                '制': '智',
                '色': '涩',
                '看': '瞰',
                '厂': '场',
                'mah': '毫安时','mAh': '毫安时','mA': '毫安','Ah': '安时',
                'mmHg': '毫米汞柱','cmH2O': '厘米水柱',
                '~': '。',
                '～': '。',
                '...': '。',
                '——': '，',
                '《': '【','》': '】','“':'【','”':'】',
                'mmol': '毫摩尔','mol': '摩尔','mol/L': '摩尔每升','mmol/L': '毫摩尔每升',
                'mg': '毫克','ml': '毫升','μg': '微克','μl': '微升',
                '但市':'但！市','但世':'但！世','重点':'众点','成分':'成奋',
            }

            pattern = re.compile("|".join(map(re.escape, sorted(repl, key=len, reverse=True))))
            input_text = pattern.sub(lambda m: repl[m.group(0)], input_text)
            input_text = re.sub(r'(?<=[\u4E00-\u9FFF])(?=[A-Za-z])', ' ', input_text)
            input_text = re.sub(r"(?<=[A-Za-z0-9'])(?=[\u4E00-\u9FFF])", ' ', input_text)

            # 1) 解析（只支持 <SPK>）
            raw_text_total = input_text
            clean_text_total = re.sub(r"<\s*SPK\s*>\s*\d+\s*<\s*/\s*SPK\s*>", "", raw_text_total, flags=re.IGNORECASE)
            spk_segs = self._parse_dialogue_segments(raw_text_total)  # [(sid, content)]
            if len(spk_segs) == 0:
                raise RuntimeError("未解析到任何说话人片段，请检查输入（只支持 <SPK>1</SPK>...）！")

            # 2) normalize + chunk
            ssml_root = SSML(clean_text_total); ssml_root.rate = float(speech_rate)
            ph_replace_table = deepcopy(self.ph_replace_table)
            custom_ph_table = kwargs.get('custom_ph_table', None)
            if custom_ph_table is not None:
                ph_replace_table.update(custom_ph_table)

            # ===== 计算 prefix_text 最终会产生多少个有效 chunk =====
            prefix_chunk_cnt = 0
            if trim_prefix_text is not None and str(trim_prefix_text).strip():
                try:
                    prefix_chunk_cnt = self._count_effective_chunks_from_raw(
                        trim_prefix_text,
                        ph_replace_table=ph_replace_table,
                        use_sa_frontend=use_sa_frontend,
                        speech_rate=ssml_root.rate
                    )
                except Exception as e:
                    print_once(f'| trim_prefix_text 解析失败，跳过裁剪：{e}')
                    prefix_chunk_cnt = 0

            text_chunks, chunk_spk_ids, chunk_seg_ids, chunk_raw_for_dit = [], [], [], []
            for seg_i, (sid, seg_content) in enumerate(spk_segs):
                sub_ssml = SSML(seg_content); sub_ssml.rate = ssml_root.rate
                sub_chunks = self.preprocess_text(sub_ssml, ph_replace_table, use_sa_frontend)
                for ch in sub_chunks:
                    text_chunks.append(ch)
                    chunk_spk_ids.append(int(sid))
                    chunk_seg_ids.append(int(seg_i))   # 属于哪个 <SPK> 段
                    chunk_raw_for_dit.append(f"<SPK>{int(sid)}</SPK>{ch.text_str}")


            if len(text_chunks) == 0:
                raise RuntimeError("文本经 normalize/切分后为空！")

            # 3) 逐 chunk 做 G2P（phone/tone）
            chunk_items = []
            spk_mask_ph_list, words_ts_list = [], []
            for ci, (ch, sid) in enumerate(zip(text_chunks, chunk_spk_ids)):
                if not use_sa_frontend:
                    with model_lock(self.lock):
                        ph_pred, tone_pred = self.g2p(ch.text_str)
                    ph_pred, tone_pred, ph2word = self.refine_ph_tone(ch, ph_pred, tone_pred)
                else:
                    from modules.tts.frontend_lm.sa_frontend import call_sa_frontend
                    sa_ret = call_sa_frontend(ch.sa_ssml_str, debug=0)
                    if sa_ret is None:
                        assert False, f'| 跳过非法片段 #{ci}'
                    text_sa, ph_tokens, tone_tokens, _ = sa_ret

                    new_text = SSML(text_sa); new_text.rate = ch.rate
                    new_text.pause_at_start = ch.pause_at_start; new_text.pause_at_end = ch.pause_at_end
                    ch = new_text
                    ph_pred  = torch.LongTensor(self.ling_dict['phone'].encode(' '.join(ph_tokens)))[None].to(device)
                    tone_pred= torch.LongTensor(self.ling_dict['tone'].encode(' '.join(tone_tokens)))[None].to(device)
                    ph2word = None
                    print(f'text_sa: {text_sa},ph_pred: {ph_tokens}, tone_pred: {tone_tokens}')
                chunk_items.append({'sid': int(sid), 'ch': ch, 'ph': ph_pred, 'tone': tone_pred, 'ph2word': ph2word})
                words_ts_list.append({'words': [], 'timestamps': []})
                spk_mask_ph_list.append(torch.full((1, ph_pred.shape[1]), int(sid), dtype=torch.long, device=device))

            if len(chunk_items) == 0:
                raise RuntimeError("所有 chunk 都被跳过，无法生成！")

            # 4) 时长预测
            if self.use_old_dur:
                dur_pred_by_chunk = self._dur_predict_old_ar_grouped_by_speaker(
                    chunk_items,
                    resource_context,
                    dur_disturb=dur_disturb,
                    mode="reset",
                    normalize_dur=normalize_dur,
                    disturb_in_predictor=False, 
                )

            else:
                dur_pred_by_chunk = None
                if prefer_onepass_dur:
                    dur_pred_by_chunk = self._dur_predict_onepass_with_spk(
                        chunk_items, resource_context,
                        dur_disturb=dur_disturb, normalize_dur=normalize_dur
                    )
                if dur_pred_by_chunk is None:
                    try:
                        dur_pred_by_chunk = self._dur_predict_per_chunk_by_speaker(
                            chunk_items, resource_context,
                            dur_disturb=dur_disturb, normalize_dur=normalize_dur
                        )
                    except TypeError:
                        dur_pred_by_chunk = self._dur_predict_per_chunk_by_speaker(
                            chunk_items, resource_context,
                            dur_disturb=dur_disturb
                        )

            # 5) 后处理 + vq 对齐 + 词级时间戳
            ph_list_all, tone_list_all, dur_list_all = [], [], []
            vqs = hparams.get('vq_stride', 8)

            dur_code_max = self.hp_dur_model.get('dur_code_size', self.hp_dur_model.get('dur_max_value', 128)) - 1
            chunk_dur_units_list = []

            for ci, it in enumerate(chunk_items):
                sid       = it['sid']
                ch        = it['ch']
                ph_pred   = it['ph']
                tone_pred = it['tone']
                ph2word   = it['ph2word']
                dur_pred = dur_pred_by_chunk[ci].to(device).long()

                # 1) Control Speech Speed（与 process_text_seg 对齐）
                dur_pred = torch.round(dur_pred.float() / float(ch.rate)).long()

                dur_code_max = self.hp_dur_model.get('dur_code_size', self.hp_dur_model.get('dur_max_value', 128)) - 1
                dur_pred = dur_pred.clamp(0, dur_code_max)

                # 2) 末句最后音素 clamp（与你原逻辑一致；放在 tail add 之前）
                if ci == len(chunk_items) - 1:
                    dur_pred[:, -1] = dur_pred[:, -1].clamp(32, 80)

                # 3) old_dur 的扰动 + dur_alpha（放在 rate/clamp 之后，贴近 process_text_seg）
                if self.use_old_dur:
                    if dur_disturb and dur_disturb > 0:
                        choice = (torch.rand_like(dur_pred.float()) > 0.5).float()
                        r = 1.0 + torch.rand_like(dur_pred.float()) * float(dur_disturb)
                        dur_pred = dur_pred.float() * r * choice + dur_pred.float() / r * (1.0 - choice)

                    dur_pred = torch.round(dur_pred * float(dur_alpha)).long()
                    dur_pred = dur_pred.clamp(0, dur_code_max)

                # 4) 扰动/alpha 后要再做一遍标点/静音最小值约束（否则会被扰动破坏）
                for sil_token in [148, 145]:  # 。 sil
                    m = (ph_pred == sil_token)
                    if m.any():
                        dur_pred[m] = dur_pred[m].clamp_min(24)
                        dur_pred[m] = dur_pred[m].clamp_max(32)
                for sil_token in [147]:  # sp
                    m = (ph_pred == sil_token)
                    if m.any():
                        dur_pred[m] = dur_pred[m].clamp_min(4)
                        dur_pred[m] = dur_pred[m].clamp_max(8)

                for sil_token in [163, 165]:            # ， ； 
                    m = (ph_pred == sil_token)
                    if m.any():
                        dur_pred[m] = dur_pred[m].clamp_min(16)
                        dur_pred[m] = dur_pred[m].clamp_max(24)
                        
                for sil_token in [153, 166]:            #  ! ?
                    m = (ph_pred == sil_token)
                    if m.any():
                        dur_pred[m] = dur_pred[m].clamp_min(24)
                        dur_pred[m] = dur_pred[m].clamp_max(32)
                for sil_token in [153, 166]:            #  ! ?
                    m = (ph_pred == sil_token)
                    if m.any():
                        dur_pred[m] = dur_pred[m].clamp_min(24)
                        dur_pred[m] = dur_pred[m].clamp_max(32)

                dur_pred[:, 0] = 8

                if not use_sa_frontend:
                    ph_pred, tone_pred, dur_pred, ph2word = self.add_breaks(
                        ch, ph_pred, tone_pred, dur_pred, ph2word, break_token=163, break_tone=3
                    )
                    if return_timestamp and ph2word is not None:
                        try:
                            words_ts = self.make_word_timestamps(ch, dur_pred.squeeze().cpu().numpy(), ph2word)
                        except Exception:
                            words_ts = {'words': [], 'timestamps': []}
                    else:
                        words_ts = {'words': [], 'timestamps': []}
                    words_ts_list[ci] = words_ts

                # === 在最后一个音素上加 80ms ===
                TAIL_ADD_MS = 80
                tail_add_units = int(round(TAIL_ADD_MS / 10.0))  # 10ms/unit

                if ci == len(chunk_items) - 1 and tail_add_units > 0:
                    dur_max = self.hp_dur_model.get('dur_code_size', self.hp_dur_model.get('dur_max_value', 128)) - 1
                    dur_pred[:, -1] = (dur_pred[:, -1] + tail_add_units).clamp(0, dur_max)

                # ===== 每个 <SPK> 段末尾：延长最后一个音素 80ms=====
                SEG_TAIL_ADD_MS = 80
                seg_tail_units = int(round(SEG_TAIL_ADD_MS / 10.0))  # 10ms/unit

                # 判断当前 chunk 是否是该 <SPK> 段的最后一个 chunk
                is_last_chunk_of_seg = (ci == len(chunk_items) - 1) or (chunk_seg_ids[ci + 1] != chunk_seg_ids[ci])

                if is_last_chunk_of_seg and seg_tail_units > 0:
                    dur_max = self.hp_dur_model.get('dur_code_size', self.hp_dur_model.get('dur_max_value', 128)) - 1
                    dur_pred[0, -1] = (dur_pred[0, -1] + seg_tail_units).clamp(0, dur_max)

                dur_sum = int(dur_pred.sum().item())
                npad = vqs - dur_sum % vqs
                if npad < vqs:
                    dur_pred[:, -1] += npad

                ph_list_all.append(ph_pred)
                tone_list_all.append(tone_pred)
                dur_list_all.append(dur_pred)
                
                # 记录每个 chunk 的 duration 总和（单位 0.01s），后面分段 LUFS 用
                chunk_dur_units_list.append(float(dur_pred.sum().item()))

            ph_pred_all   = torch.cat(ph_list_all,   dim=1)
            tone_pred_all = torch.cat(tone_list_all, dim=1)
            dur_pred_all  = torch.cat(dur_list_all,  dim=1)
            dur_pred_all[:, -1] = dur_pred_all[:, -1]
            mel2ph_pred_all = self.length_regulator(dur_pred_all).to(device)
            spk_ids_pred_all = torch.cat(spk_mask_ph_list, dim=1).long()

            # 词级时间戳拼接
            if return_timestamp:
                offsets = [0.0]
                for i in range(len(dur_list_all)-1):
                    offsets.append(offsets[-1] + float(dur_list_all[i].sum().item())/100.0)
                words_all, ts_all = [], []
                for (wts, off) in zip(words_ts_list, offsets):
                    words_all.extend(wts['words'])
                    ts_all.extend([[a+off, b+off] for (a,b) in wts['timestamps']])
                words_timestamps = {'words': words_all, 'timestamps': ts_all}
                words_timestamps_post = None
            else:
                words_timestamps = words_timestamps_post = None

            # ===== 用预测 dur 计算 prefix 对应的秒数（基于 prefix_chunk_cnt -> prefix_phone_len -> sum(dur)） =====
            prefix_trim_sec = 0.0
            if prefix_chunk_cnt > 0:
                prefix_chunk_cnt_eff = min(prefix_chunk_cnt, len(ph_list_all))
                if prefix_chunk_cnt_eff > 0:
                    prefix_phone_len = int(sum(p.shape[1] for p in ph_list_all[:prefix_chunk_cnt_eff]))
                    if prefix_phone_len > 0:
                        prefix_trim_sec = float(dur_pred_all[:, :prefix_phone_len].sum().item()) / 100.0  # 0.01s per unit

            # ===== timestamps 同步裁掉 prefix，并做平移（若存在）=====
            if prefix_trim_sec > 0.0 and return_timestamp and words_timestamps is not None:
                new_words, new_ts = [], []
                for w, (a, b) in zip(words_timestamps['words'], words_timestamps['timestamps']):
                    a2 = float(a) - prefix_trim_sec
                    b2 = float(b) - prefix_trim_sec
                    if b2 <= 0:
                        continue
                    new_words.append(w)
                    new_ts.append([max(0.0, a2), b2])
                words_timestamps = {'words': new_words, 'timestamps': new_ts}
                # words_timestamps_post 这里保持 None

            # 6) Caption & Text（整段：统一 <SPK>sid</SPK> 前缀式）
            ref_segs = self._parse_dialogue_segments(resource_context['text_ref_raw'])
            seq_for_all = ref_segs + [(sid, ch.text_str) for sid, ch in zip(chunk_spk_ids, text_chunks)]

            def _pack_spk_markup(segs: List[Tuple[int,str]]) -> str:
                out = []
                for sid_i, content_i in segs:
                    content_i = (content_i or '').strip()
                    if not content_i:
                        continue
                    out.append(f"<SPK>{int(sid_i)}</SPK>{content_i}")
                return ''.join(out)

            caption_merged_all = _pack_spk_markup(seq_for_all)
            train_text_all     = caption_merged_all

            # 7) 组装 DiT 输入
            ph_ref     = resource_context['ph_ref'].to(device)
            tone_ref   = resource_context['tone_ref'].to(device)
            dur_ref    = resource_context['dur_ref'].to(device)
            mel2ph_ref = resource_context['mel2ph_ref'].to(device)
            vae_latent = resource_context['vae_latent'].to(device)
            spk_ids_ref= resource_context['spk_ids_ref'].to(device).long()

            # 在参考末尾插 0.01s 静音
            # sil_token, sil_tone, gap_units = 145, 3, 1
            # sil_ph = torch.full((1, 1), sil_token, dtype=ph_ref.dtype, device=device)
            # sil_tn = torch.full((1, 1), sil_tone,  dtype=tone_ref.dtype, device=device)
            # sil_du = torch.full((1, 1), gap_units, dtype=dur_ref.dtype, device=dur_ref.device)
            # ph_ref_ext   = torch.cat([ph_ref,   sil_ph],   dim=1)
            # tone_ref_ext = torch.cat([tone_ref, sil_tn],   dim=1)
            # dur_ref_ext  = torch.cat([dur_ref,  sil_du],   dim=1)
            # new_idx = int(ph_ref.shape[1] + 1)
            # mel2ph_gap = torch.full((mel2ph_ref.shape[0], gap_units),
            #                         new_idx, dtype=mel2ph_ref.dtype, device=device)
            # mel2ph_ref_ext = torch.cat([mel2ph_ref, mel2ph_gap], dim=1)
            # last_sid = int(spk_ids_ref[0, -1].item()) if spk_ids_ref.numel() > 0 else 1
            # spk_ids_ref_ext = torch.cat([
            #     spk_ids_ref,
            #     torch.full((1, 1), last_sid, dtype=spk_ids_ref.dtype, device=device)
            # ], dim=1)
            
            mel2ph_ref_ext = mel2ph_ref
            spk_ids_ref_ext = spk_ids_ref
            dur_ref_ext  = dur_ref
            ph_ref_ext   = ph_ref
            tone_ref_ext = tone_ref

            # 目标侧拼接
            ph_seq   = torch.cat([ph_ref_ext,   ph_pred_all],   dim=1)
            tone_seq = torch.cat([tone_ref_ext, tone_pred_all], dim=1)
            en_tone_idx = ~((tone_seq == 4) | ((11 <= tone_seq) & (tone_seq <= 15)) | (tone_seq == 0))
            tone_seq[en_tone_idx] = 3
            spk_seq_base = torch.cat([spk_ids_ref_ext, spk_ids_pred_all], dim=1).long()
            if spk_seq_base.shape[1] != ph_seq.shape[1]:
                raise RuntimeError("spk_seq_base 与 ph_seq 长度不一致！")
            mel2ph_pred_full = torch.cat([mel2ph_ref_ext, mel2ph_pred_all + ph_ref_ext.shape[1]], dim=1)
            # 计算需要补多少帧到 fm 对齐
            cur_len = mel2ph_pred_full.size(1)
            pad = (-cur_len) % self.fm
            if pad:
                last = mel2ph_pred_full[:, -1:]                 # [B,1]
                mel2ph_pred_full = torch.cat([mel2ph_pred_full, last.repeat(1, pad)], dim=1)

                # 可选：让 duration 也一致（推荐）
                dur_pred_all[:, -1] += pad

                # NEW：同步到 chunk_dur_units_list（pad 属于最后一个 chunk/段落）
                if len(chunk_dur_units_list) > 0:
                    chunk_dur_units_list[-1] += float(pad)


            target_size = mel2ph_pred_full.shape[1] // 4

            use_caption = bool(hparams.get('use_caption_encoder', False))

            def _run_caption(caps, device_):
                inputs = self.caption_tokenizer(caps, padding=True, return_tensors="pt")
                ids = inputs.input_ids.to(device_)
                am  = inputs.attention_mask.to(device_)
                embs = self.caption_encoder(ids, return_dict=False, attention_mask=am)[0]
                return embs * am[..., None], am

            if use_caption:
                with model_lock(self.lock):
                    caption_embs_all, caption_mask_all = _run_caption([caption_merged_all], device)
                caption_lens_all = caption_mask_all.sum(-1)
            else:
                caption_embs_all = None
                caption_lens_all = None

            text_inputs_all = self.dit_text_tokenizer(train_text_all, padding=True, return_tensors='pt').to(device)
            txt_tokens_all = text_inputs_all['input_ids']; txt_mask_all = text_inputs_all['attention_mask'].bool()
            txt_tokens_all[~txt_mask_all] = self.cfg_mask_text_token

            ctx_mask = torch.ones_like(vae_latent[:, :, 0:1])
            lat = F.pad(vae_latent, (0,0,0, target_size - vae_latent.size(1)), mode='constant', value=0)
            ctx_mask = F.pad(ctx_mask, (0,0,0, target_size - ctx_mask.size(1)), mode='constant', value=0)

            self._print_dit_inputs_debug(tag="FULL", caption_str=caption_merged_all,
                                        prompt_text=train_text_all, text_inputs=text_inputs_all)

            self._print_target_ph_durations_seconds(ph_ref, tone_ref, dur_ref, tag="MFA_REF")
            self._print_target_ph_durations_seconds(ph_pred_all, tone_pred_all, dur_pred_all, tag="TARGET")

            zeros_phone = torch.full_like(ph_seq,  self.cfg_mask_token_phone)
            zeros_tone  = torch.full_like(tone_seq, self.cfg_mask_token_tone)
            zeros_txt   = torch.full_like(txt_tokens_all, self.cfg_mask_text_token)
            zeros_cap = torch.zeros_like(caption_embs_all) if use_caption else None
            zeros_lat   = torch.zeros_like(lat)

            mel2ph_sparse_1d = None
            if hparams.get('use_sparse_dur', False):
                dur_concat = torch.cat([dur_ref_ext.to('cpu').long().squeeze(0),
                                        dur_pred_all.to('cpu').long().squeeze(0)], dim=0)
                dur_list = dur_concat.numpy().tolist()
                mel2ph_sparse_1d = compute_mel2aug_from_dur(
                    dur_list,
                    gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                    gap_frames=hparams.get('sparse_dur_frames', 4),
                    gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                    min_keep=hparams.get('sparse_dur_min_keep', 1),
                    keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                    symmetric=hparams.get('sparse_dur_symmetric', True),
                )

            def _pack_3way():
                phone   = torch.cat([ph_seq,          ph_seq,          zeros_phone], dim=0)
                tone    = torch.cat([tone_seq,        tone_seq,        zeros_tone ], dim=0)
                txt_tok = torch.cat([txt_tokens_all,  txt_tokens_all,  zeros_txt  ], dim=0)
                txt_msk = torch.cat([txt_mask_all] * 3, dim=0)
                if use_caption:
                    cap     = torch.cat([caption_embs_all, caption_embs_all, zeros_cap], dim=0)
                    cap_len = torch.cat([caption_lens_all] * 3, dim=0).long()
                else:
                    cap, cap_len = None, None
                lat_ctx = torch.cat([lat, zeros_lat, zeros_lat], dim=0)
                ctx_msk = torch.cat([ctx_mask] * 3, dim=0)
                m2p     = mel2ph_pred_full.repeat(3, 1)
                spk_ids = torch.cat([spk_seq_base,
                                    spk_seq_base,
                                    torch.zeros_like(spk_seq_base)], dim=0).long()
                if mel2ph_sparse_1d is not None:
                    m2p_sparse = torch.stack([mel2ph_sparse_1d]*3).to(device)
                    m2p_sparse = m2p_sparse[:, :m2p.shape[1]]
                else:
                    m2p_sparse = None
                return phone, tone, txt_tok, txt_msk, cap, cap_len, lat_ctx, ctx_msk, m2p, m2p_sparse, spk_ids

            def _pack_5way():
                phone   = torch.cat([ph_seq, ph_seq, zeros_phone, zeros_phone, zeros_phone], dim=0)
                tone    = torch.cat([tone_seq, tone_seq, zeros_tone,  zeros_tone,  zeros_tone ], dim=0)
                txt_tok = torch.cat([txt_tokens_all, txt_tokens_all, zeros_txt, zeros_txt, zeros_txt], dim=0)
                txt_msk = torch.cat([txt_mask_all] * 5, dim=0)
                if use_caption:
                    cap     = torch.cat([caption_embs_all, zeros_cap, caption_embs_all, zeros_cap, zeros_cap], dim=0)
                    cap_len = torch.cat([caption_lens_all] * 5, dim=0).long()
                else:
                    cap, cap_len = None, None
                lat_ctx = torch.cat([lat, zeros_lat, zeros_lat, lat, zeros_lat], dim=0)
                ctx_msk = torch.cat([ctx_mask] * 5, dim=0)
                m2p     = mel2ph_pred_full.repeat(5, 1)
                spk_ids = torch.cat([spk_seq_base,
                                    spk_seq_base,
                                    torch.zeros_like(spk_seq_base),
                                    torch.zeros_like(spk_seq_base),
                                    torch.zeros_like(spk_seq_base)], dim=0).long()
                if mel2ph_sparse_1d is not None:
                    m2p_sparse = torch.stack([mel2ph_sparse_1d]*5).to(device)
                    m2p_sparse = m2p_sparse[:, :m2p.shape[1]]
                else:
                    m2p_sparse = None
                return phone, tone, txt_tok, txt_msk, cap, cap_len, lat_ctx, ctx_msk, m2p, m2p_sparse, spk_ids

            use_2step = (seq_cfg_w is not None and len(seq_cfg_w) == 2)
            use_4step = (seq_cfg_w is not None and len(seq_cfg_w) == 4)
            if use_2step:
                ph_pack, tone_pack, txt_pack, txt_mask_pack, cap_pack, cap_lens_pack, \
                lat_pack, ctx_mask_pack, m2p_pack, m2p_sparse_pack, spk_pack = _pack_3way()
            elif use_4step:
                ph_pack, tone_pack, txt_pack, txt_mask_pack, cap_pack, cap_lens_pack, \
                lat_pack, ctx_mask_pack, m2p_pack, m2p_sparse_pack, spk_pack = _pack_5way()
            else:
                raise RuntimeError("未提供合法的 seq_cfg_w（长度需为 2 或 4），建议传入 [1.5, 3.0]！")

            inputs = {
                'phone': ph_pack, 'tone': tone_pack,
                'spk_ids': spk_pack,
                "lat_ctx": lat_pack * ctx_mask_pack, "ctx_mask": ctx_mask_pack,
                "mel2ph": m2p_pack,
                "txt_tokens": txt_pack, 'txt_mask': txt_mask_pack,
            }
            if use_caption:
                inputs["caption_emb"] = cap_pack
                inputs["caption_lens"] = cap_lens_pack
            if m2p_sparse_pack is not None:
                inputs["mel2ph_sparse"] = m2p_sparse_pack

            with model_lock(self.lock):
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    x = self.dit.inference(
                        inputs,
                        timesteps=time_step,
                        seq_cfg_w=seq_cfg_w,
                        timestep_annealing_w=timestep_annealing_w,
                        use_amo_sampler=use_amo_sampler
                    )

            x[:, :vae_latent.size(1)] = vae_latent
            with model_lock(self.lock):
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    wav_pred = self.vae.decode(x)[0,0].to(torch.float32)

            hop_size = self.hp_vae['hop_size']; vae_stride = self.hp_vae['vae_stride']
            pre_samples = int(vae_latent.size(1) * vae_stride * hop_size)
            # gap_samples = int(round(gap_units * 0.01 * self.sr))
            gap_samples = 0
            trim0 = min(pre_samples + gap_samples, int(wav_pred.shape[-1]))
            wav_pred = wav_pred[trim0:]

            if wav_pred.abs().max() > 1:
                wav_pred = wav_pred / (wav_pred.abs().max())
            wav_np = wav_pred.cpu().numpy()

            # ===== 按预测 dur 裁掉 prefix 对应的音频（只动输出，不动生成过程）=====
            if prefix_trim_sec > 0.0 and wav_np is not None and wav_np.shape[0] > 0:
                n_trim = int(round(prefix_trim_sec * self.sr))
                if n_trim > 0:
                    if n_trim >= wav_np.shape[0]:
                        wav_np = wav_np[:0]
                    else:
                        wav_np = wav_np[n_trim:]

            # ===== 只跑一次 VAD：后面“边界吸附”和“非语音置静音”都复用这份结果 =====
            vad_st = []
            if wav_np is not None and wav_np.shape[0] > 0:
                vad_st = self._get_vad_speech_timestamps(
                    wav_np,
                    vad_thresholds=(0.50, 0.35, 0.25),
                    min_speech_ms=160,
                    min_silence_ms=80,
                )
                
            # ===== 按 VAD 静音切段做逐段 LUFS=-23 归一化=====
            if wav_np is not None and wav_np.shape[0] > 0:
                total_samples = wav_np.shape[0]

                # 用“更适合段落切分”的 VAD 参数再跑一次（静音阈值更大，减少碎切）
                vad_para = self._get_vad_speech_timestamps(
                    wav_np,
                    vad_thresholds=(0.50, 0.35, 0.25),
                    min_speech_ms=150,
                    min_silence_ms=100,   # 关键：更大 => 只有更长静音才切段
                )

                boundaries = self._boundaries_from_vad_silence(
                    vad_para,
                    total_samples=total_samples,
                    sr=self.sr,
                    min_gap_ms=200,        # 与 min_silence_ms 对齐即可
                    cut_mode="mid",        # 推荐：切在静音中点
                )

                segs = [wav_np[boundaries[i]:boundaries[i+1]] for i in range(len(boundaries) - 1)]
                if len(segs) == 0:
                    segs = [wav_np]

                norm_segs = []
                target_loudness = -23.0
                for seg in segs:
                    if seg.size == 0:
                        norm_segs.append(seg)
                        continue
                    try:
                        meter = pyln.Meter(self.sr)
                        loudness = meter.integrated_loudness(seg.astype(np.float32))
                        if not np.isfinite(loudness):
                            norm_segs.append(seg)
                            continue
                        seg_n = pyln.normalize.loudness(seg.astype(np.float32), loudness, target_loudness)
                        peak = float(np.max(np.abs(seg_n)) + 1e-9)
                        if peak > 0.999:
                            seg_n = seg_n * (0.999 / peak)
                        norm_segs.append(seg_n.astype(np.float32))
                    except Exception as e:
                        print_once(f'| per-paragraph loudness normalize failed: {e}')
                        norm_segs.append(seg)

                wav_np = np.concatenate(norm_segs, axis=0) if len(norm_segs) > 0 else wav_np


            # >>> 最后再加头 静音（保持你原来的习惯） <<<
            sil_sec = 0.1
            n_sil = int(round(sil_sec * self.sr))
            n_sil = min(n_sil, wav_np.shape[0])
            wav_np[:n_sil] = 0.0
            wav_np[-n_sil:] = 0.0

            # ===== 分段归一化之后，再做一次全局 LUFS 归一化到 -23 =====
            if wav_np is not None and wav_np.shape[0] > 0:
                try:
                    target_loudness = -23.0
                    meter = pyln.Meter(self.sr)
                    loudness = meter.integrated_loudness(wav_np.astype(np.float32))

                    # 全静音/极端情况下 loudness 可能是 -inf
                    if np.isfinite(loudness):
                        wav_g = pyln.normalize.loudness(
                            wav_np.astype(np.float32),
                            loudness,
                            target_loudness
                        )
                        # 防爆：再次 peak 限幅
                        peak = float(np.max(np.abs(wav_g)) + 1e-9)
                        if peak > 0.999:
                            wav_g = wav_g * (0.999 / peak)
                        wav_np = wav_g.astype(np.float32)
                except Exception as e:
                    print_once(f'| global loudness normalize failed: {e}')

            # 生成 bytes（在裁剪+静音+归一化之后）
            wav_bytes = to_wav_bytes(wav_np.astype(float), self.sr) if wav_np is not None else b""
            if return_format == 'mp3':
                wav_bytes = wav_bytes_to_mp3_bytes(wav_bytes)


            ph_pred_list   = self.ling_dict['phone'].decode(ph_pred_all.squeeze().cpu().numpy()).split(' ')
            tone_pred_list = self.ling_dict['tone'].decode(tone_pred_all.squeeze().cpu().numpy()).split(' ')

            return MegaTTS3Output(
                wav_bytes=wav_bytes,
                wav=wav_np,
                words_timestamps=words_timestamps,
                words_timestamps_post=words_timestamps_post,
                duration=(0.0 if wav_np is None else wav_np.shape[-1] / self.sr),
                ph_pred=ph_pred_list,
                tone_pred=tone_pred_list
            )



    def _clone_tensor_tree_to(self, obj, device, clone=True, detach=True):
        """
        递归把任意嵌套结构（dict/list/tuple/Tensor/None）里的 Tensor
        转成叶子副本并迁移到 device：
            Tensor -> (detach? t.detach(): t).(clone? clone(): t).to(device)
        其余类型原样返回（或递归处理）！
        """
        import torch
        if not isinstance(device, torch.device):
            device = torch.device(device)

        if torch.is_tensor(obj):
            t = obj
            if detach:
                t = t.detach()
            if clone:
                t = t.clone()
            return t.to(device, non_blocking=True)

        if isinstance(obj, dict):
            return {k: self._clone_tensor_tree_to(v, device, clone, detach) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._clone_tensor_tree_to(v, device, clone, detach) for v in obj]

        if isinstance(obj, tuple):
            return tuple(self._clone_tensor_tree_to(v, device, clone, detach) for v in obj)

        # 其他（None、标量等）
        return obj

    def _build_old_dur_prefill_by_speaker(self, ref_by_spk):
        """
        为每位说话人分别构建 ARDurPredictor 的 prefill：
        - 仅用该说话人的参考 ph/tone/dur 做 prefill；
        - 生成并返回 {'incremental_state','ctx_dur_tokens','last_pos'} 三件套（CPU 保存，便于并发安全）！
        返回: {sid: {'incremental_state':..., 'ctx_dur_tokens':Tensor[1,L], 'last_pos':int}}
        """
        import torch
        device = self.device
        hp = getattr(self, "hp_dur_model", {})
        dur_prefill = {}

        for sid, pack in ref_by_spk.items():
            ph_ref   = pack['ph_ref'].to(device)
            tone_ref = pack['tone_ref'].to(device)
            # 统一 2D 形状
            dur_ref  = pack['dur_ref']
            dur_ref  = dur_ref if dur_ref.dim() == 2 else dur_ref[None]
            dur_ref  = dur_ref.to(device)

            # dur_ref -> mel2ph -> ctx dur token
            mel2ph_ref = self.length_regulator(dur_ref).to(device)
            mel2ph_ref = mel2ph_ref[:, :mel2ph_ref.size(1)//self.fm*self.fm]
            dur_tokens_2d = mel2token_to_dur(
                mel2ph_ref, ph_ref.shape[1]
            ).clamp(max=hp.get('dur_code_size', hp.get('dur_max_value', 128)) - 1) + 1

            ctx_dur_tokens = dur_tokens_2d.clone().flatten(0, 1).to(device)
            txt_tokens_flat = ph_ref.flatten(0, 1)
            ctx_dur_tokens = ctx_dur_tokens[txt_tokens_flat > 0][None]  # [1, Lctx]
            last_pos = ctx_dur_tokens.shape[1]
            spk_pos_ids_flat = torch.arange(0, last_pos, device=device)[None, :].long()

            # 预填充，拿到该说话人的增量状态
            with model_lock(self.lock):
                _, inc_state = self.dur_model.infer(
                    ph_ref, {'tone': tone_ref},
                    None, None, None,
                    ctx_vqcodes=ctx_dur_tokens,
                    spk_pos_ids_flat=spk_pos_ids_flat,
                    return_state=True
                )

            # 存成 CPU 上的叶子副本
            safe_cpu_state = self._clone_tensor_tree_to(inc_state, torch.device('cpu'))

            dur_prefill[int(sid)] = {
                'incremental_state': safe_cpu_state,               
                'ctx_dur_tokens': ctx_dur_tokens.detach().to('cpu'),
                'last_pos': int(last_pos),
            }

        return dur_prefill


import multiprocessing as mp
from pathlib import Path
import traceback
import re
import os
import torch

def _safe_name(*parts, maxlen=120):
    base = "+".join(parts)
    base = re.sub(r'[^0-9a-zA-Z\u4e00-\u9fa5\-\+\._\[\]\(\)]+', '_', base)
    return base[:maxlen]

from collections import OrderedDict

def _make_get_resource_context(infer_ins, max_cache=8):
    cache = OrderedDict()

    def get_resource_context(ref_paths, ref_texts_pair):
        key = (str(ref_paths[0]), str(ref_paths[1]),
               None if ref_texts_pair is None else (str(ref_texts_pair[0]), str(ref_texts_pair[1])))

        if key in cache:
            cache.move_to_end(key)
            return cache[key]

        with open(ref_paths[0], 'rb') as fa, open(ref_paths[1], 'rb') as fb:
            ref_bytes = [fa.read(), fb.read()]
        rc = infer_ins.preprocess(ref_bytes, ref_texts=ref_texts_pair)

        cache[key] = rc
        cache.move_to_end(key)
        if len(cache) > max_cache:
            cache.popitem(last=False)  # 淘汰最旧的
        return rc

    return get_resource_context


def _build_dialogue_label(infer_ins, raw: str, clauses_per_seg=2, per_seg_chars=24, max_segments=2):
    segs = infer_ins._parse_dialogue_segments(raw)  # [(sid, content), ...]
    parts = []
    for sid, content in segs[:max_segments]:
        s = re.sub(r'\s+', ' ', (content or '')).strip()
        clauses = [c for c in re.split(r'[，！,\.！!？?\n]', s) if c]
        summary = '_'.join(clauses[:clauses_per_seg])[:per_seg_chars]
        parts.append(f'[{sid}]{summary}')
    return '_'.join(parts)

def _process_one_job(
    global_idx: int,
    g: dict,
    infer_ins,
    get_resource_context,
    out_dir: str,
    time_step: int,
    seq_cfg_w,
    forward_kwargs: dict,
    quiet: bool = True,
):
    """
    单 job：rc -> forward -> save wav
    quiet=True 时屏蔽 preprocess/forward 内的所有打印
    """
    ref_paths = g["ref_audios"]
    assert isinstance(ref_paths, (list, tuple)) and len(ref_paths) == 2, \
        "ref_audios 必须是长度为2的 (path_a, path_b)"

    ref_texts_pair = g.get("ref_texts", None)
    if ref_texts_pair is not None:
        assert isinstance(ref_texts_pair, (list, tuple)) and len(ref_texts_pair) == 2, \
            "ref_texts 必须是长度为2的 (txt_a, txt_b)"

    text_gen = g["text"]                         # 生成用（可能含 prefix）
    text_for_name = g.get("text_for_name", text_gen)  # 命名用（通常不含 prefix）

    # --- preprocess + forward 全静默 ---
    with suppress_output(enabled=quiet):
        rc = get_resource_context(ref_paths, ref_texts_pair)

    with suppress_output(enabled=quiet):
        out = infer_ins.forward(
            rc,
            text_gen,
            time_step=time_step,
            seq_cfg_w=seq_cfg_w,
            **forward_kwargs
        )

    label = _build_dialogue_label(infer_ins, text_for_name)

    prefix = g.get("out_name_prefix", "")
    if not prefix:
        # 兜底：没有就用 line_id / global_idx
        lid = g.get("line_id", global_idx)
        prefix = f"{lid:06d}_"

    save_name_core = _safe_name(
        f"{Path(ref_paths[0]).parent.name}{Path(ref_paths[0]).stem.replace('_vocal','')}",
        f"{Path(ref_paths[1]).parent.name}{Path(ref_paths[1]).stem.replace('_vocal','')}",
        label
    )
    save_name = f"{prefix}{save_name_core}"
    save_path = f'{out_dir}/{save_name}.wav'
    save_wav_bytes(out.wav_bytes, save_path)
    return save_path

def _worker(
    rank: int,
    gpu_id: int,
    job_slice,           # List[(global_idx, g)]
    queue,
    model_kwargs: dict,
    out_dir: str,
    time_step: int,
    seq_cfg_w,
    forward_kwargs: dict,
    quiet: bool = True,  # 新增：默认静默
):
    try:
        torch.cuda.set_device(gpu_id)

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        infer_ins = MegaTTS3DiTInfer(
            device=f'cuda:{gpu_id}',
            **model_kwargs
        )
        kill_void()

        get_rc = _make_get_resource_context(infer_ins)

        for (global_idx, g) in job_slice:
            try:
                p = _process_one_job(
                    global_idx=global_idx,
                    g=g,
                    infer_ins=infer_ins,
                    get_resource_context=get_rc,
                    out_dir=out_dir,
                    time_step=time_step,
                    seq_cfg_w=seq_cfg_w,
                    forward_kwargs=forward_kwargs,
                    quiet=quiet,
                )
                queue.put((global_idx, p, None))
            except Exception as e:
                # 失败只回传错误字符串，不在 worker 里 print
                queue.put((global_idx, None, repr(e)))

    except Exception as e:
        # worker 级别 fatal：所有任务都标失败
        for (global_idx, _) in job_slice:
            queue.put((global_idx, None, f'WORKER_FATAL: {repr(e)}'))


import re

_SPK_TAG_SPLIT_RE = re.compile(r'(<SPK>\d+</SPK>)')
_SPK_TAG_FULL_RE  = re.compile(r'^<SPK>(\d+)</SPK>$')

def merge_consecutive_same_spk(text: str) -> str:
    """
    如果出现 ... <SPK>k</SPK> ... <SPK>k</SPK> ...（中间无其它speaker标签，仅可能有空白）
    则去掉后面的重复 <SPK>k</SPK>，实现连续同speaker合并。
    """
    parts = _SPK_TAG_SPLIT_RE.split(text)
    out = []
    last_spk = None

    for part in parts:
        if not part:
            continue

        m = _SPK_TAG_FULL_RE.match(part.strip())
        if m:
            spk = m.group(1)
            if spk == last_spk:
                # 连续同 speaker：跳过这个重复标签
                continue
            last_spk = spk
            # 统一输出规范化标签（避免 part 里有奇怪空格）
            out.append(f"<SPK>{spk}</SPK>")
        else:
            out.append(part)

    return "".join(out)


# =========================================
# Example (批量多组)
# =========================================

if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')

    # =============================
    # 1) 配置
    # =============================
    # dit_exp_name = 'checkpoints/260106_megatts3_dit_dialogue_copy'
    # dit_exp_name = 'checkpoints/260113_megatts3_dit_dialogue_prompt'
    dit_exp_name = 'checkpoints/260127_megatts3_dit_dialogue_short'
    # dit_exp_name = '/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/checkpoints/260130_megatts3_dit_dialogue_prompt/model_ckpt_steps_200.ckpt'
    dur_exp_name = 'checkpoints/251104_dur_lm_multispk'
    frontend_exp_name = 'checkpoints/250923_lm_mfa_seq2seq_small_wavlmlarge_long_robust'

    time_step = 100
    seq_cfg_w = [1.6, 3.0]
    max_ref_duration = 60

    spk_combos = [
        # # 1: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/1-亲切美姨.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav"
        # ),
        # # 2: 男1+男2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/1-爽朗大叔.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-磁音直播男声.wav"
        # ),
        # # 3: 男1+男2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-清亮带货男声.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-爽朗大爷.wav"
        # ),
        # # 4: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-健谈大姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-热心姐姐.wav"
        # ),
        # # 5: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-知性美姨.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav"
        # ),
        # # 6: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-热心姐姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-爽快大姐.wav"
        # ),
        # # 7: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/1-稳重奶奶.wav"
        # ),
        # # 8: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-爽快大姐.wav"
        # ),
        # # 9: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-雅致靓姨.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav",
        # ),
        # # 10: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/1-亲切美姨.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav"
        # ),
        # # 11: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-健谈大姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-热心姐姐.wav"
        # ),
        # # 12: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-知性美姨.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav"
        # ),
        # # 13: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-知性美姨.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav"
        # ),
        # # 14: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/1-稳重奶奶.wav"
        # ),
        # # 15: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-爽快大姐.wav"
        # ),
        # # 16: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-雅致靓姨.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-爽快大姐.wav"
        # ),
        # # 17: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/1-亲切美姨.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav"
        # ),
        # # 18: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-健谈大姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-热心姐姐.wav"
        # ),
        # # 19: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-知性美姨.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav"
        # ),
        # # 20: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-热心姐姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-爽快大姐.wav"
        # ),
        # # 21: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/1-稳重奶奶.wav"
        # ),
        # # 22: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-爽快大姐.wav"
        # ),
        # # 23: 男1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-磁音直播男声.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-热心姐姐.wav",
        # ),
        # # 24: 男1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/无声场男/1-雅致大爷.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/无声场女/2-雅致靓姨.wav"
        # ),
        # # 25: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-爽快大姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/1-亲切美姨.wav"
        # ),
        # # 26: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/1-亲切美姨.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-热心姐姐.wav",
        # ),
        # # 27: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-健谈大姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/1-亲切美姨.wav",
        # ),
        # # 28: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-知性美姨.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav"
        # ),
        # # 29: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-热心姐姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/1-亲切美姨.wav",
        # ),
        # 30: 男1+男2
        (
            "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-清亮带货男声.wav",
            "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-爽朗大爷.wav"
        ),
        # # 31: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/1-稳重奶奶.wav"
        # ),
        # # 32: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-爽快大姐.wav"
        # ),
        # # 33: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-雅致靓姨.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav",
        # ),
        # # 34: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/1-亲切美姨.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav"
        # ),
        # # 35: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-健谈大姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-热心姐姐.wav"
        # ),
        # # 36: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-知性美姨.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav"
        # ),
        # 37: 女1+女2
        (
            "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-知性美姨.wav",
            "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav"
        ),
        # 38: 女1+女2
        (
            "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/2-雅致靓姨.wav",
            "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav",
        ),
        # # 39: 男1+男2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/1-爽朗大叔.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/3-磁音直播男声.wav"
        # ),
        # # 40: 女1+女2
        # (
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-爽快大姐.wav",
        #     "/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/prompts/双人bench_final/test/4-飒爽小姐姐.wav",
        # ),
    ]

    text_cases = [
        # {"text": "<SPK>1</SPK>宝妈们别再乱选奶粉了！选不对成分不仅白花钱，宝宝还不吸收！最近有款超火的奶粉，居然藏着宝宝成长的秘密武器。我今天必须扒给你们看！就是这款【圣元优博盖诺安奶粉】。好多宝妈反馈说奶粉粉质细腻，味道清淡。刚转奶的宝宝也挺适应的。关键是营养特别全！我仔细查了成分表。里面不仅有高蛋白、钙铁锌，这些基础营养。还加了益生元。能减少宝宝便秘，让大便更软！你们知道吗？钙铁锌，被称为成长铁三角。宝宝长身体的时候可不能缺！而且它的蛋白质含量特别高。还是优质的乳清蛋白。营养特别全面！这款是最近超火的热门复购款。已经有4000多个宝妈看过了。大家都说宝宝喝了之后变化特别大！关键现在还有新客专享优惠。真的太划算了<SPK>2</SPK>真的可以给宝宝试试！"},
        # {"text": "<SPK>1</SPK>百年易武普洱生茶。<SPK>2</SPK>撬散小袋竟只要这个价？<SPK>1</SPK>你们有没有花高价买过普洱茶。喝起来却像树叶？今天这款易武同庆号普洱生茶。<SPK>2</SPK>直接帮你把性价比拉满！<SPK>1</SPK>咱们直接跟【易武核心产区】茶农拿的原料。只选三年转化的优质古树茶。<SPK>2</SPK>保证是正宗的易武味道！<SPK>1</SPK>入口先是清甜的花果香。咽下去喉咙里还留着回甘。完全没有那种生涩的苦感！平时买这么一款正宗易武普洱生茶得花不少钱吧？<SPK>2</SPK>今天为了冲销量。<SPK>1</SPK>直接给到地板价！<SPK>2</SPK>百年易武普洱生茶。<SPK>1</SPK>撬散小袋只要989，还支持先试喝。<SPK>2</SPK>不满意直接退！"},
        # {"text": "<SPK>1</SPK>鞋底防滑耐磨，行走安全不易滑。质感好舒适又耐穿。纹理清晰手感细腻。上脚还合脚不挤脚。<SPK>2</SPK>喜欢的朋友可以带一双回去感受下！咱们是源头厂家直接发货。没有中间商赚差价，价格非常实惠，性价比超高！喜欢的朋友赶紧趁活动抢一双！"},
        # {"text": "<SPK>1</SPK>秋冬换季宝宝脸干得起皮、红得像小苹果？<SPK>2</SPK>别慌，这款宝宝面霜帮你搞定！<SPK>1</SPK>它加了大牌同款的金盏花成分。舒缓修护超给力，敏感肌宝宝也能放心用。涂完脸软fufu的。就像刚剥的鸡蛋。它其实是早晚霜搭配用的。早霜白天涂，锁水保湿一整天。宝宝在外面疯跑也不怕脸干；晚霜晚上涂，修护舒缓。泛红敏感涂它就对了。两个配起来用，宝宝整个秋冬的护肤都搞定了。关键价格才几十块钱，比一瓶成人面霜还便宜。<SPK>2</SPK>真的划算到哭！"},
        # {"text": "<SPK>1</SPK>颈霜是智商税？我用28天实测打脸！数据显示颈纹数量真的减少了30.66%！<SPK>2</SPK>我也被这效果征服了<SPK>1</SPK>我是个成分党，以前买过贵的、跟风买过网红的，颈纹反而更深了。<SPK>2</SPK>直到用了SDX。<SPK>1</SPK>它的三重滚珠按摩设计太绝了。挤出来直接滚。不脏手还能促进吸收。里面有重组胶原蛋白。针对颈纹松弛、暗沉粗糙。从根源淡化。每天早晚滚一滚。现在颈部状态特别好。真的离不开。<SPK>2</SPK>150g大容量两支，好评率96%。快去试试！"},
        # {"text": "<SPK>1</SPK>谁能想到啊，我熬大夜之后素颜不垮脸的秘密。竟然是这个绝色家的【熬夜急救奶皮面膜】！敷之前先涂自带的水光霜打底。打开毛孔让营养更好吸收。敷完感觉皮肤喝饱了水。摸起来滑滑嫩嫩的，干纹细纹都淡了点。膜布像奶皮一样软软糯糯的，而且做了分区设计。眼部和法令纹加厚，更服帖。能更好地照顾到容易垮的地方。十五分钟就能把熬夜的暗沉黄气压下去。敷完脸又亮又透，跟开了磨皮一样。就算素颜出门也完全没问题！<SPK>2</SPK>而且它性价比真的很高。不到一杯奶茶钱就能 get 贵妇级享受。链接我已经给你们挂好了。<SPK>1</SPK>想拥有好皮肤的姐妹赶紧冲，错过真的会后悔！"},
        # {"text": "<SPK>1</SPK>卫生间的马桶边角藏污纳垢。传统刷子根本刷不到，还容易滋生细菌发臭！<SPK>2</SPK>试试这个360度无死角马桶刷。今天下单立减2元！<SPK>1</SPK>别再用这种藏污纳垢的脏刷子了！我们家这个是加长马桶刷，加长手柄免弯腰。小黄鸭造型美观又实用。关键它是独特加液设计，把洁厕液加进去。按一下就能出液，精准去污。自带吸盘牢固又方便。你看这刷头，360度贴合马桶边角。陈年污渍刷一遍就掉，自带清洁剂，省时又省力。用完水冲即净。还能壁挂上墙收纳，完全不占空间！<SPK>2</SPK>家里有老人孕妇的赶紧备上。趁活动还在，快冲！"},
        # {"text": "<SPK>1</SPK>俄罗斯进口0蔗糖巧克力糖。很多家长还不知道！我本来以为是智商税。结果一吃就上瘾，真的惊艳到了。真的太香了。关键是0蔗糖添加更健康，里面还有夹心。黑巧克力味道浓郁，口感丝滑香醇。越嚼越香，关键不甜腻。现在下单价格真的很划算。<SPK>2</SPK>这价格能买进口巧克力真的太值了。<SPK>1</SPK>而且还是纯可可脂制作。之前卖五十四块多，现在只要二十七块八。喜欢吃巧克力的赶紧去尝尝。年前囤好我的本命零食，就等过年啦。"},
        # {"text": "<SPK>1</SPK>圣诞送礼还在纠结？想送大牌又不想花冤枉钱？这套【纪梵希明星双色组合】我直接锁死！内含G细管N05号色，搭配限量散粉15号色。一套拿下热门明星产品。这就很划算！散粉是柔雾妆感的神。粉质细腻到像烟雾一样。轻轻一拍毛孔瞬间隐形。秒变妈生好皮！这支N05号色就是富家千金本千金。上嘴丝绒哑光，高级又显白。谁涂谁好看！重点来了！买套组就赠同款色号唇膏、柔肤面霜。还有个超洋气的化妆包。这羊毛必须薅！全套加赠品价值真的绝了。节日限定优惠别错过，官旗发货保证正品。<SPK>2</SPK>姐妹们快冲！"},
        # {"text": "<SPK>1</SPK>家人们！双11薅羊毛的机会来了！<SPK>2</SPK>华为平板直降好几百！我是真的没想到。一块平板如何让办公效率翻倍？直到同事把笔记本电脑扔了换成这个华为平板。我才明白。原来平板办公真的能碾压电脑！11.5英寸超大屏，比普通平板大一圈。分屏操作、多任务处理。效率直接起飞！鸿蒙系统搭配PC级WPS。手机平板多屏协同，文件互传秒搞定。这才是打工人的梦中情机！144Hz高刷屏，滑动丝滑不卡顿。看剧办公都像开了4K特效。视觉体验绝了！<SPK>1</SPK>现在入手还享6期免息！<SPK>2</SPK>赠品多到抱不过来，直播间还有限时加赠。<SPK>1</SPK>手慢真的无！"},
        # {"text": "<SPK>1</SPK>圣诞想和孩子一起做手工？<SPK>2</SPK>试试这个能让娃远离手机的圣诞树钻石画！<SPK>1</SPK>把胶水涂在画布上。用点钻笔轻轻点上彩钻。从空白画布到闪闪发光的圣诞树，看着超治愈。有四种颜色可选。贴完装在白色相框里。挂在圣诞树上或者摆在客厅，瞬间拉满节日氛围感！孩子亲手做的圣诞树。既锻炼动手能力，又能当装饰品。<SPK>2</SPK>比买的礼物更有意义～"},
        # {"text": "<SPK>1</SPK>降温了，给你老公准备两件换洗的加绒加厚卫衣。来看看这款奥粒绒一体绒的高端卫衣。采用亲肤保暖的加厚奥利绒面料。柔软细腻贴身穿不扎不痒。还抗静电抗起球。衣服上的图案是高温压花工艺。不是劣质胶印，久穿久洗不脱落。精致，美观，上档次。穿出去一看就是专卖店的品质。面料弹力十足，一百到二百一十斤都能穿。遮肉显瘦谁穿谁好看。版型立体又有型，穿起来时尚百搭。休闲还显风度，宽松版型不压气场。商务休闲都能穿。立领设计实用又好看，自带小拉链细节满满。还能防风保暖。前侧做了拿褶上线工艺。门襟、底板、袖口都是无痕工艺。显瘦洋气显气质。<SPK>2</SPK>有品位有内涵的兄弟姐妹们，这款卫衣别错过。给老公选两件真的不亏。"},
        # {"text": "<SPK>1</SPK>宝子们！国风通勤包还能这样有文化感？<SPK>2</SPK>我挖到了一款绝了的包包！<SPK>1</SPK>它的手工刺绣，工艺真的绝。每一针都超精致。而且用的是牛皮材质，摸起来细腻又耐用。再来说说设计，外观是国风图案。超有品位，同事看到都问链接。容量也很大，通勤带的东西都能装下。日常出行完全够用。我最喜欢的是它的图案设计。太有国风那味了！它是可可棕/月光白的配色，搭配国风图案。悲出去超有文化感，朋友看到都夸有品位。而且特别百搭。不管配日常装还是新中式都好看。<SPK>2</SPK>朋友都说我很有文化品位，我真的太爱了！宝子们，喜欢国风的一定要入手！"},
        # {"text": "<SPK>1</SPK>一袋嫌少、两袋不够吃？这款500g量贩装夏威夷果。直接承包你全家的追剧零食库！<SPK>2</SPK>这可不是普通零食。<SPK>1</SPK>是很多用户都在夸的健康美味。多位用户评价奶香味足、个头大又饱满。CTR数据比同行高出60%。拒绝那些剥壳费劲的坚果。咱这款是环形大开口。手指轻轻一捏，果肉直接蹦出来。老人小孩都能轻松吃。原料只选当季大果。高温去皮后，低温慢烤。<SPK>2</SPK>0油0糖无负担。<SPK>1</SPK>每一口都是原果清香，酥脆不腻。重点看这里！日常价四十八岁。现在尝鲜价直接打骨折。<SPK>2</SPK>到手只要十五块八！<SPK>1</SPK>年前囤货季，这波福利别错过。全家老小都爱吃，趁活动赶紧囤。手慢真的就没有啦！"},
        # {"text": "<SPK>1</SPK>小月龄宝宝冬天别只穿袜子！脚底板老发凉，婆婆赶紧备了这双两面可穿的步前鞋。软到能折成小团。宝宝踩在地板上都暖乎乎的。最贴心的是抽拉式设计。一拉一系，牢牢裹住脚踝。宝宝怎么蹬腿都不掉鞋，也不会钻风。内里加绒加厚，零下几度穿都没问题。软fufu的像踩在棉花上。宝宝上脚特别舒服。<SPK>2</SPK>我看好多宝妈反馈说收到货，后悔没多拍两双。现在才二十多块钱。之前都要四十呢！有需要的家人们赶紧趁活动给宝宝备上。等天冷了不仅涨价。发货还慢！"},
        # {"text": "<SPK>1</SPK>家人们！孩子考试总看不清时间？或者手表滴答响，打扰同学？今天这款Fila手表直接解决你所有烦恼！这款手表太适合学生党了。<SPK>2</SPK>石英机芯走时精准还静音。<SPK>1</SPK>完全不用担心考试时滴答声吵到别人，帮孩子专注考试不被打扰。指针和刻度都有夜光设计。<SPK>2</SPK>光线暗也能看时间。<SPK>1</SPK>还有贴心日历设计，帮孩子随时掌握日期，做好时间规划。经典款式表盘设计简洁。大数字读时轻松。<SPK>2</SPK>孩子考试看时间一眼就能看清，不用浪费时间眯眼找刻度。<SPK>1</SPK>还有清新配色设计。实用又好看。<SPK>2</SPK>学生党肯定喜欢，日常佩戴也超合适。家人们放心冲。我是专柜卖家正品有保障。这就给你们上链接，喜欢的赶紧去拍！"},
        # {"text": "<SPK>1</SPK>在家开演唱会的快乐谁懂？别再乱买一堆了！<SPK>2</SPK>这套金运K88，让你花小钱办大事！<SPK>1</SPK>以前那是真麻烦，声卡话筒加音箱。接线让人头秃。现在这一台，全给你集成了！这就相当于把录音棚搬回家了！五核芯片实时修音。跑调也能给你掰回来，开口就是原唱！哈曼音效加持。那种360度环绕的通透感，听一次就上头。聚会绝对是气氛组C位。手提式移动KTV。内置小度AI语音点歌。出门露营拎着就走，随时随地都能燥起来！<SPK>2</SPK>别犹豫了，这快乐值得拥有。赶紧点进去看看吧！"},
        # {"text": "<SPK>1</SPK>有没有妈妈跟我一样？孩子的牙刷牙膏总堆在洗手台，台面乱得没处放杯子。牙膏还老积水发霉？自从用了这个星星收纳架。我家洗手台瞬间空出一大半！它是免打孔的，直接贴墙上就行。不用钻墙破坏瓷砖。我家卫生间贴了半年都没掉！上面的格子刚好放孩子的牙刷、牙膏。连电动牙刷都能插。下面还有沥水孔，再也不会积水发霉了！关键是星星造型孩子超喜欢。现在每次刷完牙都主动把东西摆好。再也不用我跟在后面收拾啦！<SPK>2</SPK>之前原价16块9，现在只要9块9就能拿下。妈妈们赶紧给孩子安排一个。让卫生间变整洁又有趣！"},
        # {"text": "<SPK>1</SPK>宝子们！卫仕家的主食猫条平时都要七十多。现在才三十九块九！这波不薅真的亏！他家这个是真主食猫条。营养成分跟主食罐一样。不用怕吃多了没营养，就是主食级的！重点是加了深海胶原！平时猫咪掉毛掉得沙发全是。用这个养养毛囊，掉毛都少了！你看我家猫，每次都舔得连袋子都不剩。真的是太爱吃了！<SPK>2</SPK>平时不怎么吃猫条的宝子，这个一定要试！<SPK>1</SPK>现在这么低的价格，还能买这么多口味。真的绝了！<SPK>2</SPK>刷到的宝子别错过，这波机制也就这几天。<SPK>1</SPK>错过就没这价格了！每人只能拍一单啊。多了不给发，赶紧冲！"},
        # {"text": "<SPK>1</SPK>家长们别再手动抄题了！这款能打印A4纸的零门槛打印机，无需加墨。手机直连，打印速度快到飞起！你知道为什么学霸整理错题快吗？因为他们不手写！用这个A40a错题打印机，抄题半小时变成打印5秒。省下时间刷两套卷子不香吗？它是无需墨的A4打印机。体积小巧不占地，随时随地都能打。平时孩子的作业、试卷、甚至是手工涂色画。手机连蓝牙，一键就能搞定。打印效果清晰流畅。字体锐利不模糊，重点内容还能加粗标记。关键是不用买碳粉、墨水和墨盒，一次投入长期使用。性价比真的很高！<SPK>2</SPK>现在的家长真的要学会用科技帮孩子减负！这款A40a错题打印机，操作简单易上手。<SPK>1</SPK>孩子自己就能搞定。<SPK>2</SPK>别让抄题浪费孩子的学习时间。赶紧给孩子安排一台吧！"},
        # {"text": "<SPK>1</SPK>戴眼镜的宝子们，有没有发现？为什么擦过的镜片既不起雾又没水印？戴眼镜的朋友都知道，镜片特别容易脏。沾指纹、油污，还容易起雾。用普通镜布擦了之后全是划痕。别再用镜布擦眼镜了。这就相当于用细砂纸在擦镜片！<SPK>2</SPK>这款眼镜清洁湿巾，真的解决了我所有的清洁烦恼。<SPK>1</SPK>我实测过好几款，这款真的绝了。一擦即净，而且它速干不留痕。也不伤镜片。之前实体店买这一盒老贵了。现在厂家搞活动，不到一杯奶茶钱就能带走一大盒！它是独立包装，出门带着也方便。像我这种戴眼镜的，每天都要擦好几次。这个擦完特别透亮。再也不用怕起雾尴尬了！关键它还不含酒精，不会腐蚀镜片。擦完之后镜片上还有一层保护膜。不容易沾灰。眼镜布用久了容易滋生细菌。这种独立包装的湿巾，一次一片。干净又卫生。不管是眼镜、手机屏幕还是电脑屏幕。都能擦，一擦就干净！独立包装出门带着也方便。<SPK>2</SPK>这一大盒有100片，够用大半年了。真的是性价比之王！"},
        # {"text": "<SPK>1</SPK>姐妹们！去年冬天冻得睡不着。经期肚子凉得直冒冷汗的举个手！<SPK>2</SPK>我今年早早就备上了这个卡通加强款暖贴。<SPK>1</SPK>真的绝了！<SPK>2</SPK>之前我花四十块钱买的就不说了。<SPK>1</SPK>现在十七块钱到手整整十贴！<SPK>2</SPK>关键是萌趣卡通造型。<SPK>1</SPK>每片都超可爱。普通暖贴只能管三五个小时。这款加强版能持续发热八到十个小时。女生那几天贴肚子上，暖乎乎的。粘性巨牢，狂甩都不掉。撕下还不留背胶。怕冷的姐妹贴哪里都可以。趁现在没涨价，赶紧囤！<SPK>2</SPK>马上要降温了。<SPK>1</SPK>别等冻得发抖再买！"},
        # {"text": "<SPK>1</SPK>家人们别划走！不是所有三七都来自文山。这是咱们源头厂家直供的三年生地道三七。今天给你们把价格打下来了！平时买三七怕买到假货，更怕价格虚高。中间商赚差价。我们直接从文山发货。省去所有环节。只为让你们吃到放心的好三七。这款三七精粹饮，采用植物小分子提取技术。吸收更快更好。口感纯正。微苦回甘，都是真材实料。不添加任何精糖和防腐剂。为了回馈新老客户，今天活动机制特别给力！买得越多，送得越多。折算下来每天成本超低。<SPK>2</SPK>全家人都能喝，养生要趁早。<SPK>1</SPK>千万别错过！源头厂家直供，品质有保障。趁着活动还在，赶紧给家里人囤起来。健康才是最重要的！"},
        # {"text": "<SPK>1</SPK>宿舍晚上学习灯光不够？普通台灯伤眼还占地方？别再用那种烂大街的台灯了！这款公牛护眼台灯才是学生党的神！A+级照度认证，LED无频闪光源。护眼这块拿捏得死死的。关键是它的夹式设计！不占桌面空间，随便夹哪里都稳固。宿舍床上、书桌边都能用。4000毫安大电池，充插两用。续航能力超强。晚上通宵赶作业也不怕断电。多种灯光模式随意切换。学习、阅读都适配。这氛围感，谁用谁知道！学生党闭眼冲！<SPK>2</SPK>宿舍必备好物，价格还不贵。<SPK>1</SPK>链接就在下面，手慢无！"},
        # {"text": "<SPK>1</SPK>冬天给娃洗澡总受凉？传统澡盆又浅又占地方！这款夹棉折叠泡澡桶，一甩即开免安装。我家娃现在天天喊着要泡澡！三百六十度夹棉加厚，锁温效果超好。冬天泡澡水温不易凉。再也不怕宝宝受凉感冒。像我这种怕冷的宝妈，偶尔也能和娃一起泡。底部双排水口，排水快捷不费力。用完一折就收纳。不占空间还美观。图案也很可爱，孩子特别喜欢。每天洗澡都超开心。材质厚实无异味，大人小孩都能放心用。容量超大，零到十二岁都能用。活动价也很划算。原价九十九，现价才八十九。<SPK>2</SPK>喜欢的宝妈们赶紧入手吧！让宝宝爱上洗澡，这款泡澡桶真的值得拥有！"},
        # {"text": "<SPK>1</SPK>姐妹们谁懂啊！熬夜追剧压力大。发缝宽得能透光，梳个头掉一把。真的显老十岁！这是我换老公都不换的洗发水。用空好几瓶。发缝真的肉眼可见变窄了。好多姐妹问真的有效果吗？它是持证上岗的防脱洗发水。控油防脱双功效。以前我也是大油田，每天洗头还油得贴头皮。换了这个。三天不洗都蓬松。里面加了何首乌当归这些植萃精华。温和清洁。洗完头皮像在呼吸。之前买了很多没用的，这瓶是真能处！头发顺滑有光泽。氛围感拉满。销量摆在这，用过的都回购。<SPK>2</SPK>趁优惠赶紧冲。远离脱发焦虑！"},
        # {"text": "<SPK>1</SPK>你花大价钱买的平板。<SPK>2</SPK>真不一定有这款实用！<SPK>1</SPK>华为MatePad 11.5英寸平板，原价2049元，现在只要1175元！看一下评价，好多人说它运行速度快。续航能力强，还能当学习机、办公本、娱乐屏用。<SPK>2</SPK>性价比直接拉满。<SPK>1</SPK>11.5英寸的120Hz护眼柔光屏。孩子用来看动画不伤眼。<SPK>2</SPK>家长用来办公也舒服。<SPK>1</SPK>鸿蒙系统加上8核CPU。运行速度非常流畅。<SPK>2</SPK>有276个人评价不卡顿。<SPK>1</SPK>256G大内存，能存孩子一学期的网课视频。还有1300万像素摄像头。<SPK>2</SPK>拍作业、上网课都清楚。<SPK>1</SPK>还有7700毫安大电池。续航能力特别强，出门不用带充电宝。全金属一体化机身。<SPK>2</SPK>拿在手里特别有质感。<SPK>1</SPK>现在限时低价1175元。<SPK>2</SPK>家里有娃的赶紧冲。错过就恢复原价了！"},
        # {"text": "<SPK>1</SPK>天天低头看电脑、写作业。肩颈硬得像块砖？<SPK>2</SPK>10分钟快速舒缓散压，40分钟脖颈焕活。<SPK>1</SPK>这个天鹅贴真的绝了！它是黑科技与传统结合。按压精油仓让蓝艾精油渗透进艾草包。古法艾灸加现代科技。原料都能看得见。里面有高原艾草和蓝艾精油。全是草本成分很安全，温度刚好不烫伤。贴上像做了场spa。肩颈瞬间放松。不管是办公久坐、居家休息还是出差旅行。一贴就能灸，舒缓肩颈疲劳特别好用。销量和评价都超好！<SPK>2</SPK>就是这个天鹅贴。一贴即灸，每天十分钟。肩颈舒服了，做什么都有精神！"},
        # {"text": "<SPK>1</SPK>孩子学校一坐就是一天，板凳硬得硌屁股。特别是秋冬到了，板凳又冷又硬。孩子总喊屁股疼、腿麻。写作业都坐不住？给孩子试试这款记忆棉坐垫！尺寸正好适配学校凳子。底部有防滑颗粒，加上双搭扣弹力绑带设计。绑在凳子上稳固得很，根本不会窜来窜去。记忆棉内里软fufu的，还能慢回弹。孩子坐上去像陷进云朵里，久坐都不觉得累。上课、写作业注意力都能集中。学习效率都变高了。<SPK>2</SPK>这款坐垫我给我们家孩子买了，反馈特别好。<SPK>1</SPK>不仅软弹舒适，久坐不变形。关键是防滑稳固，冬天坐也不冷。<SPK>2</SPK>真的是学习好帮手！"},
        {"text": "<SPK>1</SPK>茅台镇私藏酱酒，五升大坛仅需成本价？<SPK>2</SPK>今天我要让大家知道。什么是真正的源头好酒！<SPK>1</SPK>咱们酱香酒爱好者最担心啥？怕买到假酒、怕花冤枉钱！喝到假酒，那心里比针扎还疼啊！<SPK>2</SPK>今天我这个做了几十年酒的老伙计。<SPK>1</SPK>给大家掏心窝子来送福利了！五升大坛装，够喝大半年！很多酒友担心酒质？放心！支持开盖试喝。口感醇厚不上头！<SPK>2</SPK>怕买到假酒？我直接给你承诺：<SPK>1</SPK>收到货不是纯粮好酒。<SPK>2</SPK>或者口感不好，直接退！运费我出！只有今天！为了口碑不搞那些虚头巴脑的。<SPK>1</SPK>这波福利<SPK>2</SPK>错过就没了！"},
        # {"text": "<SPK>1</SPK>兰黛智妍紧塑精华霜75ml才二百九十六！<SPK>2</SPK>之前卖四百九十六。现在的活动价真的太划算了！<SPK>1</SPK>不管你是熬夜后的脸松垮垮、细纹冒得快，还是法令纹越来越明显。<SPK>2</SPK>怕买贵又怕没效果的姐妹都得看！好多达人都推荐过这款。<SPK>1</SPK>能抗皱紧致、去黄淡纹还补水护肤。用户反馈特别好。<SPK>2</SPK>大家都说它效果很好不油腻。<SPK>1</SPK>质地细腻吸收快！现在它在面霜品类里排第五名。<SPK>2</SPK>还添加了马齿苋提取物。<SPK>1</SPK>能改善熬夜后皮肤松垮、细纹滋生的问题。用完皮肤又紧又细！<SPK>2</SPK>现在的活动机制真的绝了。你们点开我下方专属链接看看。之前舍不得买的现在必须囤！<SPK>1</SPK>但库存有限。<SPK>2</SPK>手慢的姐妹可就抢不到了！"},
        # {"text": "<SPK>1</SPK>姐妹们，这台X80Pro手机拿在手里真的是又好看又好用。直接戳中我的审美点！它的屏幕真的很大，而且超高清。看剧玩游戏都超爽！现在手机外观设计真的越来越同质化了。买个手机跟撞衫似的，一点新鲜感都没有。我喜欢用手机记录生活，普通手机内存小。没拍多少照片就提示内存不足，根本存不下我想留住的瞬间。真的超烦！这款手机还支持全网通双卡双待。工作生活号码分开，超方便！屏幕小的手机看剧真的超难受，画面挤得慌。一点细节都看不清，看久了眼睛还酸。<SPK>2</SPK>姐妹们，这款X80Pro手机真的值得冲。外观好看又实用，别错过！"},
        # {"text": "<SPK>1</SPK>哎你出差咋不带行李箱啊？<SPK>2</SPK>带啥行李箱啊，我这包比22寸行李箱能装多了！<SPK>1</SPK>平时出门我都带这种轻便拉杆旅行袋。里面能装下我出差五天的换洗衣物。还能塞下电脑、平板、充电宝这些电子设备。你别看它装满了近二十公斤的东西。它用的是轻便拉杆。我拉着走了半小时都没觉得累！而且它用的是牛津布材质，耐磨又防泼。我上次下雨赶高铁。包湿了里面衣服都没潮。底部的鞋位能把鞋子单独收起来。再也不用跟衣服混着挤，干净又卫生！万向轮超顺滑，拉着跑都不卡轮。单手就能推！内部还有个独立拉链暗袋。能放身份证、银行卡这些重要东西，安全得很！对了，它还能折叠收纳。不用的时候折起来放柜子里，一点都不占地方！出差带它比行李箱方便十倍，装得多还轻。<SPK>2</SPK>再也不用跟行李箱较劲了！"},
        # {"text": "<SPK>1</SPK>有没有化妆新手跟我一样？用单指粉扑总握不住。拍粉底要么卡粉要么拍不匀。浪费半小时还像没化？今天这个粉扑真的救了我！三指带设计特别好握。怎么拍都不滑。像我这样的手残党也能三两下搞定底妆！而且它巨省粉！普通粉扑拍三下就吃一半粉。这个粉扑拍十下都不怎么吸粉。一瓶粉底能多用半个月！上脸巨服帖，拍出来是那种清透的奶油肌。瑕疵都能盖住，完全不会卡粉浮粉。比我之前用的粉扑好用一百倍！还能干湿两用，干用拍粉底。湿用拍遮瑕。一个顶两个用，真的超实用！弹力丝带特别结实，怎么拉都不会断。用久了也不变形，性价比绝了！<SPK>2</SPK>现在活动才九块九两盒，真的太划算了！化妆新手姐妹赶紧冲。用过你就知道有多香！"},
        # {"text": "<SPK>1</SPK>注意看！家里燃气灶出现点火后松手火苗熄灭的情况。<SPK>2</SPK>千万别大意！<SPK>1</SPK>90%的人都不知道，这其实是点火针坏了！点火针是燃气灶的易损件，长期高温炙烤容易老化。导致打不着火或者松手灭火。<SPK>2</SPK>找师傅上门修，一次就得好几十。其实完全不用花那冤枉钱！<SPK>1</SPK>这款通用型燃气灶点火针，自己在家就能换。不用焊接不用接线。直接插上去就行！换好之后，轻轻一按就着。火苗稳稳的，比以前灵敏多了！<SPK>2</SPK>家里燃气灶坏了别着急。<SPK>1</SPK>点下方链接，几块钱就能解决大麻烦！"},
        # {"text": "<SPK>1</SPK>高层擦窗还在爬窗台？太危险了！<SPK>2</SPK>赶紧把这个擦窗机器人请回家。<SPK>1</SPK>洒拖T30，它是真的懂我家的窗户！自带智能喷水，哪里脏擦哪里。以前擦窗像打仗，现在喝杯茶的功夫。全屋玻璃锃亮如新，这才是科技改变生活。吸力大到离谱，加上防坠绳设计。<SPK>2</SPK>双重保障，高层外窗也能放心大胆用。<SPK>1</SPK>浴室镜子、厨房瓷砖，它都能搞定。一键启动，脏活累活全交给它。<SPK>2</SPK>别再自己冒险了，这款洒拖T30。<SPK>1</SPK>让你家的窗户天天都像新的一样！"},
        {"text": "<SPK>1</SPK>宝子们！今天我挖到个绝了的粉刺针。<SPK>2</SPK>轻轻一挑，脸上的黑头粉刺直接一颗一颗冒出来。<SPK>1</SPK>巨爽！平时化妆卡粉、鼻子上全是小颗粒的。听我一句劝，赶紧把手里的粗针扔了！我之前用那种，不仅扎得手疼，还留红印子。但这个粉刺针不一样，针头是0.01毫米的超细款。能精准挑破闭口和黑头，还不伤皮肤。另一头是压圈，能把毛孔里的脏东西挤得干干净净。而且是带盖设计，用完直接盖上。再也不怕弄脏针头等下次用了。新手也能轻松上手，像我这种手残党都能用。真的好用到哭！平时脸上爱涨闭口、黑头多的宝子。<SPK>2</SPK>用它准没错！<SPK>1</SPK>尤其是鼻头这种难清理的地方，用它一挑一压。脏东西全出来了，脸瞬间清爽！记得搭配酒精包一起用。更干净卫生哦！<SPK>2</SPK>想拥有干净好皮肤的宝子。<SPK>1</SPK>别犹豫，赶紧冲！"},
        {"text": "<SPK>1</SPK>孩子抄错题抄到半夜？尤其是数学几何、物理电路图。光画图就半小时，手都酸了还容易错！别再手写了！这个错题打印机。<SPK>2</SPK>三步搞定，比抄题快十倍！<SPK>1</SPK>手机拍照就能把题目存进题库。自动识别题目，不会的题一键打印。孩子考前刷错题超方便。像这种带图形的数学几何题、物理电路图。手机拍一下，去除手写痕迹。直接打印出来，连图都超清晰。整理错题再也不用费劲画了！整理好的错题可以直接打印成word。直接打印出来。手机蓝牙直接连，不用插线。放宿舍放家里都方便。不用墨水，不会漏墨堵头。随时用都超省心！<SPK>2</SPK>真的别再让孩子手写抄题了！这个错题打印机，三步搞定。比抄题快十倍！"},
        # {"text": "<SPK>1</SPK>夜骑党看过来！这车灯一开，整条路都亮得像白天。再也不用怕黑了！以前晚上骑车总被对面车灯晃得睁不开眼。自己骑车又怕灯不够亮看不清路。现在这款车灯直接解决了我所有烦恼！塔斯队长这款车灯设计真的太懂夜骑党了。它能发出截止线光斑。骑行时完全不晃对向视线，安全又礼貌。还有两千流明亮度。照亮一百到两百米完全没问题。防水级别IPX6。风雨无阻。不管是下雨还是路过积水都不怕进水。内置3200毫安锂电池。续航能达10-15小时。晚上出门骑车再也不用担心半路没电。六种发光模式随意切换。不管是夜骑、通勤还是露营都能用。它是双安装模式。装在车把或者头盔上都很方便。原价二百七十五，现在只要一百六十八。<SPK>2</SPK>真的太划算了！"},
        # {"text": "<SPK>1</SPK>盖白发总染到头皮？染后发质像稻草？固色不到两月就露白？<SPK>2</SPK>试试美飘扬的这个单剂染发膏！<SPK>1</SPK>不用调不用配。挤出来就能直接用。染发轻松不脏皮肤。普通染膏要配双氧奶。双氧奶是强氧化剂，打开毛鳞片让色素进去。但会伤头皮还让头发变干；美飘扬用的是天然植物提取色素，不用双氧奶。直接裹住毛鳞片，既能盖白发。又不刺激头皮，染完头发还顺滑。白发多、刚染完又露白的姐妹。<SPK>2</SPK>还有对化学染膏敏感的宝子。<SPK>1</SPK>都能用。而且固色能达到一百八十天。白发显老影响形象的。<SPK>2</SPK>别再用普通染膏折腾自己了，赶紧试试美飘扬单剂染发膏。<SPK>1</SPK>好用又方便！"}, 
        ]

    # prefix（可选）：如果你还想做 prefix 拼接 + trim_prefix_text
    prefix_text ="<SPK>1</SPK>敏感肌想美白祛斑？用错产品泛红干痒，烂脸风险太高！别再踩坑了！ <SPK>2</SPK>我是皮肤科李医生，专注敏感肌研究十年，今天给大家推荐这款——相宜本草红景天氧白霜，敏感肌专研，美白祛斑双认证！ <SPK>1</SPK>喝水留唇印、吃饭掉色？通勤 8 小时补 3 次妆！普通口红真的太坑了！ <SPK>2</SPK>但我最近挖到宝了！2025 新款鹃汐子锁色雨衣口红。"
    # prefix_text = ''
    # =============================
    # 4) 自动交叉匹配：spk_combos × text_cases
    # =============================
    groups = []
    for si, spk_item in enumerate(spk_combos):
        if isinstance(spk_item, (list, tuple)):
            ref_audios = spk_item
            ref_texts = None
            spk_tag = f"spk{si}"
        else:
            ref_audios = spk_item["ref_audios"]
            ref_texts = spk_item.get("ref_texts", None)
            spk_tag = spk_item.get("tag", f"spk{si}")

        # for ti, tc in enumerate(text_cases):
        tc = text_cases[si]
        base_text = tc["text"]
        # text_id = tc.get("id", f"text{ti}")
        text_id = tc.get("id", f"text{si}")

        gen_text = prefix_text + merge_consecutive_same_spk(base_text)

        groups.append({
            "ref_audios": ref_audios,
            "ref_texts": ref_texts,
            "text_for_name": f"<SPK>1</SPK>{spk_tag}__{text_id} " + base_text,
            "text": gen_text, 
        })


    # =============================
    # 5) 模型参数
    # =============================
    model_kwargs = dict(
        dit_exp_name=dit_exp_name,
        dur_exp_name=dur_exp_name,
        frontend_exp_name=frontend_exp_name,
        wavvae_exp_name='checkpoints/1231_megatts3_wavvae_v3_25hz_kl001_fix4',
        use_old_aligner=False,
        use_old_dur=True,
        max_ref_duration=max_ref_duration,
        use_tqdm=False,
        use_ema=False,
        precision='bf16',
        compile_models=True,
        use_shm_ckpt=False,
        chunk_num_words_zh=100,
        chunk_num_words_en=130,
    )

    forward_kwargs = dict(
        timestep_annealing_w=(0.6, 0.4, 1.0),
        use_sa_frontend=True,
        return_format='wav',
        use_amo_sampler=True,
        speech_rate=1.1,
        custom_ph_table={
            'en': {'@': 'at', '&': 'and'},
            'zh': {'@': '艾特', '&': '和'}
        },
        dur_disturb=0.2,
        num_parallel_workers=5,
        normalize_dur=True,
        prefer_onepass_dur=False,
        trim_prefix_text=prefix_text,  # 配合 prefix_text 做裁剪
    )

    if model_kwargs['use_old_dur']:
        forward_kwargs['dur_disturb'] = 0.1
        model_kwargs['max_ref_duration'] = 10

    # out_dir = f'infer_out/multi/{Path(dit_exp_name).stem}_final/fix_-23_vadsil_cfg{seq_cfg_w}_refdur{model_kwargs["max_ref_duration"]}_amo{forward_kwargs["use_amo_sampler"]}_chunk{model_kwargs["chunk_num_words_zh"]}_rate{forward_kwargs["speech_rate"]}_timeanneal{forward_kwargs["timestep_annealing_w"]}_timestep{time_step}'
    # 新分镜
    out_dir = f'infer_out/multi/{Path(dit_exp_name).stem}_final/fenjin12'
    os.makedirs(out_dir, exist_ok=True)

    # =============================
    # 6) 多 GPU 并行 + 总进度条
    # =============================
    ngpu = torch.cuda.device_count()
    if ngpu <= 0:
        raise RuntimeError("没检测到 GPU；这套 pipeline 不建议 CPU 跑！")

    total = len(groups)
    nproc = min(ngpu, total)

    jobs = list(enumerate(groups))
    shards = [jobs[i::nproc] for i in range(nproc)]  # round-robin

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []

    QUIET = False  # 总开关：静默所有 worker 打印，只留主进度条

    for rank in range(nproc):
        gpu_id = rank
        p = ctx.Process(
            target=_worker,
            args=(
                rank,
                gpu_id,
                shards[rank],
                queue,
                model_kwargs,
                out_dir,
                time_step,
                seq_cfg_w,
                forward_kwargs,
                QUIET,
            )
        )
        p.start()
        procs.append(p)

    from tqdm import tqdm
    results = [None] * total
    succ = 0
    fail = 0

    pbar = tqdm(total=total, desc="Generating", dynamic_ncols=True)
    for _ in range(total):
        idx, path, err = queue.get()
        results[idx] = (idx, path, err)
        if path is not None:
            succ += 1
            pbar.set_postfix_str(Path(path).name)
        else:
            fail += 1
            pbar.set_postfix_str("ERR")
        pbar.update(1)
    pbar.close()

    for p in procs:
        p.join()

    print(f'| ALL DONE: success={succ} fail={fail} out_dir={out_dir}')
    if fail > 0:
        print('| failed cases:')
        for (idx, path, err) in results:
            if err is not None:
                print(f'  - idx={idx+1}: {err}')