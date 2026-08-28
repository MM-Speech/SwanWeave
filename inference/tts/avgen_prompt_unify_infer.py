import os
import random
import re
import tempfile
from datetime import datetime, timedelta
import collections
import collections.abc
from glob import glob

import numpy as np

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
from attrdictionary import AttrDict
from typing import List, Optional, Dict

import yaml
from argparse import ArgumentParser
import torch.distributed as dist

import torch
import soundfile as sf
import librosa
from multiprocessing import Process, set_start_method

from utils.commons.os_utils import kill_void
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint, torch_load_dist
from utils.commons.hparams import set_hparams, hparams

from modules.asr.sensevoice.sensevoice_api import build_asr_model, run_asr_model
from modules.tts.scriptspeech.build_model_utils import DiTBuildModelMixin, SemanticLMBuildModelMixin, build_vae, \
    build_audio_tokenizer
from utils.commons.upload_tos_utils import send_file_to_tos

import json
from utils.text.text_encoder import TokenTextEncoder
from utils.text.ssml_utils import SSML
from modules.tts.frontend_lm.sa_frontend import call_sa_frontend

from utils.text.ph_tone_convert import split_ph
from utils.text.normalize import normalize_text, isChinese
from utils.text.split_text import get_word_list
from utils.text import YUNMU, SHENGMU, ALL_PHONE, PUNC, ENG_PHONE
from utils.commons.ckpt_utils import load_ckpt
from tasks.tts.task_utils.prompttts_task_utils import build_audio_mask_from_ids


cfg_weight = None
infer_step = 100
extend_dur = 0
vad_len = 0

def merge_model_weights(model, new_ckpt_path, ignore_module=[], weight=0.5):
    """
    Args:
        model: 已经加载了原始权重的 torch.nn.Module
        new_ckpt_path: 新的 checkpoint 文件路径 (state_dict)
        weight: 融合比例，merged = weight * old + (1 - weight) * new
    """
    # 加载新的 ckpt
    if os.path.isfile(new_ckpt_path):
        base_dir = os.path.dirname(new_ckpt_path)
        ckpt_path = new_ckpt_path
        new_state_dict = torch_load_dist(new_ckpt_path, map_location='cpu', mmap=None)
    else:
        base_dir = new_ckpt_path
        new_state_dict, ckpt_path = get_last_checkpoint(new_ckpt_path)
    # new_state_dict = torch.load(new_ckpt_path, map_location="cpu")
    print(f'merge model from {ckpt_path} with weight {1 - weight}')
    # 拿到旧模型参数
    old_state_dict = model.state_dict()

    merged_state_dict = {}
    new_state_dict = new_state_dict['dit']
    for k, old_param in old_state_dict.items():
        if k in new_state_dict and old_param.shape == new_state_dict[k].shape and not any(ign in k for ign in ignore_module):
            new_param = new_state_dict[k]
            merged_state_dict[k] = weight * old_param + (1 - weight) * new_param
        else:
            # 如果没有对应权重，就保留旧的
            merged_state_dict[k] = old_param

    # 加载融合后的权重
    model.load_state_dict(merged_state_dict)

    return model

def gen_audio_html(infos: Dict[int, dict], output_fp: Optional[str] = None,
                   title_name=None, extra_desc=None):
    if output_fp is None:
        output_fp = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name

    num_per_row = 5
    total = len(infos)
    rows = (total + num_per_row - 1) // num_per_row

    with open(output_fp, 'w') as f:
        print('<html lang="en">', file=f)
        print('<head>', file=f)
        print('<meta charset="UTF-8">', file=f)
        print('<meta name="viewport" content="width=device-width, initial-scale=1.0">', file=f)
        print('<style>', file=f)
        print('''
            body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
            .container { max-width: 1280px; margin: 0 auto; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            h1 {
                  font-size: 2em;
                  margin-bottom: 0.2em;
                }
            p.description {
                  color: #666;
                  margin-bottom: 1.5em;
            }
            td { padding: 10px; border: 2px solid DodgerBlue; vertical-align: top; text-align: center; }
            audio { width: 100%; }
            .desc { margin-top: 10px; white-space: pre-wrap; text-align: left; font-size: 14px; background: #f8f8f8; padding: 8px; border-radius: 5px; }
        ''', file=f)
        print('</style>', file=f)
        print('</head>', file=f)
        print('<body>', file=f)
        if title_name is not None:
            print(f'  <h1>{title_name}</h1>', file=f)
        if extra_desc is not None:
            print(f'  <p class="description">{extra_desc}</p>', file=f)
        print('<div class="container">', file=f)
        print('<table>', file=f)

        keys = list(infos.keys())

        for row in range(rows):
            print('<tr>', file=f)
            for col in range(num_per_row):
                idx = row * num_per_row + col
                if idx >= total:
                    print('<td></td>', file=f)
                    continue
                info = infos[idx]
                tos_url = info.get('tos_url', '')
                if 'global_description' in info:
                    global_description = info.get('global_description', '')
                    fine_grained_transcription = info.get('fine_grained_transcription', '')
                    env_bgm = info.get('env_bgm', '')
                    speaker = info.get('speaker', '')
                elif 'subjects' in info:
                    subjects = info['subjects']
                    narration = info['narration']
                    visual = info['visual']
                elif 'caption' in info:
                    caption = info['caption']
                print('<td>', file=f)
                print(f'<audio controls preload="none">', file=f)
                print(f'  <source src="{tos_url}" type="audio/mpeg">', file=f)
                print('  Your browser does not support the audio element.', file=f)
                print('</audio>', file=f)
                print('<div class="desc">', file=f)
                if 'global_description' in info:
                    print(f'Global Prompt: {global_description}\n', file=f)
                    print(f'BGM Prompt: {env_bgm}\n', file=f)
                    print(f'Local Prompt: {fine_grained_transcription.replace("<", "&lt;").replace(">", "&gt;")}\n', file=f)
                    print(f'Speaker: {speaker}', file=f)
                elif 'subjects' in info:
                    print(f'subjects: {subjects}\n', file=f)
                    print(f'narration: {narration.replace("<", "&lt;").replace(">", "&gt;")}\n', file=f)
                    print(f'visual: {visual}\n', file=f)
                else:
                    print(f'caption: {caption.replace("<", "&lt;").replace(">", "&gt;")}\n', file=f)
                print('</div>', file=f)
                print('</td>', file=f)
            print('</tr>', file=f)

        print('</table>', file=f)
        print('</div>', file=f)
        print('</body>', file=f)
        print('</html>', file=f)

    return output_fp

def upload_tos_html(yml=None, out_path=None, title_name=None, extra_desc=None):
    sub_dir = 'scriptspeech_gqj'
    if yml is None:
        yml = './egs/inference/inference_na_sample0728.yaml'
    if out_path is None:
        out_path = 'infer_out/tts/250625_scriptspeech_semanticlm_singlespk_01_250828_scriptspeech_dit_s2_gokul_c_vm_1_20250901_081432'

    from collections import defaultdict
    infos = defaultdict(dict)

    with open(yml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        samples = cfg['samples']
    files = glob(f'{out_path}/*.wav')
    # assert len(files) == len(samples)
    for idx, sample in enumerate(samples):
            wav = os.path.join(out_path, f"out_{idx}.wav")
            if not os.path.exists(wav):
                continue
            tos_url = send_file_to_tos(wav, sub_dir=sub_dir)
            # tos_url = f"https://tosv-sg.tiktok-row.org/obj/sa-ag-sg-research-sg/scriptspeech_gqj/tts_250625_scriptspeech_semanticlm_singlespk_01_250723_scriptspeech_dit_s2_c_1_20250724_070952_out_{idx}.wav"
            print("tos_url: ", tos_url)
            infos[idx]['tos_url'] = tos_url
            infos[idx].update(sample)

    # 调用gen_html函数时传递names参数
    html_path = gen_audio_html(infos, title_name=title_name, extra_desc=extra_desc)
    print(f"生成的HTML文件路径：{html_path}")
    html_tos = send_file_to_tos(html_path, sub_dir=sub_dir)
    print(f"生成的HTML文件路径：{html_tos}")
    return html_tos

def build_g2p_model(device='cuda', model='qwen'):
    ling_dict = json.load(open(f"egs/tts/megatts3_dict.json"))
    ling_keys = ['phone', 'tone']
    ling_dict = {k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov='<UNK>') for k in ling_keys}

    if model == 'llama':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("checkpoints/llama_tokenizer", padding_side="right")
        tokenizer.add_tokens(['[ASR_BOS]'], special_tokens=True)
        tokenizer.add_tokens(['[ASR_EOS]'], special_tokens=True)
        tokenizer.add_tokens(['[FULL]'], special_tokens=True)
        tokenizer.add_tokens(['[PARTIAL]'], special_tokens=True)

        from modules.tts.frontend_lm.frontend_lm_g2p import Frontend_G2P_Interface, LLAMA_Config
        config = LLAMA_Config()
        config.encoder_dim = 512
        config.encoder_n_layers = 8
        config.bpe_pad = tokenizer.pad_token_id = tokenizer.eos_token_id # pad_token = eos_token
        config.use_qk_norm = True
        g2p_lm = Frontend_G2P_Interface(config)
        load_ckpt(g2p_lm, f'checkpoints/g2p', 'model')
        g2p_lm.eval()
        g2p_lm.to(device)
        g2p_lm.to(torch.float16)

    elif model == 'qwen':
        from transformers import logging
        logging.set_verbosity_error()
        from transformers import AutoTokenizer, AutoModelForCausalLM
        g2p_exp_name = 'checkpoints/g2p_qwen'
        tokenizer = AutoTokenizer.from_pretrained(g2p_exp_name, padding_side="right")
        tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        g2p_lm = AutoModelForCausalLM.from_pretrained(g2p_exp_name).eval().to(device)

    return g2p_lm, tokenizer, ling_dict


def run_g2p_model(text_inp, g2p_lm, tokenizer, ling_dict, device='cuda'):
    from modules.tts.frontend_lm.frontend_lm_g2p import Frontend_G2P_Interface
    from transformers import Qwen2ForCausalLM

    # print(type(g2p_lm))
    # print(g2p_lm)
    if isinstance(g2p_lm, Frontend_G2P_Interface):
        text_inp = normalize_text(text_inp)
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
            text_inp = '[ASR_BOS]' + '[FULL]' + text_inp + '[ASR_EOS]'
            token_dict = tokenizer(text_inp, return_tensors="pt", padding=True).to(device)
            bpe_tokens = token_dict['input_ids']
            bpe_lengths = token_dict['attention_mask'].sum(dim=-1).long()
            phone_tokens = torch.LongTensor([798])[None, ...].to(device)
            phone_len = torch.LongTensor([1]).to(device)
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                tokens_pred = g2p_lm.g2p(bpe_tokens, bpe_lengths, phone_tokens, phone_len)
            ph_pred, tone_pred = split_ph(tokens_pred[0])

    
    elif isinstance(g2p_lm, Qwen2ForCausalLM):
    # else:
        speech_start_idx = tokenizer.encode('<Reserved_TTS_0>')[0]
        txt_token = tokenizer('<BOT>' + text_inp + '<BOS>')['input_ids']
        input_ids = torch.LongTensor([txt_token + [145 + speech_start_idx]]).to(device)
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
            outputs = g2p_lm.generate(input_ids, max_new_tokens=256, do_sample=True, top_k=1, eos_token_id=800 + 1 + speech_start_idx)
        ph_tokens = outputs[:, len(txt_token): -1] - speech_start_idx
        ph_pred, tone_pred = split_ph(ph_tokens[0])
        
    ph_tokens = ph_pred.view(-1).cpu().numpy()
    tone_tokens = tone_pred.view(-1).cpu().numpy()


    ph_pred = ling_dict['phone'].decode(ph_tokens).split(' ')
    # tone_pred = ling_dict['tone'].decode(tone_tokens).split(' ')      # don't decode this!
    tone_pred = tone_tokens.tolist()

    ret = dict(
        ph_tokens=ph_tokens,
        tone_tokens=tone_tokens,
        ph_pred=ph_pred,
        tone_pred=tone_pred,
    )
    
    return ret

def isEnglish(c):
    for c in c.split(' '):
        if not (c.isalnum() and not ('\u4e00' <= c <= '\u9fff') and not (c == 'sil')):
            return False
    return True

def align_word_phone(text, ph):
    if isinstance(text, str):
        text = get_word_list(text)
    if ph[0] == 'sil' and text[0] != 'sil':
        text = ['sil'] + text

    # 处理英文：将相邻的连续英文单词聚合变成单个单词，并将对应的英文音素都对应到该联合单词上
    if len(text) > 1:
        new_text = [text[0]]
        for i in range(1, len(text)):
            if isEnglish(text[i]):
                if isEnglish(new_text[-1]):
                    new_text[-1] = new_text[-1] + ' ' + text[i]
                else:
                    new_text.append(text[i])
            else:
                new_text.append(text[i])
        text = new_text
    
    assert len(ph) >= len(text)
    ph2word = []
    word_idx = 0

    for ph_idx, p in enumerate(ph):
        if word_idx >= len(text):
            break
        if p in ALL_PHONE:
            ph2word.append(word_idx)
            if p in YUNMU:
                word_idx += 1
        elif p == text[word_idx] or (p in PUNC and text[word_idx] in PUNC):
            ph2word.append(word_idx)
            word_idx += 1
        elif p in ENG_PHONE and isEnglish(text[word_idx]):
            ph2word.append(word_idx)
            if ph_idx + 1 < len(ph) and ph[ph_idx + 1] not in ENG_PHONE:
                word_idx += 1
        else:
            ph2word.append(word_idx)
    return text, ph, ph2word


def print_align(text, ph, ph2word):
    from pypinyin import lazy_pinyin
    text = [lazy_pinyin(t)[0] if t != 'sil' and not isEnglish(t) else t for t in text]

    text_print = [f"{text[0]:<8s}"]
    ph_print = [f"{ph[0]:<8s}"]
    word_idx = 1
    for i in range(1, len(ph)):
        if ph2word[i] != ph2word[i-1]:
            text_width = max(8, len(text[word_idx]) + 1)
            text_print.append(f"{text[word_idx]:<{text_width}s}")
            ph_print.append(f"{ph[i]:<{text_width}s}")
            word_idx += 1
        else:
            text_print.append(f"{' ':<8s}")
            ph_print.append(f"{ph[i]:<8s}")
    
    import math
    to_print = ''
    n_words = 12
    for i in range(math.ceil(len(text_print) / n_words)):
        to_print = to_print + '|'.join(text_print[i * n_words: (i + 1) * n_words]) + '\n'
        to_print = to_print + '|'.join(ph_print[i * n_words: (i + 1) * n_words]) + '\n'
        to_print = to_print + '-' * n_words * 8 + '\n'
    print(to_print)


def align_word_phone2(text, ph):
    # experimental
    text = get_word_list(text)
    if ph[0] == 'sil' and text[0] != 'sil':
        text = ['sil'] + text
    assert len(ph) >= len(text)
    text2ph = []
    ph_idx = 0
    for word in text:
        t2p = []
        if isChinese(word):
            while True:
                t2p.append(ph_idx)
                ph_idx += 1
                if ph[ph_idx] in YUNMU:
                    break
        elif (word in PUNC and ph[ph_idx] in PUNC) or word == ph[ph_idx]:
            t2p.append(ph_idx)
            ph_idx += 1
        else:
            print('something is wrong')
        text2ph.append(t2p)
    
    print('text2ph', text2ph)
    
class ScriptSpeechInfer(DiTBuildModelMixin, SemanticLMBuildModelMixin):
    def __init__(self, device, dit_ckpt,
                 vae_ckpt=None,
                 merge_ckpt=None, merge_weight=0.5,
                 use_sa_front: bool = False,
                 g2p_model: str = 'qwen'):
        self.device = device
        self.build_model(
            dit_ckpt, 
            vae_ckpt=vae_ckpt,
            merge_ckpt=merge_ckpt,
            merge_weight=merge_weight,
            use_sa_front=use_sa_front,
            g2p_model=g2p_model,
        )

    def build_model(self, dit_ckpt, 
                    vae_ckpt=None,
                    merge_ckpt=None, merge_weight=0.5,
                    use_sa_front: bool = False,
                    g2p_model: str = 'qwen'):
        self.asr_model = build_asr_model(self.device)  # 你如果需要自动 ASR prompt，可以把这行解开

        # ====== hparams & config ======
        set_hparams(config=os.path.join(dit_ckpt, 'config.yaml'),
                    print_hparams=False, global_hparams=True)
        hparams["exp_name"] = 'infer'
        self.config = AttrDict(hparams)

        # sa_front / g2p 相关开关
        self.use_sa_front = hparams.get('use_sa_front', use_sa_front)
        self.g2p_model = hparams.get('g2p_model', g2p_model)

        self.g2p_lm = None
        self.g2p_tokenizer = None
        self.g2p_ling_dict = None

        # ====== VAE & audio tokenizer ======
        vae_ckpt_path = vae_ckpt or hparams.get('vae_ckpt')
        self.vae, self.hp_vae = build_vae(vae_ckpt_path)
        self.vae.to(self.device)

        self.audio_token_feature_extractor, self.audio_tokenizer, self.audio_vocab_size = build_audio_tokenizer(
            hparams.get('audio_tokenizer', 'glm4v'))
        self.audio_tokenizer.to(self.device)

        # ====== DiT & 文本 tokenizer ======
        self.dit_text_tokenizer, self.dit_vocab_size = self.build_dit_text_tokenizer()
        self.dit = self.build_dit(hparams)
        load_ckpt(self.dit, dit_ckpt, 'dit', strict=False)
        if merge_ckpt is not None:
            self.dit = merge_model_weights(self.dit, merge_ckpt,
                                           ignore_module=['cross', 'caption_proj'],
                                           weight=merge_weight)
        self.dit.eval()
        self.dit.to(self.device)

        # ====== caption 相关 encoder ======
        self.use_caption = hparams.get('load_sd_text_encoder', False)
        if self.use_caption:
            if hparams.get('model_size', 'base') == 'seedance_7b':
                self.build_sd_text_encoder(hparams['text'])
                self.sd_text_encoder.eval()
                self.sd_text_encoder.to(self.device)
            elif 'goku' in hparams.get('model_size', 'base'):
                self.build_goku_text_encoder(hparams)
                self.goku_text_encoder.eval()
                self.goku_text_encoder.to(self.device)
        else:
            self.sd_text_encoder = None
            self.goku_text_encoder = None

        # ====== phone / tone 词表 & mask id，与训练 task 对齐 ======
        ling_dict_path = hparams.get('ling_dict', 'egs/tts/megatts3_dict.json')
        with open(ling_dict_path, 'r', encoding='utf-8') as f:
            ling_dict_raw = json.load(f)

        self.ling_dict = {
            'phone': TokenTextEncoder(None, vocab_list=ling_dict_raw['phone'], replace_oov=''),
            'tone': TokenTextEncoder(None, vocab_list=ling_dict_raw['tone'], replace_oov='')
        }
        # 默认使用词表最后一个 token 作为 mask（和训练 task 中的 302-1 / 32-1 一致）
        self.cfg_mask_token_phone = hparams.get(
            'cfg_mask_token_phone',
            len(ling_dict_raw['phone']) - 1
        )
        self.cfg_mask_token_tone = hparams.get(
            'cfg_mask_token_tone',
            len(ling_dict_raw['tone']) - 1
        )

    def _encode_phone_tone(self, text: str):
        """
        根据开关选择 SA Front 或 g2p。
        返回 [1, T] 的 LongTensor。
        """
        if getattr(self, "use_sa_front", False):
            return self._encode_phone_tone_from_sa(text)
        else:
            return self._encode_phone_tone_from_g2p(text)

    def _encode_phone_tone_from_g2p(self, text: str):
        from modules.tts.frontend_lm.frontend_lm_g2p import Frontend_G2P_Interface
        from transformers import Qwen2ForCausalLM

        if self.g2p_lm is None:
            self.g2p_lm, self.g2p_tokenizer, self.g2p_ling_dict = build_g2p_model(
                device=self.device,
                model=getattr(self, "g2p_model", "qwen"),
            )

        g2p_result = run_g2p_model(
            text_inp=text,
            g2p_lm=self.g2p_lm,
            tokenizer=self.g2p_tokenizer,
            ling_dict=self.g2p_ling_dict,
            device=self.device,
        )
        ph_tokens = g2p_result["ph_tokens"]
        tone_tokens = g2p_result["tone_tokens"]

        if ph_tokens.size == 0 or tone_tokens.size == 0:
            return None, None

        ph_seq = torch.from_numpy(ph_tokens).long().unsqueeze(0).to(self.device)
        tone_seq = torch.from_numpy(tone_tokens).long().unsqueeze(0).to(self.device)

        # 与训练一致：非中文声调归一为 3
        en_tone_idx = ~(
            (tone_seq == 4) |
            ((tone_seq >= 11) & (tone_seq <= 15)) |
            (tone_seq == 0)
        )
        tone_seq[en_tone_idx] = 3
        return ph_seq, tone_seq

    def _encode_phone_tone_from_sa(self, text: str):
        """
        使用 SA Front 提取 phone / tone，与训练前端一致。
        """
        if self.ling_dict is None:
            return None, None

        ssml = SSML(text or "")
        text_sa, ph_tokens, tone_tokens, alignment_sa = call_sa_frontend(
            ssml.sa_ssml_str, debug=0
        )

        ph_ids = self.ling_dict['phone'].encode(' '.join(ph_tokens))
        tone_ids = self.ling_dict['tone'].encode(' '.join(tone_tokens))
        ph_seq = torch.LongTensor(ph_ids)[None].to(self.device)
        tone_seq = torch.LongTensor(tone_ids)[None].to(self.device)

        en_tone_idx = ~(
            (tone_seq == 4) |
            ((tone_seq >= 11) & (tone_seq <= 15)) |
            (tone_seq == 0)
        )
        tone_seq[en_tone_idx] = 3
        return ph_seq, tone_seq

    def run_goku_text_encoder(self, captions: list):
        """
        和 ScriptSpeechDiTTask 中保持一致：
        - padding=True, truncation=True
        - 可选生成 caption_text_mark (0/1/2)
        """
        inputs = self.goku_tokenizer(
            captions,
            max_length=hparams['text_max_token_length'],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        device = self.device

        input_ids = inputs.input_ids.to(self.device)
        attention_masks = inputs.attention_mask.to(self.device)

        encoder_hidden_states = self.goku_text_encoder(
            input_ids,
            return_dict=False,
            attention_mask=attention_masks,
        )[0]  # [B, T, C]

        if hparams.get('use_caption_text_mark', False):
            caption_text_mark = build_audio_mask_from_ids(
                input_ids=input_ids,
                attention_mask=attention_masks,
                tokenizer=self.goku_tokenizer,
            )  # [B, T]
        else:
            caption_text_mark = None

        return encoder_hidden_states, caption_text_mark, attention_masks

    @torch.no_grad()
    def forward(self, text, ref_audio=None, ref_text=None, prompt=None, cfg_w=None,
                negative_prompt=None, infer_length=5.0,
                start_time=0.2, end_time=0.2, num_step=100):
        """
        infer_length 语义（这里按你想要的方式改了）：
            - 无 ref_audio：infer_length = 纯生成时长（秒）
            - 有 ref_audio：infer_length = 目标“新生成”时长（秒）
              内部会自动做 tgt_len = ref_lat_len + gen_lat_len

        推理入口：整体输入组织方式与 ScriptSpeechDiTTask.run_model + Diffusion.inference 对齐。
        """
        speech = len(text) > 0
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']

        # ====== 参考音频 & 参考文本 ======
        if ref_audio is not None:
            ref_wav = torch.from_numpy(
                np.concatenate([ref_audio, np.zeros(0, dtype=np.float16)])
            )[None, :].to(self.device)

            ref_wav_lens = torch.LongTensor(
                [ref_wav.shape[1] // fm_wav * fm_wav]
            ).to(self.device)
            ref_wav = ref_wav[:, :ref_wav_lens[0]]

            # 自动 ASR 得到 ref_text（你已在 build_model 里开启 asr_model）
            if ref_text is None and hasattr(self, "asr_model") and self.asr_model is not None:
                # 1) 按参考实现：直接传 16k numpy
                wav_16k = librosa.resample(
                    ref_audio.astype(np.float32),
                    orig_sr=24000,
                    target_sr=16000
                )

                # 2) 新接口常见写法
                asr_out = run_asr_model([wav_16k], self.asr_model, with_segments=False)

                # 3) 兼容返回 list/dict
                if isinstance(asr_out, (list, tuple)) and len(asr_out) > 0:
                    asr_item = asr_out[0] or {}
                else:
                    asr_item = asr_out or {}

                print('asr_result', asr_item)

                ref_text = asr_item.get('text_normed') or asr_item.get('text') or ''
                if ref_text and not ref_text.endswith(('.', '。', '!', '！', '?', '？')):
                    ref_text = ref_text + '.'


            # 默认 cfg
            if cfg_w is None:
                cfg_w = [2, 2, 2, 5]
        else:
            ref_text = None
            ref_wav = None
            if cfg_w is None:
                # 纯文本 + caption 的 CFG 配置
                cfg_w = [5, 10, 5, 0]

        print('| CFG: ', cfg_w)

        # ====== 先算“要新生成的 latent 长度” ======
        # infer_length 这里解释为 “新生成时长（秒）”
        gen_lat_len = int(
            infer_length * 24000 / self.hp_vae['hop_size'] / self.hp_vae['vae_stride']
        )
        gen_lat_len = max(gen_lat_len, 0)

        # ====== caption 部分：和训练保持一致的 embedding & mask 组织 ======
        caption = prompt
        print('| caption:', caption)

        use_caption = getattr(self, "use_caption", False) and (caption is not None)
        caption_emb = None
        caption_lens = None
        caption_text_mark = None

        if use_caption and 'goku' in self.config.model_size:
            # 正向 caption
            gk_text_embs, captions_cmask, text_att_mask = self.run_goku_text_encoder([caption])
            text_embs = gk_text_embs * text_att_mask[..., None]
            caption_lens = text_att_mask.sum(-1)

            # negative prompt
            if negative_prompt is None:
                negative_prompt = '<I>distorted audio</I><I>background static</I>...'
            uc_gk_text_embs, uc_captions_cmask, uc_text_att_mask = self.run_goku_text_encoder(
                [negative_prompt]
            )
            uc_text_embs = uc_gk_text_embs * uc_text_att_mask[..., None]
            uc_caption_lens = uc_text_att_mask.sum(-1)

            pos_caption_emb = text_embs
            neg_caption_emb = uc_text_embs

            # 5 路 CFG（保持你原来的组织）
            caption_emb = torch.cat([
                pos_caption_emb,
                neg_caption_emb,
                pos_caption_emb,
                neg_caption_emb,
                neg_caption_emb,
            ], dim=0)
            caption_lens = torch.cat([
                caption_lens,
                uc_caption_lens,
                caption_lens,
                uc_caption_lens,
                uc_caption_lens,
            ], dim=0).long()

            if captions_cmask is not None:
                caption_text_mark = torch.cat([
                    captions_cmask,
                    uc_captions_cmask,
                    captions_cmask,
                    uc_captions_cmask,
                    uc_captions_cmask,
                ], dim=0)
        else:
            caption_emb = None
            caption_lens = None
            caption_text_mark = None

        # ====== 推断 latent_dim（避免写死 32） ======
        latent_dim = None
        if hasattr(self.vae, "latent_dim"):
            latent_dim = getattr(self.vae, "latent_dim")
        if latent_dim is None and hasattr(self.dit, "hp") and hasattr(self.dit.hp, "in_channels"):
            latent_dim = self.dit.hp.in_channels
        if latent_dim is None:
            latent_dim = 32  # 兜底

        # ====== VAE 编码参考音频，构造 lat_ctx / ctx_mask ======
        if ref_wav is not None:
            with torch.inference_mode():
                lat_ctx_ref = self.vae.encode_latent(ref_wav)  # [1, L_ref, C]
                # 用真实 ref latent dim 覆盖兜底
                latent_dim = lat_ctx_ref.size(-1)

                ref_lat_len = lat_ctx_ref.size(1)
                tgt_len = ref_lat_len + gen_lat_len  # <<<<<< 关键改动：总长度 = ref + 新生成

                ctx_mask_ref = torch.ones_like(lat_ctx_ref[:, :, 0:1])

                # pad 到目标长度
                lat = torch.nn.functional.pad(
                    lat_ctx_ref, (0, 0, 0, tgt_len - ref_lat_len),
                    mode='constant', value=0
                )
                ctx_mask = torch.nn.functional.pad(
                    ctx_mask_ref, (0, 0, 0, tgt_len - ref_lat_len),
                    mode='constant', value=0
                )

                # 文本 token：有 ref_text 就拼上，没有就只用目标 text
                if ref_text is not None:
                    text_inputs = self.dit_text_tokenizer(
                        ref_text + text, padding=True, return_tensors='pt'
                    ).to(self.device)
                else:
                    text_inputs = self.dit_text_tokenizer(
                        text, padding=True, return_tensors='pt'
                    ).to(self.device)
        else:
            tgt_len = gen_lat_len  # 纯生成
            lat_ctx_ref = torch.zeros(1, 0, latent_dim).to(self.device)
            lat = torch.zeros(1, tgt_len, latent_dim).to(self.device)
            ctx_mask = torch.zeros(1, tgt_len, 1).to(self.device)
            text_inputs = self.dit_text_tokenizer(
                text, padding=True, return_tensors='pt'
            ).to(self.device)

        # ====== 文本 token / mask / 长度 ======
        txt_tokens = text_inputs['input_ids'].clone()
        txt_mask = text_inputs['attention_mask'].bool()
        txt_tokens[~txt_mask] = self.cfg_mask_text_token
        txt_lens = txt_mask.long().sum(-1)  # [1]

        # ====== VAD mask（纯上游信息，Diffusion 目前不直接用，但保持接口一致）======
        vad_mask = torch.zeros_like(lat[:, :, :1])
        if not self.config.get('drop_vad', False) and speech:
            vad_mask[:, int(start_time * 25):-int(end_time * 25)] = 1.0

        # 5 路 CFG：在 batch 维度上复制
        vad_mask = torch.cat([vad_mask] * 5, dim=0)

        # 文本 CFG：第 3,4,5 路全部打 mask
        txt_tokens = torch.cat([
            txt_tokens,
            txt_tokens,
            torch.full_like(txt_tokens, self.cfg_mask_text_token),
            torch.full_like(txt_tokens, self.cfg_mask_text_token),
            torch.full_like(txt_tokens, self.cfg_mask_text_token),
        ], dim=0)
        txt_mask = torch.cat([txt_mask] * 5, dim=0)
        txt_lens = torch.cat([txt_lens] * 5, dim=0)

        # latent / ctx_mask 也复制 5 路
        lat = torch.cat([
            lat,
            torch.zeros_like(lat),
            torch.zeros_like(lat),
            lat,
            torch.zeros_like(lat),
        ], dim=0)
        ctx_mask = torch.cat([ctx_mask] * 5, dim=0)

        # ====== phone / tone：使用 SA/g2p 前端 + 5 路 CFG ======
        front_text = (ref_text + text) if ref_text is not None else text
        ph_seq, tone_seq = self._encode_phone_tone(front_text)

        if ph_seq is not None and tone_seq is not None \
                and self.cfg_mask_token_phone is not None \
                and self.cfg_mask_token_tone is not None:
            ph_seq = torch.cat([
                ph_seq,
                ph_seq,
                torch.full_like(ph_seq, self.cfg_mask_token_phone),
                torch.full_like(ph_seq, self.cfg_mask_token_phone),
                torch.full_like(ph_seq, self.cfg_mask_token_phone),
            ], dim=0)

            tone_seq = torch.cat([
                tone_seq,
                tone_seq,
                torch.full_like(tone_seq, self.cfg_mask_token_tone),
                torch.full_like(tone_seq, self.cfg_mask_token_tone),
                torch.full_like(tone_seq, self.cfg_mask_token_tone),
            ], dim=0)
        else:
            ph_seq = None
            tone_seq = None

        batch_size = lat.shape[0]

        # ====== 组装 Diffusion.inference 所需 inputs ======
        inputs = {
            'txt_tokens': txt_tokens if not hparams.get('drop_xt', False) else None,
            'txt_mask': txt_mask,
            'txt_lens': txt_lens,
            'ctx_mask': ctx_mask,
            'lat_ctx': lat,
            'caption_emb': caption_emb,
            'caption_lens': caption_lens,
            'vad_mask': vad_mask,
            'tgt_len': torch.full(
                (batch_size,),
                tgt_len,
                dtype=torch.long,
                device=self.device,
            ),
        }

        if caption_text_mark is not None:
            inputs['caption_text_mark'] = caption_text_mark

        if ph_seq is not None and tone_seq is not None:
            inputs['phone'] = ph_seq
            inputs['tone'] = tone_seq

        global cfg_weight, infer_step, extend_dur, vad_len
        cfg_weight, infer_step, extend_dur, vad_len = cfg_w, num_step, infer_length, [start_time, end_time]

        # ====== 调 Diffusion.inference 生成 latent，再用 VAE decode 回波形 ======
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x = self.dit.inference(inputs, timesteps=num_step, seq_cfg_w=cfg_w)

            # 把前面 ref 的部分替换为真实 prompt latent
            if lat_ctx_ref.shape[1] > 0:
                x[:, :lat_ctx_ref.shape[1]] = lat_ctx_ref

            wav_pred = self.vae.decode(x)[0, 0].to(torch.float32)

            hop_size = self.hp_vae['hop_size']
            vae_stride = self.hp_vae['vae_stride']

            # 裁掉前面的 prompt，对应 ref 部分
            if lat_ctx_ref.shape[1] > 0:
                wav_pred = wav_pred[lat_ctx_ref.size(1) * vae_stride * hop_size:]

            # clip 防止溢出
            if wav_pred.abs().max() > 1:
                print('Wav amplitude exceed 1, clip it.')
                wav_pred = wav_pred / (wav_pred.abs().max())

            wav_pred = wav_pred.cpu().numpy()

        return wav_pred



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_w_text(s: str) -> str:
    # 提取所有 <S1>...</S1> 中间的内容
    s = s.replace('<S1>', '<S1>').replace('</S1>', '</S1>')
    matches = re.findall(r"<S1>(.*?)</S1>", s, flags=re.DOTALL)
    # 用 ". " 拼接
    return ". ".join(m.strip() for m in matches if m.strip())

def worker(rank, world_size, args, cfg, out_path):
    device = f'cuda:{rank}'
    torch.cuda.set_device(device)
    if world_size > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'  # 本机通信地址
        os.environ['MASTER_PORT'] = '10198'      # 通信端口
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(rank)
        from utils.commons import trainer
        trainer.LOCAL_RANK = rank
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, device_id=torch.device(rank), timeout=timedelta(seconds=3000))


    dit_ckpt = args.dit_ckpt
    infer_ins = ScriptSpeechInfer(
        device,
        dit_ckpt=dit_ckpt,
        vae_ckpt=args.vae_ckpt,   # <- 新增
        merge_ckpt=args.merge_ckpt,
        merge_weight=args.merge_weight,
        use_sa_front=cfg.get('use_sa_front', False),
    )
    # 每个 rank 单独处理属于自己的样本
    os.makedirs(out_path, exist_ok=True)
    negative_prompt = cfg.get('negative_prompt', None)
    for idx, sample in enumerate(cfg['samples']):
        if idx % world_size != rank:
            continue
        print(f"[Rank {rank}] Processing sample {idx}: {sample}")
        if 'env_bgm' in sample:
            env_bgm = sample['env_bgm']
            fine_grained_transcription = sample['fine_grained_transcription']
            global_description = sample['global_description']
            speaker = sample['speaker']
            speech = sample['speech']
            text = extract_w_text(fine_grained_transcription)
            fine_grained_transcription = fine_grained_transcription.replace('<tag>', '<TAG>').replace('</tag>', '</TAG>').replace('<S1>', '<S1>').replace('</S1>', '</S1>')
            caption = f'{global_description} The voice is very clean and clear. {env_bgm} {speaker if speaker != "" else "There is no one speaking in the scene."} {fine_grained_transcription}'
        elif 'subjects' in sample:
            subjects = sample['subjects']
            narration = sample['narration']
            visual = sample['visual']
            text = extract_w_text(narration)
            # if len(subjects) == 0:
            #     subjects_str = 'No subjects. '
            # else:
            #     subjects_str = ''
            #     for sub in subjects:
            #         subjects_str = subjects_str + sub + ' '
            caption = f'Subjects: {subjects} Visual: {visual} Narration: {narration}'.replace('<tag>', '<TAG>').replace('</tag>',
                                                                                                       '</TAG>').replace('<S1>', '<S1>').replace('</S1>', '</S1>')
        else:
            caption = sample['caption']
            text = extract_w_text(caption)
        set_seed(len(text) + idx)


        if 'prompt_audio' in sample:
            audio = sample['prompt_audio']
            audio, _ = librosa.load(audio, sr=24000)
        else:
            audio = None
        wav = infer_ins.forward(text, ref_audio=audio, prompt=caption,
                                cfg_w=cfg.get('cfg_w', None),
                                infer_length=cfg.get('infer_length', 0),
                                num_step=cfg.get('num_step', 100),
                                start_time=cfg.get('vad_len', 0.2), end_time=cfg.get('vad_len', 0.2),
                                negative_prompt=negative_prompt)
        if wav is not None:
            print(f'save wav at {out_path}/out_{idx}.wav')
            sf.write(f'{out_path}/out_{idx}.wav', wav, 24000, 'PCM_16')

    print(f"[Rank {rank}] Finished all samples.")

if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv

        load_dotenv('.env.local')

    kill_void()

    try:
        set_start_method('spawn')  # 多进程启动方式，Linux/Windows 通用
    except RuntimeError:
        pass

    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument("--dit_ckpt", help="Path to model", type=str,
                        default='checkpoints/250622_scriptspeech_dit_singlespk_01')
    parser.add_argument("--merge_ckpt", help="Path to merge model", type=str)
    parser.add_argument("--merge_weight", help="Weight to merge model", type=float)
    parser.add_argument("--vae_ckpt", help="Path to VAE ckpt", type=str)
    args = parser.parse_args()
    # 读取 config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_path = f'{cfg["out_path"]}/{os.path.basename(args.dit_ckpt)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # 启动多进程，每个进程绑定一张 GPU
    processes = []
    gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    for rank in range(gpus):
        p = Process(target=worker, args=(rank, gpus, args, cfg, out_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All ranks finished. 可以做后处理或上传结果")
    # upload_tos_html(out_path)  # 可选，上传结果
    desc = (f"Inference setting: cfg weight: {cfg.get('cfg_w', None)}, inference step: {infer_step}, "
            f"extend duration: {extend_dur}, vad length (silence duration at bugin and tail): {vad_len}")
    upload_tos_html(yml=args.config, out_path=out_path, title_name=os.path.basename(out_path), extra_desc=desc)
    # CUDA_VISIBLE_DEVICES=0 python inference/tts/scriptspeech_infer.py

