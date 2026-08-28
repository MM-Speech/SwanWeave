import os
import argparse
import yaml
import glob
from queue import Empty
import time
import tempfile
import shutil
import math
import re

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import torchaudio

from utils.commons.multiprocess_utils import LocalDataPipeQueueService
from utils.commons.seq_utils import seq_match
from utils.commons.os_utils import kill_void
from utils.text import PUNC
from utils.metrics.wer import compute_wer_tokens

from modules.tts.frontend_lm.sa_frontend import call_sa_frontend

def batch_replace(text, old, new):
    for o in old:
        text = text.replace(o, new)
    return text

def run_reward_gemini3pro(model, inputs):
    wav = inputs['wav']
    sr = inputs['sr']
    text = inputs['text']
    
    with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
        temp_path = os.path.join(temp_dir, f"audio.wav")
        sf.write(temp_path, wav, sr, 'PCM_16')
        result = model.process(temp_path, text)
    return result

def run_reward_lai(model, inputs):
    wav = inputs['wav']
    sr = inputs['sr']
    text = inputs['text']
    
    def batch_check_in(text, items):
        if text is None:
            return False
        for item in items:
            if item in text:
                return True
        return False

    # 去掉 <S1> / </S1> 等标签，保留原始标点
    text = batch_replace(
        text,
        ['<S1>', '</S1>', '<S2>', '</S2>', '<S3>', '</S3>', '<S4>', '</S4>'],
        ''
    )
    
    # 调用 Lai 对齐
    with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
        temp_path = os.path.join(temp_dir, "audio.wav")
        sf.write(temp_path, wav, sr, 'PCM_16')
        alignment = model.process_alignment(temp_path, text)

    if alignment is None:
        return None

    # word_info: [(start, end, text, conf), ...]
    word_info = [
        (w['start'], w['end'], w['text'], w['conf'])
        for w in alignment['word_info']
    ]

    # 目标：文本中的「应该停顿」(基于标点)
    # 预测：音频中的「实际停顿」(基于静音段)
    target = []   # 0/1，文本中这个 token 后是否有标点对应的停顿
    pred = []     # 0/1，音频中这个 token 后是否有静音段

    pause_punct = "，,、。.;；:：?!？!…···"
    n = len(word_info)

    for word_idx, word in enumerate(word_info):
        start, end, token_text, conf = word

        # 静音段：Lai 里用空字符串表示（有时也可能是 None，这里一起兼容）
        if token_text is None or token_text == "":
            # 不考虑开头和结尾的静音（你说只关心中间的停顿）
            # if word_idx == 0 or word_idx == n - 1:
            if word_idx == 0:
                continue

            # 中间的静音：认为是前一个非静音 token 后的停顿
            if len(pred) > 0:
                pred[-1] = 1
            continue

        # 到这里是「字/词」token（非静音）
        # target: 看这个 token 里面有没有任何一个停顿标点
        if batch_check_in(token_text, pause_punct):
            target.append(1)
        else:
            target.append(0)

        # 先假设没有停顿，后面如果遇到静音再把它改成 1
        pred.append(0)

    # 如果没有有效 token，就直接返回 0 分
    if len(target) == 0 or len(pred) == 0:
        pause_reward = 0.0
        match_track = []
    else:
        pause_reward = float(np.mean(np.array(target) == np.array(pred)))

    # 打包结果：保留对齐信息 + 停顿 reward 和序列，方便调试
    result = {
        "text": alignment["text"],
        "sentence_conf": alignment["sentence_conf"],
        "word_info": word_info,
        "pause_target": target,
        "pause_pred": pred,
        "pause_reward": float(pause_reward),
        # "pause_match_track": match_track,
    }

    return result


def run_reward_qwen3aligner(model, inputs):
    from data_gen.qwen.qwen3aligner import (
        detect_language_for_qwen3aligner,
        sentence_conf_geomean,
        sentence_conf_min,
        sentence_conf_p10,
    )
    from utils.text import isPUNC
    from utils.text.split_text import get_word_list

    wav = inputs['wav']
    sr = inputs['sr']
    text = inputs['text']

    def batch_check_in(text, items):
        if text is None:
            return False
        for item in items:
            if item in text:
                return True
        return False

    def extract_words_and_punc_after(raw_text: str):
        """
        把原始 text 拆成：
        - words: 非标点 token 序列
        - punc_after: 每个 word 后面紧跟的标点串（可能为空）
        用于构造 target：该 word 后是否“应该停顿”（基于标点）
        """
        tokens = get_word_list(raw_text)
        words = []
        punc_after = []
        for tok in tokens:
            if isPUNC(tok):
                if len(punc_after) > 0:
                    punc_after[-1] += tok
                continue
            words.append(tok)
            punc_after.append("")
        return words, punc_after

    def normalize_for_match(token: str, lang: str) -> str:
        """
        用于把 aligner token 和 text token 做顺序匹配（尤其英文大小写/符号差异）。
        Qwen3AlignerInfer 的 language 是 'english'/'chinese' 这种。
        """
        token = (token or "").strip()
        if not token:
            return ""
        if lang == "english":
            token = token.lower()
            token = re.sub(r"[^a-z0-9]+", "", token)
        return token

    # 去掉 <S1> / </S1> 等标签，保留原始标点
    text = batch_replace(
        text,
        ['<S1>', '</S1>', '<S2>', '</S2>', '<S3>', '</S3>', '<S4>', '</S4>'],
        ''
    )

    language = detect_language_for_qwen3aligner(text)

    # 调用 Qwen3Aligner 对齐（注意：Qwen3AlignerInfer 只有 process，没有 process_alignment）
    with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
        temp_path = os.path.join(temp_dir, "audio.wav")
        sf.write(temp_path, wav, sr, 'PCM_16')
        align_results = model.process(temp_path, text, language, return_conf=True, return_json=False)

    if align_results is None or len(align_results) == 0:
        return None

    # 单条输入 -> align_results[0] 是 ForcedAlignResultWithConf，可迭代得到 item
    items = list(align_results[0])
    if len(items) == 0:
        return None

    # word_info: [(start, end, text, conf), ...] 统一成 Lai 风格，便于对比/调试
    word_info = [(it.start_time, it.end_time, it.text, float(getattr(it, "conf", 0.0))) for it in items]

    # target: 文本里“这个词后面是否有停顿标点”（基于标点）
    # pred: 音频里“这个词后面是否有实际停顿”（基于对齐 token gap）
    pause_punct = "，,、。.;；:：?!？!…···"
    pause_token = "<|sp|>"

    words, punc_after = extract_words_and_punc_after(text)

    # 让 align token 和 text token 尽量对齐（长度不一致时做顺序匹配，失败再截断）
    if len(words) == len(items):
        idx_map = list(range(len(items)))
    else:
        idx_map = []
        w_i = 0
        for it in items:
            it_n = normalize_for_match(it.text, language)
            while w_i < len(words) and normalize_for_match(words[w_i], language) != it_n:
                w_i += 1
            if w_i >= len(words):
                break
            idx_map.append(w_i)
            w_i += 1

        if len(idx_map) != len(items):
            n = min(len(words), len(items))
            idx_map = list(range(n))
            items = items[:n]

    target = []
    pred = []

    # gap 阈值：认为出现“真实停顿”的最小间隙（秒）
    # Lai 用显式静音段；Qwen3Aligner 没有静音 token，所以只能用 gap 判定
    min_pause_gap_s = 0.2

    for i, it in enumerate(items):
        w_idx = idx_map[i]

        # target：看这个 word 后面的标点串里有没有“停顿标点”
        existing = punc_after[w_idx]
        if batch_check_in(existing, pause_punct) or (pause_token in existing):
            target.append(1)
        else:
            target.append(0)

        # pred：看下一个 token 的 start 是否明显晚于当前 token 的 end
        if i + 1 < len(items):
            gap = float(getattr(items[i + 1], "start_time", 0.0)) - float(getattr(it, "end_time", 0.0))
            pred.append(1 if gap >= min_pause_gap_s else 0)
        else:
            # 句尾没有 next token，尾静音无法从对齐结果直接得到；先按 0 处理
            pred.append(0)

    if len(target) == 0 or len(pred) == 0:
        pause_reward = 0.0
    else:
        pause_reward = float(np.mean(np.array(target) == np.array(pred)))

    sentence_conf = {
        "geomean": float(sentence_conf_geomean(items)),
        "min": float(sentence_conf_min(items)),
        "p10": float(sentence_conf_p10(items)),
    }

    result = {
        "text": text,
        "language": language,
        "sentence_conf": sentence_conf,
        "word_info": word_info,
        "pause_target": target,
        "pause_pred": pred,
        "pause_reward": float(pause_reward),
    }

    return result


def run_reward_phone(model, inputs):
    wav = inputs['wav']
    sr = inputs['sr']
    text = inputs['text']
    text = batch_replace(text, ['<S1>', '</S1>', '<S2>', '</S2>', '<S3>', '</S3>', '<S4>', '</S4>'], '')

    alpha = 1.0

    def remove_punc(ph, tone):
        ph_, tone_ = [], []
        for ph_idx in range(len(ph)):
            if ph[ph_idx] not in PUNC and ph[ph_idx] != 'sil':
                ph_.append(ph[ph_idx])
                tone_.append(tone[ph_idx])
        return ph_, tone_
    
    ph_res, tone_res, dur_res = model.forward(wav, sr, timesteps=20)
    ph_res, tone_res = remove_punc(ph_res, tone_res)

    normed_text, phone, tone, alignment = call_sa_frontend(f"<speak>{text}</speak>")
    phone, tone = remove_punc(phone, tone)

    ph_gt = [f"{p}_{t}" for p, t in zip(phone, tone)]
    ph_pred = [f"{p}_{t}" for p, t in zip(ph_res, tone_res)]

    try:
        assert len(ph_gt) > 0
        wer_result = compute_wer_tokens(ph_gt, ph_pred)
        score = math.exp(-alpha * wer_result['wer'])
    except IndexError:
        wer_result = None
        score = 0.0

    result = {
        "score": score,
        "wer_result": wer_result,
        "phone_gt": phone,
        "tone_gt": tone,
        "phone_pred": ph_res,
        "tone_pred": tone_res
    }

    return result


def run_reward_sim(model_pack, inputs):
    wav = inputs['wav']
    sr = int(inputs['sr'])

    ref_wav = inputs.get('ref_wav', None)
    ref_sr = int(inputs.get('ref_sr', sr) or sr)
    ref_wav_path = inputs.get('ref_wav_path', None)

    if ref_wav is None:
        if ref_wav_path is None or not os.path.isfile(ref_wav_path):
            return {
                "score": 0.0,
                "error": "missing_ref_wav",
            }
        ref_wav, ref_sr = sf.read(ref_wav_path, dtype='float32')

    wav = np.asarray(wav, dtype=np.float32)
    ref_wav = np.asarray(ref_wav, dtype=np.float32)

    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if ref_wav.ndim > 1:
        ref_wav = ref_wav.mean(axis=-1)

    if wav.size == 0 or ref_wav.size == 0:
        return {
            "score": 0.0,
            "error": "empty_wav",
        }

    sim_model = model_pack['model']
    device = next(sim_model.parameters()).device

    wav_t = torch.from_numpy(wav).to(device=device, dtype=torch.float32).unsqueeze(0)
    ref_wav_t = torch.from_numpy(ref_wav).to(device=device, dtype=torch.float32).unsqueeze(0)

    if sr != 16000:
        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
    if ref_sr != 16000:
        ref_wav_t = torchaudio.functional.resample(ref_wav_t, ref_sr, 16000)

    with torch.no_grad():
        embeds_pred = sim_model(wav_t)
        embeds_ref = sim_model(ref_wav_t)
        score = F.cosine_similarity(embeds_pred, embeds_ref, dim=-1).mean()

    score = float(score.item()) if torch.isfinite(score) else 0.0
    return {
        "score": score,
    }


class RewardWorker:
    def __init__(self, config, device='cpu', worker_id=0, num_workers=1, shared_dict=None, shared_lock=None):
        self.config = config
        
        reward_model_pack = {}
        reward_fns = {}
        
        if config['reward'].get('gemini3pro'):
            from tasks.tts.posttrain.reward_models.gemini import GeminiRewardModel
            reward_model_pack['gemini3pro'] = GeminiRewardModel()
            reward_fns['gemini3pro'] = run_reward_gemini3pro
        
        if config['reward'].get('lai'):
            from data_gen.lai.lai import LaiAligner
            reward_model_pack['lai'] = LaiAligner(model_name=config.get('lai_model_path', 'LattifAI/Lattice-1'), device=device)
            reward_fns['lai'] = run_reward_lai

        if config['reward'].get('qwen3aligner'):
            from data_gen.qwen.qwen3aligner import Qwen3AlignerInfer
            reward_model_pack['qwen3aligner'] = Qwen3AlignerInfer(device=device)
            reward_fns['qwen3aligner'] = run_reward_qwen3aligner

        if config['reward'].get('phone'):
            from inference.asr.nar_mfa_infer import MFAInfer
            mfa_ckpt = 'checkpoints/251104_nar_mfa_v6_long_base_robust/model_ckpt_steps_90000.ckpt'
            model = MFAInfer(device, mfa_ckpt, torch_compile=False, precision=torch.bfloat16)
            reward_model_pack['phone'] = model
            reward_fns['phone'] = run_reward_phone

        if config['reward'].get('sim'):
            from modules.asr.speaker_verification.ecapa_tdnn import ECAPA_TDNN_SMALL
            model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
            sim_ckpt = config.get('sim_ckpt', 'checkpoints/wavlm/wavlm_large_finetune.pth')
            state_dict = torch.load(sim_ckpt, map_location='cpu')
            model.load_state_dict(state_dict['model'], strict=False)
            model.eval()
            model.to(device)
            for param in model.parameters():
                param.requires_grad = False
            reward_model_pack['sim'] = {
                'model': model,
            }
            reward_fns['sim'] = run_reward_sim
        
        self.reward_model_pack = reward_model_pack
        self.reward_fns = reward_fns
        
    def process(self, inputs_path):
        data = np.load(inputs_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object and data.size == 1:
            data = data.item()
    
        total_results = {}
        
        for reward_k in self.config['reward']:
            if self.config['reward'][reward_k]:
                results = self.reward_fns[reward_k](self.reward_model_pack[reward_k], data)
                total_results[reward_k] = results
        
        return total_results


if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')
    # kill_void()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--exp_name')
    parser.add_argument('--work_dir', default=None)
    parser.add_argument('--reset', action='store_true')
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config['work_dir'] = args.work_dir if args.work_dir is not None else config['work_dir']
    config['work_dir'] = work_dir = os.path.join(config['work_dir'], args.exp_name)
    if args.reset and os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    config['inputs_dir'] = inputs_dir = os.path.join(work_dir, 'inputs')
    config['outputs_dir'] = outputs_dir = os.path.join(work_dir, 'outputs')
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    poll_interval = config.get("poll_interval", 0.5)
    
    stats_interval = config.get("stats_interval", 60.0)  # 每多少秒打印一次统计
    last_stats_time = time.time()
    total_submitted = 0
    total_completed = 0
    interval_submitted = 0
    interval_completed = 0
    
    service = LocalDataPipeQueueService(
        data_pipe_cls=RewardWorker,
        cls_init_kwargs={
            'config': config
        },
        n_devices=config['n_devices'],
        workers_per_device=config['workers_per_device'],
        ordered=False,
        q_max_size=1000,
        daemon=config['daemon'],
        worker_proc_name='reward_worker'
    )
    print(f"Service is established, listening to {inputs_dir}...")
    
    in_progress = set()
    pending_jobs = {}
    
    try:
        
        while True:
            
            cur_inputs_paths = glob.glob(f"{inputs_dir}/*.npy")
            cur_inputs_paths = [p for p in cur_inputs_paths if p not in in_progress]
            
            for inputs_path in sorted(cur_inputs_paths):
                job_id = service.submit(inputs_path)
                pending_jobs[job_id] = inputs_path
                in_progress.add(inputs_path)
                
                total_submitted += 1
                interval_submitted += 1
                
            while True:
                try:
                    job_id, result = service.output_queue.get_nowait()
                except Empty:
                    break
                
                inputs_path = pending_jobs.pop(job_id, None)
                # if inputs_path is None: continue
                in_progress.discard(inputs_path)
                
                base = os.path.splitext(os.path.basename(inputs_path))[0]
                output_path = os.path.join(outputs_dir, base + ".npy")
                
                with open(output_path + '.tmp', 'wb') as f:
                    np.save(f, result, allow_pickle=True)
                os.rename(output_path + '.tmp', output_path)
                
                if config.get('delete_inputs_after_done', True):
                    try:
                        os.remove(inputs_path)
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        print(f"[WARN] Failed to delete input file {inputs_path}: {e}")
                        
                total_completed += 1
                interval_completed += 1
                
            now = time.time()
            elapsed = now - last_stats_time
            if elapsed >= stats_interval:
                if interval_submitted > 0 or interval_completed > 0:
                    throughput = interval_completed / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[STATS] last {elapsed:.1f}s | "
                        f"submitted: {interval_submitted} | "
                        f"completed: {interval_completed} | "
                        f"throughput: {throughput:.2f} req/s | "
                        f"in_progress_files: {len(in_progress)} | "
                        f"pending_jobs: {len(pending_jobs)} | "
                        f"total_completed: {total_completed}"
                    )
                # 重置窗口计数和时间
                interval_submitted = 0
                interval_completed = 0
                last_stats_time = now
                
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        service.close()
        os.system("pkill -f reward_worker")
    
    
# python tasks/tts/posttrain/reward_models/reward_server.py --config tasks/tts/posttrain/reward_models/configs/gemini3pro.yaml --exp_name 260116_prompt_base_dialogue_nft_unsync_pos_fix

# python tasks/tts/posttrain/reward_models/reward_server.py --config tasks/tts/posttrain/reward_models/configs/lai.yaml --exp_name 260125_prompt_base_dialogue_nft_lai

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tasks/tts/posttrain/reward_models/reward_server.py --config tasks/tts/posttrain/reward_models/configs/lai_phone.yaml --exp_name 260128_prompt_base_dialogue_nft_lai_phone --reset

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tasks/tts/posttrain/reward_models/reward_server.py --config tasks/tts/posttrain/reward_models/configs/qwen3aligner_phone.yaml --exp_name 260131_prompt_base_dialogue_nft_qwen3aligner_phone --reset
