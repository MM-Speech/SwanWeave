import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import os
import random
import json
from copy import deepcopy
import pickle
import re
import traceback
import math
import time
import tempfile
from pathlib import Path
import uuid
import io

import setproctitle
import torch
import torchaudio
import numpy as np
import torch.utils
import torch.utils.data
import librosa
from dataloader import FalconReader, KVReader

from utils.commons.import_utils import import_module_bystr
from utils.commons.hparams import hparams
from utils.commons.os_utils import multiprocess_glob, handle_exacption
from utils.commons.io import get_wav_duration, print_once, load_samples_from_tsv, load_samples_from_jsonl
from utils.commons.base_shm_dataset import BaseFalconReaderShmDataset, get_from_global_stores, save_samples_to_shm
from utils.commons.dataset_utils import collate_xd, pad_or_cut_xd, SkipLogger
from utils.commons.tensor_utils import convert_to_tensor, convert_to_np
from tasks.tts.dataset_utils.base_fastdataset_v2 import shuffle_speaker_ids, build_speaker_shuffle_map, apply_speaker_shuffle
from utils.commons.tos_utils_v2 import TosClient
from utils.commons.hdfs_utils import HDFSClient
from utils.commons.jsonl_utils import get_jsonl_line_by_number, count_jsonl_n_lines, JsonlChunkReader, get_jsonl_lines_by_range, build_jsonl_index
from utils.commons.parquet_utils import ParquetChunkReader
from utils.dataset.batcher import BucketBatcher
from utils.audio.vad import build_vad_model, run_vad_trim
from utils.audio.io import to_wav_bytes
from utils.text.split_text import get_word_list
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text import is_chinese, is_english
from utils.text.pinyin_aug import augment_text_with_pinyin_advanced
from utils.service.file_service import FileQueueClient

from tasks.tts.dataset_utils.tts_datasets import MegaTTSDataset, FrontendLMDataset
from tasks.tts.dataset_utils.tts_fastdataset_v2 import BaseTTSShmDataset, raw_text_process, valid_item_kv
from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import build_spk_mask_from_text_tokens, _get_sx_token_patterns, augment_text_with_pinyin_s1s2_safe

DEBUG = False


def _count_speaker_tags_in_text(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    ids_found = set(re.findall(r"<\s*/?\s*(S\d+)\s*>", text, flags=re.IGNORECASE))
    return len(ids_found)



class DialogueShmDataset(BaseTTSShmDataset):

    def get_batcher(self, hparams, global_stores):
        batcher = get_from_global_stores(
            'batcher', global_stores,
            lambda: BucketBatcher(
                buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 
                            600, 650, 700, 750, 800, 850, 900, 950, 1000, 1200, 1400, 
                            1600, 1800, 2000, 2400, 2800, 3000, 4000, 5000, 6000, 10000],
                dynamic_batch=hparams.get("dynamic_batch", True),
                batch_size=hparams['max_sentences'],
                maximum_bucket_size=hparams['max_tokens'],
                length_fn=lambda x: x['len'],
            )
        )
        return batcher

    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):

        hop_size = hparams['hop_size']
        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']
        sr = hparams['audio_sample_rate']

        speech_augmentor = None
        if hparams.get('wav_add_noise', False) or hparams.get('wav_add_effect', False):
            from tasks.tts.dataset_utils.augment import SpeechAugment
            speech_augmentor = get_from_global_stores(
                'speech_augmentor', global_stores, 
                lambda: SpeechAugment(
                    hparams.get('wav_add_noise', False), hparams.get('wav_add_effect', False), hparams.get('musan_dir', None), 
                    noise_prob=hparams.get('wav_add_noise_prob', 0.5), effect_prob=hparams.get('wav_add_effect_prob', 0.5), 
                    noise_snr=(6.0, 20.0), with_speech=hparams.get('musan_with_speech', False)
                )
            )

        if hparams.get('use_cosyvoice2_text_tokenizer', False):
            from utils.text.cosyvoice2_tokenizer import get_tokenizer
            cosyvoice2_text_tokenizer = get_tokenizer(multilingual=True, num_languages=100)
            
        skip_logger: SkipLogger = get_from_global_stores(
            'skip_logger', global_stores, 
            lambda: SkipLogger([
                'no_score_cnt',
                'no_text_cnt',
                'no_caption_cnt',
            ], interval=1000, i_worker=i_worker, n_worker=n_worker)
        )
        
        items = processer_fn(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
        if items is None:
            if DEBUG:
                print(f'processer {i_worker}/{n_worker}: {items = }')
            return
            
        ##################
        # merge same spk #
        ##################
        def merge_samples(samples):
            sample_merged = {
                'id': 0,
                'item_name': '|||'.join([s['item_name'] for s in samples]),
                'wav': torch.cat([s['wav'] for s in samples], 0) if valid_item_kv(samples[0], 'wav') else None,
                'txt': ' '.join([s['txt'] for s in samples]),
                'spk_name': samples[0]['spk_name'],
                'wav_len': sum([s['wav_len'] for s in samples])
            }
            return sample_merged
            
        merge_same_spk = True
        if merge_same_spk:
            items_merged = []
            last_spk = ''
            total_frames = 0
            items_to_merge = []
            merge_multi_spk = hparams.get('merge_multi_spk', False)
            for item in items:
                if item.get('skip_merge_same_spk', False):
                    items_merged.append(item)
                    continue
                wav_len = item['wav_len']
                if len(items_to_merge) > 0:
                    if (
                            ((not merge_multi_spk) and item['spk_name'] != last_spk) or 
                            (tgt_size is not None and total_frames > 0 and (total_frames + wav_len // hparams['hop_size']) > tgt_size)
                        ):
                        items_merged.append(merge_samples(items_to_merge))
                        items_to_merge = []
                        total_frames = 0
                items_to_merge.append(item)
                last_spk = item['spk_name']
                total_frames += wav_len // hparams['hop_size']
            if len(items_to_merge) > 0:
                items_merged.append(merge_samples(items_to_merge))
            items = items_merged

        #########################
        # online text alignment #
        #########################
        if hparams.get('online_text_alignment_dataloader', False):
            asr_client: FileQueueClient = get_from_global_stores(
                'asr_client', global_stores, 
                lambda: FileQueueClient(
                    work_dir=hparams.get('online_text_alignment_work_dir', 'user/service_cache/asr'),
                )
            )
            job_ids = []
            job_id2batch_size = {}
            payload = {"wav_bytes": [], "texts": [], "durations": []}
            for item in items:
                payload['wav_bytes'].append(to_wav_bytes(item['wav'].numpy(), sr))
                payload['texts'].append(item['txt'])
                payload['durations'].append(item['wav_len'] / sr)
                if len(payload['wav_bytes']) >= hparams.get('online_text_alignment_batch_size', 32):
                    job_id = asr_client.submit(payload)
                    job_ids.append(job_id)
                    job_id2batch_size[job_id] = len(payload['wav_bytes'])
                    payload = {"wav_bytes": [], "texts": [], "durations": []}
            if len(payload['wav_bytes']) > 0:
                job_id = asr_client.submit(payload)
                job_ids.append(job_id)
                job_id2batch_size[job_id] = len(payload['wav_bytes'])
            texts_aligned_total = []
            for job_id in job_ids:
                try:
                    asr_results = asr_client.wait_result(job_id, poll_s=0.5, timeout_s=300)
                    if asr_results is None:
                        print(f'asr results is None, job_id: {job_id}')
                        texts_aligned = [None] * job_id2batch_size[job_id]
                    else:
                        texts_aligned = asr_results['result']['asr_results']['pause_punct_texts']
                        assert len(texts_aligned) == job_id2batch_size[job_id], f"asr results len {len(texts_aligned)} != batch size {job_id2batch_size[job_id]}"
                except TimeoutError:
                    # traceback.print_exc()
                    texts_aligned = [None] * job_id2batch_size[job_id]
                except:
                    traceback.print_exc()
                    texts_aligned = [None] * job_id2batch_size[job_id]
                texts_aligned_total.extend(texts_aligned)
            assert len(texts_aligned_total) == len(items), f"texts_aligned_total len {len(texts_aligned_total)} != items len {len(items)}"
            items_ = []
            for item_idx in range(len(items)):
                if texts_aligned_total[item_idx] is not None:
                    items[item_idx]['txt'] = texts_aligned_total[item_idx]
                    items_.append(items[item_idx])
            items = items_
            if len(items) == 0:
                return

        ########################
        # task specific process #
        ########################
        for item_tgt in items:
            if not (hparams['max_frames'] >= item_tgt['wav_len'] // hop_size > hparams['min_frames']):
                skip_logger.update(1); continue

            item_tgt['text'] = item_tgt['txt']
            item_tgt['orig_text'] = deepcopy(item_tgt['text'])
            
            if hparams.get('load_wav', True):
                item_tgt['wav'] = item_tgt['wav'][:len(item_tgt['wav']) // fm_wav * fm_wav]
            mel_len = len(item_tgt['wav']) // hop_size

            if hparams.get('use_cosyvoice2_text_tokenizer', False):
                if hparams.get('mix_text_pinyin', {}).get('enable', False):
                    item_tgt['text'] = augment_text_with_pinyin_s1s2_safe(item_tgt['text'], hparams)
                text_tokens = cosyvoice2_text_tokenizer.encode(item_tgt['text'])
                text_tokens = torch.tensor(text_tokens).long()

                vae_stride = int(hparams.get('vae_stride', 4))
                latent_len = int(item_tgt['wav_len'] // hop_size // vae_stride)

                if latent_len <= 0 or text_tokens.numel() > latent_len:
                    skip_logger.update(1); continue

                item_tgt['txt_tokens'] = text_tokens

                # ===== spk_mask =====
                sx_patterns = get_from_global_stores(
                    'cosyvoice2_sx_token_patterns',
                    global_stores,
                    lambda: _get_sx_token_patterns(cosyvoice2_text_tokenizer)
                )
                item_tgt['spk_mask'] = build_spk_mask_from_text_tokens(text_tokens, sx_patterns)
                if item_tgt['spk_mask'].shape != text_tokens.shape:
                    skip_logger.update(1); continue
            
            if item_tgt.get('ctx_wav') is None:
                min_idx = max(int(mel_len * 0.1), 200)
                max_idx = min(int(mel_len * 0.9), mel_len - 200)
                if min_idx > max_idx:
                    min_idx = int(mel_len * 0.4)
                    max_idx = int(mel_len * 0.6)
                rand_length = random.randint(min_idx, max_idx) // fm * fm
                ctx_mask = torch.zeros((item_tgt['wav'].shape[0] // hparams['hop_size'], 1))
                ctx_mask[:rand_length] = 1.0
                item_tgt['ctx_mask'] = ctx_mask[::hparams['vae_stride']]
                item_tgt['ctx_wav'] = deepcopy(item_tgt['wav'])
                item_tgt['ctx_wav'] = item_tgt['ctx_wav'][:rand_length*hparams['hop_size']]
                if speech_augmentor is not None:
                    item_tgt['ctx_wav'] = speech_augmentor(item_tgt['ctx_wav'], sr)
            
            item_tgt['len'] = mel_len // 4
            yield item_tgt
            skip_logger.step(1)


def processer_fn_zyxc_dialogue(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    def get_tos_client():
        cluster = os.environ.get('CLUSTER', '').lower()
        if cluster == 'va':
            tos_bucket = 'sa-ag-sg-research-sg'
        else:
            tos_bucket = 'humanaigc-ads'
        return TosClient(bucket=tos_bucket)

    tos_client: TosClient = get_from_global_stores(
        'tos_client', global_stores,
        get_tos_client
    )

    sr = hparams['audio_sample_rate']
    fm = hparams['frames_multiple']
    hop = hparams['hop_size']
    stride = hparams.get('vae_stride', 8)
    fm_wav = fm * hop
    max_spk_num = hparams.get('max_spk_num', 8)

    if not raw_item:
        return []

    # TOS 磁盘缓存辅助（复用 swan_caption_fastdataset 的模式）
    import hashlib

    def _sha1(s: str) -> str:
        return hashlib.sha1(s.encode('utf-8')).hexdigest()

    def _get_cache_dir():
        cache_dir = get_from_global_stores(
            'tos_cache_dir', global_stores,
            lambda: hparams.get('tos_cache_dir', '/dev/shm/zyxc_tos_cache')
        )
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass
        return cache_dir

    def _tos_to_local_path(wav_k: str) -> str:
        """TOS key → 本地缓存路径，未命中则下载。返回 None 表示失败。"""
        cache_dir = _get_cache_dir()
        ext = os.path.splitext(wav_k)[1] or '.m4a'
        cached_path = os.path.join(cache_dir, f"{_sha1(wav_k)}{ext}")
        if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
            return cached_path
        data = tos_client.get_object(wav_k, verbose=False)
        if data is None:
            return None
        try:
            tmp = f"{cached_path}.tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, cached_path)
        except Exception:
            return None
        return cached_path

    def _load_segment(wav_path: str, start_sec: float, end_sec: float, target_sr: int):
        """用 torchaudio 片段读取，不加载整条音频。"""
        try:
            info = torchaudio.info(wav_path)
            org_sr = int(info.sample_rate)
        except Exception:
            return None

        s_idx = int(start_sec * org_sr)
        e_idx = int(end_sec * org_sr)
        num_frames = max(e_idx - s_idx, 1)
        try:
            wav, _ = torchaudio.load(wav_path, frame_offset=s_idx, num_frames=num_frames)
        except Exception:
            return None
        wav = wav.mean(dim=0).numpy()
        if len(wav) == 0:
            return None
        if org_sr != target_sr:
            wav = librosa.resample(wav, orig_sr=org_sr, target_sr=target_sr)
        return wav

    with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
        items = []
        for item_ in raw_item:
            if not isinstance(item_, dict):
                skip_logger.report(k='dialogue')
                skip_logger.step()
                continue
            if not isinstance(item_.get('turns'), list) or not isinstance(item_.get('segments'), list):
                skip_logger.report(k='dialogue')
                skip_logger.step()
                continue
            try:
                turns = item_['turns']
                segments = item_['segments']

                for segment in segments:
                    # 临时容器（只有 turn 全部校验通过才 append）
                    spk_map = {}  # 原 spk -> 连续 1-based

                    # 文本/说话人
                    res_text_parts, res_cap_parts = [], []
                    kept_turn_sids = []     # 只统计“真正保留”的 turn
                    ref_wav_start_turn_idx = 0  # 仍然用“第几条 turn”来定位 ctx 起点

                    # 新旧两种音频组织方式：
                    #   - 老格式：逐 turn 读裁好的 wav，再拼接 -> wav_lst / frame_spk_mask_lst
                    #   - 新格式：segment/turn 带 start/end（秒），按 segment 的 [start,end] 在一整条 wav 上切
                    seg_has_ts = isinstance(segment, dict) and ('start' in segment and 'end' in segment)
                    seg_start_sec = float(segment.get('start', 0.0)) if seg_has_ts else None
                    seg_end_sec   = float(segment.get('end', 0.0)) if seg_has_ts else None

                    # 新 jsonl：只记录“整条 wav 的来源”和每个 turn 的时间区间
                    segment_audio_src = None  # ('local'|'tos', path_or_key)
                    turn_time_info = []       # [{'sid': sid, 'start_sec': s, 'end_sec': e}, ...]
                    turn_sample_lens = []     # 与 kept_turn_sids 对齐，用于估计 ctx 起点

                    # 读取 segment 里的每个 turn
                    for turn_idx in segment['turn_idxs']:
                        turn = turns[turn_idx]

                        # 先过滤 text 为空
                        text = turn.get('text')
                        if not text or not str(text).strip():
                            continue

                        # 说话人 sid（1-based）
                        raw_spk = turn.get('spk', 'spk_unk')
                        sid = spk_map.setdefault(raw_spk, len(spk_map) + 1)

                        t_start = float(turn.get('start', seg_start_sec if seg_start_sec is not None else 0.0))
                        t_end   = float(turn.get('end', t_start))
                        # 与 segment 区间求交集，保证落在 [seg_start, seg_end] 内
                        if seg_start_sec is not None:
                            t_start = max(t_start, seg_start_sec)
                        if seg_end_sec is not None:
                            t_end = min(t_end, seg_end_sec)
                        if t_end <= t_start:
                            # 该 turn 在本 segment 内没有有效时长
                            continue

                        wav_path = turn.get('wav_path', None)
                        wav_k = turn.get('wav_k', None)
                        # if i_worker in [0, 1, 2]:
                        #     print(f"{i_worker = } {wav_path = }, {wav_k = }")
                        if segment_audio_src is None:
                            if wav_path:
                                if wav_path.startswith('XYZ_20w'):
                                    wav_path = os.path.join('/mnt/bn/sa-ag-data/zhangyu.34/data/speech', wav_path)
                                segment_audio_src = ('local', wav_path)
                            elif wav_k:
                                segment_audio_src = ('tos', wav_k)

                        # 暂时不读 wav，只记录时间区间，后面按整段 segment 一次性裁切
                        turn_time_info.append({
                            'sid': sid,
                            'start_sec': t_start,
                            'end_sec': t_end,
                        })
                        # 用于估计“第三次换人”的采样点位置
                        turn_sample_lens.append(int(max(0.0, (t_end - t_start)) * sr))

                        # —— 文本与 caption：统一包裹为 <S{sid}>...</S{sid}> —— #
                        seg_txt = f'<S{sid}>{text}</S{sid}>'
                        res_text_parts.append(seg_txt)
                        res_cap_parts.append(seg_txt)

                        kept_turn_sids.append(sid)
                                
                    # === 段级检查（按“保留下来的 turn”统计）===
                    if len(kept_turn_sids) == 0:
                        continue

                    candidate_split_idxs = list(range(len(kept_turn_sids)))
                    ref_wav_start_turn_idx = None
                    for split_idx in candidate_split_idxs:
                        ref_speakers = set(kept_turn_sids[:split_idx+1])
                        tgt_speakers = set(kept_turn_sids[split_idx:])
                        if tgt_speakers.issubset(ref_speakers):
                            ref_wav_start_turn_idx = split_idx
                            break

                    if ref_wav_start_turn_idx is None:
                        continue

                    # 按 segment 的 [start, end] 片段读取，不加载整条音频
                    if segment_audio_src is None or len(turn_time_info) == 0:
                        continue

                    kind, value = segment_audio_src
                    s_sec = seg_start_sec if seg_start_sec is not None else min(t['start_sec'] for t in turn_time_info)
                    e_sec = seg_end_sec   if seg_end_sec is not None else max(t['end_sec']   for t in turn_time_info)

                    # 获取本地路径（TOS 走磁盘缓存）
                    if kind == 'local':
                        local_path = value
                    elif kind == 'tos':
                        feat_tos_dir = str(Path(value).with_name(Path(value).stem + '_features').with_suffix(''))
                        vocal_k = os.path.join(feat_tos_dir, 'vocal.m4a')
                        local_path = _tos_to_local_path(vocal_k)
                        if local_path is None:
                            continue
                    else:
                        continue

                    wav_seg = _load_segment(local_path, s_sec, e_sec, sr)
                    if wav_seg is None or len(wav_seg) == 0:
                        continue
                    wav_seg = wav_seg[:len(wav_seg) // fm_wav * fm_wav]
                    if wav_seg.size == 0:
                        continue

                    # === 文本与 caption（统一为 <S{sid}>...</S{sid}>）===
                    text_merged = ''.join(res_text_parts)
                    if hparams.get('shuffle_spk_ids', True):
                        text_merged = shuffle_speaker_ids(text_merged)

                    # ctx 起点
                    pre_len = 0
                    if ref_wav_start_turn_idx > 0:
                        pre_len = int(sum(turn_sample_lens[:ref_wav_start_turn_idx]))

                    ref_wav_start = (pre_len // fm_wav) * fm_wav
                    max_idx = min(int(wav_seg.shape[0] * 0.9), wav_seg.shape[0] - 20000)
                    if max_idx > ref_wav_start:
                        ref_wav_start = (random.randint(ref_wav_start, max_idx) // fm_wav) * fm_wav

                    ctx_wav = wav_seg[:ref_wav_start]
                    ctx_mask = torch.zeros((wav_seg.shape[0], 1), dtype=torch.float32)
                    ctx_mask[:ref_wav_start] = 1.0
                    ctx_mask = ctx_mask[:: hop * stride]

                    # 产出
                    item = {
                        'item_name': str(uuid.uuid4()),
                        'wav': torch.from_numpy(wav_seg),
                        'wav_len': len(wav_seg),
                        'txt': text_merged,
                        'ctx_wav': torch.from_numpy(ctx_wav),
                        'ctx_mask': ctx_mask,
                    }

                    item['len'] = int(item['wav'].shape[0] / hop / stride)
                    item['skip_merge_same_spk'] = True

                    items.append(item)

            except:
                traceback.print_exc()
                skip_logger.report(k='dialogue')
                continue
            skip_logger.step()
    return items


_DIALOGUE_LOCAL_RE = re.compile(r'<(S(\d+))>(.*?)</\1>', flags=re.IGNORECASE | re.DOTALL)


def _build_dialogue_local(text: str) -> str:
    """从 <S1>xxx</S1><S2>yyy</S2> 构造 Content: { Subject 1说<S1>xxx</S1>, Subject 2说<S2>yyy</S2> }."""
    if not isinstance(text, str) or not text.strip():
        return ""
    matches = _DIALOGUE_LOCAL_RE.findall(text)
    if not matches:
        return ""
    parts = [f"Speaker{n}说<{tag}>{content}</{tag}>" for tag, n, content in matches]
    return f"Content: {{ {', '.join(parts)} }}."


def processer_fn_zyxc_dialogue_for_prompt_caption(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    """
    适配 PromptDiTTask + SwanCaptionShmDataset:
    - 复用现有多人 dialogue 的 ctx/ref 构造
    - 始终构造 Content local（Subject N说格式）
    - 强制保留 ref：force_ref=True
    - 按当前 caption 体系的最大说话人数做过滤
    """
    items = processer_fn_zyxc_dialogue(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
    if not items:
        return items

    max_spk = int(hparams.get('max_speakers_in_caption', hparams.get('max_spk_num', 4)))
    items_out = []
    for item in items:
        spk_cnt = _count_speaker_tags_in_text(item.get('txt', ''))
        if spk_cnt <= 0 or spk_cnt > max_spk:
            continue

        item['caption'] = ""
        item['global'] = ""
        # 多人 zeroshot: 始终提供 content local，避免空 local 批次。
        item['local'] = _build_dialogue_local(item.get('txt', ''))
        # Keep dialogue samples schema-compatible with swan_caption batches.
        item['bgm_flag'] = 2
        item['quality_flag'] = 3
        item['force_ref'] = True
        item['disable_ref'] = False
        items_out.append(item)

    return items_out


def processer_fn_megatts3(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    sr = hparams['audio_sample_rate']
    items = []
    for item_ in raw_item:
        try:
            item = {}
            if hparams.get('load_wav', True):
                wav = item_['wav'].astype(float)
                if sr != 24000:
                    wav = librosa.resample(wav, orig_sr=24000, target_sr=sr)
                item['wav'] = torch.FloatTensor(wav)
                item['wav_len'] = item['wav'].shape[0]
            else:
                item['wav_len'] = int(float(item_['sec']) * hparams['audio_sample_rate'])
            item['item_name'] = item_['item_name']
            txt = raw_text_process(item_['txt_raw'], wav_len=item['wav_len'])
            if txt is None:
                continue
            item['txt'] = f"<S1>{txt}</S1>"
            if hparams.get('shuffle_spk_ids', True):
                item['txt'] = shuffle_speaker_ids(item['txt'])
            item['spk_name'] = item_['spk_name']
            items.append(item)
        except:
            skip_logger.report(k='megatts3')
            continue
        skip_logger.step()
    return items


def processer_fn_robust_mega3(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    fm_wav = hparams['frames_multiple'] * hparams['hop_size']
    sr = hparams['audio_sample_rate']
    
    items = []
    for item_ in raw_item:
        try:
            item = {}
            wav = item_['wav'].astype(float)
            if sr != 24000:
                wav = librosa.resample(wav, orig_sr=24000, target_sr=sr)
            item['wav'] = torch.FloatTensor(wav)
            item['wav'] = item['wav'][:len(item['wav']) // fm_wav * fm_wav]
            item['ctx_wav'] = torch.FloatTensor(item_['ref_wav'])
            item['ctx_wav'] = item['ctx_wav'][:len(item['ctx_wav']) // fm_wav * fm_wav]
            item['wav'] = torch.cat(item['ctx_wav'], item['wav'])
            item['wav_len'] = item['wav'].shape[0]
            ctx_mask = torch.zeros((item['wav'].shape[0] // hparams['hop_size'], 1))
            ctx_mask[:item['ctx_wav'].shape[0] // hparams['hop_size']] = 1.0
            item['ctx_mask'] = ctx_mask[::hparams['vae_stride']]
            item['item_name'] = item_['item_name']
            item['txt'] = item_['ref_text'] + item_['text']
            txt = raw_text_process(item['txt'], wav_len=item['wav_len'])
            if txt is None:
                continue
            item['txt'] = f"<S1>{txt}</S1>"
            if hparams.get('shuffle_spk_ids', True):
                item['txt'] = shuffle_speaker_ids(item['txt'])
            ds_name = item_['ds_name']
            item['spk_name'] = f"{ds_name}#{item_['spk']}"
            item['skip_merge_same_spk'] = True
            items.append(item)
        except:
            skip_logger.report(k='robust_megatts3')
            continue
        skip_logger.step()
    return items
