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
from utils.commons.tos_utils_v2 import TosClient
from utils.commons.hdfs_utils import HDFSClient
from utils.commons.jsonl_utils import get_jsonl_line_by_number, count_jsonl_n_lines, JsonlChunkReader, get_jsonl_lines_by_range, build_jsonl_index
from utils.commons.parquet_utils import ParquetChunkReader
from utils.dataset.batcher import BucketBatcher
from utils.audio.vad import build_vad_model, run_vad_trim
from utils.audio.align import mel2token_to_dur
from utils.audio.align import mel2token_to_dur
from utils.text.split_text import get_word_list
from utils.text.ph_tone_convert import map_phone_to_tokendict
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text import is_chinese, is_english
from utils.text.pinyin_aug import augment_text_with_pinyin_advanced

from tasks.tts.dataset_utils.tts_datasets import MegaTTSDataset, FrontendLMDataset
from tasks.tts.dataset_utils.tts_fastdataset_v2 import BaseTTSShmDataset, raw_text_process, valid_item_kv
from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import build_spk_mask_from_text_tokens, _get_sx_token_patterns, augment_text_with_pinyin_s1s2_safe
from tasks.tts.dataset_utils.dialogue_fastdataset_v2 import shuffle_speaker_ids

DEBUG = False


class DialogueConcatShmDataset(BaseTTSShmDataset):
    def process_item(self, index, reader_pack, global_stores, hparams, i_worker, n_worker):
        
        if DEBUG:
            print(f'processer {i_worker}/{n_worker}: {index = }')
        
        def init_new_batch():
            tgt_size = random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])
            return tgt_size

        extra_random_reads = max(int(hparams.get('extra_random_reads', 0)), 2)
        n_chunks = self.ds_len
        g = torch.Generator(); g.manual_seed(i_worker % 1001 + hparams.get('dataloader_seed', 1231))
        indices = torch.randperm(n_chunks, generator=g).tolist()
        other_idx = 0
        
        read_res = self.read_fn(index, reader_pack, global_stores)
        if read_res is None:
            return
        raw_item, processer_fn = read_res

        raw_items = [raw_item]
        processer_fns = [processer_fn]
        for _ in range(extra_random_reads):
            try:
                read_res = self.read_fn(indices[other_idx], reader_pack, global_stores)
                other_idx = (other_idx + 1) % len(indices)
                raw_item, processer_fn = read_res
                raw_items.append(raw_item)
                processer_fns.append(processer_fn)
            except:
                pass
        
        if self.use_fast_dataloader:
            batcher = self.get_batcher(hparams, global_stores)
            tgt_size = init_new_batch()
        
        for item in self._process_item(processer_fns, raw_items, tgt_size, hparams, global_stores, i_worker, n_worker):
            if item is None:
                continue
            if self.use_fast_dataloader:
                batch = batcher.collate_batch(item)
                if batch is not None and len(batch) > 0:
                    # print(f"{len(batch) = } {batch[0]['wav'].shape = } {tgt_size = }")
                    tgt_size = init_new_batch()
                    yield batch
            else:
                yield [item]

    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):

        hop = hparams['hop_size']
        stride = hparams.get('vae_stride', 8)
        sr = hparams['audio_sample_rate']
        fm = hparams['frames_multiple']
        fm_wav = fm * hop
        min_frames = hparams['min_frames']
        max_tokens = hparams.get('max_tokens', 40000)
        prefetch_steps = hparams.get('prefetch_steps', 200)
        min_spk_num = hparams.get('min_spk_num', 2)
        max_spk_num = hparams.get('max_spk_num', 2)

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
            sx_patterns = get_from_global_stores(
                'cosyvoice2_sx_token_patterns',
                global_stores,
                lambda: _get_sx_token_patterns(cosyvoice2_text_tokenizer)
            )
            
        skip_logger: SkipLogger = get_from_global_stores(
            'skip_logger', global_stores, 
            lambda: SkipLogger([
                'no_score_cnt',
                'no_text_cnt',
                'no_caption_cnt',
            ], interval=1000, i_worker=i_worker, n_worker=n_worker)
        )
        
        items = []
        for processer_fn_, raw_item_ in zip(processer_fn, raw_item):
            items_ = processer_fn_(raw_item_, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
            if items_ is None:
                continue
            items.extend(items_)
        if len(items) == 0:
            if DEBUG:
                print(f'processer {i_worker}/{n_worker}: {items = }')
            return

        items_ = []
        spk2idx = {}
        for it in items:
            if ('wav' not in it) or ('txt' not in it):
                continue
            wav = it['wav'] if isinstance(it['wav'], torch.Tensor) else torch.as_tensor(it['wav'], dtype=torch.float32)
            frames = wav.shape[0] // hop
            if frames < min_frames:
                continue
            items_.append({
                'item_name': it['item_name'],
                'spk_name':  it['spk_name'],  # 如果有 subset，可在此处拼接 subset 以去歧义
                'wav': wav,
                'frames': frames,
                'text': it['txt'],
            })
            spk2idx.setdefault(it['spk_name'], []).append(len(items_) - 1)

        items = items_

        singles = {}

        blocks, rest_idxs = [], []
        while len(spk2idx) >= min_spk_num:
            cur_spk_num = min(len(spk2idx), random.randint(min_spk_num, max_spk_num))
            spk_names = random.sample(list(spk2idx.keys()), cur_spk_num)

            picks, total_f = [], 0
            # 先每个说话人拿一条，确保真正的“多说话人”
            for sn in spk_names:
                if spk2idx[sn]:
                    idx = spk2idx[sn].pop()
                    picks.append(idx)
                    total_f += items[idx]['frames']
            if len(picks) <= 1:
                for x in picks: rest_idxs.append(x)
                break

            # 轮转追加，直到接近目标长度
            turn = 0
            while True:
                sn = spk_names[turn % cur_spk_num]; turn += 1
                if not spk2idx[sn]:
                    if all(len(spk2idx[x]) == 0 for x in spk_names):
                        break
                    continue
                nxt = spk2idx[sn][-1]
                if total_f + items[nxt]['frames'] > tgt_size:
                    break
                picks.append(spk2idx[sn].pop())
                total_f += items[picks[-1]]['frames']

            blocks.append((picks, cur_spk_num))

            # === CHANGED: 现在才把只剩 0/1 条的说话人转移到 singles，避免下一轮死循环
            for sn in list(spk_names):
                if len(spk2idx.get(sn, [])) <= 1:
                    if len(spk2idx.get(sn, [])) == 1:
                        singles[sn] = spk2idx[sn]
                    spk2idx.pop(sn, None)
        
        # 剩余样本下放
        for sn, lst in spk2idx.items(): rest_idxs.extend(lst)
        for sn, lst in singles.items(): rest_idxs.extend(lst)

        # 合并样本（多说话人）
        for picks, cur_spk_num in blocks:
            wav = torch.cat([items[i]['wav'] for i in picks], dim=0)
            if wav.numel() < min_frames * hop:
                continue
            wav = wav[: (wav.shape[0] // fm_wav) * fm_wav]

            # === CHANGED: 统一生成 text/caption：每段使用 <S{sid}>...</S{sid}> 包裹
            seg_parts = []
            spk_map = {}      # name -> sid (1-based)
            ctx_wavs = []
            pick_sids = []    # 记录每个 pick 的 sid（供后面的 phone-level spk_mask 用）
            for j, idx in enumerate(picks):
                if j < cur_spk_num:
                    ctx_wavs.append(items[idx]['wav'])
                sn = items[idx]['spk_name']
                sid = spk_map.setdefault(sn, len(spk_map) + 1)  # 1-based，与原下游兼容
                pick_sids.append(sid)
                txt = items[idx]['text']
                seg_parts.append(f'<S{sid}>{txt}</S{sid}>')

            text_merged = ''.join(seg_parts)
            if hparams.get('shuffle_spk_ids', False):
                text_merged = shuffle_speaker_ids(text_merged)

            if len(picks) > cur_spk_num:
                ctx_cat = torch.cat(ctx_wavs, dim=0) if len(ctx_wavs) > 0 else wav[:0]
                ref_wav_start = (ctx_cat.shape[0] // fm_wav) * fm_wav
            else:
                if len(ctx_wavs) > 1:
                    pre = torch.cat(ctx_wavs[:-1], dim=0)
                    ref_wav_start = max(int(pre.shape[0] + ctx_wavs[-1].shape[0] * 0.1), pre.shape[0] + 20000)
                else:
                    ref_wav_start = 20000
                ref_wav_start = (ref_wav_start // fm_wav) * fm_wav
            max_idx = min(int(wav.shape[0] * 0.7), wav.shape[0] - 20000)
            if max_idx > ref_wav_start:
                ref_wav_start = (random.randint(ref_wav_start, max_idx) // fm_wav) * fm_wav

            ctx_wav = wav[:ref_wav_start]
            if speech_augmentor is not None:
                ctx_wav = speech_augmentor(ctx_wav, sr)
            ctx_mask = torch.zeros((wav.shape[0], 1), dtype=torch.float32)
            ctx_mask[:ref_wav_start] = 1.0
            ctx_mask = ctx_mask[:: hop * stride]

            mel_len = wav.shape[0] // hop
            
            item_tgt = {
                'id': 0,
                'item_name': '|||'.join([items[i]['item_name'] for i in picks]),
                'wav': wav,
                'wav_len': wav.shape[0],
                'text': text_merged,
                'ctx_wav': ctx_wav,
                'ctx_mask': ctx_mask,
                'len': mel_len // stride
            }

            if hparams.get('use_cosyvoice2_text_tokenizer', False):
                if hparams.get('mix_text_pinyin', {}).get('enable', False):
                    text_merged = augment_text_with_pinyin_s1s2_safe(text_merged, hparams)
                    item_tgt['text'] = text_merged
                text_tokens = cosyvoice2_text_tokenizer.encode(text_merged)
                text_tokens = torch.tensor(text_tokens).long()

                latent_len = int(item_tgt['wav_len'] // hop // stride)

                if latent_len <= 0 or text_tokens.numel() > latent_len:
                    skip_logger.update(1); continue

                # ===== spk_mask =====
                spk_mask = build_spk_mask_from_text_tokens(text_tokens, sx_patterns)
                if spk_mask.shape != text_tokens.shape:
                    skip_logger.update(1); continue

                item_tgt['txt_tokens'] = text_tokens
                item_tgt['spk_mask'] = spk_mask

            yield item_tgt

        for idx in rest_idxs:
            it = items[idx]
            wav = it['wav']
            if wav.numel() < min_frames * hop:
                continue
            wav = wav[: (wav.shape[0] // fm_wav) * fm_wav]

            mel_len = wav.shape[0] // hop

            text_single = f"<S1>{it['text']}</S1>"
            if hparams.get('shuffle_spk_ids', False):
                text_single = shuffle_speaker_ids(text_single)

            item_tgt = {
                'id': 0,
                'item_name': it['item_name'],
                'wav': wav,
                'wav_len': wav.shape[0],
                'text': text_single,
                'len': mel_len // stride
            }
            
            min_idx = max(int(wav.shape[0] * 0.1), 20000)
            max_idx = min(int(wav.shape[0] * 0.9), wav.shape[0] - 20000)
            if min_idx > max_idx:
                min_idx = int(wav.shape[0] * 0.4); max_idx = int(wav.shape[0] * 0.6)
            ref_wav_start = (random.randint(min_idx, max_idx) // fm_wav) * fm_wav

            ctx_wav = wav[:ref_wav_start]
            if speech_augmentor is not None:
                ctx_wav = speech_augmentor(ctx_wav, sr)
            ctx_mask = torch.zeros((wav.shape[0], 1), dtype=torch.float32)
            ctx_mask[:ref_wav_start] = 1.0
            ctx_mask = ctx_mask[:: hop * stride]
            item_tgt['ctx_wav'] = ctx_wav
            item_tgt['ctx_mask'] = ctx_mask

            if hparams.get('use_cosyvoice2_text_tokenizer', False):
                if hparams.get('mix_text_pinyin', {}).get('enable', False):
                    text_single = augment_text_with_pinyin_s1s2_safe(text_single, hparams)
                    item_tgt['text'] = text_single
                text_tokens = cosyvoice2_text_tokenizer.encode(text_single)
                text_tokens = torch.tensor(text_tokens).long()

                latent_len = int(item_tgt['wav_len'] // hop // stride)

                if latent_len <= 0 or text_tokens.numel() > latent_len:
                    skip_logger.update(1); continue

                # ===== spk_mask =====
                spk_mask = build_spk_mask_from_text_tokens(text_tokens, sx_patterns)
                if spk_mask.shape != text_tokens.shape:
                    skip_logger.update(1); continue

                item_tgt['txt_tokens'] = text_tokens
                item_tgt['spk_mask'] = spk_mask

            yield item_tgt


def processer_fn_megatts3(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    items = []
    for item_ in raw_item:
        try:
            item = {}
            if hparams.get('load_wav', True):
                item['wav'] = torch.FloatTensor(item_['wav'])
                item['wav_len'] = item['wav'].shape[0]
            else:
                item['wav_len'] = int(float(item_['sec']) * hparams['audio_sample_rate'])
            item['item_name'] = item_['item_name']
            txt = raw_text_process(item_['txt_raw'], wav_len=item['wav_len'])
            if txt is None:
                continue
            item['txt'] = txt
            item['spk_name'] = item_['spk_name']
            items.append(item)
        except:
            continue
    return items


def processer_fn_robust_mega3_noref(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    fm_wav = hparams['frames_multiple'] * hparams['hop_size']
    
    items = []
    for item_ in raw_item:
        try:
            item = {}

            item['wav'] = torch.FloatTensor(item_['wav'])
            item['wav'] = item['wav'][:len(item['wav']) // fm_wav * fm_wav]
            item['wav_len'] = item['wav'].shape[0]
            item['item_name'] = item_['item_name']
            item['txt'] = item_['text']
            txt = raw_text_process(item['txt'], wav_len=item['wav_len'])
            if txt is None:
                continue
            item['txt'] = txt
            ds_name = item_['ds_name']
            item['spk_name'] = f"{ds_name}#{item_['spk']}"
            items.append(item)
        except:
            continue
    return items

