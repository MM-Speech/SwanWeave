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
from utils.commons.seq_utils import adjust_list_to_sum
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
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator
from tasks.tts.dataset_utils.tts_fastdataset_v2 import BaseTTSShmDataset, raw_text_process, valid_item_kv, merge_A2B
from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import build_spk_mask_from_text_tokens, _get_sx_token_patterns, augment_text_with_pinyin_s1s2_safe

DEBUG = False


class ASRShmDataset(BaseTTSShmDataset):
    def get_batcher(self, hparams, global_stores):
        batcher = get_from_global_stores(
            'batcher', global_stores,
            lambda: BucketBatcher(
                buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 
                            600, 650, 700, 750, 800, 850, 900, 950, 1000, 1200, 1400, 
                            1600, 1800, 2000, 2400, 2800, 3000, 4000, 5000, 6000, 7000,
                            8000, 9000, 10000, 11000, 12000, 20000],
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

        from utils.text.cosyvoice2_tokenizer import get_tokenizer
        cosyvoice2_text_tokenizer = get_from_global_stores(
            'cosyvoice2_text_tokenizer', global_stores, 
            lambda: get_tokenizer(multilingual=True, num_languages=100)
        )

        skip_logger: SkipLogger = get_from_global_stores(
            'skip_logger', global_stores, 
            lambda: SkipLogger([
                'no_score_cnt',
                'no_text_cnt',
                'no_caption_cnt',
                'no_phone_cnt',
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
                'wav_len': sum([s['wav_len'] for s in samples]),
                'seg_wav_lens': [s['wav_len'] for s in samples],
                'seg_txts': [s['txt'] for s in samples],
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

        ########################
        # task specific process #
        ########################
        for item_tgt in items:
            if not (hparams['max_frames'] >= item_tgt['wav_len'] // hop_size > hparams['min_frames']):
                skip_logger.update(1); continue
            
            item_tgt['text'] = item_tgt['txt']

            if hparams.get('load_wav', True):
                item_tgt['wav'] = item_tgt['wav'][:len(item_tgt['wav']) // fm_wav * fm_wav]
                if speech_augmentor is not None:
                    item_tgt['wav'] = speech_augmentor(item_tgt['wav'], sr)

            mel_len = len(item_tgt['wav']) // hop_size

            seg_dur = []
            token_seg_id = []
            text_tokens = []
            seg_wav_offsets = np.cumsum([0] + item_tgt['seg_wav_lens'])
            for seg_i in range(len(item_tgt['seg_txts'])):
                text_tokens_ = cosyvoice2_text_tokenizer.encode(item_tgt['seg_txts'][seg_i])
                text_tokens.extend(text_tokens_)
                seg_dur.append((seg_wav_offsets[seg_i + 1] - seg_wav_offsets[seg_i]) // hop_size)
                token_seg_id.extend([seg_i] * len(text_tokens_))
            if sum(seg_dur) != mel_len:
                seg_dur = adjust_list_to_sum(seg_dur, mel_len) 
            item_tgt['txt_tokens'] = torch.tensor(text_tokens).long()
            item_tgt['seg_dur'] = torch.tensor(seg_dur).long()
            item_tgt['token_seg_id'] = torch.tensor(token_seg_id).long()
            if item_tgt['token_seg_id'].max() >= len(item_tgt['seg_dur']):
                print(f"{item_tgt['token_seg_id'].max() = } {len(item_tgt['seg_dur']) = }")
                skip_logger.update(1); continue
                
            item_tgt['len'] = mel_len
            yield item_tgt
            skip_logger.step(1)

    def collater(self, samples):
        batch = super().collater(samples)

        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]

        if 'seg_dur' not in batch and 'seg_dur' in samples[0]:
            batch['seg_dur'] = collate_xd([s['seg_dur'] for s in samples], 0)
            batch['seg_dur_len'] = torch.tensor([len(s['seg_dur']) for s in samples]).long()
            batch['token_seg_id'] = collate_xd([s['token_seg_id'] for s in samples], 0)
        
        return batch

            
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
            item['txt'] = item_['txt_raw']
            item['txt'] = raw_text_process(item['txt'], wav_len=item['wav_len'])
            if item['txt'] is None:
                continue
            item['item_name'] = item_['item_name']
            item['spk_name'] = item_['spk_name']
            items.append(item)
        except:
            continue
    return items


def processer_fn_robust_mega3(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):    
    items = []
    for item_ in raw_item:
        try:
            item = {}

            item['wav'] = torch.FloatTensor(item_['wav'])
            item['wav_len'] = item['wav'].shape[0]
            
            item['item_name'] = item_['item_name']
            item['txt'] = item_['text']
            ds_name = item_['ds_name']
            item['spk_name'] = f"{ds_name}#{item_['spk']}"
            items.append(item)
        except:
            continue
    return items

