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
from utils.commons.base_shm_dataset import BaseFalconReaderShmDataset, get_from_global_stores, save_samples_to_shm
from utils.commons.dataset_utils import collate_xd, pad_or_cut_xd, SkipLogger
from utils.dataset.batcher import BucketBatcher
from utils.nn.seq_utils import repeat_or_chunk_1d

from tasks.tts.dataset_utils.tts_datasets import MegaTTSDataset, FrontendLMDataset
from tasks.tts.dataset_utils.tts_fastdataset_v2 import BaseTTSShmDataset, raw_text_process, valid_item_kv
from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import build_spk_mask_from_text_tokens, _get_sx_token_patterns, augment_text_with_pinyin_s1s2_safe

DEBUG = False


class NepaShmDataset(BaseTTSShmDataset):

    def get_batcher(self, hparams, global_stores):
        batcher = get_from_global_stores(
            'batcher', global_stores,
            lambda: BucketBatcher(
                buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 
                            600, 650, 700, 750, 800, 850, 900, 950, 1000, 1200, 1400, 
                            1600, 1800, 2000, 2400, 2800, 3000, 4000, 5000, 6000, 10000, 
                            12000, 14000, 16000, 18000, 20000, 30000, 40000],
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
        tgt_size = tgt_size // fm * fm

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
        
        ########################
        # task specific process #
        ########################
        for item_tgt in items:
            if not (hparams['max_frames'] >= item_tgt['wav_len'] // hop_size > hparams['min_frames']):
                skip_logger.update(1); continue
            
            item_tgt['wav'] = repeat_or_chunk_1d(item_tgt['wav'], tgt_size * hop_size)[0]
            if speech_augmentor is not None:
                item_tgt['wav'] = torch.from_numpy(speech_augmentor(item_tgt['wav'].numpy()))
            mel_len = len(item_tgt['wav']) // hop_size

            item_tgt['len'] = mel_len
            yield item_tgt
            skip_logger.step(1)

    def collater(self, samples):
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]
        if len(samples) == 0:
            if hasattr(self, 'backup_batch') and self.backup_batch is not None:
                print('use backup batch!')
                return self.backup_batch
            else:
                print('no batch to take!')
                return {}
        wavs = collate_xd([s['wav'] for s in samples], 0.0) if 'wav' in samples[0] and samples[0]['wav'] is not None else None
        wav_lengths = torch.LongTensor([s['wav'].shape[0] for s in samples]) if wavs is not None else None
        
        if wavs is not None:
            mix_prob = hparams.get('wav_mix_prob', 0.0)
            alpha_range = hparams.get('wav_mix_alpha_range', (0.4, 0.6))
            if mix_prob > 0.0:
                B, T = wavs.shape
                for i in range(B):
                    if random.random() < mix_prob:
                        j = random.randint(0, B - 1)
                        if j == i:
                            continue
                        alpha = random.uniform(alpha_range[0], alpha_range[1])
                        li = wav_lengths[i].item()
                        lj = wav_lengths[j].item()
                        L = max(li, lj)
                        wavs[i, :L] = alpha * wavs[i, :L] + (1.0 - alpha) * wavs[j, :L]
                        wav_lengths[i] = L
        
        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
        }
        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch
