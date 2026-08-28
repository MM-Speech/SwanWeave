import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import os
import random
import traceback
import tempfile
import bisect
import math
from copy import deepcopy
import torch
import torch.nn.functional as F
import torchaudio
import librosa
from utils.commons.base_shm_dataset import get_from_global_stores
from utils.commons.dataset_utils import collate_xd, SkipLogger
from utils.commons.tos_utils_v2 import TosClient
from utils.dataset.batcher import BucketBatcher
from utils.audio.io import to_wav_bytes
from tasks.tts.dataset_utils.augment import SpeechAugment
from utils.text.pinyin_aug import augment_text_with_pinyin_s1s2_safe
from utils.service.file_service import FileQueueClient
from tasks.tts.dataset_utils.base_fastdataset_v2 import BaseShmDataset, valid_item_kv, raw_text_process, shuffle_speaker_ids
from tasks.tts.dataset_utils.swan_base_fastdataset import SwanTTSShmDataset

DEBUG = False

class SwanVCShmDataset(SwanTTSShmDataset):

    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):

        hop_size = hparams['hop_size']
        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']
        sr = hparams['audio_sample_rate']

        speech_augmentor = None
        if hparams.get('wav_add_noise', False) or hparams.get('wav_add_effect', False):
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

        try:
            items = processer_fn(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
        except Exception:
            fn_name = getattr(processer_fn, "__name__", str(processer_fn))
            raw_type = type(raw_item)
            raw_len = len(raw_item) if hasattr(raw_item, "__len__") else "NA"
            print(f"| SwanTTSShmDataset: processer_fn crashed worker={i_worker}/{n_worker} fn={fn_name} raw_type={raw_type} raw_len={raw_len}")
            if isinstance(raw_item, (list, tuple)) and len(raw_item) > 0 and isinstance(raw_item[0], dict):
                print(f"| SwanTTSShmDataset: raw_item[0].keys={list(raw_item[0].keys())[:50]}")
            traceback.print_exc()
            return
        
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
            
            if item_tgt['txt'] is None:
                skip_logger.update(1); continue
            item_tgt['text'] = item_tgt['txt']
            item_tgt['orig_text'] = deepcopy(item_tgt['text'])
            
            if hparams.get('load_wav', True):
                item_tgt['wav'] = item_tgt['wav'][:len(item_tgt['wav']) // fm_wav * fm_wav]
            mel_len = len(item_tgt['wav']) // hop_size

            if speech_augmentor is not None:
                item_tgt['wav'] = torch.from_numpy(speech_augmentor(item_tgt['wav'].numpy()))

            # normalize to [-1, 1]
            if torch.max(torch.abs(item_tgt['wav'])) > 1.0:
                item_tgt['wav'] = item_tgt['wav'] / torch.max(torch.abs(item_tgt['wav']))

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
                ctx_len = rand_length * hparams['hop_size']
                item_tgt['ctx_wav'] = item_tgt['wav'][:ctx_len]   
            
            item_tgt['len'] = mel_len // 4
            yield item_tgt
            skip_logger.step(1)
          
          