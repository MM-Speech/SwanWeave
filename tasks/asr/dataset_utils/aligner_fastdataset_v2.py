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
from utils.text.pinyin_aug import augment_text_with_pinyin_advanced

from tasks.tts.dataset_utils.tts_datasets import MegaTTSDataset, FrontendLMDataset
from tasks.tts.dataset_utils.tts_fastdataset_v2 import BaseTTSShmDataset, raw_text_process, valid_item_kv
from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import build_spk_mask_from_text_tokens, _get_sx_token_patterns, augment_text_with_pinyin_s1s2_safe

DEBUG = False


class AlignerShmDataset(BaseTTSShmDataset):

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

        if hparams['model_version'] in ['aligner_2tower', 'aligner_2tower_v2', 'aligner_2tower_v3', 'aligner_2tower_v4', 'aligner_2tower_v5', 'aligner_2tower_v6']:
            from modules.asr.forced_align.aligner_2tower import build_text_tokenizer
            text_tokenizer = build_text_tokenizer(hparams)
        elif hparams['model_version'] == 'aligner_mdm':
            from modules.asr.forced_align.aligner_mdm import build_text_tokenizer
            text_tokenizer = build_text_tokenizer(hparams)
            text_tokenizer.timestamp_start_id = text_tokenizer.encode('<|TS0.00|>')[0]
            text_tokenizer.timestamp_end_id = text_tokenizer.encode('<|TS300.00|>')[0]
        
        items = processer_fn(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
        if items is None:
            if DEBUG:
                print(f'processer {i_worker}/{n_worker}: {items = }')
            return
            
        ########################
        # task specific process #
        ########################
        for item_tgt in items:

            if speech_augmentor is not None:
                item_tgt['wav'] = torch.from_numpy(speech_augmentor(item_tgt['wav'].numpy()))
            item_tgt['wav'] = pad_or_cut_xd(item_tgt['wav'], math.ceil(len(item_tgt['wav']) / fm_wav) * fm_wav)
            mel_len = len(item_tgt['wav']) // hop_size

            if hparams['model_version'] in ['aligner_2tower', 'aligner_2tower_v2', 'aligner_2tower_v3', 'aligner_2tower_v4', 'aligner_2tower_v5', 'aligner_2tower_v6']:
                words = item_tgt['words']
                text = '<|wbd|>'.join(words) + '<|wbd|>'
                item_tgt['text'] = text

                text_tokens = text_tokenizer.encode(item_tgt['text'])
                text_tokens = torch.tensor(text_tokens).long()
                item_tgt['txt_tokens'] = text_tokens

            elif hparams['model_version'] == 'aligner_mdm':
                text = []
                words = item_tgt['words']
                word_start_times = item_tgt['word_start_times']
                word_end_times = item_tgt['word_end_times']
                word_conf = item_tgt['word_conf']
                for word_idx in range(len(words)):
                    text.append(words[word_idx])
                    text.append(f"<|TS{float(word_start_times[word_idx].item()):.2f}|>")
                    text.append(f"<|TS{float(word_end_times[word_idx].item()):.2f}|>")
                text = ''.join(text)
                item_tgt['text'] = text
                text_tokens = text_tokenizer.encode(text)
                text_tokens = torch.tensor(text_tokens).long()
                item_tgt['txt_tokens'] = text_tokens

                timestamp_mask = torch.zeros(len(text_tokens), dtype=torch.bool)
                for token_idx, token in enumerate(text_tokens):
                    if text_tokenizer.timestamp_start_id <= token <= text_tokenizer.timestamp_end_id:
                        timestamp_mask[token_idx] = True
                item_tgt['timestamp_mask'] = timestamp_mask
                
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
        
        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
            'text': [s['text'] for s in samples],
            'txt_tokens': collate_xd([s['txt_tokens'] for s in samples]),
            'txt_lengths': torch.LongTensor([s['txt_tokens'].shape[0] for s in samples]),
            'word_start_times': collate_xd([s['word_start_times'] for s in samples], -1.0),
            'word_end_times': collate_xd([s['word_end_times'] for s in samples], -1.0),
            'word_conf': collate_xd([s['word_conf'] for s in samples], 0.0),
        }
        if 'timestamp_mask' in samples[0] and samples[0]['timestamp_mask'] is not None:
            batch['timestamp_mask'] = collate_xd([s['timestamp_mask'] for s in samples], False)
        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch


def processer_fn_qwen3aligner(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):    
    items = []
    for item_ in raw_item:
        try:
            item = {}

            item['item_name'] = item_['item_name']

            if 'wav' in item_:

                item['wav'] = torch.FloatTensor(item_['wav'])
                item['wav_len'] = item['wav'].shape[0]

            else:

                wav_path = item_['wav_path']
                if isinstance(wav_path, str):
                    # wav, _ = librosa.load(wav_path, sr=hparams['audio_sample_rate'])
                    wav, orig_sr = torchaudio.load(wav_path)
                    wav = wav.mean(dim=0)
                    if len(wav) == 0:
                        continue
                    if orig_sr != hparams['audio_sample_rate']:
                        wav = torchaudio.functional.resample(wav, orig_sr, hparams['audio_sample_rate'])
                    wav = wav.numpy()
                elif isinstance(wav_path, list):
                    # wav = np.concatenate([librosa.load(p, sr=hparams['audio_sample_rate'])[0] for p in wav_path])
                    wavs = []
                    for p in wav_path:
                        wav, orig_sr = torchaudio.load(p)
                        wav = wav.mean(dim=0)
                        if len(wav) == 0:
                            continue
                        if orig_sr != hparams['audio_sample_rate']:
                            wav = torchaudio.functional.resample(wav, orig_sr, hparams['audio_sample_rate'])
                        wavs.append(wav.numpy())
                    wav = np.concatenate(wavs)
                item['wav'] = torch.FloatTensor(wav)
                item['wav_len'] = item['wav'].shape[0]
                
            words = []
            word_start_times = []
            word_end_times = []
            word_conf = []
            for word in item_['qwen3aligner']:
                words.append(word['text'])
                word_start_times.append(word['start_time'])
                word_end_times.append(word['end_time'])
                word_conf.append(word['conf'])

            item['words'] = words
            item['word_start_times'] = torch.FloatTensor(word_start_times)
            item['word_end_times'] = torch.FloatTensor(word_end_times)
            item['word_conf'] = torch.FloatTensor(word_conf)

            items.append(item)
        except:
            continue
    return items

