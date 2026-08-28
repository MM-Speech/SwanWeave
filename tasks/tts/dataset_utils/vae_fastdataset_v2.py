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
import uuid

import setproctitle
import torch
import torchaudio
import numpy as np
import torch.utils
import torch.utils.data
import librosa
from dataloader import FalconReader, KVReader

from utils.commons.hparams import hparams
from utils.commons.base_shm_dataset import BaseFalconReaderShmDataset, get_from_global_stores, save_samples_to_shm
from utils.commons.dataset_utils import collate_xd, pad_or_cut_xd, SkipLogger
from utils.commons.tos_utils_v2 import TosClient
from utils.dataset.batcher import BucketBatcher
from utils.nn.seq_utils import repeat_or_chunk_1d

from tasks.tts.dataset_utils.tts_datasets import MegaTTSDataset, FrontendLMDataset
from tasks.tts.dataset_utils.tts_fastdataset_v2 import BaseTTSShmDataset, safe_read_path
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator

DEBUG = False

import math
import torch

class VAEShmDataset(BaseTTSShmDataset):
    def get_batcher(self, hparams, global_stores):
        return get_from_global_stores(
            'batcher', global_stores,
            lambda: BucketBatcher(
                buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,
                         600, 650, 700, 750, 800, 850, 900, 950, 1000, 1200, 1400,
                         1600, 1800, 2000, 2400, 2800, 3000, 3500, 4000, 4500, 5000,
                         6000, 7000, 8000, 9000, 10000, 11000, 12000, 14000, 16000, 18000, 20000],
                dynamic_batch=hparams.get("dynamic_batch", True),   # False
                batch_size=hparams['max_sentences'],
                maximum_bucket_size=hparams.get('max_tokens', None),
                length_fn=lambda x: x['len'],
            )
        )
    
    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):
        fm = hparams['frames_multiple']
        hop_size = hparams['hop_size']
        fm_wav = fm * hop_size
        sr = hparams['audio_sample_rate']
        tgt_size = tgt_size // fm * fm
        
        skip_logger: SkipLogger = get_from_global_stores(
            'skip_logger', global_stores, 
            lambda: SkipLogger(interval=1000, i_worker=i_worker, n_worker=n_worker)
        )
        
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
        
        items = processer_fn(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
        if items is None or len(items) <= 0:
            return
        
        for item in items:
            wav = item['wav']
            chunked_wavs = repeat_or_chunk_1d(wav, tgt_size=tgt_size * hop_size, drop_last=False)

            for i in range(chunked_wavs.shape[0]):
                
                if speech_augmentor is not None:
                    chunked_wavs[i] = speech_augmentor(chunked_wavs[i], sr)

                if hparams.get('low_sample_rate_enhance_prob', 0.0) > 0:
                    if random.random() < hparams.get('low_sample_rate_enhance_prob', 0.0):
                        low_sr = random.choice(hparams.get('low_sample_rate', [sr]))
                        if low_sr != sr:
                            chunked_wav = chunked_wavs[i].numpy()
                            chunked_wav = librosa.resample(chunked_wav, orig_sr=sr, target_sr=low_sr)
                            chunked_wav = librosa.resample(chunked_wav, orig_sr=low_sr, target_sr=sr)
                            chunked_wav = chunked_wav[: len(chunked_wav) // fm_wav * fm_wav]
                            chunked_wavs[i] = torch.from_numpy(chunked_wav)

                # normalize to [-1, 1]
                if torch.max(torch.abs(chunked_wavs[i])) > 1.0:
                    chunked_wavs[i] = chunked_wavs[i] / torch.max(torch.abs(chunked_wavs[i]))
                
                yield {
                    'wav': chunked_wavs[i],
                    'len': len(chunked_wavs[i]) // hop_size
                }

        
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


class VAEFixedLenProcessMixShmDataset(VAEShmDataset):
    """
    VAE dataset variant for fixed-length training and worker-local wav mix.

    Key differences from VAEShmDataset:
    - training tgt_size can be fixed by hparams['fixed_tgt_size']
    - wav mix happens inside processor workers, so it no longer depends on
      runtime batch size in collater
    - collater only stacks tensors and does not apply extra mix
    """

    def _sample_tgt_size(self, hparams):
        fixed_tgt_size = int(hparams.get('fixed_tgt_size', 0) or 0)
        if fixed_tgt_size > 0:
            return fixed_tgt_size
        return random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])

    def _get_process_mix_prob(self, hparams):
        mix_prob = hparams.get('process_wav_mix_prob', None)
        if mix_prob is None:
            return float(hparams.get('wav_mix_prob', 0.0))
        return float(mix_prob)

    def _get_process_mix_alpha_range(self, hparams):
        return hparams.get('process_wav_mix_alpha_range', hparams.get('wav_mix_alpha_range', (0.4, 0.6)))

    def _maybe_process_mix(self, wav, wav_len_frames, hparams, global_stores):
        mix_prob = self._get_process_mix_prob(hparams)
        pool_size = int(hparams.get('process_wav_mix_pool_size', 128))
        min_pool_size = int(hparams.get('process_wav_mix_min_pool_size', 16))
        alpha_range = self._get_process_mix_alpha_range(hparams)

        mix_pools = get_from_global_stores(
            'process_wav_mix_pools', global_stores,
            lambda: collections.defaultdict(collections.deque)
        )
        pool_key = int(wav_len_frames)
        pool = mix_pools[pool_key]

        mixed_wav = wav
        if (
                mix_prob > 0.0 and
                len(pool) >= min_pool_size and
                random.random() < mix_prob
            ):
            partner = pool[random.randrange(len(pool))]
            alpha = random.uniform(alpha_range[0], alpha_range[1])
            mixed_wav = alpha * wav + (1.0 - alpha) * partner

        pool.append(wav.clone())
        while len(pool) > pool_size:
            pool.popleft()

        return mixed_wav

    def process_item(self, *args):
        index, reader_pack, global_stores, hparams, i_worker, n_worker = args
        if DEBUG:
            print(f'processer {i_worker}/{n_worker}: {index = }')

        read_res = self.read_fn(index, reader_pack, global_stores)
        if read_res is None:
            return
        raw_item, processer_fn = read_res

        if self.use_fast_dataloader:
            batcher = self.get_batcher(hparams, global_stores)
            tgt_size = self._sample_tgt_size(hparams)

        for item in self._process_item(processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):
            if item is None:
                continue
            if self.use_fast_dataloader:
                batch = batcher.collate_batch(item)
                if batch is not None and len(batch) > 0:
                    tgt_size = self._sample_tgt_size(hparams)
                    yield batch
            else:
                yield [item]
        return

    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):
        fm = hparams['frames_multiple']
        hop_size = hparams['hop_size']
        fm_wav = fm * hop_size
        sr = hparams['audio_sample_rate']
        tgt_size = tgt_size // fm * fm
        tgt_wav_len = tgt_size * hop_size

        skip_logger: SkipLogger = get_from_global_stores(
            'skip_logger', global_stores,
            lambda: SkipLogger(interval=1000, i_worker=i_worker, n_worker=n_worker)
        )

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

        items = processer_fn(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
        if items is None or len(items) <= 0:
            return

        for item in items:
            wav = item['wav']
            chunked_wavs = repeat_or_chunk_1d(wav, tgt_size=tgt_wav_len, drop_last=False)

            for i in range(chunked_wavs.shape[0]):
                chunk_wav = chunked_wavs[i]

                if speech_augmentor is not None:
                    chunk_wav = speech_augmentor(chunk_wav, sr)

                if hparams.get('low_sample_rate_enhance_prob', 0.0) > 0:
                    if random.random() < hparams.get('low_sample_rate_enhance_prob', 0.0):
                        low_sr = random.choice(hparams.get('low_sample_rate', [sr]))
                        if low_sr != sr:
                            chunk_np = chunk_wav.numpy()
                            chunk_np = librosa.resample(chunk_np, orig_sr=sr, target_sr=low_sr)
                            chunk_np = librosa.resample(chunk_np, orig_sr=low_sr, target_sr=sr)
                            chunk_np = chunk_np[: len(chunk_np) // fm_wav * fm_wav]
                            chunk_wav = torch.from_numpy(chunk_np)

                # keep shapes compile-friendly even after resample/effect changes
                if chunk_wav.shape[0] != tgt_wav_len:
                    chunk_wav = pad_or_cut_xd(chunk_wav, tgt_wav_len, pad_value=0.0)

                if torch.max(torch.abs(chunk_wav)) > 1.0:
                    chunk_wav = chunk_wav / torch.max(torch.abs(chunk_wav))

                base_chunk = chunk_wav.float()
                mixed_chunk = self._maybe_process_mix(
                    base_chunk, tgt_size, hparams, global_stores
                )

                yield {
                    'wav': mixed_chunk,
                    'len': tgt_size
                }

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
        }
        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch
    

def processer_fn_megatts3(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    sr = hparams['audio_sample_rate']
    items = []
    for item_ in raw_item:
        try:
            item = {}
            wav = item_['wav'].astype(float)
            if sr != 24000:
                wav = librosa.resample(wav, orig_sr=24000, target_sr=sr)
            item['wav'] = torch.FloatTensor(wav)
            item['wav_len'] = item['wav'].shape[0]
            item['item_name'] = item_['item_name']
            item['spk_name'] = item_['spk_name']
            items.append(item)
            skip_logger.step(1)
        except:
            traceback.print_exc()
            skip_logger.report(1, 'megatts3')
            continue
    return items


def processer_fn_zyxc_1spk(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):

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
    
    with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
        items = []
        for item_ in raw_item:
            try:
                item_name = item_['item_name']
                wav_k = item_['wav_k']
            
                data = tos_client.get_object(wav_k, verbose=False)
                if data is None:
                    continue
                global_wav_path = os.path.join(temp_dir, f'global.m4a')
                with open(global_wav_path, 'wb') as f:
                    f.write(data)
                # global_wav, sr = torchaudio.load(io.BytesIO(data))
                try:
                    global_wav, sr_ = torchaudio.load(global_wav_path)
                    global_wav = global_wav.mean(dim=0).numpy()
                except:
                    continue
                if len(global_wav) == 0:
                    continue

                for segment_idx, segment_meta in enumerate(item_['segments_1spk']):
                    item = {}
                
                    wav_start, wav_end = segment_meta['start'], segment_meta['end']
                    wav = global_wav[int(wav_start * sr_): int(wav_end * sr_)]
                    if len(wav) == 0:
                        continue
                    if sr_ != sr:
                        wav = librosa.resample(wav, orig_sr=sr_, target_sr=sr)
                    item['wav'] = torch.FloatTensor(wav)
                    item['wav_len'] = wav.shape[0]
                        
                    item['item_name'] = item_name + '#' + f'{segment_idx}'

                    items.append(item)
                
            except:
                traceback.print_exc()
                continue
            
    return items


def processer_fn_prompttts(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    fm = hparams['frames_multiple']
    hop_size = hparams['hop_size']
    fm_wav = fm * hop_size
    sr = hparams['audio_sample_rate']
    
    items = []
    for item_ in raw_item:
        try:
            wav = item_.get('wav')
            if wav is None:
                return
            wav = torch.FloatTensor(wav)
            if wav.numel() == 0:
                return

            org_sr = item_.get('sr', sr)

            if sr != org_sr:
                wav = librosa.resample(wav.numpy(), orig_sr=org_sr, target_sr=sr)
                wav = torch.from_numpy(wav)
            
            wav = pad_or_cut_xd(wav, math.ceil(len(wav) / fm_wav) * fm_wav)

            item = {}
            item['wav'] = wav
            item['wav_len'] = item['wav'].shape[0]
            items.append(item)
            skip_logger.step(1)
        except Exception as e:
            traceback.print_exc()
            skip_logger.report(1, 'prompttts')
            continue

    return items

def processer_fn_mtg_jamendo(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    fm = hparams['frames_multiple']
    hop_size = hparams['hop_size']
    fm_wav = fm * hop_size
    sr = hparams['audio_sample_rate']

    hdfs_clients = get_from_global_stores(
        'hdfs_clients', global_stores,
        lambda: {}
    )
    
    with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
        items = []
        for item_ in raw_item:
            wav_path = safe_read_path(item_['wav_path'], os.path.join(temp_dir, f"{uuid.uuid4()}.wav"), hdfs_clients)
            try:
                wav, _ = librosa.load(wav_path, sr=sr)
                if wav.size > 0 and fm_wav > 0:
                    wav = pad_or_cut_xd(torch.from_numpy(wav), math.ceil(len(wav) / fm_wav) * fm_wav).numpy()
                if wav.size == 0:
                    return
                item = {}
                item['wav'] = torch.FloatTensor(wav)
                item['wav_len'] = item['wav'].shape[0]
                items.append(item)
                skip_logger.step(1)
            except:
                skip_logger.report(1, 'mtg_jamendo')
                continue
        
    return items

def processer_fn_audioset(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    fm = hparams['frames_multiple']
    hop_size = hparams['hop_size']
    fm_wav = fm * hop_size
    sr = hparams['audio_sample_rate']

    items = []
    for item_ in raw_item:
        try:
            wav_path = item_['wav_path']
            wav, _ = librosa.load(wav_path, sr=sr)
            if wav.size > 0 and fm_wav > 0:
                wav = pad_or_cut_xd(torch.from_numpy(wav), math.ceil(len(wav) / fm_wav) * fm_wav).numpy()
            if wav.size == 0:
                return
            item = {}
            item['wav'] = torch.FloatTensor(wav)
            item['wav_len'] = item['wav'].shape[0]
            items.append(item)
            skip_logger.step(1)
        except:
            skip_logger.report(1, 'audioset')
            continue
    
    return items

def processer_fn_classic(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    fm = hparams['frames_multiple']
    hop_size = hparams['hop_size']
    fm_wav = fm * hop_size
    sr = hparams['audio_sample_rate']
    cluster = os.environ.get('CLUSTER', '').lower()

    items = []
    for item_ in raw_item:
        try:
            wav_path = item_['wav_path']
            if cluster == 'va':
                wav_path = os.path.join('/mnt/bn/genai-data2/liruiqi/data/music', wav_path)
            else:
                wav_path = os.path.join('/mnt/bn/sa-ag-data/liruiqi/data/music', wav_path)
            wav, _ = librosa.load(wav_path, sr=sr)
            if wav.size > 0 and fm_wav > 0:
                wav = pad_or_cut_xd(torch.from_numpy(wav), math.ceil(len(wav) / fm_wav) * fm_wav).numpy()
            if wav.size == 0:
                return
            item = {}
            item['wav'] = torch.FloatTensor(wav)
            item['wav_len'] = item['wav'].shape[0]
            items.append(item)
            skip_logger.step(1)
        except:
            traceback.print_exc()
            skip_logger.report(1, 'classic')
            continue
    
    return items

if __name__ == '__main__':
    # client = HDFSClient(namespace='harunava')
    from tasks.tts.dataset_utils.tts_fastdataset_v2 import get_hdfs_file
    with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
        wav_path = get_hdfs_file('hdfs://harunava/home/byte_advertising_genai/20250808/liruiqi/data/music/mtg-jamendo/train_sp/193/1024417[0012].wav', f"{temp_dir}/audio.wav", {})
        print(f"{wav_path = }")
