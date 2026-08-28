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
from functools import partial
import subprocess
import multiprocessing
import glob
import shutil

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
from utils.commons.base_shm_dataset import BaseFalconReaderShmDataset, get_from_global_stores, save_samples_to_shm, save_fnames_to_shm
from utils.commons.dataset_utils import collate_xd, pad_or_cut_xd, SkipLogger
from utils.commons.tensor_utils import convert_to_tensor, convert_to_np
from utils.commons.tos_utils_v2 import TosClient
from utils.commons.hdfs_utils import HDFSClient
from utils.commons.jsonl_utils import get_jsonl_line_by_number, count_jsonl_n_lines, JsonlChunkReader, get_jsonl_lines_by_range
from utils.commons.parquet_utils import ParquetChunkReader
from utils.dataset.batcher import BucketBatcher
from utils.audio.vad import build_vad_model, run_vad_trim
from utils.audio.align import mel2token_to_dur
from utils.audio.align import mel2token_to_dur
from utils.text.split_text import get_word_list
from utils.text.ph_tone_convert import map_phone_to_tokendict
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text import is_chinese, is_english, PHONE_VOCAB, TONE_VOCAB
from utils.text.text_encoder import TokenTextEncoder

from tasks.tts.dataset_utils.tts_datasets import MegaTTSDataset, FrontendLMDataset
from tasks.tts.dataset_utils.tts_fastdataset_v2 import BaseTTSShmDataset, raw_text_process, merge_A2B, valid_item_kv
from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import build_spk_mask_from_text_tokens, _get_sx_token_patterns, augment_text_with_pinyin_s1s2_safe
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator

DEBUG = False

class RepeatBatcher:
    def __init__(self, repeat_k=1, batch_size=1):
        self.repeat_k = repeat_k
        self.batch_size = batch_size
        self.buffer = []
        
    def collate_batch(self, item):
        self.buffer.append(item)
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer * self.repeat_k
        self.buffer = []
        return batch

def _replicate_paths_for_all_ranks(paths, world_size):
    replicas = {r: [] for r in range(world_size)}
    for src in paths:
        # 先把 src 原子重命名为 .r0，避免短暂同时存在两个名字
        dst0 = f"{src}.r0"
        try:
            os.rename(src, dst0)  # 同目录原子操作
        except FileExistsError:
            pass
        except Exception:
            shutil.copyfile(src, dst0)
            try:
                os.remove(src)
            except FileNotFoundError:
                pass
        replicas[0].append(dst0)
        # 再为其他 rank 建立硬链接
        for r in range(1, world_size):
            dstr = f"{src}.r{r}"
            try:
                os.link(dst0, dstr)
            except FileExistsError:
                pass
            except OSError:
                shutil.copyfile(dst0, dstr)
            replicas[r].append(dstr)
    return replicas

class PosttrainTTSRepeatShmDataset(BaseTTSShmDataset):

    def build_fast_dataloader(self, shm_base, seed, world_size,
                              hparams, n_processor=16, max_epoch=-1, auto_restart=True, ds_len=None):
        if hparams.get('dataloader_mp_type', 'fork') == 'spawn':
            ctx = multiprocessing.get_context('spawn')
        else:
            ctx = multiprocessing.get_context('fork')
        setproctitle.setproctitle(
            f'data_processor:{hparams["exp_name"]}:constructor')
        sp_size = hparams.get('sp_size', 1)
        q_to_pull = ctx.Queue(2048 * world_size // sp_size)
        q_to_push = ctx.Queue(10000)
        os.makedirs(shm_base, exist_ok=True)
        print(f"| training dataset len: {ds_len}")
        self.seed = seed
        proc_controller = ctx.Process(
            target=self.controller_fn, args=(ds_len, seed, q_to_pull, hparams, max_epoch, n_processor), daemon=True
        )
        proc_controller.start()
        batch_counter = ctx.Value('i', 0)

        def create_batch_saver(rank_i):
            return ctx.Process(target=self.batch_saver_fn,
                               args=(q_to_push, hparams, seed, shm_base, world_size // sp_size, rank_i, batch_counter),
                               daemon=True)

        # MODIFIED: 只启动一个batch_saver
        proc_batch_saver = [create_batch_saver(0)]
        for p in proc_batch_saver:
            p.start()

        counter = ctx.Value('i', 0)

        def create_processor(worker_i):
            return ctx.Process(target=self.process_fn,
                               args=(q_to_pull, q_to_push, world_size // sp_size, shm_base, counter, hparams,
                                     seed, worker_i, n_processor),
                               daemon=True)

        proc_processor = [create_processor(i) for i in range(n_processor)]
        for p in proc_processor:
            p.start()
        time.sleep(60)
        if auto_restart:
            while True:
                if not proc_controller.is_alive():
                    break
                for p in proc_processor:
                    if not p.is_alive():
                        print("| Restarting process")
                        i = proc_processor.index(p)
                        new_p = create_processor(i)
                        new_p.start()
                        proc_processor[i] = new_p

                time.sleep(5)
        proc_controller.join()
        for p in proc_processor:
            p.join()

    def batch_saver_fn(self, q_to_push, hparams_, seed, shm_base, world_size, rank_i, batch_counter):
        hparams.update(hparams_)
        setproctitle.setproctitle(f'data_processor:{hparams["exp_name"]}:batch_saver#{rank_i}/{world_size}')
        print(f"| init batch_saver#{rank_i}/{world_size}")

        dataloader_num_replica = int(hparams.get('dataloader_num_replica', 1) or 1)
        if dataloader_num_replica <= 0:
            dataloader_num_replica = 1
        if dataloader_num_replica > int(self.world_size):
            raise ValueError(
                f"dataloader_num_replica ({dataloader_num_replica}) must be <= local world_size ({int(self.world_size)})"
            )
        if int(self.world_size) % dataloader_num_replica != 0:
            raise ValueError(
                f"dataloader_num_replica ({dataloader_num_replica}) must divide local world_size ({int(self.world_size)}) to keep prompt-sharing groups aligned"
            )

        # repeat_interleaved 模式下，只让 rank0 的 batch_saver 真正工作，其他直接退出
        # if self.hparams.get('repeat_interleaved', False) and rank_i != 0:
        #     print(f"| repeat_interleaved enabled: batch_saver#{rank_i} exits (only rank0 writes)")
        #     return

        g = torch.Generator()
        g.manual_seed(seed)
        random.seed(seed)
        shuffle_buffer = self.shuffle_buffer
        items_buffer = []

        local_batch_cnt = 0
        while True:
            try:
                while True:
                    cur_batch_names = glob.glob(f'{shm_base}/*.json')
                    if len(cur_batch_names) < self.prefetch_steps * self.world_size:
                        break
                    with batch_counter.get_lock():
                        cur_batch_cnt = batch_counter.value
                    cur_max_batch_idx = max([int(Path(p).stem) for p in cur_batch_names]) if cur_batch_names else -1
                    if cur_max_batch_idx > cur_batch_cnt * dataloader_num_replica:
                        break
                    if DEBUG:
                        print(f"batch_saver#{rank_i}/{world_size}: json 文件太多"
                            f"({len(cur_batch_names)} >= {self.prefetch_steps} x {self.world_size}), 休息1s等等getitem")
                    time.sleep(1)

                if q_to_push.empty():
                    if len(items_buffer) < shuffle_buffer:
                        if DEBUG:
                            print(f"batch_saver#{rank_i}/{world_size}: q_to_push 为空, 休息1s等等processor")
                        time.sleep(1)
                else:
                    items_buffer.append(q_to_push.get())

                if DEBUG:
                    print(f"| length of items_buffer: {len(items_buffer)}")

                while len(items_buffer) > shuffle_buffer:
                    with batch_counter.get_lock():
                        batch_cnt = batch_counter.value
                        batch_counter.value += 1
                        local_batch_cnt += 1

                    if local_batch_cnt % 16 == 0:
                        random.shuffle(items_buffer)

                    batch, items_buffer = self.create_batch(items_buffer)

                    if dataloader_num_replica > 1:
                        # 关键：为所有 rank 复制同一 batch 的样本文件（硬链接），并分别写 JSON
                        replicas = _replicate_paths_for_all_ranks(batch, dataloader_num_replica)
                        for r in range(dataloader_num_replica):
                            global_idx = batch_cnt * dataloader_num_replica + r
                            out_path = save_fnames_to_shm(replicas[r], global_idx, shm_base)
                            if DEBUG:
                                print(f"batch saver#{rank_i}: broadcast batch_cnt={batch_cnt} to rank={r}, saved {out_path}")
                    else:
                        # 原来的单 rank 保存逻辑（如果你保留多个 saver，注意它们会写不同内容）
                        out_path = save_fnames_to_shm(batch, batch_cnt, shm_base)
                        if DEBUG:
                            print(f"batch saver#{rank_i}/{world_size}: saved to {out_path}")
                    batch_cnt += 1

            except:
                traceback.print_exc()


    def get_batcher(self, hparams, global_stores):
        batcher = get_from_global_stores(
            'batcher', global_stores,
            lambda: RepeatBatcher(
                repeat_k=1,
                batch_size=hparams['sample']['train_batch_size'],
            )
        )
        return batcher
    
    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):

        hop_size = hparams['hop_size']
        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']
        sr = hparams['audio_sample_rate']
        vae_stride = hparams['vae_stride']

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
                'no_phone_cnt',
            ], interval=1000, i_worker=i_worker, n_worker=n_worker)
        )

        ref_wav_manifests = get_from_global_stores(
            'ref_wav_manifests', global_stores,
            lambda: {
                'zh': json.load(open('/mnt/bn/sa-ag-data/liruiqi/data/speech/robust_mega3/ref_251211/manifest_zh.json')),
                'en': json.load(open('/mnt/bn/sa-ag-data/liruiqi/data/speech/robust_mega3/ref_251211/manifest_en.json')),
            }
        )

        from utils.text.cosyvoice2_tokenizer import get_tokenizer
        cosyvoice2_text_tokenizer = get_tokenizer(multilingual=True, num_languages=100)
        sx_patterns = get_from_global_stores(
            'cosyvoice2_sx_token_patterns',
            global_stores,
            lambda: _get_sx_token_patterns(cosyvoice2_text_tokenizer)
        )

        items = processer_fn(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
        if items is None:
            if DEBUG:
                print(f'processer {i_worker}/{n_worker}: {items = }')
            return

        ########################
        # task specific process #
        ########################
        # for dialogue training

        n_spks = 2
        item_idx = 0
        while item_idx < len(items):
            
            is_multispk = random.random() < 0.6
            
            if not is_multispk:
                item_name = items[item_idx]['item_name']
                text = items[item_idx]['txt']
                if is_chinese(text):
                    ref_item = random.choice(ref_wav_manifests['zh'])
                else:
                    ref_item = random.choice(ref_wav_manifests['en'])
                ref_wav_path = os.path.join('/mnt/bn/sa-ag-data/liruiqi/data/speech/robust_mega3/ref_251211', ref_item['wav_path'])
                ref_wav, _ = librosa.load(ref_wav_path, sr=sr)
                ref_wav_paths = [ref_wav_path]
                ref_wav = ref_wav[int(ref_item['start'] * sr): int(ref_item['end'] * sr)]
                ref_text = ref_item['text']
                spk_id = random.choice([1, 2, 3, 4])
                if random.random() < 0.5:
                    ref_text = f"<S{spk_id}>{ref_text}"
                    text = f"{text}</S{spk_id}>"
                else:
                    ref_text = f"<S{spk_id}>{ref_text}</S{spk_id}>"
                    text = f"<S{spk_id}>{text}</S{spk_id}>"

                item_idx += 1
                
            else:
                spk_ids = [1, 2, 3, 4]
                random.shuffle(spk_ids)
                spk_ids = spk_ids[:n_spks]
                n_turns = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                text_merged = ''
                item_names = []
                for t_i in range(n_turns):
                    item_names.append(items[item_idx]['item_name'])
                    text = items[item_idx]['txt']
                    spk_id = spk_ids[t_i % len(spk_ids)]
                    text_merged += f"<S{spk_id}>{text}</S{spk_id}>"
                    item_idx += 1
                    if item_idx >= len(items):
                        break
                item_name = '|||'.join(item_names)
                text = text_merged
                if is_chinese(text):
                    ref_item_ids = list(range(len(ref_wav_manifests['zh'])))
                    random.shuffle(ref_item_ids)
                    ref_item_ids = ref_item_ids[:n_spks]
                    ref_items = [ref_wav_manifests['zh'][ref_item_ids[i]] for i in range(n_spks)]
                else:
                    ref_item_ids = list(range(len(ref_wav_manifests['en'])))
                    random.shuffle(ref_item_ids)
                    ref_item_ids = ref_item_ids[:n_spks]
                    ref_items = [ref_wav_manifests['en'][ref_item_ids[i]] for i in range(n_spks)]
                ref_wavs = []
                ref_texts = []
                ref_wav_paths = []
                for spk_id, ref_item in zip(spk_ids, ref_items):
                    ref_wav_path = os.path.join('/mnt/bn/sa-ag-data/liruiqi/data/speech/robust_mega3/ref_251211', ref_item['wav_path'])
                    ref_wav, _ = librosa.load(ref_wav_path, sr=sr)
                    ref_wav_paths.append(ref_wav_path)
                    ref_wav = ref_wav[int(ref_item['start'] * sr): int(ref_item['end'] * sr)]
                    ref_wavs.append(ref_wav)
                    ref_texts.append(f"<S{spk_id}>{ref_item['text']}</S{spk_id}>")
                ref_wav = np.concatenate(ref_wavs, axis=0)
                ref_text = ''.join(ref_texts)
                
            if hparams.get('mix_text_pinyin', {}).get('enable', False):
                ref_text = augment_text_with_pinyin_s1s2_safe(ref_text, hparams)
                text = augment_text_with_pinyin_s1s2_safe(text, hparams)
            
            ref_text_tokens = cosyvoice2_text_tokenizer.encode(ref_text)
            ref_text_tokens = torch.tensor(ref_text_tokens).long()

            text_tokens = cosyvoice2_text_tokenizer.encode(text)
            text_tokens = torch.tensor(text_tokens).long()

            ref_tok_n = int(ref_text_tokens.numel())
            tgt_tok_n = int(text_tokens.numel())
            if ref_tok_n <= 1 or tgt_tok_n <= 0:
                skip_logger.update(1, 'bad_text_tokens')
                continue

            wav_len = float(len(ref_wav)) / float(ref_tok_n) * float(ref_tok_n + tgt_tok_n)
            if not math.isfinite(wav_len):
                skip_logger.update(1, 'nonfinite_wav_len')
                continue

            if wav_len > hparams['max_frames'] * hop_size:
                skip_logger.update(1, 'wav_len_exceed_max_frames')
                continue

            wav_len = int(round(wav_len * (1 + random.random() * 0.04 - 0.02)))
            if wav_len <= 0:
                skip_logger.update(1, 'nonpos_wav_len')
                continue

            # Align to frames_multiple and enforce sane length bounds
            wav_len = (wav_len // fm_wav) * fm_wav
            mel_len = wav_len // hop_size
            min_frames = int(hparams.get('min_frames', 1) or 1)
            max_frames = int(hparams.get('max_frames', 0) or 0)
            if max_frames > 0 and not (max_frames >= mel_len > min_frames):
                skip_logger.update(1, 'bad_wav_len')
                continue

            wav = np.zeros(wav_len, dtype=np.float32)
            wav[:len(ref_wav)] = ref_wav

            ctx_mask = torch.zeros((wav_len, 1), dtype=torch.float32)
            ctx_mask[:len(ref_wav)] = 1.0
            ctx_mask = ctx_mask[:: hop_size * vae_stride]
            
            latent_len = wav_len // hop_size // vae_stride
            
            text_tokens = torch.cat([ref_text_tokens, text_tokens], dim=0)

            # ===== spk_mask =====
            spk_mask = build_spk_mask_from_text_tokens(text_tokens, sx_patterns)
            if spk_mask.shape != text_tokens.shape:
                skip_logger.update(1); continue

            item_tgt = {
                'id': 0,
                'item_name': item_name,
                'wav': torch.from_numpy(wav),
                'wav_len': wav.shape[0],
                'tgt_text': text,
                'text': ref_text + text,
                'txt_tokens': text_tokens,
                'ctx_wav': torch.from_numpy(ref_wav),
                'ctx_mask': ctx_mask,
                'spk_mask': spk_mask,
                'len': latent_len,
                'ref_wav_paths': ref_wav_paths,
            }

            yield item_tgt
            skip_logger.step(1)
            
    def collater(self, samples):
        batch = super().collater(samples)

        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]

        if 'tgt_text' not in batch and 'tgt_text' in samples[0]:
            batch['tgt_text'] = [s['tgt_text'] for s in samples]
            
        if 'ref_wav_paths' not in batch and 'ref_wav_paths' in samples[0]:
            batch['ref_wav_paths'] = [s['ref_wav_paths'] for s in samples]
        
        return batch


def processer_fn_robust_mega3_text(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    items = []
    for item_ in raw_item:
        try:
            item = {}
            item['item_name'] = item_['item_name']
            item['txt'] = item_['text']
            for s in '‘’《》“”':
                item['txt'] = item['txt'].replace(s, '')
            for s in '：:、':
                item['txt'] = item['txt'].replace(s, ', ')
            items.append(item)
        except:
            continue
    return items


if __name__ == '__main__':
    ph = torch.tensor([145,  86,  50,  13,  44,  70,  28, 163,  57,  34,  50,  65,  28,  28,                                                                                                                     163, 100,   4,  70,  17,  28, 163,  50,  57,  28,  70,  44,  28, 145,                                                                                                                                             69,  40,  69,   7,  70,  30,  98,  39,  90, 163,  55,  31,  26,  17,                                                                                                                                             67,  20,  97,  31,  89, 163,  98,  10,  16,  60,  16,  17,  97,  43,                                                                                                                                    
        163,  57,   6,  54,   7,  53,  89, 148, 145,  69,  10,  57,  30,  69,
         40,  16,  17,  66,  28,  28, 163,  28,  16,  30,  16,  30, 148, 145,                       
         66,  89, 148, 145,  57,  28,  16,  17,  27,  72,  89, 163,  70,   3,
         98,   4, 163,  66,  44,  66,  44,  70,  44,   2])
    ph_tokenizer = TokenTextEncoder(None, vocab_list=PHONE_VOCAB, replace_oov='<UNK>')

    ph_tokens = ph_tokenizer.decode(ph.numpy()).split(' ')

    print(ph_tokens)