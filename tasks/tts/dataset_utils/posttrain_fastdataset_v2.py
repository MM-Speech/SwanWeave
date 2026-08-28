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

class RepeatSampler(torch.utils.data.Sampler):
    def __init__(self, ds_len, repeat=1, shuffle=False, seed=0):
        self.ds_len = ds_len
        self.repeat = repeat
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.ds_len * self.repeat
    
    def __iter__(self):
        n = self.ds_len
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(n, generator=g).tolist()
        else:
            order = list(range(n))
        for i in order:
            for _ in range(self.repeat):
                yield i
    
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

        dataloader_num_replica = hparams.get('dataloader_num_replica', 1)

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
                repeat_k=hparams['sample']['num_generation_per_prompt'],
                batch_size=hparams['sample']['train_batch_size'],
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

        ph_tokenizer: TokenTextEncoder = get_from_global_stores(
            'ph_tokenizer', global_stores, 
            lambda: TokenTextEncoder(None, vocab_list=PHONE_VOCAB, replace_oov='<UNK>')
        )
        sil_ph = get_from_global_stores(
            'sil_ph', global_stores, 
            lambda: ph_tokenizer.sil_phonemes()
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
                'phone': torch.cat([s['phone'] for s in samples], 0),
                'wav': torch.cat([s['wav'] for s in samples], 0) if valid_item_kv(samples[0], 'wav') else None,
                'mel2ph': merge_A2B([s['mel2ph'] for s in samples], [len(s['phone']) for s in samples]),
                'tone': torch.cat([s['tone'] for s in samples], 0),
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
            
            txt = raw_text_process(item_tgt['txt'], wav_len=item_tgt['wav_len'])
            if txt is None:
                skip_logger.update(1); continue
            item_tgt['text'] = txt
            item_tgt['ph_token'] = item_tgt['phone']
            if item_tgt['ph_token'].shape[0] > item_tgt['wav_len'] // hop_size // 4:
                skip_logger.update(1); continue
            
            if hparams.get('load_wav', True):
                item_tgt['wav'] = item_tgt['wav'][:len(item_tgt['wav']) // fm_wav * fm_wav]
                if speech_augmentor is not None:
                    item_tgt['wav'] = speech_augmentor(item_tgt['wav'], sr)
                    
            mel_len = len(item_tgt['wav']) // hop_size
            if mel_len > len(item_tgt['mel2ph']):
                mel2ph = torch.zeros(mel_len).long()
                mel2ph[:len(item_tgt['mel2ph'])] = item_tgt['mel2ph']
                mel2ph[len(item_tgt['mel2ph']):] = mel2ph[len(item_tgt['mel2ph'])-1]
                item_tgt['mel2ph'] = mel2ph
            
            item_tgt['mel2ph'] = item_tgt['mel2ph'][:len(item_tgt['mel2ph']) // fm * fm]
            if 'dur' not in item_tgt:
                item_tgt['dur'] = mel2token_to_dur(item_tgt['mel2ph'])

            if hparams.get('load_wav', True) and len(item_tgt['mel2ph']) != len(item_tgt['wav']) // hop_size:
                skip_logger.update(1); continue
                
            if hparams.get('use_ph_timestamp', False):
                try:
                    item_tgt['ph_timestamp'] = FrontendLMDataset.get_ph_timestamp(item_tgt)
                except:
                    skip_logger.update(1); continue
            if hparams.get('use_merged_ph', False):
                try:
                    item_tgt['merged_ph_token'] = map_phone_to_tokendict({
                        'phone': item_tgt['phone'], 'tone': item_tgt['tone']
                    }, pad_bos_eos=False)
                except:
                    skip_logger.update(1); continue
            if hparams.get('use_merged_ph', False) and 'dur' in hparams['task_cls']:
                if item_tgt['merged_ph_token'].shape[0] != item_tgt['dur'].shape[0]:
                    skip_logger.update(1); continue
            if hparams.get('valid_ph_dur', False):
                if item_tgt['phone'].shape[0] != item_tgt['dur'].shape[0]:
                    skip_logger.update(1); continue

            ph_offsets = np.cumsum(item_tgt['dur'].numpy())
            phone = ph_tokenizer.decode(item_tgt['ph_token'].numpy()).split(' ')
            sil_ph_idxs = [p_i for p_i in range(len(phone)) if phone[p_i] in sil_ph and 0 < p_i < len(phone) - 1]
            if len(sil_ph_idxs) == 0:
                skip_logger.update(1, 'no sil'); continue
            rand_ph_idx = random.choice(sil_ph_idxs)
            rand_length = round(ph_offsets[rand_ph_idx] / fm) * fm

            ctx_mask = torch.zeros((item_tgt['wav'].shape[0] // hparams['hop_size'], 1))
            ctx_mask[:rand_length] = 1.0
            item_tgt['ctx_mask'] = ctx_mask[::hparams['vae_stride']]
            item_tgt['ctx_wav'] = deepcopy(item_tgt['wav'])
            item_tgt['ctx_wav'] = item_tgt['ctx_wav'][:rand_length*hparams['hop_size']]

            ctx_ph_mask = torch.zeros_like(item_tgt['ph_token'])
            ctx_ph_mask[:rand_ph_idx + 1] = 1
            item_tgt['ctx_ph_mask'] = ctx_ph_mask
            
            item_tgt['vad_mask'] = None
                
            item_tgt['len'] = mel_len // 4
            yield item_tgt
            skip_logger.step(1)


if __name__ == '__main__':
    ph = torch.tensor([145,  86,  50,  13,  44,  70,  28, 163,  57,  34,  50,  65,  28,  28,                                                                                                                     163, 100,   4,  70,  17,  28, 163,  50,  57,  28,  70,  44,  28, 145,                                                                                                                                             69,  40,  69,   7,  70,  30,  98,  39,  90, 163,  55,  31,  26,  17,                                                                                                                                             67,  20,  97,  31,  89, 163,  98,  10,  16,  60,  16,  17,  97,  43,                                                                                                                                    
        163,  57,   6,  54,   7,  53,  89, 148, 145,  69,  10,  57,  30,  69,
         40,  16,  17,  66,  28,  28, 163,  28,  16,  30,  16,  30, 148, 145,                       
         66,  89, 148, 145,  57,  28,  16,  17,  27,  72,  89, 163,  70,   3,
         98,   4, 163,  66,  44,  66,  44,  70,  44,   2])
    ph_tokenizer = TokenTextEncoder(None, vocab_list=PHONE_VOCAB, replace_oov='<UNK>')

    ph_tokens = ph_tokenizer.decode(ph.numpy()).split(' ')

    print(ph_tokens)
    
