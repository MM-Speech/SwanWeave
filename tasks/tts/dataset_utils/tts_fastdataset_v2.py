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
from utils.audio.io import to_wav_bytes
from utils.text.split_text import get_word_list
from utils.text.ph_tone_convert import map_phone_to_tokendict
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text import is_chinese, is_english
from utils.text.pinyin_aug import augment_text_with_pinyin_advanced
from utils.service.file_service import FileQueueClient

from tasks.tts.dataset_utils.tts_datasets import MegaTTSDataset, FrontendLMDataset
from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur
from modules.tts.ar_dur.commons.nar_tts_modules import LengthRegulator

DEBUG = False

def valid_item_kv(item, k):
    return k in item and item[k] is not None

def merge_A2B(A2B, B_lens):
    token_lens_cumsum = np.cumsum([0] + B_lens[:-1])
    token_lens_cumsum = torch.LongTensor(token_lens_cumsum)
    for i in range(len(B_lens)):
        A2B[i] = A2B[i] + token_lens_cumsum[i]
    A2B = torch.cat(A2B, 0)
    return A2B

def raw_text_process(txt, wav=None, wav_len=None):
    txt = txt.strip()
    if txt.startswith('sil '):
        txt = txt[4:]
    txt = txt.replace(' sil ', ' ')
    txt = txt.replace(' ,', ',').replace(',,', ',').replace(' ，', '，').replace('， ', '，')
    txt = txt.replace(' .', '.').replace(' 。', '。').replace('。 ', '。').replace('。 ', '。')
    txt = txt.replace(' ?', '?').replace(' ？', '？').replace('？ ', '？').replace('？ ', '？')
    txt = txt.replace(' !', '!').replace(' ！', '！').replace('！ ', '！').replace('！ ', '！')
    txt = txt.replace(' ;', ',').replace(' ；', '，').replace('； ', '，').replace('； ', '，').replace(';', ',').replace('；', '，')
    txt = txt.replace(' :', ',').replace(' ：', '，').replace('： ', '，').replace(':', ',').replace('：', '，')
    txt = txt.replace(' 、', '，').replace('、 ', '，').replace('、', '，')
    txt = txt.replace('"', '').replace('“', '').replace('”', '')
    txt = txt.replace('- ', ' ')
    txt = txt.replace('+', ' ')
    txt = txt.replace('，。', '。').replace('。，', '。')
    txt = txt.replace(':。', '。').replace('：。', '。')
    txt = txt.replace('……', '，')
    txt = remove_spaces_between_chinese(txt)
    if txt[-1] not in '.,?!;。，？！；、':
        if is_chinese(txt):
            txt = txt + '。'
        else:
            txt = txt + '. '
    if wav is not None:
        wav_len = wav.shape[0]
    if len(get_word_list(txt)) > wav_len // hparams['hop_size'] // 4:
        return
    return txt

def get_hdfs_file(hdfs_path, save_path, hdfs_clients: dict = None):
    namespace = None
    if hdfs_path.startswith('hdfs://'):
        namespace = hdfs_path.split('://')[1].split('/')[0]
    if hdfs_clients is None:
        hdfs_clients = {}
    if namespace is not None:
        if namespace not in hdfs_clients:
            client = hdfs_clients[namespace] = HDFSClient(namespace=namespace)
        else:
            client = hdfs_clients[namespace]
    else:
        if 'default' not in hdfs_clients:
            client = hdfs_clients['default'] = HDFSClient()
        else:
            client = hdfs_clients['default']
    if not client.check_file_exists(hdfs_path):
        return None
    data = client.get_object(hdfs_path)
    # print(f"{client.namespace = }, {hdfs_path = }, {data is not None = }")
    if data is None:
        return
    os.makedirs(Path(save_path).parent, exist_ok=True)
    with open(save_path, 'wb') as f:
        f.write(data)
    return save_path

def safe_read_path(path, save_path, hdfs_clients: dict = None):
    if path.startswith('hdfs://'):
        return get_hdfs_file(path, save_path, hdfs_clients)
    else:
        return path

class BaseTTSShmDataset(BaseFalconReaderShmDataset):

    def controller_fn(self, ds_len, seed, q_to_pull, hparams_, max_epoch=0, n_processor=0):
        hparams.update(hparams_)
        setproctitle.setproctitle(f'data_processor:{hparams["exp_name"]}:controller_fn')
        print(f"| init controller")
        try:
            g = torch.Generator()
            g.manual_seed(seed)

            dataloader_use_weight = hparams.get('dataloader_use_weight', False)

            if dataloader_use_weight:

                packs = self.dataset_meta['datasets']
                offsets = torch.LongTensor([p['offset'] for p in packs])
                n_chunks = torch.LongTensor([p['n_chunks'] for p in packs])
                weights = torch.tensor([float(p.get('weight', 1.0)) for p in packs], dtype=torch.float)

                active = (n_chunks > 0)
                probs = torch.where(active, weights, torch.zeros_like(weights))
                s = probs.sum()
                probs = probs / (s if s > 0 else 1.0)

                # 为每个数据集准备循环 randperm 与指针（持久化跨 epoch，避免“新一轮又从头覆盖之前访问过的chunk”）
                if not hasattr(self, '_ds_chunk_perm'):
                    self._ds_chunk_perm = []
                    for i in range(len(packs)):
                        nc = n_chunks[i].item()
                        if nc > 0:
                            self._ds_chunk_perm.append(torch.randperm(nc, generator=g))
                        else:
                            self._ds_chunk_perm.append(torch.empty(0, dtype=torch.long))
                if not hasattr(self, '_ds_chunk_pos'):
                    self._ds_chunk_pos = torch.zeros(len(packs), dtype=torch.long)

                def build_weighted_indices_looping(generator):
                    out = []
                    # 如果所有可用权重为0，直接返回空（防御）
                    if probs.sum() <= 0:
                        return out

                    # 生成本轮所需个数
                    for _ in range(ds_len):
                        # 选数据集
                        ds_id = torch.multinomial(probs, 1, replacement=True, generator=generator).item()

                        # 如果被选中数据集没有 chunk（极少见于配置），跳过寻找下一个
                        if n_chunks[ds_id].item() == 0:
                            continue

                        # 取该数据集的下一个 chunk（循环 randperm）
                        pos = int(self._ds_chunk_pos[ds_id].item())
                        nc = n_chunks[ds_id].item()
                        # pos 可能由于多节点切片等原因偏大，做一次取模保护
                        pos_mod = pos % nc
                        chunk_local = int(self._ds_chunk_perm[ds_id][pos_mod].item())
                        self._ds_chunk_pos[ds_id] = pos + 1

                        # 如果刚好走完一轮，重洗并从头再循环
                        if (pos_mod + 1) >= nc:
                            self._ds_chunk_perm[ds_id] = torch.randperm(nc, generator=generator)
                            # pos+1 已经写回；下一次会从0开始按新顺序取

                        out.append(int(offsets[ds_id].item() + chunk_local))
                    return out

                indices = build_weighted_indices_looping(g)
                indices = torch.tensor(indices, dtype=torch.long)
                if self.node_id is not None:
                    indices = indices[self.node_id::self.node_size]
                indices = indices.tolist()

            else:

                indices = torch.randperm(ds_len, generator=g).tolist()
                if self.node_id is not None:
                    indices = indices[self.node_id::self.node_size]

            pull_i = 0
            epoch = 0
            while max_epoch <= 0 or epoch < max_epoch:
                while not q_to_pull.full():
                    if pull_i == len(indices):
                        epoch += 1
                        if dataloader_use_weight:
                            # 新一轮继续生成（注意：不重置 _ds_chunk_pos，从而跨 epoch 继续循环覆盖）
                            new_indices = build_weighted_indices_looping(g)
                            new_indices = torch.tensor(new_indices, dtype=torch.long)
                            if self.node_id is not None:
                                new_indices = new_indices[self.node_id::self.node_size]
                            indices = new_indices.tolist()
                        else:
                            indices = torch.randperm(ds_len, generator=g).tolist()
                            if self.node_id is not None:
                                indices = indices[self.node_id::self.node_size]
                        pull_i = 0
                        break
                    q_to_pull.put(indices[pull_i])
                    pull_i += 1
                if DEBUG:
                    print("controller: q_to_pull满了, 休息1s等等processor")
                time.sleep(1)
            for i in range(n_processor * 2):
                q_to_pull.put(None)
            print("| Controller worker finished...")
        except:
            traceback.print_exc()
            
    def get_binary_reader(self, data_paths, reader_chunk_size, worker_id=0, worker_world_size=1, reader_cache_name='cache'):
        fd_cache_size = 1024
        io_thread_num = 64
        io_retry = 5
        reader = FalconReader(data_paths, fd_cache_size, io_thread_num, io_retry, 
                            reader_cache_name, worker_world_size, worker_id, reader_chunk_size)
        ds_len = reader.get_entry_num(list(range(len(data_paths))), False)
        return reader, ds_len

    def get_manifest_reader(self, data_paths, reader_chunk_size, worker_id=0, worker_world_size=1, reader_cache_name='cache'):
        return {}
    
    def get_dataset_meta(self):
        cluster = os.environ.get('CLUSTER', '').lower()
        dataset_meta = {
            'datasets': []
        }
        total_ds_len = 0
        idx_offset = 0
        reader_chunk_size = hparams.get('reader_chunk_size', 64)
        if not hasattr(self, 'hdfs_clients'):
            self.hdfs_clients = {}
        print(f'| training datasets:')

        with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:

            hp_datasets = hparams['datasets']
            if hparams.get('is_debug_run', False):
                hp_datasets = hparams.get('debug_run_dataset', hp_datasets)
                print_once('| Use debug_run_dataset')
                
            for dataset_group_name in hp_datasets:

                if hp_datasets[dataset_group_name] is None:
                    continue

                print(f'| - dataset group [{dataset_group_name}]')
                dataset_group = hp_datasets[dataset_group_name]
                dataset_processer_fn = import_module_bystr(dataset_group['processer_fn'])
                dataset_group_weight = float(dataset_group.get('weight', 1.0))

                if dataset_group.get('binary_data_root'):
                    if f"{cluster}_hdfs" in dataset_group['binary_data_root']:
                        binary_data_root = dataset_group['binary_data_root'][f"{cluster}_hdfs"]
                    elif f"{cluster}_nas" in dataset_group['binary_data_root']:
                        binary_data_root = dataset_group['binary_data_root'][f"{cluster}_nas"]
                    else:
                        binary_data_root = dataset_group['binary_data_root']["default"]
                else:
                    binary_data_root = ''
                
                for rel_path in dataset_group['train_sets']:
                    if rel_path.endswith('.json'):
                        manifest_path = os.path.join(binary_data_root, rel_path)
                        manifest = json.load(open(safe_read_path(manifest_path, os.path.join(temp_dir, rel_path), self.hdfs_clients)))
                        ds_len = len(manifest)
                        dataset_meta_ = {
                            'data_path': manifest_path,
                            'manifest': manifest,
                            'reader_type': 'manifest'
                        }
                    elif rel_path.endswith('.jsonl'):
                        manifest_path = os.path.join(binary_data_root, rel_path)
                        if not manifest_path.startswith('hdfs://') and dataset_group.get('read_idx', True):  # jsonl.idx can't be in hdfs
                            with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
                                jsonl_idx_path = manifest_path + '.idx'
                                if not os.path.isfile(jsonl_idx_path):
                                    if self.node_id is not None and self.node_id > 0:
                                        jsonl_idx_path = os.path.join(temp_dir, 'jsonl.idx')
                                    print(f'| building jsonl idx: {jsonl_idx_path}')
                                    build_jsonl_index(manifest_path, jsonl_idx_path, use_tqdm=False)
                                ds_len = count_jsonl_n_lines(jsonl_idx_path)
                                dataset_meta_ = {
                                    'data_path': manifest_path,
                                    'reader_type': 'jsonl_idx',
                                }
                        else:
                            manifest = load_samples_from_jsonl(safe_read_path(manifest_path, os.path.join(temp_dir, rel_path), self.hdfs_clients))
                            ds_len = len(manifest)
                            dataset_meta_ = {
                                'data_path': manifest_path,
                                'manifest': manifest,
                                'reader_type': 'manifest'
                            }
                    elif rel_path.endswith('.tsv'):
                        manifest_path = os.path.join(binary_data_root, rel_path)
                        manifest = load_samples_from_tsv(safe_read_path(manifest_path, os.path.join(temp_dir, rel_path), self.hdfs_clients))
                        ds_len = len(manifest)
                        dataset_meta_ = {
                            'data_path': manifest_path,
                            'manifest': manifest,
                            'reader_type': 'manifest'
                        }
                    elif rel_path.endswith('.parquet') or rel_path.endswith('.pq'):
                        manifest_path = os.path.join(binary_data_root, rel_path)
                        reader_chunk_size_ = dataset_group.get('reader_chunk_size', reader_chunk_size)
                        reader = ParquetChunkReader(manifest_path, reader_chunk_size_)
                        ds_len = len(reader)
                        dataset_meta_ = {
                            'data_path': manifest_path,
                            'reader_type': 'pq_reader'
                        }
                    else:
                        binary_data_path = os.path.join(binary_data_root, rel_path, 'data')
                        reader_chunk_size_ = dataset_group.get('reader_chunk_size', reader_chunk_size)
                        _, ds_len = self.get_binary_reader([binary_data_path], reader_chunk_size_)
                        n_chunks = math.ceil(ds_len / reader_chunk_size_)
                        dataset_meta_ = {
                            'data_path': binary_data_path,
                            'reader_type': 'binary'
                        }
                        
                    reader_chunk_size_ = dataset_group.get('reader_chunk_size', reader_chunk_size)
                    n_chunks = math.ceil(ds_len / reader_chunk_size_)
                    dataset_meta_.update({
                        'ds_len': ds_len,
                        'n_chunks': math.ceil(ds_len / reader_chunk_size_),
                        'offset': idx_offset,
                        'processer_fn': dataset_processer_fn,
                        'reader_chunk_size': reader_chunk_size_,
                        'weight': dataset_group_weight / len(dataset_group['train_sets'])
                    })

                    dataset_meta['datasets'].append(dataset_meta_)
                    total_ds_len += ds_len
                    idx_offset += n_chunks
                    print(f'|   - {dataset_meta_["data_path"]}')
                    print(f'|     - length: {ds_len}')
                    print(f'|     - reader_chunk_size: {reader_chunk_size_}')
                    print(f'|     - n_chunks: {n_chunks}')
        print(f"| Total training data length: {total_ds_len}")
        print(f"| Total num chunks: {idx_offset}")
        return dataset_meta, idx_offset
    
    
    def process_fn(
            self, q_to_pull, q_to_push, world_size,
            shm_base, counter, hparams_, seed, i_worker, n_worker
    ):
        hparams.update(hparams_)
        setproctitle.setproctitle(f'data_processor:{hparams["exp_name"]}:processor_fn#{i_worker}/{n_worker}')
        self.seed = seed
        print(f"| Starting processor_fn_worker#{i_worker}/{n_worker}.")

        try:
            global_stores = {}
            reader_pack = self.prepare_reader(self.dataset_meta, global_stores, i_worker, n_worker)
            print(f"| init processor (dataset_reader)#{i_worker}/{n_worker}.")
            restart_countdown = 10000
            while True:
                try:
                    idx = q_to_pull.get()
                    if idx is None:
                        return None
                    for item in self.process_item(idx, reader_pack, global_stores, hparams, i_worker, n_worker):
                        if isinstance(item, tuple):
                            item, item_meta = item
                        else:
                            item_meta = ''
                        item = convert_to_np(item)
                        with counter.get_lock():
                            cnt = counter.value
                            counter.value += 1
                        out_path = save_samples_to_shm(item, cnt, shm_base, item_meta)
                        if DEBUG:
                            print(f"processor#{i_worker}/{n_worker}: save to {out_path}")
                        q_to_push.put(out_path)
                        restart_countdown -= 1
                        if restart_countdown == 0:
                            return
                        while q_to_push.qsize() > self.shuffle_buffer * world_size * 2:
                            if DEBUG:
                                print(
                                    f"processor#{i_worker}/{n_worker}: q_to_push里面积压的太多了, 休息1s等等batch_saver")
                            time.sleep(1)
                        self.after_process_item(item, hparams, global_stores)
                except:
                    traceback.print_exc()
                    continue
        except:
            traceback.print_exc()
    
    def prepare_reader(self, dataset_meta, global_stores, i_worker, n_worker):
        reader_pack = []
        for dataset_meta_ in dataset_meta['datasets']:
            reader_pack_ = {
                'ds_len': dataset_meta_['ds_len'],
                'n_chunks': dataset_meta_['n_chunks'],
                'offset': dataset_meta_['offset'],
                'data_path': dataset_meta_['data_path'],
                'processer_fn': dataset_meta_['processer_fn'],
                'reader_chunk_size': dataset_meta_['reader_chunk_size'],
                'reader_type': dataset_meta_['reader_type'],
            }
            if reader_pack_['reader_type'] == 'binary':
                reader_pack_['reader'] = self.get_binary_reader([dataset_meta_['data_path']], dataset_meta_['reader_chunk_size'])[0]
            elif reader_pack_['reader_type'] == 'jsonl_idx':
                reader_pack_['reader'] = JsonlChunkReader(dataset_meta_['data_path'], dataset_meta_['data_path'] + '.idx')
            elif reader_pack_['reader_type'] == 'pq_reader':
                reader_pack_['reader'] = ParquetChunkReader(dataset_meta_['data_path'], dataset_meta_['reader_chunk_size'])
            reader_pack.append(reader_pack_)
        return reader_pack
    
    def read_fn(self, idx, reader_pack, global_stores):
        """
        idx: controller 传进来的全局 chunk index（跨所有数据集的 chunk 下标）
        """
        # 1. 找到对应的数据集 reader & reader_idx
        reader = None
        reader_idx = None
        for i, rp in enumerate(reader_pack):
            start = rp['offset']                  # 该数据集 chunk 全局起始下标
            end = start + rp['n_chunks']         # 该数据集 chunk 全局结束下标（开区间）
            if start <= idx < end:
                reader = rp
                reader_idx = i
                break

        if reader is None:
            print(f"| [read_fn] invalid global chunk idx: {idx}")
            return

        try:
            # 2. 计算该数据集内的本地 chunk index 和样本起始下标
            local_chunk_idx = idx - reader['offset']
            sample_start = local_chunk_idx * reader['reader_chunk_size']
            chunk_size = reader['reader_chunk_size']

            # 3. 按不同 reader_type 读取
            if reader['reader_type'] == 'binary':
                # FalconReader.read_many 的参数是 sample 下标（不是 chunk 下标）
                items = [pickle.loads(x) for x in reader['reader'].read_many([sample_start])[0]]

            elif reader['reader_type'] == 'manifest':
                # manifest 已经在 dataset_meta 里完整加载，直接 slice
                manifest = self.dataset_meta['datasets'][reader_idx]['manifest']
                items = manifest[sample_start: sample_start + chunk_size]

            elif reader['reader_type'] == 'jsonl_idx':
                # JsonlChunkReader.read_range 的下标是行号（样本下标）
                start_line = sample_start
                end_line = min(reader['ds_len'] - 1, sample_start + chunk_size - 1)
                items = reader['reader'].read_range(start_line, end_line)

            elif reader['reader_type'] == 'pq_reader':
                # ParquetChunkReader 一般是按 chunk 编号读
                items = reader['reader'].read_chunk(local_chunk_idx)

            else:
                raise ValueError(f"Unknown reader_type: {reader['reader_type']}")

            return items, reader['processer_fn']

        except:
            traceback.print_exc()
            return

        
    def get_batcher(self, hparams, global_stores):
        batcher = get_from_global_stores(
            'batcher', global_stores,
            lambda: BucketBatcher(
                buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 
                            600, 650, 700, 750, 800, 850, 900, 950, 1000, 1200, 1400, 
                            1600, 1800, 2000, 2400, 2800, 3000, 4000, 5000, 6000, 8000, 10000],
                dynamic_batch=hparams.get("dynamic_batch", True),
                batch_size=hparams['max_sentences'],
                maximum_bucket_size=hparams['max_tokens'],
                length_fn=lambda x: x['len'],
            )
        )
        return batcher
    
    def process_item(self, index, reader_pack, global_stores, hparams, i_worker, n_worker):
        
        if DEBUG:
            print(f'processer {i_worker}/{n_worker}: {index = }')
        
        def init_new_batch():
            tgt_size = random.randint(hparams['tgt_size_min'], hparams['tgt_size_max'])
            return tgt_size
        
        read_res = self.read_fn(index, reader_pack, global_stores)
        if read_res is None:
            return
        raw_item, processer_fn = read_res
        
        if self.use_fast_dataloader:
            batcher = self.get_batcher(hparams, global_stores)
            tgt_size = init_new_batch()
        
        for item in self._process_item(processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):
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
            
        if hparams.get('add_vad_mask', False):
            from utils.audio.vad import get_vad_model
            vad_model = get_from_global_stores(
                'vad_model', global_stores,
                lambda: get_vad_model()
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
            
            # txt = raw_text_process(item_tgt['txt'], wav_len=item_tgt['wav_len'])
            txt = item_tgt['txt']
            if txt is None:
                skip_logger.update(1); continue
            item_tgt['text'] = txt
            item_tgt['orig_text'] = deepcopy(item_tgt['text'])
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

            if hparams.get('use_cosyvoice2_text_tokenizer', False):
                if hparams.get('mix_text_pinyin', {}).get('enable', False):
                    item_tgt['text'] = augment_text_with_pinyin_advanced(
                        item_tgt['text'],
                        p_augment=hparams.get('mix_text_pinyin', {}).get('enable_prob', 0.3),
                        p_bernoulli_mode=0.1,
                        poly_weight_bernoulli=3.0,
                        ratio_gamma=3.0,
                        poly_weight_ratio=3.0,
                        tone3=True,
                        pinyin_tokenizer=lambda x: f"<|py_{x}|>"
                    )
                text_tokens = cosyvoice2_text_tokenizer.encode(item_tgt['text'])
                text_tokens = torch.tensor(text_tokens).long()
                item_tgt['txt_tokens'] = text_tokens
                    
            if hparams.get('use_sparse_dur', False):
                mel2ph_sparse = compute_mel2aug_from_dur(
                    item_tgt['dur'].numpy().tolist(),
                    gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                    gap_frames=hparams.get('sparse_dur_frames', 4),
                    gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                    min_keep=hparams.get('sparse_dur_min_keep', 1),
                    keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                    symmetric=hparams.get('sparse_dur_symmetric', True),
                )
                item_tgt['mel2ph_sparse'] = mel2ph_sparse
                
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
            item_tgt['ctx_wav'] = item_tgt['wav'][:ctx_len]   # 只是 view，不会复制整段 wav
            
            if hparams.get('add_vad_mask', False):
                from utils.audio.vad import run_vad_trim
                vad_start, vad_end = run_vad_trim(item_tgt['wav'], hparams['audio_sample_rate'], vad_model)
                vm = hparams['hop_size'] * hparams['vae_stride']
                vad_mask = np.zeros((item_tgt['wav'].shape[0] // vm))
                vad_mask[int(vad_start * hparams['audio_sample_rate'] // vm): int(vad_end * hparams['audio_sample_rate'] // vm)] = 1
                item_tgt['vad_mask'] = vad_mask # 直接是lat的shape
            else:
                item_tgt['vad_mask'] = None
                
            item_tgt['len'] = mel_len // 4
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
        ctx_wavs = collate_xd([s['ctx_wav'] for s in samples], 0.0) if 'ctx_wav' in samples[0] and samples[0]['ctx_wav'] is not None else None
        ctx_wav_lengths = torch.LongTensor([s['ctx_wav'].shape[0] for s in samples]) if ctx_wavs is not None else None
        if 'vad_mask' in samples[0] and samples[0]['vad_mask'] is not None:
            vad_mask = collate_xd([s['vad_mask'] for s in samples], 0.0)[..., None]
        else:
            vad_mask = None
        batch = {
            'item_name': [s['item_name'] for s in samples],
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
            'ctx_wavs': ctx_wavs,
            'ctx_wav_lengths': ctx_wav_lengths,
            'vad_mask': vad_mask
        }
        if valid_item_kv(samples[0], 'mel'):
            batch['mels'] = collate_xd([s['mel'] for s in samples], -6.0)
        if 'mel2ph' in samples[0]:
            batch['mel2ph'] = collate_xd([s['mel2ph'] for s in samples], 0)
        if 'dur' in samples[0]:
            batch['dur'] = collate_xd([s['dur'] for s in samples], 0)
            batch['dur_len'] = torch.LongTensor([s['dur'].shape[0] for s in samples])
        if 'mel2ph_sparse' in samples[0]:
            batch['mel2ph_sparse'] = collate_xd([s['mel2ph_sparse'] for s in samples], 0)
        if valid_item_kv(samples[0], 'ctx_mask'):
            batch['ctx_mask'] = collate_xd([s['ctx_mask'] for s in samples], 0)
        batch['text'] = [s['text'] for s in samples]
        if valid_item_kv(samples[0], 'orig_text'):
            batch['orig_text'] = [s['orig_text'] for s in samples]
        if 'caption' in samples[0]:
            batch['caption'] = [s['caption'] for s in samples]
        if 'global' in samples[0]:
            batch['global'] = [s['global'] for s in samples]
        if 'local' in samples[0]:
            batch['local'] = [s['local'] for s in samples]
        if 'caption_audio' in samples[0]:
            batch['caption_audio'] = [s['caption_audio'] for s in samples]
        if 'ph_token' in samples[0]:
            batch['ph_tokens'] = collate_xd([s['ph_token'] for s in samples], 0)
            batch['txt_lengths'] = torch.LongTensor([s['ph_token'].numel() for s in samples])
        if 'tone' in samples[0]:
            batch['tone'] = collate_xd([s['tone'] for s in samples], 0)
        if 'ph_timestamp' in samples[0]:
            batch['ph_timestamp'] = collate_xd([s['ph_timestamp'] for s in samples], 797)
            batch['ph_timestamp_len'] = torch.LongTensor([s['ph_timestamp'].shape[0] for s in samples])
        if 'merged_ph_token' in samples[0]:
            batch['merged_ph_tokens'] = collate_xd([s['merged_ph_token'] for s in samples], 797)
            batch['merged_ph_tokens_len'] = torch.LongTensor([s['merged_ph_token'].shape[0] for s in samples])
        if 'ph_dur_seq' in samples[0]:
            batch['ph_dur_seqs'] = collate_xd([s['ph_dur_seq'] for s in samples], 797)
            batch['ph_dur_seqs_len'] = torch.LongTensor([s['ph_dur_seqs'].shape[0] for s in samples])
            batch['ph_dur_seq_dur_mask'] = collate_xd([s['ph_dur_seq_dur_mask'] for s in samples], 0)
        if 'spk_mask' in samples[0]:
            batch['spk_mask'] = collate_xd([s['spk_mask'] for s in samples], 0)
        if 'audio_mask' in samples[0]:
            batch['audio_mask'] = collate_xd([s['audio_mask'] for s in samples], 0)
        if 'ctx_ph_mask' in samples[0]:
            batch['ctx_ph_mask'] = collate_xd([s['ctx_ph_mask'] for s in samples], 0)
        if valid_item_kv(samples[0], 'txt_tokens'):
            batch['txt_tokens'] = collate_xd([s['txt_tokens'] for s in samples], 0)
            batch['txt_lengths'] = torch.LongTensor([s['txt_tokens'].numel() for s in samples])
        
        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch
    

class TTSTextOnlyShmDataset(BaseTTSShmDataset):
    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):

        hop_size = hparams['hop_size']
        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']
        sr = hparams['audio_sample_rate']

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
        if hparams.get('online_text_alignment', False):
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
                    asr_results = asr_client.wait_result(job_id, poll_s=0.5, timeout_s=600)
                    if asr_results is None:
                        print(f'asr results is None, job_id: {job_id}')
                        texts_aligned = [None] * job_id2batch_size[job_id]
                    else:
                        texts_aligned = asr_results['result']['asr_results']['pause_punct_texts']
                        assert len(texts_aligned) == job_id2batch_size[job_id], f"asr results len {len(texts_aligned)} != batch size {job_id2batch_size[job_id]}"
                except TimeoutError:
                    traceback.print_exc()
                    texts_aligned = [None] * job_id2batch_size[job_id]
                except:
                    traceback.print_exc()
                    texts_aligned = [None] * job_id2batch_size[job_id]
                texts_aligned_total.extend(texts_aligned)
            assert len(texts_aligned_total) == len(items), f"texts_aligned_total len {len(texts_aligned_total)} != items len {len(items)}"
            for item_idx in range(len(items)):
                if texts_aligned_total[item_idx] is not None:
                    items[item_idx]['txt'] = texts_aligned_total[item_idx]
        
        ########################
        # task specific process #
        ########################
        for item_tgt in items:
            if not (hparams['max_frames'] >= item_tgt['wav_len'] // hop_size > hparams['min_frames']):
                skip_logger.update(1); continue
            
            # txt = raw_text_process(item_tgt['txt'], wav_len=item_tgt['wav_len'])
            txt = item_tgt['txt']
            if txt is None:
                skip_logger.update(1); continue
            item_tgt['text'] = txt
            
            if hparams.get('load_wav', True):
                item_tgt['wav'] = item_tgt['wav'][:len(item_tgt['wav']) // fm_wav * fm_wav]
            mel_len = len(item_tgt['wav']) // hop_size

            if hparams.get('use_cosyvoice2_text_tokenizer', False):
                if hparams.get('mix_text_pinyin', {}).get('enable', False):
                    item_tgt['text'] = augment_text_with_pinyin_advanced(
                        item_tgt['text'],
                        p_augment=hparams.get('mix_text_pinyin', {}).get('enable_prob', 0.3),
                        p_bernoulli_mode=0.1,
                        poly_weight_bernoulli=3.0,
                        ratio_gamma=3.0,
                        poly_weight_ratio=3.0,
                        tone3=True,
                        pinyin_tokenizer=lambda x: f"<|py_{x}|>"
                    )
                text_tokens = cosyvoice2_text_tokenizer.encode(item_tgt['text'])
                text_tokens = torch.tensor(text_tokens).long()
                item_tgt['txt_tokens'] = text_tokens
                
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
            item['phone'] = torch.LongTensor(item_['phone_encoded'])
            item['tone'] = torch.LongTensor(item_['tone_encoded'])
            item['mel2ph'] = torch.LongTensor(item_['mel2ph'])
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
    length_regulator = get_from_global_stores(
        'length_regulator', global_stores,
        lambda: LengthRegulator()
    )
    sr = hparams['audio_sample_rate']
    
    with tempfile.TemporaryDirectory(dir='/dev/shm') as temp_dir:
        items = []
        for item_ in raw_item:
            try:
                item_name = item_['item_name']
                feat_k = item_['feat_k']
                vocal_k = item_['vocal_k']
                subset = ['subset']
                
                if hparams.get('load_wav', True):
                    # if not tos_client.check_tos_file_exists(vocal_k):
                    #     if DEBUG:
                    #         print(f'processer {i_worker}/{n_worker}: key not exists: {vocal_k}, {item_ = }')
                    #     skip_logger.update(1); continue
                    # try:
                    data = tos_client.get_object(vocal_k, verbose=False)
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
                    # except Exception as err:
                    #     traceback.print_exc()
                    #     # handle_exacption(err)
                        # skip_logger.update(1); continue
                
                for segment_idx, segment_meta in enumerate(item_['segments_1spk']):
                    item = {}
                    
                    if hparams.get('load_wav', True):
                        wav_start, wav_end = segment_meta['start'], segment_meta['end']
                        wav = global_wav[int(wav_start * sr_): int(wav_end * sr_)]
                        if len(wav) == 0:
                            continue
                        if sr_ != sr:
                            wav = librosa.resample(wav, orig_sr=sr_, target_sr=sr)
                        item['wav'] = torch.FloatTensor(wav)
                        item['wav_len'] = wav.shape[0]
                    else:
                        item['wav_len'] = int(segment_meta['sec'] * sr)
                        
                    item['item_name'] = item_name + '#' + f'{segment_idx}'
                    
                    if segment_meta.get('phone_encoded') is None:
                        continue
                    item['phone'] = torch.LongTensor(segment_meta['phone_encoded'])
                    item['tone'] = torch.LongTensor(segment_meta['tone_encoded'])
                    item['dur'] = torch.LongTensor(segment_meta['dur'])
                    item['mel2ph'] = length_regulator(item['dur'][None, :])[0]
                    txt = raw_text_process(segment_meta['txt_raw'], wav_len=item['wav_len'])
                    if txt is None:
                        continue
                    item['txt'] = txt
                    item['spk_name'] = item_name + '#' + segment_meta['spk_name']
                    
                    items.append(item)
                
            except:
                traceback.print_exc()
                continue
            
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
            txt = raw_text_process(item_['txt_raw'], wav_len=item['wav_len'])
            if txt is None:
                continue
            item['txt'] = txt
            ds_name = item_['ds_name']
            item['spk_name'] = f"{ds_name}#{item_['spk']}"
            item['skip_merge_same_spk'] = True
            items.append(item)
        except:
            continue
    return items


if __name__ == '__main__':
    from utils.commons.hparams import set_hparams, hparams
    set_hparams(
        'egs/tts/megatts3_dit_v2_dataloader_v2.yaml', 
        print_hparams=False, 
    )
    exp_name = 'test_DiTT2ADataset'
    hparams.update(dict(
        exp_name=exp_name,
        sp_size=1,
        ds_workers=8,
        debug=True,
        fast_ds_shuffle_buffer=32,
        max_sentences=5,
        max_tokens=2000,
        frames_multiple=8
    ))

    ds_train = BaseTTSShmDataset('train', hparams, use_fast_dataloader=True, rank_id=0, world_size=1, batch_size=1)
    dl_train = ds_train.get_dataloader(seed=1234, num_workers=hparams['ds_workers'])
    for i, items in enumerate(dl_train):
        if 'ph_tokens' in items:
            print(items)
            break
        
    # tos_client = TosClient(bucket='humanaigc-ads')
    # vocal_k = 'tts_datasets/zhiyuexingchen/cn/podcast/apple/2CK5BNKN/apple_podcasts/cn_41/audio_01/1539659953/rssFileVip_89_features/vocal.m4a'
    # print(f"{tos_client.check_tos_file_exists(vocal_k) = }")
    
    # length_regulator = LengthRegulator()
    # dur = torch.LongTensor([ 26,  12,   9,  12,  30,  20,  35,  81,   0,  17,   6,  12,   0,  10,
    #       7,  11,   5,  39,  25,  16,  16,  15,  10,  15,   2,  17,  65,   0,
    #      63, 201,  19,   9,  12,  41,  20,  10,   6,  59,  67,   4,   8,  11,
    #      13,  13,   4,  13,  12,   9,   7,   4,  16,   7,   5,   7,  11,   6,
    #      13,  11, 114,   0,  76,  62,  10,  11,   4,   8,  27,  26,  22,  21,
    #      10,   6,  13,  10,   9,  14,  12,   4,   4,  15,   6,  15,   7, 124,
    #       8,  13,  13,  28,   9,   8,  11,   6,  16,  11,   6,  17,  19,  11,
    #      11,   6,   7,  16,   5,   6,  10,  14,   4,  10,   6,   8,   7,  14,
    #       6,  18,   8,  52])
    # mel2ph = length_regulator(dur[None, :])
    # print(mel2ph)
