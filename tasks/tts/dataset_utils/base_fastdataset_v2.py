import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import os
import random
import json
import pickle
import traceback
import math
import time
import tempfile
from pathlib import Path
import faulthandler
import signal

import setproctitle
import torch
import numpy as np
import re
from dataloader import FalconReader

from utils.commons.import_utils import import_module_bystr
from utils.commons.hparams import hparams
from utils.commons.io import print_once, load_samples_from_tsv, load_samples_from_jsonl
from utils.commons.base_shm_dataset import BaseFalconReaderShmDataset, get_from_global_stores, save_samples_to_shm
from utils.commons.tensor_utils import  convert_to_np
from utils.commons.hdfs_utils import HDFSClient
from utils.commons.jsonl_utils import count_jsonl_n_lines, JsonlChunkReader, build_jsonl_index
from utils.commons.parquet_utils import ParquetChunkReader
from utils.dataset.batcher import BucketBatcher
from utils.text.split_text import get_word_list
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text import is_chinese

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
    txt = txt.replace('"', '').replace('"', '').replace('"', '')
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
    if wav_len is not None and len(get_word_list(txt)) > wav_len // hparams['hop_size'] // 4:
        return
    return txt

def simple_text_process(txt, wav=None, wav_len=None):
    txt = txt.strip()
    if txt.startswith('sil '):
        txt = txt[4:]
    txt = txt.replace(' sil ', ' ')
    txt = remove_spaces_between_chinese(txt)
    if txt[-1] not in '.,?!;。，？！；、':
        if is_chinese(txt):
            txt = txt + '。'
        else:
            txt = txt + '. '
    if wav is not None:
        wav_len = wav.shape[0]
    if wav_len is not None and len(get_word_list(txt)) > wav_len // hparams['hop_size'] // 4:
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

_S_TAG_RE = re.compile(r"</?(S\d+)>")
# 匹配所有 speaker 引用：<S1>、</S1>、Speaker1、Speaker 1
_SPEAKER_REF_RE = re.compile(r'</?(S(\d+))>|Speaker\s*(\d+)')


def build_speaker_shuffle_map(text: str, keep_ids=None) -> dict:
    """从 text 中找 <Sx> 标签，生成随机 shuffle 映射。
    返回 {'S1': 'S2', 'S2': 'S1'} 或 None（不需要 shuffle 时）。
    """
    if keep_ids is None:
        keep_ids = []

    ids_found = set(_S_TAG_RE.findall(text))
    ids_to_shuffle = sorted([i for i in ids_found if i not in keep_ids])

    if len(ids_to_shuffle) <= 1:
        return None

    shuffled_ids = ids_to_shuffle[:]
    random.shuffle(shuffled_ids)

    id_map = {}
    for i in ids_found:
        if i in keep_ids:
            id_map[i] = i
    for old, new in zip(ids_to_shuffle, shuffled_ids):
        id_map[old] = new
    return id_map


def apply_speaker_shuffle(text: str, id_map: dict) -> str:
    """将 speaker ID 映射应用到文本的所有 speaker 引用：
    <S1>/<S2> 标签、Speaker1/Speaker2、Speaker 1/Speaker 2。
    用 regex 单次扫描替换，避免链式覆盖。
    """
    if not id_map or not isinstance(text, str) or not text:
        return text

    # 数字映射 {1: 2, 2: 1}
    num_map = {}
    for old, new in id_map.items():
        num_map[old[1:]] = new[1:]  # "S1" -> "1", "S2" -> "2"

    def _replace(match):
        full = match.group(0)
        # <S1> 或 </S1> 匹配
        if match.group(1):
            sid = match.group(1)       # "S1"
            new_sid = id_map.get(sid, sid)
            if full.startswith("</"):
                return f"</{new_sid}>"
            return f"<{new_sid}>"
        # Speaker1 / Speaker 1 匹配
        if match.group(3):
            old_num = match.group(3)   # "1"
            new_num = num_map.get(old_num, old_num)
            return full[:full.rfind(old_num)] + new_num
        return full

    return _SPEAKER_REF_RE.sub(_replace, text)


def shuffle_speaker_ids(text: str, keep_ids=None) -> str:
    """向后兼容的接口：生成映射并应用到单条 text。"""
    id_map = build_speaker_shuffle_map(text, keep_ids)
    if id_map is None:
        return text
    return apply_speaker_shuffle(text, id_map)


class BaseShmDataset(BaseFalconReaderShmDataset):

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

                # 为每个数据集准备循环 randperm 与指针（持久化跨 epoch，避免"新一轮又从头覆盖之前访问过的chunk"）
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
                dataset_group_meta = []

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
                    })

                    dataset_group_meta.append(dataset_meta_)
                    total_ds_len += ds_len
                    idx_offset += n_chunks
                    print(f'|   - {dataset_meta_["data_path"]}')
                    print(f'|     - length: {ds_len}')
                    print(f'|     - reader_chunk_size: {reader_chunk_size_}')
                    print(f'|     - n_chunks: {n_chunks}')
                group_total_n_chunks = sum(item['n_chunks'] for item in dataset_group_meta)
                fallback_weight = dataset_group_weight / max(len(dataset_group_meta), 1)
                for dataset_meta_ in dataset_group_meta:
                    if group_total_n_chunks > 0:
                        dataset_meta_['weight'] = dataset_group_weight * dataset_meta_['n_chunks'] / group_total_n_chunks
                    else:
                        dataset_meta_['weight'] = fallback_weight
                    dataset_meta['datasets'].append(dataset_meta_)
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
        # print(f"| Starting processor_fn_worker#{i_worker}/{n_worker}.")

        log_f = None
        try:
            os.makedirs(shm_base, exist_ok=True)
            log_path = os.path.join(shm_base, f"processor_fn_worker_{i_worker}_{n_worker}.log")
            log_f = open(log_path, "a", buffering=1)
            faulthandler.enable(file=log_f, all_threads=True)
            try:
                faulthandler.register(signal.SIGUSR1, file=log_f, all_threads=True)
            except Exception:
                pass
            # print(f"| processor_fn_worker#{i_worker}/{n_worker}: faulthandler -> {log_path}")
            print(f"| Starting processor_fn_worker#{i_worker}/{n_worker}. faulthandler -> {log_path}")
        except Exception:
            print(f"| Starting processor_fn_worker#{i_worker}/{n_worker}.")
            traceback.print_exc()

        try:
            global_stores = {}
            reader_pack = self.prepare_reader(self.dataset_meta, global_stores, i_worker, n_worker)
            print(f"| init processor (dataset_reader)#{i_worker}/{n_worker}.")
            restart_countdown = 10000
            while True:
                idx = None
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
                        q_to_push.put(out_path)
                        restart_countdown -= 1
                        if restart_countdown == 0:
                            return
                        while q_to_push.qsize() > self.shuffle_buffer * world_size * 2:
                            time.sleep(1)
                        self.after_process_item(item, hparams, global_stores)

                except Exception:
                    print(f"| processor_fn_worker#{i_worker}/{n_worker}: exception idx={idx}")
                    traceback.print_exc()
                    if log_f is not None:
                        try:
                            print(f"| processor_fn_worker#{i_worker}/{n_worker}: exception idx={idx}", file=log_f)
                            traceback.print_exc(file=log_f)
                        except Exception:
                            pass
                    continue
        except Exception:
            traceback.print_exc()
            if log_f is not None:
                try:
                    traceback.print_exc(file=log_f)
                except Exception:
                    pass
    
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
                items = reader['reader'].read_range(start_line, end_line, on_error='none')

            elif reader['reader_type'] == 'pq_reader':
                # ParquetChunkReader 一般是按 chunk 编号读
                items = reader['reader'].read_chunk(local_chunk_idx)

            else:
                raise ValueError(f"Unknown reader_type: {reader['reader_type']}")

            return items, reader['processer_fn']

        except Exception:
            reader_type = None if reader is None else reader.get("reader_type")
            data_path = None if reader is None else reader.get("data_path")
            print(f"| [read_fn] exception: idx={idx} reader_type={reader_type} data_path={data_path}")
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
        
        read_res = self.read_fn(index, reader_pack, global_stores)
        if read_res is None:
            return
        raw_item, processer_fn = read_res

        if self.use_fast_dataloader:
            batcher = self.get_batcher(hparams, global_stores)
        
        for item in self._process_item(processer_fn, raw_item, hparams, global_stores, i_worker, n_worker):
            if item is None:
                continue
            if self.use_fast_dataloader:
                batch = batcher.collate_batch(item)
                if batch is not None and len(batch) > 0:
                    yield batch
            else:
                yield [item]

    def _process_item(self, processer_fn, raw_item, hparams, global_stores, i_worker, n_worker, **kwargs):
        raise NotImplementedError

    def collater(self, samples):
        raise NotImplementedError
        
