import os
import random
import json
from copy import deepcopy

import torch
import numpy as np
import torch.utils
import torch.utils.data
import librosa

from utils.commons.hparams import hparams
from utils.commons.os_utils import multiprocess_glob, handle_exacption
from utils.dataset.batcher import BucketBatcher
from utils.commons.io import get_wav_duration
from utils.text.split_text import get_word_list
from utils.commons.base_shm_dataset import BaseShmDataset, get_from_global_stores
from utils.commons.dataset_utils import collate_xd
import tempfile
import torchaudio

class DialogueRandomSliceShmDataset(BaseShmDataset):
    def get_dataset_meta(self):
        hparams, prefix = self.hparams, self.prefix
        meta_dir = '/mnt/bn/lq-ads-aigc/renyi/tts_datasets_bak/InteractiveDialogue/XYZ_20w/yuzhou/metas'
        # meta_paths = multiprocess_glob(f"{meta_dir}/*/*_info.json")
        # with open('data/XYZ_20w_meta_paths.lst', 'w') as f:
        #     f.write('\n'.join(meta_paths))
        with open('data/XYZ_20w_meta_paths.lst') as f:
            meta_paths = f.read().split('\n')
        return meta_paths, len(meta_paths)

    def prepare_reader(self, dataset_meta, global_stores):
        return 1
    
    def read_fn(self, idx, reader_pack, global_stores):
        return self.dataset_meta[idx]
    
    def process_item(self, raw_item, hparams, global_stores):
        if self.use_fast_dataloader:
            buckets = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 
                       3200, 3600, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 
                       11000, 12000, 13000, 14000, 15000, 16000, 18000, 20000, 40000, 60000]
            batcher = get_from_global_stores(
                'batcher', global_stores,
                lambda: BucketBatcher(
                    buckets=buckets,
                    dynamic_batch=hparams.get("dynamic_batch", True),
                    batch_size=hparams['max_sentences'],
                    maximum_bucket_size=hparams['max_tokens'],
                    length_fn=lambda x: x['len'],
                )
            )

        try:
            item = self._process_item(raw_item, hparams)
        except Exception as err:
            handle_exacption(err, raw_item)
            return
        if item is None:
            return

        if self.use_fast_dataloader:
            batch = batcher.collate_batch(item)
            if batch is not None and len(batch) > 0:
                yield batch
        else:
            yield [item]

    def _process_item(self, raw_item, hparams):
        import re

        # ---- helpers ----

        def _mel2token_to_dur(m2p: torch.LongTensor) -> torch.LongTensor:
            if m2p.numel() == 0:
                return torch.zeros(0, dtype=torch.long)
            mx = int(m2p.max().item()) if m2p.numel() > 0 else 0
            if mx == 0:
                return torch.zeros(0, dtype=torch.long)
            cnts = torch.bincount(m2p.clamp_min(0), minlength=mx + 1)
            return cnts[1:].to(torch.long)

        def _to_int_list_maybe(x):
            """
            将多种可能形式的 tone 安全转成整型 list。
            支持：
            - list/tuple/ndarray，元素可为 str/int/float
            - 纯字符串，如 "1 2 3" / "1,2,3"
            - JSON 字符串，如 "[1, 2, 3]" 或 '["0","4","4"]'
            转换失败返回 None。
            """
            if x is None:
                return None
            if isinstance(x, str):
                s = x.strip()
                if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
                    try:
                        parsed = json.loads(s.replace('(', '[').replace(')', ']'))
                        return _to_int_list_maybe(parsed)
                    except Exception:
                        pass
                nums = re.findall(r'-?\d+\.?\d*', s)
                if len(nums) == 0:
                    return None
                try:
                    return [int(float(t)) for t in nums]
                except Exception:
                    return None
            if isinstance(x, (list, tuple, np.ndarray)):
                out = []
                for v in x:
                    try:
                        if isinstance(v, str):
                            v = v.strip()
                            if v == '':
                                out.append(0)
                                continue
                            out.append(int(float(v)))
                        elif isinstance(v, (int, np.integer)):
                            out.append(int(v))
                        elif isinstance(v, (float, np.floating)):
                            out.append(int(v))
                        else:
                            out.append(int(v))
                    except Exception:
                        out.append(0)
                return out
            try:
                return [int(x)]
            except Exception:
                return None

        # ---- params ----
        fm     = hparams['frames_multiple']
        hop    = hparams['hop_size']
        stride = hparams.get('vae_stride', 8)
        fm_wav = fm * hop
        sr     = hparams['audio_sample_rate']
        max_spk_num = hparams.get('max_spk_num', 8)
        use_sparse_dur = hparams.get('use_sparse_dur', False)

        meta_path = raw_item
        meta_item = json.load(open(meta_path))
        turns     = meta_item['turns']
        segments  = meta_item['segments']

        for segment in segments:
            # 临时容器（只有 turn 全部校验通过才 append）
            spk_map = {}  # 原 spk -> 连续 1-based
            wav_lst, frame_spk_mask_lst = [], []
            ph_list, tone_list, m2p_list, spk_mask_ph_list = [], [], [], []

            # === 统一 text/caption 到 <S{sid}>...</S{sid}> ===
            res_text_parts, res_cap_parts = [], []
            kept_turn_sids = []     # 只统计“真正保留”的 turn
            change_of_spk = 1
            ref_wav_start_turn_idx = 0

            for turn_idx in segment['turn_idxs']:
                turn = turns[turn_idx]

                # 原逻辑：先过滤 text 为空
                text = turn.get('text')
                if not text or not str(text).strip():
                    continue

                # 先准备说话人 sid（1-based）
                raw_spk = turn.get('spk', 'spk_unk')
                sid = spk_map.setdefault(raw_spk, len(spk_map) + 1)

                # 预取并校验：phone / mel2ph / wav（全部可用才真正累加）
                ph_enc = turn.get('phone_encoded')
                m2p_raw = turn.get('mel2ph')
                if ph_enc is None or len(ph_enc) == 0 or m2p_raw is None or len(m2p_raw) == 0:
                    continue  # 这条 turn 直接跳过

                # 读音频
                wav_rel = turn.get('wav_path').replace('wavs/16k160/', 'wavs/24k/')
                if not wav_rel:
                    continue
                full_wav = os.path.join('/mnt/bn/lq-ads-aigc/renyi/tts_datasets_bak/InteractiveDialogue', wav_rel)
                try:
                    wav_np, _ = librosa.load(full_wav, sr=sr)
                except Exception:
                    continue
                if wav_np is None or len(wav_np) == 0:
                    continue
                wav_np = np.asarray(wav_np, dtype=np.float32)

                # —— 文本与 caption：统一包裹为 <S{sid}>...</S{sid}> —— #
                seg_txt = f'<S{sid}>{text}</S{sid}>'
                res_text_parts.append(seg_txt)
                res_cap_parts.append(seg_txt)

                kept_turn_sids.append(sid)
                if len(kept_turn_sids) >= 2 and kept_turn_sids[-1] != kept_turn_sids[-2]:
                    change_of_spk += 1
                    if change_of_spk == 3:
                        ref_wav_start_turn_idx = len(wav_lst)

                # 累加音频与帧级 one-hot（旧 spk_mask → frame_spk_mask）
                one_hot = np.zeros((len(wav_np), max_spk_num), dtype=np.int16)
                col = min(max(sid - 1, 0), max_spk_num - 1)
                one_hot[:, col] = 1
                wav_lst.append(wav_np)
                frame_spk_mask_lst.append(one_hot)

                # 累加 phone/tone/m2p 与 phone 对齐的 spk_mask
                ph_i = torch.as_tensor(ph_enc, dtype=torch.long)

                # —— tone 的健壮读取与数值化 —— #
                tone_raw = turn.get('tone_encoded')
                tone_enc_clean = _to_int_list_maybe(tone_raw) if tone_raw is not None else None
                if (tone_enc_clean is None) or (len(tone_enc_clean) != len(ph_enc)):
                    tn_i = torch.zeros_like(ph_i, dtype=torch.long)
                else:
                    tn_i = torch.as_tensor(tone_enc_clean, dtype=torch.long)

                m2p_i = torch.as_tensor(m2p_raw, dtype=torch.long)  # 1-based
                ph_list.append(ph_i)
                tone_list.append(tn_i)
                m2p_list.append(m2p_i)
                spk_mask_ph_list.append(torch.full((ph_i.numel(),), int(sid), dtype=torch.long))

            # === 段级检查（按“保留下来的 turn”统计）===
            if len(wav_lst) == 0:
                continue

            # 至少 4 个有效 turn、且有来回
            if len(kept_turn_sids) < 4:
                continue
            num_conversations = 1
            for i in range(1, len(kept_turn_sids)):
                if kept_turn_sids[i] != kept_turn_sids[i-1]:
                    num_conversations += 1
            if num_conversations // 2 < 2:
                continue

            # === 拼接音频与帧级掩码 ===
            wav_cat = torch.from_numpy(np.concatenate(wav_lst))
            frame_spk_mask = torch.from_numpy(np.concatenate(frame_spk_mask_lst, axis=0))
            # 对齐到 fm_wav
            T = (wav_cat.shape[0] // fm_wav) * fm_wav
            wav_cat = wav_cat[:T]
            frame_spk_mask = frame_spk_mask[:T]

            # === 拼接 phones/tone/mel2ph（1-based offset 修正）===
            ph_offset = 0
            m2p_fixed = []
            for m2p_i, ph_i in zip(m2p_list, ph_list):
                if ph_offset > 0:
                    m2p_i = m2p_i + (m2p_i > 0).long() * ph_offset
                m2p_fixed.append(m2p_i)
                ph_offset += ph_i.numel()

            ph_token = torch.cat(ph_list, dim=0)
            tone     = torch.cat(tone_list, dim=0)
            mel2ph   = torch.cat(m2p_fixed, dim=0)
            spk_mask_ph = torch.cat(spk_mask_ph_list, dim=0)

            # 使 mel2ph 与当前音频的 mel 帧长度一致（再对齐到 fm）
            mel_len = wav_cat.shape[0] // hop
            if mel2ph.numel() < mel_len:
                pad_len = mel_len - mel2ph.numel()
                last = mel2ph[-1] if mel2ph.numel() > 0 else torch.tensor(0, dtype=torch.long)
                mel2ph = torch.cat([mel2ph, last.repeat(pad_len)], dim=0)
            mel2ph = mel2ph[:mel_len]
            mel2ph = mel2ph[: (mel_len // fm) * fm]

            # === 计算 dur，并按需在 worker 内生成 mel2ph_sparse（严格与 mel2ph 等长、对齐到 fm）===
            dur = _mel2token_to_dur(mel2ph)
            mel2ph_sparse = None
            if use_sparse_dur:
                # compute_mel2aug_from_dur 的输入是 “每个 phone 的 mel 帧时长”
                _m2a = compute_mel2aug_from_dur(
                    dur.cpu().numpy().tolist(),
                    gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                    gap_frames=hparams.get('sparse_dur_frames', 4),
                    gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                    min_keep=hparams.get('sparse_dur_min_keep', 1),
                    keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                    symmetric=hparams.get('sparse_dur_symmetric', True),
                )
                mel2ph_sparse = torch.as_tensor(_m2a, dtype=torch.long)
                # 等长裁剪/补齐，并保持与 mel2ph 的 fm 对齐
                target_len = mel2ph.shape[0]  # 已对齐到 fm
                if mel2ph_sparse.numel() < target_len:
                    pad_len = target_len - mel2ph_sparse.numel()
                    last = mel2ph_sparse[-1] if mel2ph_sparse.numel() > 0 else torch.tensor(0, dtype=torch.long)
                    mel2ph_sparse = torch.cat([mel2ph_sparse, last.repeat(pad_len)], dim=0)
                mel2ph_sparse = mel2ph_sparse[:target_len]

            # === 文本与 caption（统一为 <S{sid}>...</S{sid}>）===
            text_merged    = ''.join(res_text_parts)
            caption_merged = text_merged

            # ctx 起点：按“第三次换人”的 turn 边界，再随机微调
            pre_len = 0
            if 0 < ref_wav_start_turn_idx <= len(wav_lst):
                pre_len = np.concatenate(wav_lst[:ref_wav_start_turn_idx]).shape[0]
            ref_wav_start = (pre_len // fm_wav) * fm_wav
            max_idx = min(int(wav_cat.shape[0] * 0.9), wav_cat.shape[0] - 20000)
            if max_idx > ref_wav_start:
                ref_wav_start = (random.randint(ref_wav_start, max_idx) // fm_wav) * fm_wav

            ctx_wav = wav_cat[:ref_wav_start]
            ctx_mask = torch.zeros((wav_cat.shape[0], 1), dtype=torch.float32)
            ctx_mask[:ref_wav_start] = 1.0
            ctx_mask = ctx_mask[:: hop * stride]

            # 产出
            item = {
                'wav': wav_cat,
                'text': text_merged,
                'caption': caption_merged,
                'spk_mask': spk_mask_ph.to(torch.long),   # phone-level, 1-based
                'frame_spk_mask': frame_spk_mask,         # 采样点级 one-hot（旧，改名保留）
                'ctx_wav': ctx_wav,
                'ctx_mask': ctx_mask,
                'ph_token': ph_token.to(torch.long),
                'tone': tone.to(torch.long),
                'mel2ph': mel2ph.to(torch.long),
                'dur': dur.to(torch.long),
            }
            if mel2ph_sparse is not None:
                item['mel2ph_sparse'] = mel2ph_sparse.to(torch.long)

            item['len'] = int(item['wav'].shape[0] / hop / stride)

            yield item


    def collater(self, samples):
        samples = samples[0]
        if len(samples) == 0:
            if hasattr(self, 'backup_batch') and self.backup_batch is not None:
                print('use backup batch!')
                return self.backup_batch
            else:
                print('no batch to take!')
                return {}

        wavs = collate_xd([s['wav'] for s in samples], 0.0)
        wav_lengths = torch.LongTensor([s['wav'].shape[0] for s in samples])
        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
        }
        batch['text'] = [s['text'] for s in samples]
        batch['spk_mask'] = collate_xd([s['spk_mask'] for s in samples], 0.0) if 'spk_mask' in samples[0] else None

        if 'wav_w2v2' in samples[0]:
            batch['wavs_w2v2'] = collate_xd([s['wav_w2v2'] for s in samples], 0.0)
            batch['wav_w2v2_lengths'] = torch.LongTensor([s['wav_w2v2'].shape[0] for s in samples])

        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch


class DialogueSegmentShmDataset(BaseShmDataset):
    def get_dataset_meta(self):
        meta_paths = multiprocess_glob(f'/mnt/bn/sa-ag-data/liruiqi/data/speech/XYZ_20w/metas/*/*.json', num_workers=128)
        return meta_paths, len(meta_paths)
    
    def prepare_reader(self, dataset_meta, global_stores):
        return 1
    
    def read_fn(self, idx, reader_pack, global_stores):
        return self.dataset_meta[idx]
    
    def process_item(self, raw_item, hparams, global_stores):
        if self.use_fast_dataloader:
            buckets = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 
                       3200, 3600, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 
                       11000, 12000, 13000, 14000, 15000, 16000, 18000, 20000, 40000, 60000]
            batcher = get_from_global_stores(
                'batcher', global_stores,
                lambda: BucketBatcher(
                    buckets=buckets,
                    dynamic_batch=hparams.get("dynamic_batch", True),
                    batch_size=hparams['max_sentences'],
                    maximum_bucket_size=hparams['max_tokens'],
                    length_fn=lambda x: x['len'],
                )
            )

        for item in self._process_item(raw_item, hparams):
            if item is None:
                continue
            if self.use_fast_dataloader:
                batch = batcher.collate_batch(item)
                if batch is not None and len(batch) > 0:
                    yield batch
            else:
                yield [item]

    def _process_item(self, raw_item, hparams):
        meta_path = raw_item
        meta_item = json.load(open(meta_path))
        max_spk_num = hparams.get('max_spk_num', 8)
        fm = hparams['frames_multiple']
        fm_wav = hparams['frames_multiple'] * hparams['hop_size']

        turns = meta_item['turns']
        segments = meta_item['segments']

        for segment in segments:
            spk_map = {}
            spk_lst = []
            res_text = ''
            wav_lst = []
            spk_mask_lst = []
            ref_wav_start = 0
            change_of_spk = 1
            for turn_idx in segment['turn_idxs']:
                turn = turns[turn_idx]
                text = turn['text']
                if text is None:
                    continue
                spk_name = turn['spk']
                if spk_name not in spk_map:
                    spk_map[spk_name] = len(spk_map)
                spk_id = spk_map[spk_name]
                if len(spk_lst) > 0 and spk_id == spk_lst[-1]:
                    res_text += text
                else:
                    change_of_spk += 1
                    res_text += f'<SPK>{spk_id}</SPK>' + text
                spk_lst.append(spk_id)
                if change_of_spk == 3:
                    ref_wav_start = len(wav_lst)

                wav_path = turn['wav_path']
                wav_path = os.path.join('/mnt/bn/lq-ads-aigc/renyi/tts_datasets_bak/InteractiveDialogue', wav_path)
                wav, _ = librosa.load(wav_path, sr=hparams['audio_sample_rate'])
                wav_lst.append(wav)

                spk_mask = np.zeros((len(wav), max_spk_num), dtype=int)
                spk_mask[:, spk_id] = 1
                spk_mask_lst.append(spk_mask)
            
            # check turns
            if len(spk_lst) < 4:
                continue
            num_conversations = 1
            for i in range(1, len(spk_lst)):
                if spk_lst[i] != spk_lst[i-1]:
                    num_conversations += 1
            if num_conversations // 2 < 2:
                continue

            item = {
                'wav': torch.from_numpy(np.concatenate(wav_lst)),
                'text': res_text,
                'spk_mask': torch.from_numpy(np.concatenate(spk_mask_lst, axis=0)),
            }
            item['wav'] = item['wav'][:len(item['wav']) // fm_wav * fm_wav]
            item['spk_mask'] = item['spk_mask'][:len(item['spk_mask']) // fm_wav * fm_wav]
            item['len'] = int(item['wav'].shape[0] / hparams['hop_size'] / hparams['vae_stride'])

            ref_wav_start = np.concatenate(wav_lst[:ref_wav_start]).shape[0] // fm_wav * fm_wav
            max_idx = min(int(len(item['wav']) * 0.9), len(item['wav']) - 20000)
            if max_idx > ref_wav_start:
                ref_wav_start = random.randint(ref_wav_start, max_idx) // fm_wav * fm_wav
            ctx_wav = deepcopy(item['wav'])
            ctx_wav = ctx_wav[:ref_wav_start]
            item['ctx_wav'] = ctx_wav
            ctx_mask = torch.zeros_like(item['wav'])[:, None]
            ctx_mask[:ref_wav_start] = 1.0
            ctx_mask = ctx_mask[::hparams['hop_size']*hparams['vae_stride']]
            item['ctx_mask'] = ctx_mask

            yield item

    def collater(self, samples):
        samples = samples[0]
        if len(samples) == 0:
            if hasattr(self, 'backup_batch') and self.backup_batch is not None:
                print('use backup batch!')
                return self.backup_batch
            else:
                print('no batch to take!')
                return {}

        wavs = collate_xd([s['wav'] for s in samples], 0.0)
        wav_lengths = torch.LongTensor([s['wav'].shape[0] for s in samples])
        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
        }
        batch['text'] = [s['text'] for s in samples]
        batch['spk_mask'] = collate_xd([s['spk_mask'] for s in samples], 0.0) if 'spk_mask' in samples[0] else None
        batch['ctx_wavs'] = collate_xd([s['ctx_wav'] for s in samples], 0.0)
        batch['ctx_mask'] = collate_xd([s['ctx_mask'] for s in samples], 0.0)

        if 'wav_w2v2' in samples[0]:
            batch['wavs_w2v2'] = collate_xd([s['wav_w2v2'] for s in samples], 0.0)
            batch['wav_w2v2_lengths'] = torch.LongTensor([s['wav_w2v2'].shape[0] for s in samples])

        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch

from modules.tts.ar_dur.commons.align_ops import compute_mel2aug_from_dur

class DialogueSegmentEmbDataset(BaseShmDataset):
    def get_dataset_meta(self):
        meta_paths = multiprocess_glob(
            f'/mnt/bn/sa-ag-data/zhangyu.34/data/speech/XYZ_20w/metas_with_tson_16k160_final_hard/*/*.json',
            num_workers=128
        )
        return meta_paths, len(meta_paths)
    
    def prepare_reader(self, dataset_meta, global_stores):
        return 1

    def read_fn(self, idx, reader_pack, global_stores):
        return self.dataset_meta[idx]

    def process_item(self, raw_item, hparams, global_stores):
        """
        逐 meta 产生样本，若 use_fast_dataloader=True，则在本函数内部做 bucket 动态组 batch。
        输出字段与原先保持一致，但 text/caption 统一为 v2 的 <SPK>i</SPK> 格式（同 speaker 连续不重复插入 tag）。
        另外：对 ctx/ref 部分增加轻量混响或噪声增强（仅作用于 ctx_wav，不改动 wav 主体）。
        """
        if self.use_fast_dataloader:
            buckets = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2800,
                    3200, 3600, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000,
                    11000, 12000, 13000, 14000, 15000, 16000, 18000, 20000, 40000, 60000]
            batcher = get_from_global_stores(
                'batcher', global_stores,
                lambda: BucketBatcher(
                    buckets=buckets,
                    dynamic_batch=hparams.get("dynamic_batch", True),
                    batch_size=hparams['max_sentences'],
                    maximum_bucket_size=hparams['max_tokens'],
                    length_fn=lambda x: x['len'],
                )
            )

        for item in self._process_item(raw_item, hparams, global_stores):
            if item is None:
                continue
            if self.use_fast_dataloader:
                batch = batcher.collate_batch(item)
                if batch is not None and len(batch) > 0:
                    yield batch
            else:
                yield [item]

    def _process_item(self, raw_item, hparams, global_stores):
        """
        - raw_item 可能是 .json / .jsonl 的 path（与原先一致）
        - 每个 turn 只使用 wav_path 本地读取，并按 turn 顺序拼接成 segment（不读取 TOS，不用 wav_k）
        - text/caption 输出改为 v2：<SPK>i</SPK>xxx，同一说话人连续不重复插入 tag
        - ctx/ref 切分逻辑沿用原先（第三次换人处作为 ref 起点 + 随机右移微调）
        - 新增：仅对 ctx_wav（ref 部分）按 sid 做轻量 reverb/noise（可能不加）
        -  修改：去掉 frame_spk_mask
        -  修改：ctx_mask 定义与 v2 统一（latent_len 维度）
        """
        import os
        import re
        import json
        import random
        import numpy as np
        import torch
        import librosa
        import torch.nn.functional as F

        # ---- helpers ----
        def _mel2token_to_dur(m2p: torch.LongTensor) -> torch.LongTensor:
            if m2p.numel() == 0:
                return torch.zeros(0, dtype=torch.long)
            mx = int(m2p.max().item()) if m2p.numel() > 0 else 0
            if mx <= 0:
                return torch.zeros(0, dtype=torch.long)
            cnts = torch.bincount(m2p.clamp_min(0), minlength=mx + 1)
            return cnts[1:].to(torch.long)

        def _to_int_list_maybe(x):
            if x is None:
                return None

            if isinstance(x, str):
                s = x.strip()
                if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
                    try:
                        parsed = json.loads(s.replace('(', '[').replace(')', ']'))
                        return _to_int_list_maybe(parsed)
                    except Exception:
                        pass
                nums = re.findall(r'-?\d+\.?\d*', s)
                if len(nums) == 0:
                    return None
                try:
                    return [int(float(t)) for t in nums]
                except Exception:
                    return None

            if isinstance(x, (list, tuple, np.ndarray)):
                out = []
                for v in x:
                    try:
                        if isinstance(v, str):
                            v = v.strip()
                            if v == '':
                                out.append(0)
                                continue
                            out.append(int(float(v)))
                        elif isinstance(v, (int, np.integer)):
                            out.append(int(v))
                        elif isinstance(v, (float, np.floating)):
                            out.append(int(v))
                        else:
                            out.append(int(v))
                    except Exception:
                        out.append(0)
                return out

            try:
                return [int(x)]
            except Exception:
                return None

        # ====== 读 meta：支持 .json / .jsonl（保持原逻辑） ======
        def _iter_meta_items(meta_path: str):
            if meta_path.endswith('.json'):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    yield json.load(f)
            elif meta_path.endswith('.jsonl'):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        yield json.loads(line)
            else:
                raise ValueError(f'Unknown meta file type: {meta_path}')

        # ====== 读 wav：仅本地 wav_path（不读 TOS；保持“每 turn 读然后拼接”） ======
        sr = int(hparams['audio_sample_rate'])

        def _load_wav_local(wav_rel: str):
            if not wav_rel:
                return None
            wav_rel = wav_rel.replace('wavs/16k160/', 'wavs/24k/')
            full_wav = os.path.join(
                '/mnt/bn/lq-ads-aigc/renyi/tts_datasets_bak/InteractiveDialogue', wav_rel
            )
            try:
                wav_np, _ = librosa.load(full_wav, sr=sr)
            except Exception:
                return None
            if wav_np is None or len(wav_np) == 0:
                return None
            return np.asarray(wav_np, dtype=np.float32)

        # ====== ref augmentation（轻量、只改 ctx_wav） ======
        def _apply_noise(seg: torch.Tensor, snr_db: float) -> torch.Tensor:
            if seg.numel() <= 0:
                return seg
            x = seg.to(torch.float32)
            rms = torch.sqrt(torch.mean(x * x) + 1e-12)
            noise_rms = rms / (10.0 ** (float(snr_db) / 20.0))
            noise = torch.randn_like(x) * noise_rms
            return x + noise

        def _apply_noise(seg: torch.Tensor, snr_db: float) -> torch.Tensor:
            # seg: [T]
            if seg.numel() <= 0:
                return seg
            seg_f = seg.to(torch.float32)
            rms = torch.sqrt(torch.mean(seg_f * seg_f) + 1e-12)
            # snr_db 越大噪声越小
            noise_rms = rms / (10.0 ** (float(snr_db) / 20.0))
            noise = torch.randn_like(seg_f) * noise_rms
            out = seg_f + noise
            return out

        def _apply_light_reverb(seg: torch.Tensor, wet: float) -> torch.Tensor:
            """
            轻量“混响/空间感”：多 tap 延迟(很小) + 小平滑 + wet 混合
            wet < 0.2，尽量不影响可懂度
            """
            ...
            return out


        def _apply_light_reverb(seg: torch.Tensor, wet: float) -> torch.Tensor:
            if seg.numel() <= 0:
                return seg
            x = seg.to(torch.float32)
            y = x.clone()

            n_taps = random.randint(2, 4)
            for _ in range(n_taps):
                d = int(sr * random.uniform(0.010, 0.055))
                g = random.uniform(0.04, 0.14)
                if 0 < d < y.numel():
                    y[d:] = y[d:] + g * y[:-d]

            k = random.choice([3, 5, 7])
            if y.numel() > k:
                ker = torch.ones((k,), device=y.device, dtype=y.dtype) / float(k)
                y = F.conv1d(y[None, None, :], ker[None, None, :], padding=k // 2)[0, 0, :]

            x_peak = x.abs().max().clamp_min(1e-6)
            y_peak = y.abs().max().clamp_min(1e-6)
            y = y / y_peak * x_peak

            wet = float(max(0.0, min(wet, 0.199)))
            return (1.0 - wet) * x + wet * y

        # ---- params ----
        fm     = int(hparams['frames_multiple'])
        hop    = int(hparams['hop_size'])
        stride = int(hparams.get('vae_stride', 8))
        fm_wav = fm * hop
        latent_hop = hop * stride

        meta_path = raw_item

        for meta_item in _iter_meta_items(meta_path):
            turns    = meta_item['turns']
            segments = meta_item['segments']

            wav_cache = {}

            for segment in segments:
                spk_map = {}

                wav_lst = []
                ph_list, tone_list, m2p_list, spk_mask_ph_list = [], [], [], []

                res_text = ''
                last_sid_for_text = None

                kept_turn_sids = []
                change_of_spk = 1
                ref_wav_start_turn_idx = 0

                turn_ranges = []
                cum = 0

                for turn_idx in segment['turn_idxs']:
                    turn = turns[turn_idx]

                    text = turn.get('text')
                    if not text or not str(text).strip():
                        continue
                    text = str(text).strip()

                    raw_spk = turn.get('spk', 'spk_unk')
                    sid = spk_map.setdefault(raw_spk, len(spk_map) + 1)

                    ph_enc = turn.get('phone_encoded')
                    m2p_raw = turn.get('mel2ph')
                    if ph_enc is None or len(ph_enc) == 0 or m2p_raw is None or len(m2p_raw) == 0:
                        continue

                    wav_path = turn.get('wav_path', None)
                    if not wav_path:
                        continue

                    cache_key = ('local', wav_path)
                    if cache_key in wav_cache:
                        wav_np = wav_cache[cache_key]
                    else:
                        wav_np = _load_wav_local(wav_path)
                        if wav_np is None or len(wav_np) == 0:
                            continue
                        wav_cache[cache_key] = wav_np

                    kept_turn_sids.append(sid)
                    if len(kept_turn_sids) >= 2 and kept_turn_sids[-1] != kept_turn_sids[-2]:
                        change_of_spk += 1
                        if change_of_spk == 3:
                            ref_wav_start_turn_idx = len(wav_lst)

                    if last_sid_for_text is None or int(sid) != int(last_sid_for_text):
                        res_text += f'<SPK>{int(sid)}</SPK>' + text
                        last_sid_for_text = sid
                    else:
                        res_text += text

                    wav_lst.append(wav_np)

                    L = int(len(wav_np))
                    if L > 0:
                        turn_ranges.append({'sid': int(sid), 'start': int(cum), 'end': int(cum + L)})
                        cum += L

                    ph_i = torch.as_tensor(ph_enc, dtype=torch.long)

                    tone_raw = turn.get('tone_encoded')
                    tone_enc_clean = _to_int_list_maybe(tone_raw) if tone_raw is not None else None
                    if (tone_enc_clean is None) or (len(tone_enc_clean) != len(ph_enc)):
                        tn_i = torch.zeros_like(ph_i, dtype=torch.long)
                    else:
                        tn_i = torch.as_tensor(tone_enc_clean, dtype=torch.long)

                    m2p_i = torch.as_tensor(m2p_raw, dtype=torch.long)

                    ph_list.append(ph_i)
                    tone_list.append(tn_i)
                    m2p_list.append(m2p_i)
                    spk_mask_ph_list.append(torch.full((ph_i.numel(),), int(sid), dtype=torch.long))

                # --- filters（保持原先） ---
                if len(wav_lst) == 0:
                    continue
                if len(kept_turn_sids) < 4:
                    continue
                num_conversations = 1
                for i in range(1, len(kept_turn_sids)):
                    if kept_turn_sids[i] != kept_turn_sids[i - 1]:
                        num_conversations += 1
                if num_conversations // 2 < 2:
                    continue

                wav_cat = torch.from_numpy(np.concatenate(wav_lst))

                T = (int(wav_cat.shape[0]) // fm_wav) * fm_wav
                if T <= 0:
                    continue
                wav_cat = wav_cat[:T]
                if wav_cat.numel() == 0:
                    continue

                if turn_ranges:
                    new_ranges = []
                    for r in turn_ranges:
                        st = int(max(0, min(r['start'], T)))
                        ed = int(max(0, min(r['end'], T)))
                        if ed > st:
                            new_ranges.append({'sid': int(r['sid']), 'start': st, 'end': ed})
                    turn_ranges = new_ranges

                # --- mel2ph offset fix & concat（保持原先） ---
                ph_offset = 0
                m2p_fixed = []
                for m2p_i, ph_i in zip(m2p_list, ph_list):
                    if ph_offset > 0:
                        m2p_i = m2p_i + (m2p_i > 0).long() * ph_offset
                    m2p_fixed.append(m2p_i)
                    ph_offset += int(ph_i.numel())

                ph_token = torch.cat(ph_list, dim=0)
                tone     = torch.cat(tone_list, dim=0)
                mel2ph   = torch.cat(m2p_fixed, dim=0)
                spk_mask_ph = torch.cat(spk_mask_ph_list, dim=0)

                mel_len = int(wav_cat.shape[0] // hop)
                if mel_len <= 0:
                    continue
                if mel2ph.numel() < mel_len:
                    pad_len = mel_len - mel2ph.numel()
                    last = mel2ph[-1] if mel2ph.numel() > 0 else torch.tensor(0, dtype=torch.long)
                    mel2ph = torch.cat([mel2ph, last.repeat(pad_len)], dim=0)
                mel2ph = mel2ph[:mel_len]
                mel2ph = mel2ph[: (mel_len // fm) * fm]
                if mel2ph.numel() == 0:
                    continue

                dur = _mel2token_to_dur(mel2ph)

                mel2ph_sparse = None
                if hparams.get('use_sparse_dur', False):
                    from some_module import compute_mel2aug_from_dur
                    _m2a = compute_mel2aug_from_dur(
                        dur.cpu().numpy().tolist(),
                        gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                        gap_frames=hparams.get('sparse_dur_frames', 4),
                        gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                        min_keep=hparams.get('sparse_dur_min_keep', 1),
                        keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                        symmetric=hparams.get('sparse_dur_symmetric', True),
                    )
                    mel2ph_sparse = torch.as_tensor(_m2a, dtype=torch.long)
                    target_len = int(mel2ph.shape[0])
                    if mel2ph_sparse.numel() < target_len:
                        pad_len = target_len - mel2ph_sparse.numel()
                        last = mel2ph_sparse[-1] if mel2ph_sparse.numel() > 0 else torch.tensor(0, dtype=torch.long)
                        mel2ph_sparse = torch.cat([mel2ph_sparse, last.repeat(pad_len)], dim=0)
                    mel2ph_sparse = mel2ph_sparse[:target_len]

                text_merged = res_text
                caption_merged = text_merged

                # --- ctx split（保持原先语义：第三段开始 + 随机右移） ---
                pre_len = 0
                if turn_ranges and 0 < ref_wav_start_turn_idx < len(turn_ranges):
                    pre_len = int(turn_ranges[ref_wav_start_turn_idx]['start'])
                ref_wav_start = (int(pre_len) // fm_wav) * fm_wav

                max_idx = min(int(wav_cat.shape[0] * 0.9), int(wav_cat.shape[0]) - 20000)
                if max_idx > ref_wav_start:
                    ref_wav_start = (random.randint(ref_wav_start, max_idx) // fm_wav) * fm_wav
                ref_wav_start = int(max(0, min(ref_wav_start, int(wav_cat.shape[0]))))

                ctx_wav = wav_cat[:ref_wav_start].clone()

                # --- ctx(ref) augmentation（保持原先） ---
                if ctx_wav.numel() > 0 and turn_ranges:
                    sids_in_ref = set()
                    for r in turn_ranges:
                        if r['start'] < ref_wav_start and r['end'] > 0:
                            sids_in_ref.add(int(r['sid']))

                    spk_aug = {}
                    for sid_i in sids_in_ref:
                        if random.random() < 0.5:
                            spk_aug[sid_i] = ('none', None)
                        else:
                            if random.random() < 0.5:
                                wet = random.uniform(0.03, 0.19)
                                spk_aug[sid_i] = ('reverb', wet)
                            else:
                                snr_db = random.uniform(18.0, 40.0)
                                spk_aug[sid_i] = ('noise', snr_db)

                    for r in turn_ranges:
                        st = int(r['start'])
                        ed = int(r['end'])
                        if st >= ref_wav_start or ed <= 0:
                            continue
                        sid_i = int(r['sid'])
                        aug = spk_aug.get(sid_i, ('none', None))
                        if aug[0] == 'none':
                            continue

                        s0 = max(0, st)
                        e0 = min(ref_wav_start, ed)
                        if e0 <= s0:
                            continue

                        seg = ctx_wav[s0:e0]
                        if aug[0] == 'reverb':
                            seg2 = _apply_light_reverb(seg, wet=float(aug[1]))
                        else:
                            seg2 = _apply_noise(seg, snr_db=float(aug[1]))
                        ctx_wav[s0:e0] = seg2.to(ctx_wav.dtype)

                    peak = float(ctx_wav.abs().max().item()) if ctx_wav.numel() > 0 else 0.0
                    if peak > 1.2:
                        ctx_wav = ctx_wav / peak

                #  ctx_mask：统一为 v2 定义（latent_len 维度）
                latent_len = int(wav_cat.shape[0] // latent_hop)
                if latent_len <= 0:
                    continue
                ctx_latent_len = int(ref_wav_start // latent_hop)
                ctx_mask = torch.zeros((latent_len, 1), dtype=torch.float32)
                if ctx_latent_len > 0:
                    ctx_mask[:min(ctx_latent_len, latent_len)] = 1.0

                item = {
                    'wav': wav_cat,
                    'text': text_merged,
                    'caption': caption_merged,
                    'spk_mask': spk_mask_ph.to(torch.long),
                    'ctx_wav': ctx_wav,
                    'ctx_mask': ctx_mask,
                    'ph_token': ph_token.to(torch.long),
                    'tone': tone.to(torch.long),
                    'mel2ph': mel2ph.to(torch.long),
                    'dur': dur.to(torch.long),
                    'len': latent_len,  #  与 v2 一致：len = latent_len
                }
                if mel2ph_sparse is not None:
                    item['mel2ph_sparse'] = mel2ph_sparse.to(torch.long)

                yield item


    def collater(self, samples):
        """
        支持 fast-dataloader（samples 由若干条样本组成）与回退备份逻辑。
        汇总字段尽量与 DiTWavTextDataset.collater 对齐：
            - 'spk_mask'：phone 对齐的说话人 id（1-based）
        """
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]

        if len(samples) == 0:
            if hasattr(self, 'backup_batch') and self.backup_batch is not None:
                print('use backup batch!')
                return self.backup_batch
            else:
                print('no batch to take!')
                return {}

        wavs = collate_xd([s['wav'] for s in samples], 0.0)
        wav_lengths = torch.LongTensor([s['wav'].shape[0] for s in samples])
        ctx_wavs = collate_xd([s['ctx_wav'] for s in samples], 0.0)

        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
            'ctx_wavs': ctx_wavs,
            'ctx_mask': collate_xd([s['ctx_mask'] for s in samples], 0),
            'text': [s['text'] for s in samples],
            'caption': [s['caption'] for s in samples] if 'caption' in samples[0] else None,
        }

        if 'ph_token' in samples[0]:
            batch['ph_tokens'] = collate_xd([s['ph_token'] for s in samples], 0)
            batch['txt_lengths'] = torch.LongTensor([s['ph_token'].numel() for s in samples])
        if 'tone' in samples[0]:
            batch['tone'] = collate_xd([s['tone'] for s in samples], 0)
        if 'mel2ph' in samples[0]:
            batch['mel2ph'] = collate_xd([s['mel2ph'] for s in samples], 0)
        if 'dur' in samples[0]:
            batch['dur'] = collate_xd([s['dur'] for s in samples], 0)
            batch['dur_len'] = torch.LongTensor([s['dur'].shape[0] for s in samples])
        if 'mel2ph_sparse' in samples[0]:
            batch['mel2ph_sparse'] = collate_xd([s['mel2ph_sparse'] for s in samples], 0)

        if 'spk_mask' in samples[0]:
            batch['spk_mask'] = collate_xd([s['spk_mask'] for s in samples], 0)  # [B, T_ph]

        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch


from utils.commons.tos_utils_v2 import TosClient
from utils.commons.jsonl_utils import build_jsonl_index, count_jsonl_n_lines
from utils.commons.jsonl_utils import JsonlChunkReader
class DialogueSegmentEmbDataset_v2(BaseShmDataset):
    def get_dataset_meta(self):
        import os
        import math

        jsonl_paths = multiprocess_glob(
            '/mnt/bn/sa-ag-data/panchanghao/code/mega-data-pipeline/user/zhangyu/zhiyuexingchen_dialogue/meta_merged_ph_tone_encoded_final_hard/*.jsonl',
            num_workers=128
        )

        chunk_size = int(self.hparams.get('reader_chunk_size', 32))

        packs = []
        offset_chunks = 0
        for p in jsonl_paths:
            idx_path = p + '.idx'

            if not os.path.isfile(idx_path):
                try:
                    build_jsonl_index(p, idx_path, use_tqdm=False)
                except Exception:
                    pass

            try:
                n_lines = int(count_jsonl_n_lines(idx_path))
            except Exception:
                n_lines = 0

            n_chunks = int(math.ceil(n_lines / chunk_size)) if n_lines > 0 else 0

            packs.append({
                'path': p,
                'idx_path': idx_path,
                'n_lines': n_lines,
                'chunk_size': chunk_size,
                'n_chunks': n_chunks,
                'offset_chunks': offset_chunks,
            })
            offset_chunks += n_chunks

        # dataset_len 变成 “chunk 数”，不是行数
        return packs, int(offset_chunks)

    def prepare_reader(self, dataset_meta, global_stores):
        readers = []
        for pack in dataset_meta:
            if pack.get('n_lines', 0) <= 0 or pack.get('n_chunks', 0) <= 0:
                readers.append(None)
            else:
                readers.append(JsonlChunkReader(pack['path'], pack['idx_path']))
        return readers

    def read_fn(self, idx, reader_pack, global_stores):
        import bisect

        packs = self.dataset_meta
        if idx is None or idx < 0 or len(packs) == 0:
            return None

        # end 是开区间上界：offset_chunks + n_chunks
        if (not hasattr(self, '_pack_chunk_ends')) or (len(self._pack_chunk_ends) != len(packs)):
            self._pack_chunk_ends = [int(p['offset_chunks']) + int(p['n_chunks']) for p in packs]

        # 找第一个 end > idx 的 pack
        pack_id = bisect.bisect_left(self._pack_chunk_ends, int(idx) + 1)
        if pack_id < 0 or pack_id >= len(packs):
            return None

        pack = packs[pack_id]
        r = reader_pack[pack_id]
        if r is None:
            return None

        local_chunk = int(idx) - int(pack['offset_chunks'])
        if local_chunk < 0 or local_chunk >= int(pack['n_chunks']):
            return None

        chunk_size = int(pack['chunk_size'])
        start_line = local_chunk * chunk_size
        end_line = min(int(pack['n_lines']) - 1, start_line + chunk_size - 1)
        if end_line < start_line:
            return None

        try:
            items = r.read_range(start_line, end_line)  # 闭区间
            if not items:
                return None
            return items  # list[meta_item(dict)]
        except Exception:
            return None

    # ==================== 支持 phone/mel2ph，并输出与 processer_fn_* 对齐的字段 ====================
    def process_item(self, raw_item, hparams, global_stores):

        if raw_item is None:
            return

        # read_fn 返回 list[meta_item]
        meta_items = raw_item if isinstance(raw_item, list) else [raw_item]

        if self.use_fast_dataloader:
            buckets = [
                400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000,
                2400, 2800, 3200, 3600, 4000, 4500, 5000, 5500, 6000,
                7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000,
                15000, 16000, 18000, 20000, 40000, 60000
            ]
            batcher = get_from_global_stores(
                'batcher', global_stores,
                lambda: BucketBatcher(
                    buckets=buckets,
                    dynamic_batch=hparams.get("dynamic_batch", True),
                    batch_size=hparams['max_sentences'],
                    maximum_bucket_size=hparams['max_tokens'],
                    length_fn=lambda x: x['len'],
                )
            )

        for meta_item in meta_items:
            for item in self._process_item(meta_item, hparams, global_stores):
                if item is None:
                    continue
                if self.use_fast_dataloader:
                    batch = batcher.collate_batch(item)
                    if batch is not None and len(batch) > 0:
                        yield batch
                else:
                    yield [item]

    def _process_item(self, meta_item, hparams, global_stores):
        """
        meta_item: dict（jsonl 一行）
        - 支持 turn 级 phone/tone/mel2ph 拼接
        - 支持 segment/turn 带 start/end：整条 wav 解码一次后按 [s,e] slice
        - text/caption 输出：<SPK>{id}</SPK>xxx，同一说话人连续不重复插入
        - ref(ctx) 语义统一到 DialogueSegmentEmbDataset：
            ctx/ref 包含“第一轮完整 AB”，从第三段（A/B/C 的第三个说话人段）开始切分
        - 新增：仅对 ref(ctx) 部分，按 speaker(sid) 随机加轻量混响或噪声（也可能不加），不额外引参
        """
        import os
        import re
        import json
        import random
        import tempfile
        import numpy as np
        import torch
        import torchaudio
        import torchaudio.functional as AF
        import hashlib
        import subprocess
        import torch.nn.functional as F
        from collections import OrderedDict

        # --------- helpers ----------
        def _enforce_cache_budget(
            cache_dir: str,
            max_gb: float = 100.0,
            near_ratio: float = 0.90,
            keep_ratio: float = 0.10,
        ):
            try:
                import glob
                max_bytes = int(max_gb * (1024**3))
                if max_bytes <= 0:
                    return

                trigger_bytes = int(max_bytes * float(near_ratio))
                keep_bytes = int(max_bytes * float(keep_ratio))

                files = glob.glob(os.path.join(cache_dir, "*"))
                if not files:
                    return

                sizes = []
                total = 0
                for p in files:
                    try:
                        st = os.stat(p)
                        sz = int(st.st_size)
                        mt = float(st.st_mtime)
                        total += sz
                        sizes.append((mt, sz, p))
                    except Exception:
                        pass

                if total < trigger_bytes:
                    return

                sizes.sort(key=lambda x: x[0])  # old -> new
                for _, sz, p in sizes:
                    if total <= keep_bytes:
                        break
                    try:
                        os.remove(p)
                        total -= sz
                    except Exception:
                        pass
            except Exception:
                return

        def _mel2token_to_dur(m2p: torch.LongTensor) -> torch.LongTensor:
            if m2p.numel() == 0:
                return torch.zeros(0, dtype=torch.long)
            mx = int(m2p.max().item()) if m2p.numel() > 0 else 0
            if mx <= 0:
                return torch.zeros(0, dtype=torch.long)
            cnts = torch.bincount(m2p.clamp_min(0), minlength=mx + 1)
            return cnts[1:].to(torch.long)

        def _to_int_list_maybe(x):
            if x is None:
                return None
            if isinstance(x, str):
                s = x.strip()
                if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
                    try:
                        parsed = json.loads(s.replace('(', '[').replace(')', ']'))
                        return _to_int_list_maybe(parsed)
                    except Exception:
                        pass
                nums = re.findall(r'-?\d+\.?\d*', s)
                if len(nums) == 0:
                    return None
                try:
                    return [int(float(t)) for t in nums]
                except Exception:
                    return None

            if isinstance(x, (list, tuple, np.ndarray)):
                out = []
                for v in x:
                    try:
                        if isinstance(v, str):
                            v = v.strip()
                            if v == '':
                                out.append(0)
                                continue
                            out.append(int(float(v)))
                        elif isinstance(v, (int, np.integer)):
                            out.append(int(v))
                        elif isinstance(v, (float, np.floating)):
                            out.append(int(v))
                        else:
                            out.append(int(v))
                    except Exception:
                        out.append(0)
                return out

            try:
                return [int(x)]
            except Exception:
                return None

        def _sha1(s: str) -> str:
            return hashlib.sha1(s.encode('utf-8')).hexdigest()

        def _atomic_write(path: str, data: bytes):
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

        # --------- params ----------
        sr = int(hparams['audio_sample_rate'])
        fm = int(hparams['frames_multiple'])
        hop = int(hparams['hop_size'])
        stride = int(hparams.get('vae_stride', 8))
        fm_wav = fm * hop
        latent_hop = hop * stride

        # --------- worker shared caches ----------
        def _get_wav_lru():
            return get_from_global_stores('wav_lru_cache', global_stores, lambda: OrderedDict())

        def _lru_get(cache: OrderedDict, k):
            v = cache.get(k, None)
            if v is not None:
                cache.move_to_end(k)
            return v

        def _lru_put(cache: OrderedDict, k, v, max_items: int):
            cache[k] = v
            cache.move_to_end(k)
            while len(cache) > max_items:
                cache.popitem(last=False)

        def _get_cache_dir():
            cache_dir = get_from_global_stores(
                'tos_cache_dir', global_stores,
                lambda: hparams.get('tos_cache_dir', '/dev/shm/zyxc_tos_cache')
            )
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except Exception:
                pass
            return cache_dir

        # --------- audio loading ----------
        def _load_wav_local(wav_rel: str, start_sec: float = None, end_sec: float = None):
            if not wav_rel:
                return None

            wav_rel = wav_rel.replace('wavs/16k160/', 'wavs/24k/')
            full_wav = os.path.join('/mnt/bn/lq-ads-aigc/renyi/tts_datasets_bak/InteractiveDialogue', wav_rel)
            if not os.path.exists(full_wav):
                return None

            lru = _get_wav_lru()
            max_items = int(hparams.get('wav_mem_cache_items', 16))

            if start_sec is None or end_sec is None:
                ck = ('local_full', full_wav, sr)
                hit = _lru_get(lru, ck)
                if hit is not None:
                    return hit

            try:
                if start_sec is not None and end_sec is not None and end_sec > start_sec:
                    info = torchaudio.info(full_wav)
                    sr0 = int(info.sample_rate)
                    frame_offset = int(max(0, round(float(start_sec) * sr0)))
                    num_frames = int(max(1, round((float(end_sec) - float(start_sec)) * sr0)))
                    wav_t, sr_ = torchaudio.load(full_wav, frame_offset=frame_offset, num_frames=num_frames)
                else:
                    wav_t, sr_ = torchaudio.load(full_wav)
            except Exception:
                return None

            if wav_t is None or wav_t.numel() == 0:
                return None

            if wav_t.ndim == 2 and wav_t.size(0) > 1:
                wav_t = wav_t.mean(dim=0)
            else:
                wav_t = wav_t.squeeze(0)

            sr_ = int(sr_)
            if sr_ != sr:
                try:
                    wav_t = AF.resample(wav_t, orig_freq=sr_, new_freq=sr)
                except Exception:
                    try:
                        wav_t = torchaudio.transforms.Resample(orig_freq=sr_, new_freq=sr)(wav_t)
                    except Exception:
                        return None

            wav_np = wav_t.detach().cpu().to(torch.float32).numpy()
            if wav_np.size == 0:
                return None
            wav_np = np.asarray(wav_np, dtype=np.float32)

            if start_sec is None or end_sec is None:
                _lru_put(lru, ck, wav_np, max_items=max_items)
            return wav_np

        def _get_tos_client():
            cluster = os.environ.get('CLUSTER', '').lower()
            if cluster == 'va':
                return TosClient(bucket='sa-ag-sg-research-sg')
            return TosClient(bucket='humanaigc-ads')

        def _load_wav_from_tos(wav_k: str, start_sec: float = None, end_sec: float = None):
            if not wav_k:
                return None

            # ==== 这里加过滤逻辑 ====
            # 任何 key 里包含 /apple/ 或 /xmly/ 的，直接当成无效样本
            if "/apple/" in wav_k or "/xmly/" in wav_k:
                return None
            # =======================

            cache_dir = _get_cache_dir()
            key_hash = _sha1(wav_k)

            # full wav npy cache
            if start_sec is None or end_sec is None:
                npy_path = os.path.join(cache_dir, f'{key_hash}.sr{sr}.npy')
                if os.path.exists(npy_path):
                    try:
                        wav_np = np.load(npy_path, allow_pickle=False)
                        if wav_np is not None and wav_np.size > 0:
                            return np.asarray(wav_np, dtype=np.float32)
                    except Exception:
                        pass

            # m4a cache
            m4a_path = os.path.join(cache_dir, f'{key_hash}.m4a')
            if not os.path.exists(m4a_path) or os.path.getsize(m4a_path) == 0:
                tos_client: TosClient = get_from_global_stores('tos_client', global_stores, _get_tos_client)
                try:
                    data = tos_client.get_object(wav_k)
                except Exception:
                    return None
                if data is None:
                    return None
                try:
                    _atomic_write(m4a_path, data)
                except Exception:
                    m4a_path = None
            _enforce_cache_budget(cache_dir)

            def _decode_with_ffmpeg(path: str, s_sec: float, e_sec: float):
                if path is None or not os.path.exists(path):
                    return None
                if s_sec is None or e_sec is None or e_sec <= s_sec:
                    return None
                cmd = [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{float(s_sec):.6f}", "-to", f"{float(e_sec):.6f}",
                    "-i", path,
                    "-f", "f32le", "-ac", "1", "-ar", str(sr),
                    "pipe:1"
                ]
                try:
                    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                    if p.returncode != 0 or p.stdout is None or len(p.stdout) == 0:
                        return None
                    wav_np = np.frombuffer(p.stdout, dtype=np.float32)
                    if wav_np.size == 0:
                        return None
                    return wav_np
                except Exception:
                    return None

            if start_sec is not None and end_sec is not None and end_sec > start_sec:
                if m4a_path is not None:
                    seg = _decode_with_ffmpeg(m4a_path, start_sec, end_sec)
                    if seg is not None:
                        return np.asarray(seg, dtype=np.float32)

            # full decode (with mem LRU)
            lru = _get_wav_lru()
            max_items = int(hparams.get('wav_mem_cache_items', 16))
            ck = ('tos_full', wav_k, sr)
            hit = _lru_get(lru, ck)
            if hit is not None:
                return hit

            try:
                if m4a_path is not None and os.path.exists(m4a_path):
                    wav_t, sr_ = torchaudio.load(m4a_path)
                else:
                    tos_client: TosClient = get_from_global_stores('tos_client', global_stores, _get_tos_client)
                    data = tos_client.get_object(wav_k)
                    if data is None:
                        return None
                    with tempfile.NamedTemporaryFile(suffix='.m4a', dir='/dev/shm', delete=True) as f:
                        f.write(data)
                        f.flush()
                        wav_t, sr_ = torchaudio.load(f.name)
            except Exception:
                return None

            if wav_t is None or wav_t.numel() == 0:
                return None

            if wav_t.ndim == 2 and wav_t.size(0) > 1:
                wav_t = wav_t.mean(dim=0)
            else:
                wav_t = wav_t.squeeze(0)

            sr_ = int(sr_)
            if sr_ != sr:
                try:
                    wav_t = AF.resample(wav_t, orig_freq=sr_, new_freq=sr)
                except Exception:
                    try:
                        wav_t = torchaudio.transforms.Resample(orig_freq=sr_, new_freq=sr)(wav_t)
                    except Exception:
                        return None

            wav_np = wav_t.detach().cpu().to(torch.float32).numpy()
            if wav_np.size == 0:
                return None
            wav_np = np.asarray(wav_np, dtype=np.float32)

            try:
                npy_path = os.path.join(cache_dir, f'{key_hash}.sr{sr}.npy')
                if not os.path.exists(npy_path):
                    np.save(npy_path, wav_np, allow_pickle=False)
            except Exception:
                pass
            _enforce_cache_budget(cache_dir)

            _lru_put(lru, ck, wav_np, max_items=max_items)
            return wav_np

        # --------- ref augmentation helpers ( 新增) ----------
        def _apply_noise(seg: torch.Tensor, snr_db: float) -> torch.Tensor:
            # seg: [T]
            if seg.numel() <= 0:
                return seg
            seg_f = seg.to(torch.float32)
            rms = torch.sqrt(torch.mean(seg_f * seg_f) + 1e-12)
            # snr_db 越大噪声越小
            noise_rms = rms / (10.0 ** (float(snr_db) / 20.0))
            noise = torch.randn_like(seg_f) * noise_rms
            out = seg_f + noise
            return out

        def _apply_light_reverb(seg: torch.Tensor, wet: float) -> torch.Tensor:
            """
            轻量“混响/空间感”：多 tap 延迟(很小) + 小平滑 + wet 混合
            wet < 0.2，尽量不影响可懂度
            """
            if seg.numel() <= 0:
                return seg
            x = seg.to(torch.float32)
            y = x.clone()

            # 2~4 个延迟 tap，延迟 10~55ms，增益较小
            n_taps = random.randint(2, 4)
            for _ in range(n_taps):
                d = int(sr * random.uniform(0.010, 0.055))
                g = random.uniform(0.04, 0.14)  # 小一点，避免糊
                if d > 0 and d < y.numel():
                    # 用递归 y 叠加会更“混响”，但可能爆；这里用 y 做一次弱反馈，仍然安全
                    y[d:] = y[d:] + g * y[:-d]

            # 轻微平滑（模拟空气吸收），kernel 很小，开销低
            k = random.choice([3, 5, 7])
            if y.numel() > k:
                ker = torch.ones((k,), device=y.device, dtype=y.dtype) / float(k)
                y = F.conv1d(y[None, None, :], ker[None, None, :], padding=k // 2)[0, 0, :]

            # 能量归一，避免整体变响/过载
            x_peak = x.abs().max().clamp_min(1e-6)
            y_peak = y.abs().max().clamp_min(1e-6)
            y = y / y_peak * x_peak

            wet = float(max(0.0, min(wet, 0.35)))
            out = (1.0 - wet) * x + wet * y
            return out

        # --------- main ----------
        if meta_item is None or not isinstance(meta_item, dict):
            return

        turns = meta_item.get('turns', [])
        segments = meta_item.get('segments', [])
        if not turns or not segments:
            return

        wav_cache_turn = {}
        wav_cache_full = {}

        for segment in segments:
            spk_map = {}  # raw_spk -> sid(1-based)
            ph_list, tone_list, m2p_list, spk_mask_ph_list = [], [], [], []

            res_text = ''
            kept_turn_sids = []

            change_of_spk = 1
            last_sid_for_text = None

            ref_wav_start_turn_idx = 0
            ref_start_sec = None

            seg_has_ts = isinstance(segment, dict) and ('start' in segment and 'end' in segment)
            seg_start_sec = float(segment.get('start', 0.0)) if seg_has_ts else None
            seg_end_sec = float(segment.get('end', 0.0)) if seg_has_ts else None

            #  turn_ranges：记录每个 turn 在 wav_cat（slice 前的基准）里的 sample 区间，后面会映射到最终 wav_cat
            turn_ranges = []  # list of dict: {sid, start, end}

            if seg_has_ts:
                segment_audio_src = None
                turn_time_info = []
                turn_sample_lens = []
            else:
                wav_lst = []
                cum = 0  # sample 累计，用于 turn_ranges

            for turn_idx in segment.get('turn_idxs', []):
                if turn_idx < 0 or turn_idx >= len(turns):
                    continue
                turn = turns[turn_idx]

                text = turn.get('text')
                if not text or not str(text).strip():
                    continue

                raw_spk = turn.get('spk', 'spk_unk')
                sid = spk_map.setdefault(raw_spk, len(spk_map) + 1)  # 1-based

                ph_enc = turn.get('phone_encoded')
                m2p_raw = turn.get('mel2ph')
                if ph_enc is None or len(ph_enc) == 0 or m2p_raw is None or len(m2p_raw) == 0:
                    continue

                # ---------- audio ----------
                if seg_has_ts:
                    t_start = float(turn.get('start', seg_start_sec if seg_start_sec is not None else 0.0))
                    t_end = float(turn.get('end', t_start))
                    if seg_start_sec is not None:
                        t_start = max(t_start, seg_start_sec)
                    if seg_end_sec is not None:
                        t_end = min(t_end, seg_end_sec)
                    if t_end <= t_start:
                        continue

                    wav_path = turn.get('wav_path', None)
                    wav_k = turn.get('wav_k', None)
                    if segment_audio_src is None:
                        if wav_path:
                            segment_audio_src = ('local', wav_path)
                        elif wav_k:
                            segment_audio_src = ('tos', wav_k)

                    turn_time_info.append({'sid': sid, 'start_sec': t_start, 'end_sec': t_end})
                    turn_sample_lens.append(int(max(0.0, (t_end - t_start)) * sr))
                else:
                    wav_np = None
                    wav_path = turn.get('wav_path', None)
                    wav_k = turn.get('wav_k', None)

                    cache_key = None
                    if wav_path:
                        cache_key = ('local', wav_path)
                    elif wav_k:
                        cache_key = ('tos', wav_k)

                    if cache_key is not None and cache_key in wav_cache_turn:
                        wav_np = wav_cache_turn[cache_key]
                    else:
                        if wav_path:
                            wav_np = _load_wav_local(wav_path)
                        elif wav_k:
                            wav_np = _load_wav_from_tos(wav_k)
                        if wav_np is None or len(wav_np) == 0:
                            continue
                        if cache_key is not None:
                            wav_cache_turn[cache_key] = wav_np

                    wav_lst.append(wav_np)

                    #  记录该 turn 在拼接 wav_cat 内的区间
                    L = int(len(wav_np))
                    if L > 0:
                        turn_ranges.append({'sid': int(sid), 'start': int(cum), 'end': int(cum + L)})
                        cum += L

                # ---------- text & change_of_spk ----------
                spk_tag_id = int(sid)
                if last_sid_for_text is None:
                    res_text += f'<SPK>{spk_tag_id}</SPK>' + str(text)
                    last_sid_for_text = sid
                else:
                    if int(sid) != int(last_sid_for_text):
                        change_of_spk += 1

                        #  进入第三段（AB 后的第三段）：记录“第三段开始之前”的边界
                        if change_of_spk == 3:
                            if seg_has_ts:
                                ref_start_sec = float(turn_time_info[-1]['start_sec'])
                                ref_wav_start_turn_idx = max(0, len(turn_sample_lens) - 1)
                            else:
                                # wav_lst 已 append 当前段，所以要 -1 变成第三段的 index
                                ref_wav_start_turn_idx = max(0, len(wav_lst) - 1)

                        res_text += f'<SPK>{spk_tag_id}</SPK>' + str(text)
                        last_sid_for_text = sid
                    else:
                        res_text += str(text)

                kept_turn_sids.append(sid)

                # ---------- tensors ----------
                ph_i = torch.as_tensor(ph_enc, dtype=torch.long)

                tone_raw = turn.get('tone_encoded')
                tone_enc_clean = _to_int_list_maybe(tone_raw) if tone_raw is not None else None
                if (tone_enc_clean is None) or (len(tone_enc_clean) != len(ph_enc)):
                    tn_i = torch.zeros_like(ph_i, dtype=torch.long)
                else:
                    tn_i = torch.as_tensor(tone_enc_clean, dtype=torch.long)

                m2p_i = torch.as_tensor(m2p_raw, dtype=torch.long)

                ph_list.append(ph_i)
                tone_list.append(tn_i)
                m2p_list.append(m2p_i)
                spk_mask_ph_list.append(torch.full((ph_i.numel(),), int(sid), dtype=torch.long))

            # ---------- segment-level filters ----------
            if len(kept_turn_sids) < 4:
                continue
            num_conversations = 1
            for i in range(1, len(kept_turn_sids)):
                if kept_turn_sids[i] != kept_turn_sids[i - 1]:
                    num_conversations += 1
            if num_conversations // 2 < 2:
                continue

            # ====== INSERT: shuffle speaker ids within this segment (keep consistency) ======
            shuffle_spk_id = bool(hparams.get("shuffle_spk_id", False))
            sid_remap = None
            n_spk = int(len(spk_map))

            if shuffle_spk_id and n_spk > 1:
                # old sid: 1..n_spk  -> new sid: a random permutation of 1..n_spk
                new_ids = list(range(1, n_spk + 1))
                random.shuffle(new_ids)
                sid_remap = {old: new_ids[old - 1] for old in range(1, n_spk + 1)}

                # 1) remap <SPK> tags in text
                def _remap_tag(m):
                    old = int(m.group(1))
                    new = int(sid_remap.get(old, old))
                    return f"<SPK>{new}</SPK>"

                res_text = re.sub(r"<SPK>(\d+)</SPK>", _remap_tag, res_text)

                # 2) remap kept_turn_sids (可选，后面你基本不再用它，但保持一致更稳)
                kept_turn_sids = [int(sid_remap.get(int(x), int(x))) for x in kept_turn_sids]

                # 3) remap turn_time_info (ts 模式后面会用它生成 turn_ranges)
                if seg_has_ts:
                    for ti in turn_time_info:
                        ti['sid'] = int(sid_remap.get(int(ti['sid']), int(ti['sid'])))

                # 4) remap turn_ranges (非 ts 模式这时已构建)
                if (not seg_has_ts) and turn_ranges:
                    for r in turn_ranges:
                        r['sid'] = int(sid_remap.get(int(r['sid']), int(r['sid'])))

            # ---------- build wav_cat_np + build turn_ranges for ts / non-ts ----------
            if seg_has_ts:
                if segment_audio_src is None or len(turn_time_info) == 0:
                    continue

                kind, value = segment_audio_src
                s_sec = seg_start_sec if seg_start_sec is not None else min(t['start_sec'] for t in turn_time_info)
                e_sec = seg_end_sec if seg_end_sec is not None else max(t['end_sec'] for t in turn_time_info)
                if e_sec <= s_sec:
                    continue

                if kind == 'local':
                    # 本地文件：直接按 [s_sec, e_sec] 读 segment，不走 full wav
                    wav_cat_np = _load_wav_local(value, start_sec=s_sec, end_sec=e_sec)
                    if wav_cat_np is None or len(wav_cat_np) == 0:
                        continue
                    wav_cat_np = np.asarray(wav_cat_np, dtype=np.float32)

                else:
                    # TOS：保持原来的全量 + 缓存逻辑
                    full_key = (kind, value, sr)
                    wav_full = wav_cache_full.get(full_key, None)
                    if wav_full is None:
                        wav_full = _load_wav_from_tos(value)
                        if wav_full is None or len(wav_full) == 0:
                            continue
                        wav_cache_full[full_key] = wav_full

                    total_len = int(len(wav_full))
                    if total_len <= 0:
                        continue

                    s_idx = int(max(0, min(round(float(s_sec) * sr), total_len - 1)))
                    e_idx = int(max(s_idx + 1, min(round(float(e_sec) * sr), total_len)))
                    wav_cat_np = wav_full[s_idx:e_idx]
                    if wav_cat_np is None or len(wav_cat_np) == 0:
                        continue
                    wav_cat_np = np.asarray(wav_cat_np, dtype=np.float32)

                # ts 模式下构建 turn_ranges（相对 segment slice 起点 s_sec）
                turn_ranges = []
                for ti in turn_time_info:
                    sid_i = int(ti['sid'])
                    st = int(round((float(ti['start_sec']) - float(s_sec)) * sr))
                    ed = int(round((float(ti['end_sec']) - float(s_sec)) * sr))
                    st = max(0, st)
                    ed = max(st, ed)
                    if ed > st:
                        turn_ranges.append({'sid': sid_i, 'start': st, 'end': ed})

            else:
                # 非 ts：直接把每个 turn 的 wav 拼起来
                if not wav_lst:
                    continue
                # 过滤掉长度为 0 的
                wav_lst_valid = [w for w in wav_lst if w is not None and len(w) > 0]
                if not wav_lst_valid:
                    continue

                try:
                    wav_cat_np = np.concatenate(
                        [np.asarray(w, dtype=np.float32) for w in wav_lst_valid],
                        axis=0
                    )
                except Exception:
                    continue

                if wav_cat_np.size == 0:
                    continue

            # ---------- trim to frames_multiple ----------
            wav_cat = torch.from_numpy(wav_cat_np)
            T = (int(wav_cat.shape[0]) // fm_wav) * fm_wav
            if T <= 0:
                continue
            wav_cat = wav_cat[:T]
            if wav_cat.numel() == 0:
                continue

            #  turn_ranges 裁剪到 [0, T)
            if turn_ranges:
                new_ranges = []
                for r in turn_ranges:
                    st = int(max(0, min(r['start'], T)))
                    ed = int(max(0, min(r['end'], T)))
                    if ed > st:
                        new_ranges.append({'sid': int(r['sid']), 'start': st, 'end': ed})
                turn_ranges = new_ranges

            # ---------- fix mel2ph offsets & concat ----------
            ph_offset = 0
            m2p_fixed = []
            for m2p_i, ph_i in zip(m2p_list, ph_list):
                if ph_offset > 0:
                    m2p_i = m2p_i + (m2p_i > 0).long() * ph_offset
                m2p_fixed.append(m2p_i)
                ph_offset += int(ph_i.numel())

            ph_token = torch.cat(ph_list, dim=0)
            tone = torch.cat(tone_list, dim=0)
            mel2ph = torch.cat(m2p_fixed, dim=0)
            spk_mask_ph = torch.cat(spk_mask_ph_list, dim=0)

            # ====== INSERT: remap phone-level spk_mask to match <SPK> ids ======
            if sid_remap is not None and n_spk > 1 and spk_mask_ph.numel() > 0:
                lut = torch.arange(0, n_spk + 1, dtype=torch.long)  # 0 保留
                for old, new in sid_remap.items():
                    if 0 <= int(old) <= n_spk:
                        lut[int(old)] = int(new)
                spk_mask_ph = lut[spk_mask_ph.clamp(0, n_spk)]
            # ====== INSERT END ======

            mel_len = int(wav_cat.shape[0] // hop)
            if mel_len <= 0:
                continue

            if mel2ph.numel() < mel_len:
                pad_len = mel_len - mel2ph.numel()
                last = mel2ph[-1] if mel2ph.numel() > 0 else torch.tensor(0, dtype=torch.long)
                mel2ph = torch.cat([mel2ph, last.repeat(pad_len)], dim=0)
            mel2ph = mel2ph[:mel_len]
            mel2ph = mel2ph[: (mel_len // fm) * fm]
            if mel2ph.numel() == 0:
                continue

            dur = _mel2token_to_dur(mel2ph)

            mel2ph_sparse = None
            if hparams.get('use_sparse_dur', False):
                _m2a = compute_mel2aug_from_dur(
                    dur.cpu().numpy().tolist(),
                    gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                    gap_frames=hparams.get('sparse_dur_frames', 4),
                    gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                    min_keep=hparams.get('sparse_dur_min_keep', 1),
                    keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                    symmetric=hparams.get('sparse_dur_symmetric', True),
                )
                mel2ph_sparse = torch.as_tensor(_m2a, dtype=torch.long)
                target_len = mel2ph.shape[0]
                if mel2ph_sparse.numel() < target_len:
                    pad_len = target_len - mel2ph_sparse.numel()
                    last = mel2ph_sparse[-1] if mel2ph_sparse.numel() > 0 else torch.tensor(0, dtype=torch.long)
                    mel2ph_sparse = torch.cat([mel2ph_sparse, last.repeat(pad_len)], dim=0)
                mel2ph_sparse = mel2ph_sparse[:target_len]

            # text/caption
            text_merged = res_text
            caption_merged = text_merged

            # ---------- ctx split ----------
            # ref 边界精确在第三段开始处（再随机右移）
            if seg_has_ts and (ref_start_sec is not None) and (s_sec is not None):
                pre_len = int(round(max(0.0, float(ref_start_sec) - float(s_sec)) * sr))
            else:
                # 非 ts：ref_wav_start_turn_idx 是第三段对应的 turn index（0-based）
                # 想要第三段开始处：取该段 start
                pre_len = 0
                if turn_ranges and (ref_wav_start_turn_idx > 0) and (ref_wav_start_turn_idx < len(turn_ranges)):
                    pre_len = int(turn_ranges[ref_wav_start_turn_idx]['start'])

            ref_wav_start = (int(pre_len) // fm_wav) * fm_wav
            max_idx = min(int(wav_cat.shape[0] * 0.9), int(wav_cat.shape[0]) - 20000)
            if max_idx > ref_wav_start:
                ref_wav_start = (random.randint(ref_wav_start, max_idx) // fm_wav) * fm_wav
            ref_wav_start = int(max(0, min(ref_wav_start, int(wav_cat.shape[0]))))

            # ---------- build ctx_wav ----------
            ctx_wav = wav_cat[:ref_wav_start].clone()

            # 新增：ref(ctx) 内按 spk 做轻量增强（每个 sid 独立随机）
            # 概率：0.5 不加；否则 0.5 混响 / 0.5 噪声；强度都很轻
            if ctx_wav.numel() > 0 and turn_ranges:
                # 只看 ref 区间内出现过的 sid
                sids_in_ref = set()
                for r in turn_ranges:
                    if r['start'] < ref_wav_start and r['end'] > 0:
                        sids_in_ref.add(int(r['sid']))

                spk_aug = {}  # sid -> ('none'|'reverb'|'noise', param)
                for sid_i in sids_in_ref:
                    if random.random() < 0.3:
                        spk_aug[sid_i] = ('none', None)
                    else:
                        if random.random() < 0.5:
                            wet = random.uniform(0.05, 0.30)
                            spk_aug[sid_i] = ('reverb', wet)
                        else:
                            snr_db = random.uniform(18.0, 40.0)  # 高 SNR，基本不伤内容
                            spk_aug[sid_i] = ('noise', snr_db)

                # 对 ref 内每个 turn 片段（或其与 ref 的交集）应用对应 sid 的增强
                for r in turn_ranges:
                    st = int(r['start'])
                    ed = int(r['end'])
                    if st >= ref_wav_start or ed <= 0:
                        continue
                    sid_i = int(r['sid'])
                    aug = spk_aug.get(sid_i, ('none', None))
                    if aug[0] == 'none':
                        continue

                    s0 = max(0, st)
                    e0 = min(ref_wav_start, ed)
                    if e0 <= s0:
                        continue

                    seg = ctx_wav[s0:e0]
                    if aug[0] == 'reverb':
                        wet = float(aug[1])
                        seg2 = _apply_light_reverb(seg, wet=wet)
                    else:
                        snr_db = float(aug[1])
                        seg2 = _apply_noise(seg, snr_db=snr_db)

                    ctx_wav[s0:e0] = seg2.to(ctx_wav.dtype)

                # 最后轻微防爆（不硬裁剪得太狠）
                peak = ctx_wav.abs().max().item() if ctx_wav.numel() > 0 else 0.0
                if peak > 1.2:  # 给一点余量
                    ctx_wav = ctx_wav / peak

            # ctx_mask：按 latent 长度建
            latent_len = int(wav_cat.shape[0] // latent_hop)
            max_lat = int(hparams.get("max_seq_len", 7500))
            if latent_len > max_lat:
                print(f"latent_len {latent_len} > max_lat {max_lat}")
                continue
            min_lat = int(hparams.get("min_seq_len", 25))
            if latent_len < min_lat:
                print(f"latent_len {latent_len} < min_lat {min_lat}")
                continue

            ctx_latent_len = int(ref_wav_start // latent_hop)
            if latent_len <= 0:
                continue
            ctx_mask = torch.zeros((latent_len, 1), dtype=torch.float32)
            if ctx_latent_len > 0:
                ctx_mask[:min(ctx_latent_len, latent_len)] = 1.0

            item = {
                'wav': wav_cat,
                'text': text_merged,
                'caption': caption_merged,
                'spk_mask': spk_mask_ph.to(torch.long),
                'ctx_wav': ctx_wav,          #  这里已经是“ref增强后”的 ctx
                'ctx_mask': ctx_mask,
                'ph_token': ph_token.to(torch.long),
                'tone': tone.to(torch.long),
                'mel2ph': mel2ph.to(torch.long),
                'dur': dur.to(torch.long),
                'len': latent_len,
            }
            if mel2ph_sparse is not None:
                item['mel2ph_sparse'] = mel2ph_sparse.to(torch.long)

            yield item



    def collater(self, samples):
        """
        支持 fast-dataloader（samples 由若干条样本组成）与回退备份逻辑。
        汇总字段尽量与 DiTWavTextDataset.collater 对齐：
            - 'spk_mask'：phone 对齐的说话人 id（1-based）
        """
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]

        if len(samples) == 0:
            if hasattr(self, 'backup_batch') and self.backup_batch is not None:
                print('use backup batch!')
                return self.backup_batch
            else:
                print('no batch to take!')
                return {}

        wavs = collate_xd([s['wav'] for s in samples], 0.0)
        wav_lengths = torch.LongTensor([s['wav'].shape[0] for s in samples])
        ctx_wavs = collate_xd([s['ctx_wav'] for s in samples], 0.0)

        batch = {
            'nsamples': len(samples),
            'wavs': wavs,
            'wav_lengths': wav_lengths,
            'ctx_wavs': ctx_wavs,
            'ctx_mask': collate_xd([s['ctx_mask'] for s in samples], 0),
            'text': [s['text'] for s in samples],
            'caption': [s['caption'] for s in samples] if 'caption' in samples[0] else None,
        }

        if 'ph_token' in samples[0]:
            batch['ph_tokens'] = collate_xd([s['ph_token'] for s in samples], 0)
            batch['txt_lengths'] = torch.LongTensor([s['ph_token'].numel() for s in samples])
        if 'tone' in samples[0]:
            batch['tone'] = collate_xd([s['tone'] for s in samples], 0)
        if 'mel2ph' in samples[0]:
            batch['mel2ph'] = collate_xd([s['mel2ph'] for s in samples], 0)
        if 'dur' in samples[0]:
            batch['dur'] = collate_xd([s['dur'] for s in samples], 0)
            batch['dur_len'] = torch.LongTensor([s['dur'].shape[0] for s in samples])
        if 'mel2ph_sparse' in samples[0]:
            batch['mel2ph_sparse'] = collate_xd([s['mel2ph_sparse'] for s in samples], 0)

        if 'spk_mask' in samples[0]:
            batch['spk_mask'] = collate_xd([s['spk_mask'] for s in samples], 0)  # [B, T_ph]

        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch

class DialogueSegmentEmbDataset_Mix(DialogueSegmentEmbDataset_v2):

    def get_dataset_meta(self):
        import os
        import math

        xyz_glob = self.hparams.get(
            'xyz_meta_glob',
            '/mnt/bn/sa-ag-data/zhangyu.34/data/speech/XYZ_20w/metas_with_tson_16k160_final_hard_emo/*/*.json'
        )
        singlespk_glob = self.hparams.get(
            'singlespk_meta_glob',
            '/mnt/bn/sa-ag-data/zhangyu.34/code/Venus/user/singlespk_data1/*.jsonl'
        )
        prompttts_glob = self.hparams.get(
            'prompttts_meta_glob',
            '/mnt/bn/sa-ag-data/panchanghao/code/mega-data-pipeline/user/panchanghao/prompttts_multispk_mfa/multispk_meta/seg_filter_new2/*.jsonl'
        )
        zyxc_glob = self.hparams.get(
            'zyxc_meta_glob',
            '/mnt/bn/sa-ag-data/panchanghao/code/mega-data-pipeline/user/zhangyu/zhiyuexingchen_dialogue/meta_merged_ph_tone_encoded_final_hard_emo_090/*.jsonl'
        )
        expressive_glob = self.hparams.get(
            'expressive_meta_glob',
            '/mnt/bn/sa-ag-data/liruiqi/data/speech/expressive_ad/filter/*.jsonl'
        )

        enable_xyz = bool(self.hparams.get('enable_xyz', False))
        enable_singlespk = bool(self.hparams.get('enable_singlespk', False))
        enable_prompttts = bool(self.hparams.get('enable_prompttts', True))
        enable_zyxc = bool(self.hparams.get('enable_zyxc', True))
        enable_expressive = bool(self.hparams.get('enable_expressive', False))

        # ===== 1) xyz: 每个文件 = 1 个 meta_item =====
        xyz_paths = multiprocess_glob(xyz_glob, num_workers=128) if enable_xyz else []
        xyz_len = int(len(xyz_paths))

        # 统一 chunk_size
        chunk_size = int(self.hparams.get('reader_chunk_size', 128))

        # ===== 2) singlespk: jsonl 按 chunk 读，每行 = 1 个 meta_item =====
        singlespk_paths = multiprocess_glob(singlespk_glob, num_workers=128) if enable_singlespk else []
        singlespk_packs = []
        singlespk_offset_chunks = 0
        for p in singlespk_paths:
            idx_path = p + '.idx'
            if not os.path.isfile(idx_path):
                try:
                    build_jsonl_index(p, idx_path, use_tqdm=False)
                except Exception:
                    pass
            try:
                n_lines = int(count_jsonl_n_lines(idx_path))
            except Exception:
                n_lines = 0
            n_chunks = int(math.ceil(n_lines / chunk_size)) if n_lines > 0 else 0
            singlespk_packs.append({
                'source': 'singlespk',
                'path': p,
                'idx_path': idx_path,
                'n_lines': n_lines,
                'chunk_size': chunk_size,
                'n_chunks': n_chunks,
                'offset_chunks': singlespk_offset_chunks,
            })
            singlespk_offset_chunks += n_chunks
        singlespk_chunks = int(singlespk_offset_chunks)

        # ===== 3) prompttts: jsonl 按 chunk 读，每行 = 1 个 meta_item =====
        prompttts_paths = multiprocess_glob(prompttts_glob, num_workers=128) if enable_prompttts else []
        prompttts_packs = []
        prompttts_offset_chunks = 0
        for p in prompttts_paths:
            idx_path = p + '.idx'
            if not os.path.isfile(idx_path):
                try:
                    build_jsonl_index(p, idx_path, use_tqdm=False)
                except Exception:
                    pass
            try:
                n_lines = int(count_jsonl_n_lines(idx_path))
            except Exception:
                n_lines = 0
            n_chunks = int(math.ceil(n_lines / chunk_size)) if n_lines > 0 else 0
            prompttts_packs.append({
                'source': 'prompttts',
                'path': p,
                'idx_path': idx_path,
                'n_lines': n_lines,
                'chunk_size': chunk_size,
                'n_chunks': n_chunks,
                'offset_chunks': prompttts_offset_chunks,
            })
            prompttts_offset_chunks += n_chunks
        prompttts_chunks = int(prompttts_offset_chunks)

        # ===== 4) expressive 单人数据: jsonl 按 chunk 读，每行 = 1 个 meta_item =====
        expressive_paths = multiprocess_glob(expressive_glob, num_workers=128) if enable_expressive else []
        expressive_packs = []
        expressive_offset_chunks = 0
        for p in expressive_paths:
            idx_path = p + '.idx'
            if not os.path.isfile(idx_path):
                try:
                    build_jsonl_index(p, idx_path, use_tqdm=False)
                except Exception:
                    pass
            try:
                n_lines = int(count_jsonl_n_lines(idx_path))
            except Exception:
                n_lines = 0
            n_chunks = int(math.ceil(n_lines / chunk_size)) if n_lines > 0 else 0
            expressive_packs.append({
                'source': 'expressive',
                'path': p,
                'idx_path': idx_path,
                'n_lines': n_lines,
                'chunk_size': chunk_size,
                'n_chunks': n_chunks,
                'offset_chunks': expressive_offset_chunks,
            })
            expressive_offset_chunks += n_chunks
        expressive_chunks = int(expressive_offset_chunks)

        # ===== 5) zyxc: jsonl 按 chunk 读，每行 = 1 个 meta_item =====
        zyxc_paths = multiprocess_glob(zyxc_glob, num_workers=128) if enable_zyxc else []
        zyxc_packs = []
        zyxc_offset_chunks = 0
        for p in zyxc_paths:
            idx_path = p + '.idx'
            if not os.path.isfile(idx_path):
                try:
                    build_jsonl_index(p, idx_path, use_tqdm=False)
                except Exception:
                    pass
            try:
                n_lines = int(count_jsonl_n_lines(idx_path))
            except Exception:
                n_lines = 0
            n_chunks = int(math.ceil(n_lines / chunk_size)) if n_lines > 0 else 0
            zyxc_packs.append({
                'source': 'zyxc',
                'path': p,
                'idx_path': idx_path,
                'n_lines': n_lines,
                'chunk_size': chunk_size,
                'n_chunks': n_chunks,
                'offset_chunks': zyxc_offset_chunks,
            })
            zyxc_offset_chunks += n_chunks
        zyxc_chunks = int(zyxc_offset_chunks)

        meta = {
            'xyz_paths': xyz_paths,
            'xyz_len': xyz_len,

            'singlespk_packs': singlespk_packs,
            'singlespk_chunks': singlespk_chunks,

            'prompttts_packs': prompttts_packs,
            'prompttts_chunks': prompttts_chunks,

            'expressive_packs': expressive_packs,
            'expressive_chunks': expressive_chunks,

            'zyxc_packs': zyxc_packs,
            'zyxc_chunks': zyxc_chunks,
        }

        dataset_len = int(xyz_len + singlespk_chunks + prompttts_chunks + expressive_chunks + zyxc_chunks)
        
        import random
        index_perm = list(range(dataset_len))
        random.shuffle(index_perm)
        meta['index_perm'] = index_perm
        
        return meta, dataset_len

    def prepare_reader(self, dataset_meta, global_stores):
        """
        为 prompttts / zyxc / singlespk / expressive 建 JsonlChunkReader。
        xyz 不需要 reader（直接读文件）。
        """
        prompttts_readers = []
        for pack in dataset_meta.get('prompttts_packs', []):
            if pack.get('n_lines', 0) <= 0 or pack.get('n_chunks', 0) <= 0:
                prompttts_readers.append(None)
            else:
                prompttts_readers.append(JsonlChunkReader(pack['path'], pack['idx_path']))

        zyxc_readers = []
        for pack in dataset_meta.get('zyxc_packs', []):
            if pack.get('n_lines', 0) <= 0 or pack.get('n_chunks', 0) <= 0:
                zyxc_readers.append(None)
            else:
                zyxc_readers.append(JsonlChunkReader(pack['path'], pack['idx_path']))

        singlespk_readers = []
        for pack in dataset_meta.get('singlespk_packs', []):
            if pack.get('n_lines', 0) <= 0 or pack.get('n_chunks', 0) <= 0:
                singlespk_readers.append(None)
            else:
                singlespk_readers.append(JsonlChunkReader(pack['path'], pack['idx_path']))

        expressive_readers = []
        for pack in dataset_meta.get('expressive_packs', []):
            if pack.get('n_lines', 0) <= 0 or pack.get('n_chunks', 0) <= 0:
                expressive_readers.append(None)
            else:
                expressive_readers.append(JsonlChunkReader(pack['path'], pack['idx_path']))

        return {
            'prompttts': prompttts_readers,
            'zyxc': zyxc_readers,
            'singlespk': singlespk_readers,
            'expressive': expressive_readers,
        }

    def _read_single_meta_file(self, meta_path: str):
        """
        每个文件返回 1 个 meta_item(dict)。
        - .json: 读出来如果是 list，就取第一个 dict
        - .jsonl: 取第一行非空 dict
        """
        import json
        if not meta_path:
            return None

        try:
            if meta_path.endswith('.json'):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    return obj
                if isinstance(obj, list):
                    for x in obj:
                        if isinstance(x, dict):
                            return x
                return None

            if meta_path.endswith('.jsonl'):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            x = json.loads(line)
                            if isinstance(x, dict):
                                return x
                        except Exception:
                            continue
                return None
        except Exception:
            return None

        return None

    def _sanitize_meta_item(self, meta_item: dict, meta_path: str = None, source: str = None):
        import os

        if not isinstance(meta_item, dict):
            return meta_item

        # xyz / singlespk：强制删 segment start/end
        if source in ('xyz', 'singlespk'):
            segs = meta_item.get('segments', None)
            if isinstance(segs, list):
                for seg in segs:
                    if isinstance(seg, dict):
                        seg.pop('start', None)
                        seg.pop('end', None)

        # singlespk：过滤 + 补全 turn wav_path
        if source == 'singlespk':
            bad_tokens = ('姐', '姨', '奶奶', '女')

            def _has_bad_token(p: str) -> bool:
                if not isinstance(p, str) or not p:
                    return False
                return any(tok in p for tok in bad_tokens)

            turns = meta_item.get('turns')
            if isinstance(turns, list):
                for t in turns:
                    if not isinstance(t, dict):
                        continue
                    wp = t.get('wav_path')
                    if _has_bad_token(wp):
                        return None

            if _has_bad_token(meta_item.get('wav_path')):
                return None

            # 真正“补全 turn 的相对 wav_path”
            if meta_path is not None and isinstance(turns, list):
                meta_dir = os.path.dirname(meta_path)
                for t in turns:
                    if not isinstance(t, dict):
                        continue
                    wp = t.get('wav_path')
                    if isinstance(wp, str) and wp and (not os.path.isabs(wp)):
                        t['wav_path'] = os.path.join(meta_dir, wp)

        # expressive：每行 = 单句单说话人，包装成 turns / segments
        if source == 'expressive':
            meta_dir = os.path.dirname(meta_path) if meta_path is not None else None

            # 兼容几种常见字段名
            wav_path = meta_item.get('wav_path') \
                or meta_item.get('audio_path') \
                or meta_item.get('audio') \
                or meta_item.get('wav')
            if isinstance(wav_path, str) and wav_path and meta_dir is not None and (not os.path.isabs(wav_path)):
                wav_path = os.path.join(meta_dir, wav_path)

            wav_k = meta_item.get('wav_k')

            text = meta_item.get('text') \
                or meta_item.get('sentence') \
                or meta_item.get('caption') \
                or ''

            phone_encoded = meta_item.get('phone_encoded') \
                or meta_item.get('ph_token') \
                or meta_item.get('ph')

            tone_encoded = meta_item.get('tone_encoded') \
                or meta_item.get('tone')

            mel2ph = meta_item.get('mel2ph') \
                or meta_item.get('mel2phone')

            turn = {
                'spk': 'spk1',          # 单人数据，固定 spk1 -> sid=1
                'wav_path': wav_path,
                'wav_k': wav_k,
                'text': text,
                'phone_encoded': phone_encoded,
                'tone_encoded': tone_encoded,
                'mel2ph': mel2ph,
            }

            meta_item['turns'] = [turn]
            meta_item['segments'] = [{'turn_idxs': [0]}]

        return meta_item

    def read_fn(self, idx, reader_pack, global_stores):
        import bisect

        if idx is None or idx < 0:
            return None

        meta = self.dataset_meta

        perm = meta.get('index_perm', None)
        if perm is not None:
            if idx >= len(perm):
                return None
            idx = int(perm[idx])

        xyz_paths = meta.get('xyz_paths', [])
        xyz_len = int(meta.get('xyz_len', len(xyz_paths)))

        singlespk_packs = meta.get('singlespk_packs', [])
        singlespk_chunks = int(meta.get('singlespk_chunks', 0))

        prompttts_packs = meta.get('prompttts_packs', [])
        prompttts_chunks = int(meta.get('prompttts_chunks', 0))

        expressive_packs = meta.get('expressive_packs', [])
        expressive_chunks = int(meta.get('expressive_chunks', 0))

        zyxc_packs = meta.get('zyxc_packs', [])
        zyxc_chunks = int(meta.get('zyxc_chunks', 0))

        # ===== 1) xyz：每个文件 = 1 meta_item =====
        if idx < xyz_len:
            meta_path = xyz_paths[idx] if idx < len(xyz_paths) else None
            it = self._read_single_meta_file(meta_path)
            if not isinstance(it, dict):
                return None
            it = self._sanitize_meta_item(it, meta_path=meta_path, source='xyz')
            it['__source__'] = 'xyz'
            return [it]

        base = int(idx) - int(xyz_len)

        # ===== 2) singlespk：按 chunk 读 jsonl（每行一个 meta_item）=====
        if base < singlespk_chunks:
            idx2 = int(base)

            if (not hasattr(self, '_singlespk_chunk_ends')) or (len(self._singlespk_chunk_ends) != len(singlespk_packs)):
                self._singlespk_chunk_ends = [
                    int(p['offset_chunks']) + int(p['n_chunks']) for p in singlespk_packs
                ]

            pack_id = bisect.bisect_left(self._singlespk_chunk_ends, idx2 + 1)
            if pack_id < 0 or pack_id >= len(singlespk_packs):
                return None

            pack = singlespk_packs[pack_id]
            r_list = reader_pack.get('singlespk', []) if isinstance(reader_pack, dict) else []
            r = r_list[pack_id] if pack_id < len(r_list) else None
            if r is None:
                return None

            local_chunk = idx2 - int(pack['offset_chunks'])
            if local_chunk < 0 or local_chunk >= int(pack['n_chunks']):
                return None

            chunk_size = int(pack['chunk_size'])
            start_line = local_chunk * chunk_size
            end_line = min(int(pack['n_lines']) - 1, start_line + chunk_size - 1)
            if end_line < start_line:
                return None

            try:
                items = r.read_range(start_line, end_line)  # list[dict]
                if not items:
                    return None

                new_items = []
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    it2 = self._sanitize_meta_item(it, meta_path=pack['path'], source='singlespk')
                    if not isinstance(it2, dict):
                        continue  # 被过滤掉（None）或异常的直接丢弃
                    it2['__source__'] = 'singlespk'
                    new_items.append(it2)

                return new_items if new_items else None
            except Exception:
                return None

        base -= int(singlespk_chunks)

        # ===== 3) prompttts：按 chunk 读 jsonl =====
        if base < prompttts_chunks:
            idx2 = int(base)

            if (not hasattr(self, '_prompttts_chunk_ends')) or (len(self._prompttts_chunk_ends) != len(prompttts_packs)):
                self._prompttts_chunk_ends = [
                    int(p['offset_chunks']) + int(p['n_chunks']) for p in prompttts_packs
                ]

            pack_id = bisect.bisect_left(self._prompttts_chunk_ends, idx2 + 1)
            if pack_id < 0 or pack_id >= len(prompttts_packs):
                return None

            pack = prompttts_packs[pack_id]
            r_list = reader_pack.get('prompttts', []) if isinstance(reader_pack, dict) else []
            r = r_list[pack_id] if pack_id < len(r_list) else None
            if r is None:
                return None

            local_chunk = idx2 - int(pack['offset_chunks'])
            if local_chunk < 0 or local_chunk >= int(pack['n_chunks']):
                return None

            chunk_size = int(pack['chunk_size'])
            start_line = local_chunk * chunk_size
            end_line = min(int(pack['n_lines']) - 1, start_line + chunk_size - 1)
            if end_line < start_line:
                return None

            try:
                items = r.read_range(start_line, end_line)  # list[dict]
                if not items:
                    return None
                for it in items:
                    if isinstance(it, dict):
                        it['__source__'] = 'prompttts'
                return items
            except Exception:
                return None

        base -= int(prompttts_chunks)

        # ===== 4) expressive：按 chunk 读 jsonl（每行一个 meta_item）=====
        if base < expressive_chunks:
            idx2 = int(base)

            if (not hasattr(self, '_expressive_chunk_ends')) or (len(self._expressive_chunk_ends) != len(expressive_packs)):
                self._expressive_chunk_ends = [
                    int(p['offset_chunks']) + int(p['n_chunks']) for p in expressive_packs
                ]

            pack_id = bisect.bisect_left(self._expressive_chunk_ends, idx2 + 1)
            if pack_id < 0 or pack_id >= len(expressive_packs):
                return None

            pack = expressive_packs[pack_id]
            r_list = reader_pack.get('expressive', []) if isinstance(reader_pack, dict) else []
            r = r_list[pack_id] if pack_id < len(r_list) else None
            if r is None:
                return None

            local_chunk = idx2 - int(pack['offset_chunks'])
            if local_chunk < 0 or local_chunk >= int(pack['n_chunks']):
                return None

            chunk_size = int(pack['chunk_size'])
            start_line = local_chunk * chunk_size
            end_line = min(int(pack['n_lines']) - 1, start_line + chunk_size - 1)
            if end_line < start_line:
                return None

            try:
                items = r.read_range(start_line, end_line)  # list[dict]
                if not items:
                    return None

                new_items = []
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    it2 = self._sanitize_meta_item(it, meta_path=pack['path'], source='expressive')
                    if not isinstance(it2, dict):
                        continue
                    it2['__source__'] = 'expressive'
                    new_items.append(it2)

                return new_items if new_items else None
            except Exception:
                return None

        base -= int(expressive_chunks)

        # ===== 5) zyxc：按 chunk 读 jsonl（原逻辑不变）=====
        if base < 0 or base >= zyxc_chunks or len(zyxc_packs) == 0:
            return None

        idx3 = int(base)

        if (not hasattr(self, '_zyxc_chunk_ends')) or (len(self._zyxc_chunk_ends) != len(zyxc_packs)):
            self._zyxc_chunk_ends = [
                int(p['offset_chunks']) + int(p['n_chunks']) for p in zyxc_packs
            ]

        pack_id = bisect.bisect_left(self._zyxc_chunk_ends, idx3 + 1)
        if pack_id < 0 or pack_id >= len(zyxc_packs):
            return None

        pack = zyxc_packs[pack_id]
        r_list = reader_pack.get('zyxc', []) if isinstance(reader_pack, dict) else []
        r = r_list[pack_id] if pack_id < len(r_list) else None
        if r is None:
            return None

        local_chunk = idx3 - int(pack['offset_chunks'])
        if local_chunk < 0 or local_chunk >= int(pack['n_chunks']):
            return None

        chunk_size = int(pack['chunk_size'])
        start_line = local_chunk * chunk_size
        end_line = min(int(pack['n_lines']) - 1, start_line + chunk_size - 1)
        if end_line < start_line:
            return None

        try:
            items = r.read_range(start_line, end_line)
            if not items:
                return None
            for it in items:
                if isinstance(it, dict):
                    it['__source__'] = 'zyxc'
            return items
        except Exception:
            return None
        
    def _process_item(self, meta_item, hparams, global_stores):
        """
        meta_item: dict（jsonl 一行）
        - 支持 turn 级 phone/tone/mel2ph 拼接
        - 支持 segment/turn 带 start/end：整条 wav 解码一次后按 [s,e] slice
        - text/caption 输出：<SPK>{id}</SPK>xxx，同一说话人连续不重复插入
        - ref(ctx) 语义统一到 DialogueSegmentEmbDataset：
            ctx/ref 包含“第一轮完整 AB”，从第三段（A/B/C 的第三个说话人段）开始切分
        - 新增：仅对 ref(ctx) 部分，按 speaker(sid) 随机加轻量混响或噪声（也可能不加），不额外引参
        """
        import os
        import re
        import json
        import random
        import tempfile
        import numpy as np
        import torch
        import torchaudio
        import torchaudio.functional as AF
        import hashlib
        import subprocess
        import torch.nn.functional as F
        from collections import OrderedDict

        # --------- helpers ----------
        def _enforce_cache_budget(
            cache_dir: str,
            max_gb: float = 100.0,
            near_ratio: float = 0.90,
            keep_ratio: float = 0.10,
        ):
            try:
                import glob
                max_bytes = int(max_gb * (1024**3))
                if max_bytes <= 0:
                    return

                trigger_bytes = int(max_bytes * float(near_ratio))
                keep_bytes = int(max_bytes * float(keep_ratio))

                files = glob.glob(os.path.join(cache_dir, "*"))
                if not files:
                    return

                sizes = []
                total = 0
                for p in files:
                    try:
                        st = os.stat(p)
                        sz = int(st.st_size)
                        mt = float(st.st_mtime)
                        total += sz
                        sizes.append((mt, sz, p))
                    except Exception:
                        pass

                if total < trigger_bytes:
                    return

                sizes.sort(key=lambda x: x[0])  # old -> new
                for _, sz, p in sizes:
                    if total <= keep_bytes:
                        break
                    try:
                        os.remove(p)
                        total -= sz
                    except Exception:
                        pass
            except Exception:
                return

        def _mel2token_to_dur(m2p: torch.LongTensor) -> torch.LongTensor:
            if m2p.numel() == 0:
                return torch.zeros(0, dtype=torch.long)
            mx = int(m2p.max().item()) if m2p.numel() > 0 else 0
            if mx <= 0:
                return torch.zeros(0, dtype=torch.long)
            cnts = torch.bincount(m2p.clamp_min(0), minlength=mx + 1)
            return cnts[1:].to(torch.long)

        def _to_int_list_maybe(x):
            if x is None:
                return None
            if isinstance(x, str):
                s = x.strip()
                if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
                    try:
                        parsed = json.loads(s.replace('(', '[').replace(')', ']'))
                        return _to_int_list_maybe(parsed)
                    except Exception:
                        pass
                nums = re.findall(r'-?\d+\.?\d*', s)
                if len(nums) == 0:
                    return None
                try:
                    return [int(float(t)) for t in nums]
                except Exception:
                    return None

            if isinstance(x, (list, tuple, np.ndarray)):
                out = []
                for v in x:
                    try:
                        if isinstance(v, str):
                            v = v.strip()
                            if v == '':
                                out.append(0)
                                continue
                            out.append(int(float(v)))
                        elif isinstance(v, (int, np.integer)):
                            out.append(int(v))
                        elif isinstance(v, (float, np.floating)):
                            out.append(int(v))
                        else:
                            out.append(int(v))
                    except Exception:
                        out.append(0)
                return out

            try:
                return [int(x)]
            except Exception:
                return None

        def _sha1(s: str) -> str:
            return hashlib.sha1(s.encode('utf-8')).hexdigest()

        def _atomic_write(path: str, data: bytes):
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

        # --------- params ----------
        sr = int(hparams['audio_sample_rate'])
        fm = int(hparams['frames_multiple'])
        hop = int(hparams['hop_size'])
        stride = int(hparams.get('vae_stride', 8))
        fm_wav = fm * hop
        latent_hop = hop * stride

        # --------- worker shared caches ----------
        def _get_wav_lru():
            return get_from_global_stores('wav_lru_cache', global_stores, lambda: OrderedDict())

        def _lru_get(cache: OrderedDict, k):
            v = cache.get(k, None)
            if v is not None:
                cache.move_to_end(k)
            return v

        def _lru_put(cache: OrderedDict, k, v, max_items: int):
            cache[k] = v
            cache.move_to_end(k)
            while len(cache) > max_items:
                cache.popitem(last=False)

        def _get_cache_dir():
            cache_dir = get_from_global_stores(
                'tos_cache_dir', global_stores,
                lambda: hparams.get('tos_cache_dir', '/dev/shm/zyxc_tos_cache')
            )
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except Exception:
                pass
            return cache_dir

        # --------- audio loading ----------
        def _load_wav_local(wav_rel: str, start_sec: float = None, end_sec: float = None):
            if not wav_rel:
                return None

            wav_rel = wav_rel.replace('wavs/16k160/', 'wavs/24k/')
            full_wav = os.path.join('/mnt/bn/lq-ads-aigc/renyi/tts_datasets_bak/InteractiveDialogue', wav_rel)
            if not os.path.exists(full_wav):
                return None

            lru = _get_wav_lru()
            max_items = int(hparams.get('wav_mem_cache_items', 16))

            if start_sec is None or end_sec is None:
                ck = ('local_full', full_wav, sr)
                hit = _lru_get(lru, ck)
                if hit is not None:
                    return hit

            try:
                if start_sec is not None and end_sec is not None and end_sec > start_sec:
                    info = torchaudio.info(full_wav)
                    sr0 = int(info.sample_rate)
                    frame_offset = int(max(0, round(float(start_sec) * sr0)))
                    num_frames = int(max(1, round((float(end_sec) - float(start_sec)) * sr0)))
                    wav_t, sr_ = torchaudio.load(full_wav, frame_offset=frame_offset, num_frames=num_frames)
                else:
                    wav_t, sr_ = torchaudio.load(full_wav)
            except Exception:
                return None

            if wav_t is None or wav_t.numel() == 0:
                return None

            if wav_t.ndim == 2 and wav_t.size(0) > 1:
                wav_t = wav_t.mean(dim=0)
            else:
                wav_t = wav_t.squeeze(0)

            sr_ = int(sr_)
            if sr_ != sr:
                try:
                    wav_t = AF.resample(wav_t, orig_freq=sr_, new_freq=sr)
                except Exception:
                    try:
                        wav_t = torchaudio.transforms.Resample(orig_freq=sr_, new_freq=sr)(wav_t)
                    except Exception:
                        return None

            wav_np = wav_t.detach().cpu().to(torch.float32).numpy()
            if wav_np.size == 0:
                return None
            wav_np = np.asarray(wav_np, dtype=np.float32)

            if start_sec is None or end_sec is None:
                _lru_put(lru, ck, wav_np, max_items=max_items)
            return wav_np

        def _get_tos_client():
            cluster = os.environ.get('CLUSTER', '').lower()
            if cluster == 'va':
                return TosClient(bucket='sa-ag-sg-research-sg')
            return TosClient(bucket='humanaigc-ads')

        def _load_wav_from_tos(wav_k: str, start_sec: float = None, end_sec: float = None):
            if not wav_k:
                return None

            # ==== 这里加过滤逻辑 ====
            # 任何 key 里包含 /apple/ 或 /xmly/ 的，直接当成无效样本
            # if "/apple/" in wav_k or "/xmly/" in wav_k:
            #     return None
            # =======================

            cache_dir = _get_cache_dir()
            key_hash = _sha1(wav_k)

            # full wav npy cache
            if start_sec is None or end_sec is None:
                npy_path = os.path.join(cache_dir, f'{key_hash}.sr{sr}.npy')
                if os.path.exists(npy_path):
                    try:
                        wav_np = np.load(npy_path, allow_pickle=False)
                        if wav_np is not None and wav_np.size > 0:
                            return np.asarray(wav_np, dtype=np.float32)
                    except Exception:
                        pass

            # m4a cache
            m4a_path = os.path.join(cache_dir, f'{key_hash}.m4a')
            if not os.path.exists(m4a_path) or os.path.getsize(m4a_path) == 0:
                tos_client: TosClient = get_from_global_stores('tos_client', global_stores, _get_tos_client)
                try:
                    data = tos_client.get_object(wav_k)
                except Exception:
                    return None
                if data is None:
                    return None
                try:
                    _atomic_write(m4a_path, data)
                except Exception:
                    m4a_path = None
            _enforce_cache_budget(cache_dir)

            def _decode_with_ffmpeg(path: str, s_sec: float, e_sec: float):
                if path is None or not os.path.exists(path):
                    return None
                if s_sec is None or e_sec is None or e_sec <= s_sec:
                    return None
                cmd = [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{float(s_sec):.6f}", "-to", f"{float(e_sec):.66f}",
                    "-i", path,
                    "-f", "f32le", "-ac", "1", "-ar", str(sr),
                    "pipe:1"
                ]
                try:
                    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                    if p.returncode != 0 or p.stdout is None or len(p.stdout) == 0:
                        return None
                    wav_np = np.frombuffer(p.stdout, dtype=np.float32)
                    if wav_np.size == 0:
                        return None
                    return wav_np
                except Exception:
                    return None

            if start_sec is not None and end_sec is not None and end_sec > start_sec:
                if m4a_path is not None:
                    seg = _decode_with_ffmpeg(m4a_path, start_sec, end_sec)
                    if seg is not None:
                        return np.asarray(seg, dtype=np.float32)

            # full decode (with mem LRU)
            lru = _get_wav_lru()
            max_items = int(hparams.get('wav_mem_cache_items', 16))
            ck = ('tos_full', wav_k, sr)
            hit = _lru_get(lru, ck)
            if hit is not None:
                return hit

            try:
                if m4a_path is not None and os.path.exists(m4a_path):
                    wav_t, sr_ = torchaudio.load(m4a_path)
                else:
                    tos_client: TosClient = get_from_global_stores('tos_client', global_stores, _get_tos_client)
                    data = tos_client.get_object(wav_k)
                    if data is None:
                        return None
                    with tempfile.NamedTemporaryFile(suffix='.m4a', dir='/dev/shm', delete=True) as f:
                        f.write(data)
                        f.flush()
                        wav_t, sr_ = torchaudio.load(f.name)
            except Exception:
                return None

            if wav_t is None or wav_t.numel() == 0:
                return None

            if wav_t.ndim == 2 and wav_t.size(0) > 1:
                wav_t = wav_t.mean(dim=0)
            else:
                wav_t = wav_t.squeeze(0)

            sr_ = int(sr_)
            if sr_ != sr:
                try:
                    wav_t = AF.resample(wav_t, orig_freq=sr_, new_freq=sr)
                except Exception:
                    try:
                        wav_t = torchaudio.transforms.Resample(orig_freq=sr_, new_freq=sr)(wav_t)
                    except Exception:
                        return None

            wav_np = wav_t.detach().cpu().to(torch.float32).numpy()
            if wav_np.size == 0:
                return None
            wav_np = np.asarray(wav_np, dtype=np.float32)

            try:
                npy_path = os.path.join(cache_dir, f'{key_hash}.sr{sr}.npy')
                if not os.path.exists(npy_path):
                    np.save(npy_path, wav_np, allow_pickle=False)
            except Exception:
                pass
            _enforce_cache_budget(cache_dir)

            _lru_put(lru, ck, wav_np, max_items=max_items)
            return wav_np

        # --------- ref augmentation helpers ----------
        def _apply_noise(seg: torch.Tensor, snr_db: float) -> torch.Tensor:
            # seg: [T]
            if seg.numel() <= 0:
                return seg
            seg_f = seg.to(torch.float32)
            rms = torch.sqrt(torch.mean(seg_f * seg_f) + 1e-12)
            # snr_db 越大噪声越小
            noise_rms = rms / (10.0 ** (float(snr_db) / 20.0))
            noise = torch.randn_like(seg_f) * noise_rms
            out = seg_f + noise
            return out
        
        # === 新增：RMS dBFS 归一化到 target_db（默认 -23 dB） ===
        def _rms_norm_np(wav_np: np.ndarray, target_db: float = -23.0, eps: float = 1e-8) -> np.ndarray:
            if wav_np is None:
                return None
            wav_np = np.asarray(wav_np, dtype=np.float32)
            if wav_np.size == 0:
                return wav_np
            rms = np.sqrt(np.mean(wav_np ** 2) + eps)
            if rms < eps:
                # 静音或者接近静音，不动
                return wav_np
            cur_db = 20.0 * np.log10(rms + eps)
            gain_db = float(target_db) - float(cur_db)
            gain = 10.0 ** (gain_db / 20.0)
            out = wav_np * gain
            # 防止极端情况下过载，超过 1.0 就等比缩放回来
            max_abs = float(np.max(np.abs(out)))
            if max_abs > 1.0:
                out = out / max_abs
            return out.astype(np.float32)

        def _apply_light_reverb(seg: torch.Tensor, wet: float) -> torch.Tensor:
            """
            轻量“混响/空间感”：多 tap 延迟(很小) + 小平滑 + wet 混合
            wet < 0.2，尽量不影响可懂度
            """
            if seg.numel() <= 0:
                return seg
            x = seg.to(torch.float32)
            y = x.clone()

            # 2~4 个延迟 tap，延迟 10~55ms，增益较小
            n_taps = random.randint(2, 4)
            for _ in range(n_taps):
                d = int(sr * random.uniform(0.010, 0.055))
                g = random.uniform(0.04, 0.14)  # 小一点，避免糊
                if d > 0 and d < y.numel():
                    # 用递归 y 叠加会更“混响”，但可能爆；这里用 y 做一次弱反馈，仍然安全
                    y[d:] = y[d:] + g * y[:-d]

            # 轻微平滑（模拟空气吸收），kernel 很小，开销低
            k = random.choice([3, 5, 7])
            if y.numel() > k:
                ker = torch.ones((k,), device=y.device, dtype=y.dtype) / float(k)
                y = F.conv1d(y[None, None, :], ker[None, None, :], padding=k // 2)[0, 0, :]

            # 能量归一，避免整体变响/过载
            x_peak = x.abs().max().clamp_min(1e-6)
            y_peak = y.abs().max().clamp_min(1e-6)
            y = y / y_peak * x_peak

            wet = float(max(0.0, min(wet, 0.35)))
            out = (1.0 - wet) * x + wet * y
            return out

        # --------- main ----------
        if meta_item is None or not isinstance(meta_item, dict):
            return

        turns = meta_item.get('turns', [])
        segments = meta_item.get('segments', [])
        if not turns or not segments:
            return

        wav_cache_turn = {}
        wav_cache_full = {}

        for segment in segments:
            spk_map = {}  # raw_spk -> sid(1-based)
            ph_list, tone_list, m2p_list, spk_mask_ph_list = [], [], [], []

            res_text = ''
            kept_turn_sids = []

            change_of_spk = 1
            last_sid_for_text = None

            ref_wav_start_turn_idx = 0
            ref_start_sec = None

            seg_has_ts = isinstance(segment, dict) and ('start' in segment and 'end' in segment)
            seg_start_sec = float(segment.get('start', 0.0)) if seg_has_ts else None
            seg_end_sec = float(segment.get('end', 0.0)) if seg_has_ts else None

            #  turn_ranges：记录每个 turn 在 wav_cat（slice 前的基准）里的 sample 区间，后面会映射到最终 wav_cat
            turn_ranges = []  # list of dict: {sid, start, end}

            if seg_has_ts:
                segment_audio_src = None
                turn_time_info = []
                turn_sample_lens = []
            else:
                wav_lst = []
                cum = 0  # sample 累计，用于 turn_ranges

            for turn_idx in segment.get('turn_idxs', []):
                if turn_idx < 0 or turn_idx >= len(turns):
                    continue
                turn = turns[turn_idx]

                text = turn.get('text')
                if not text or not str(text).strip():
                    continue

                raw_spk = turn.get('spk', 'spk_unk')
                sid = spk_map.setdefault(raw_spk, len(spk_map) + 1)  # 1-based

                ph_enc = turn.get('phone_encoded')
                m2p_raw = turn.get('mel2ph')
                if ph_enc is None or len(ph_enc) == 0 or m2p_raw is None or len(m2p_raw) == 0:
                    continue

                # ---------- audio ----------
                if seg_has_ts:
                    t_start = float(turn.get('start', seg_start_sec if seg_start_sec is not None else 0.0))
                    t_end = float(turn.get('end', t_start))
                    if seg_start_sec is not None:
                        t_start = max(t_start, seg_start_sec)
                    if seg_end_sec is not None:
                        t_end = min(t_end, seg_end_sec)
                    if t_end <= t_start:
                        continue

                    wav_path = turn.get('wav_path', None)
                    wav_k = turn.get('wav_k', None)
                    if segment_audio_src is None:
                        if wav_path:
                            segment_audio_src = ('local', wav_path)
                        elif wav_k:
                            segment_audio_src = ('tos', wav_k)

                    turn_time_info.append({'sid': sid, 'start_sec': t_start, 'end_sec': t_end})
                    turn_sample_lens.append(int(max(0.0, (t_end - t_start)) * sr))
                else:
                    wav_np = None
                    wav_path = turn.get('wav_path', None)
                    wav_k = turn.get('wav_k', None)

                    cache_key = None
                    if wav_path:
                        cache_key = ('local', wav_path)
                    elif wav_k:
                        cache_key = ('tos', wav_k)

                    if cache_key is not None and cache_key in wav_cache_turn:
                        wav_np = wav_cache_turn[cache_key]
                    else:
                        if wav_path:
                            wav_np = _load_wav_local(wav_path)
                        elif wav_k:
                            wav_np = _load_wav_from_tos(wav_k)
                        if wav_np is None or len(wav_np) == 0:
                            continue

                        # ★ 每个 turn 单独归一化到 -23 dB RMS
                        wav_np = _rms_norm_np(wav_np, target_db=-23.0)

                        if cache_key is not None:
                            wav_cache_turn[cache_key] = wav_np

                    wav_lst.append(wav_np)

                    # 记录该 turn 在拼接 wav_cat 内的区间
                    L = int(len(wav_np))
                    if L > 0:
                        turn_ranges.append({'sid': int(sid), 'start': int(cum), 'end': int(cum + L)})
                        cum += L

                # ---------- text & change_of_spk ----------
                spk_tag_id = int(sid)
                if last_sid_for_text is None:
                    res_text += f'<SPK>{spk_tag_id}</SPK>' + str(text)
                    last_sid_for_text = sid
                else:
                    if int(sid) != int(last_sid_for_text):
                        change_of_spk += 1

                        # 进入第三段（AB 后的第三段）：记录“第三段开始之前”的边界
                        if change_of_spk == 3:
                            if seg_has_ts:
                                ref_start_sec = float(turn_time_info[-1]['start_sec'])
                                ref_wav_start_turn_idx = max(0, len(turn_sample_lens) - 1)
                            else:
                                # wav_lst 已 append 当前段，所以要 -1 变成第三段的 index
                                ref_wav_start_turn_idx = max(0, len(wav_lst) - 1)

                        res_text += f'<SPK>{spk_tag_id}</SPK>' + str(text)
                        last_sid_for_text = sid
                    else:
                        res_text += str(text)

                kept_turn_sids.append(sid)

                # ---------- tensors ----------
                ph_i = torch.as_tensor(ph_enc, dtype=torch.long)

                tone_raw = turn.get('tone_encoded')
                tone_enc_clean = _to_int_list_maybe(tone_raw) if tone_raw is not None else None
                if (tone_enc_clean is None) or (len(tone_enc_clean) != len(ph_enc)):
                    tn_i = torch.zeros_like(ph_i, dtype=torch.long)
                else:
                    tn_i = torch.as_tensor(tone_enc_clean, dtype=torch.long)

                m2p_i = torch.as_tensor(m2p_raw, dtype=torch.long)

                ph_list.append(ph_i)
                tone_list.append(tn_i)
                m2p_list.append(m2p_i)
                spk_mask_ph_list.append(torch.full((ph_i.numel(),), int(sid), dtype=torch.long))

            # ---------- segment-level filters ----------
            # expressive 单人数据不过滤句数/轮次；其它源保持原逻辑
            is_expressive_single = (str(meta_item.get('__source__', '')) == 'expressive')
            if not is_expressive_single:
                if len(kept_turn_sids) < 4:
                    continue
                num_conversations = 1
                for i in range(1, len(kept_turn_sids)):
                    if kept_turn_sids[i] != kept_turn_sids[i - 1]:
                        num_conversations += 1
                if num_conversations // 2 < 2:
                    continue

            num_conversations = 1
            for i in range(1, len(kept_turn_sids)):
                if kept_turn_sids[i] != kept_turn_sids[i - 1]:
                    num_conversations += 1

            # ====== INSERT: shuffle speaker ids within this segment (keep consistency) ======
            if self.use_fast_dataloader:
                buckets = [
                    400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000,
                    2400, 2800, 3200, 3600, 4000, 4500, 5000, 5500, 6000,
                    7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000,
                    15000, 16000, 18000, 20000, 40000, 60000
                ]
                batcher = get_from_global_stores(
                    'batcher', global_stores,
                    lambda: BucketBatcher(
                        buckets=buckets,
                        dynamic_batch=hparams.get("dynamic_batch", True),
                        batch_size=hparams['max_sentences'],
                        maximum_bucket_size=hparams['max_tokens'],
                        length_fn=lambda x: x['len'],
                    )
                )

            shuffle_spk_id = bool(hparams.get("shuffle_spk_id", False))
            sid_remap = None
            n_spk = int(len(spk_map))

            if shuffle_spk_id and n_spk > 1:
                # old sid: 1..n_spk  -> new sid: a random permutation of 1..n_spk
                new_ids = list(range(1, n_spk + 1))
                random.shuffle(new_ids)
                sid_remap = {old: new_ids[old - 1] for old in range(1, n_spk + 1)}

                # 1) remap <SPK> tags in text
                def _remap_tag(m):
                    old = int(m.group(1))
                    new = int(sid_remap.get(old, old))
                    return f"<SPK>{new}</SPK>"

                res_text = re.sub(r"<SPK>(\d+)</SPK>", _remap_tag, res_text)

                # 2) remap kept_turn_sids
                kept_turn_sids = [int(sid_remap.get(int(x), int(x))) for x in kept_turn_sids]

                # 3) remap turn_time_info
                if seg_has_ts:
                    for ti in turn_time_info:
                        ti['sid'] = int(sid_remap.get(int(ti['sid']), int(ti['sid'])))

                # 4) remap turn_ranges
                if (not seg_has_ts) and turn_ranges:
                    for r in turn_ranges:
                        r['sid'] = int(sid_remap.get(int(r['sid']), int(r['sid'])))

            # ---------- build wav_cat_np + build turn_ranges for ts / non-ts ----------
            if seg_has_ts:
                if segment_audio_src is None or len(turn_time_info) == 0:
                    continue

                kind, value = segment_audio_src
                s_sec = seg_start_sec if seg_start_sec is not None else min(t['start_sec'] for t in turn_time_info)
                e_sec = seg_end_sec if seg_end_sec is not None else max(t['end_sec'] for t in turn_time_info)
                if e_sec <= s_sec:
                    continue

                if kind == 'local':
                    wav_cat_np = _load_wav_local(value, start_sec=s_sec, end_sec=e_sec)
                    if wav_cat_np is None or len(wav_cat_np) == 0:
                        continue
                    wav_cat_np = np.asarray(wav_cat_np, dtype=np.float32)

                else:
                    full_key = (kind, value, sr)
                    wav_full = wav_cache_full.get(full_key, None)
                    if wav_full is None:
                        wav_full = _load_wav_from_tos(value)
                        if wav_full is None or len(wav_full) == 0:
                            continue
                        wav_cache_full[full_key] = wav_full

                    total_len = int(len(wav_full))
                    if total_len <= 0:
                        continue

                    s_idx = int(max(0, min(round(float(s_sec) * sr), total_len - 1)))
                    e_idx = int(max(s_idx + 1, min(round(float(e_sec) * sr), total_len)))
                    wav_cat_np = wav_full[s_idx:e_idx]
                    if wav_cat_np is None or len(wav_cat_np) == 0:
                        continue
                    wav_cat_np = np.asarray(wav_cat_np, dtype=np.float32)
                    
                wav_cat_np = _rms_norm_np(wav_cat_np, target_db=-23.0)

                # ts 模式下构建 turn_ranges（相对 segment slice 起点 s_sec）
                turn_ranges = []
                for ti in turn_time_info:
                    sid_i = int(ti['sid'])
                    st = int(round((float(ti['start_sec']) - float(s_sec)) * sr))
                    ed = int(round((float(ti['end_sec']) - float(s_sec)) * sr))
                    st = max(0, st)
                    ed = max(st, ed)
                    if ed > st:
                        turn_ranges.append({'sid': sid_i, 'start': st, 'end': ed})

            else:
                if not wav_lst:
                    continue
                wav_lst_valid = [w for w in wav_lst if w is not None and len(w) > 0]
                if not wav_lst_valid:
                    continue

                try:
                    wav_cat_np = np.concatenate(
                        [np.asarray(w, dtype=np.float32) for w in wav_lst_valid],
                        axis=0
                    )
                except Exception:
                    continue

                if wav_cat_np.size == 0:
                    continue

            # ---------- trim to frames_multiple ----------
            wav_cat = torch.from_numpy(wav_cat_np)
            T = (int(wav_cat.shape[0]) // fm_wav) * fm_wav
            if T <= 0:
                continue
            wav_cat = wav_cat[:T]
            if wav_cat.numel() == 0:
                continue

            # turn_ranges 裁剪到 [0, T)
            if turn_ranges:
                new_ranges = []
                for r in turn_ranges:
                    st = int(max(0, min(r['start'], T)))
                    ed = int(max(0, min(r['end'], T)))
                    if ed > st:
                        new_ranges.append({'sid': int(r['sid']), 'start': st, 'end': ed})
                turn_ranges = new_ranges

            # ---------- fix mel2ph offsets & concat ----------
            ph_offset = 0
            m2p_fixed = []
            for m2p_i, ph_i in zip(m2p_list, ph_list):
                if ph_offset > 0:
                    m2p_i = m2p_i + (m2p_i > 0).long() * ph_offset
                m2p_fixed.append(m2p_i)
                ph_offset += int(ph_i.numel())

            ph_token = torch.cat(ph_list, dim=0)
            tone = torch.cat(tone_list, dim=0)
            mel2ph = torch.cat(m2p_fixed, dim=0)
            spk_mask_ph = torch.cat(spk_mask_ph_list, dim=0)

            # ====== INSERT: remap phone-level spk_mask to match <SPK> ids ======
            if sid_remap is not None and n_spk > 1 and spk_mask_ph.numel() > 0:
                lut = torch.arange(0, n_spk + 1, dtype=torch.long)  # 0 保留
                for old, new in sid_remap.items():
                    if 0 <= int(old) <= n_spk:
                        lut[int(old)] = int(new)
                spk_mask_ph = lut[spk_mask_ph.clamp(0, n_spk)]
            # ====== INSERT END ======

            mel_len = int(wav_cat.shape[0] // hop)
            if mel_len <= 0:
                continue

            if mel2ph.numel() < mel_len:
                pad_len = mel_len - mel2ph.numel()
                last = mel2ph[-1] if mel2ph.numel() > 0 else torch.tensor(0, dtype=torch.long)
                mel2ph = torch.cat([mel2ph, last.repeat(pad_len)], dim=0)
            mel2ph = mel2ph[:mel_len]
            mel2ph = mel2ph[: (mel_len // fm) * fm]
            if mel2ph.numel() == 0:
                continue

            dur = _mel2token_to_dur(mel2ph)

            mel2ph_sparse = None
            if hparams.get('use_sparse_dur', False):
                _m2a = compute_mel2aug_from_dur(
                    dur.cpu().numpy().tolist(),
                    gap_mode=hparams.get('sparse_dur_mode', 'proportional'),
                    gap_frames=hparams.get('sparse_dur_frames', 4),
                    gap_alpha=hparams.get('sparse_dur_alpha', 0.2),
                    min_keep=hparams.get('sparse_dur_min_keep', 1),
                    keep_ratio=hparams.get('sparse_dur_keep_ratio'),
                    symmetric=hparams.get('sparse_dur_symmetric', True),
                )
                mel2ph_sparse = torch.as_tensor(_m2a, dtype=torch.long)
                target_len = mel2ph.shape[0]
                if mel2ph_sparse.numel() < target_len:
                    pad_len = target_len - mel2ph_sparse.numel()
                    last = mel2ph_sparse[-1] if mel2ph_sparse.numel() > 0 else torch.tensor(0, dtype=torch.long)
                    mel2ph_sparse = torch.cat([mel2ph_sparse, last.repeat(pad_len)], dim=0)
                mel2ph_sparse = mel2ph_sparse[:target_len]

            # text/caption
            text_merged = res_text
            caption_merged = text_merged

            # ---------- ctx split ----------
            # ref 边界精确在第三段开始处（再随机右移）
            if seg_has_ts and (ref_start_sec is not None) and (seg_start_sec is not None):
                pre_len = int(round(max(0.0, float(ref_start_sec) - float(seg_start_sec)) * sr))
            else:
                pre_len = 0
                if turn_ranges and (ref_wav_start_turn_idx > 0) and (ref_wav_start_turn_idx < len(turn_ranges)):
                    pre_len = turn_ranges[ref_wav_start_turn_idx]['start']

            ref_wav_start = (int(pre_len) // fm_wav) * fm_wav
            max_idx = min(int(wav_cat.shape[0] * 0.9), int(wav_cat.shape[0]) - 20000)
            if max_idx > ref_wav_start:
                ref_wav_start = (random.randint(ref_wav_start, max_idx) // fm_wav) * fm_wav
            ref_wav_start = int(max(0, min(ref_wav_start, int(wav_cat.shape[0]))))

            # ---------- build ctx_wav ----------
            ctx_wav = wav_cat[:ref_wav_start].clone()

            # 新增：ref(ctx) 内按 spk 做轻量增强（每个 sid 独立随机）
            # 概率：0.5 不加；否则 0.5 混响 / 0.5 噪声；强度都很轻
            if ctx_wav.numel() > 0 and turn_ranges:
                sids_in_ref = set()
                for r in turn_ranges:
                    if r['start'] < ref_wav_start and r['end'] > 0:
                        sids_in_ref.add(int(r['sid']))

                spk_aug = {}  # sid -> ('none'|'reverb'|'noise', param)
                for sid_i in sids_in_ref:
                    if random.random() < 0.3:
                        spk_aug[sid_i] = ('none', None)
                    else:
                        if random.random() < 0.5:
                            wet = random.uniform(0.05, 0.20)
                            spk_aug[sid_i] = ('reverb', wet)
                        else:
                            snr_db = random.uniform(18.0, 25.0)
                            spk_aug[sid_i] = ('noise', snr_db)

                for r in turn_ranges:
                    st = int(r['start'])
                    ed = int(r['end'])
                    if st >= ref_wav_start or ed <= 0:
                        continue
                    sid_i = int(r['sid'])
                    aug = spk_aug.get(sid_i, ('none', None))
                    if aug[0] == 'none':
                        continue

                    s0 = max(0, st)
                    e0 = min(ref_wav_start, ed)
                    if e0 <= s0:
                        continue

                    seg = ctx_wav[s0:e0]
                    if aug[0] == 'reverb':
                        wet = float(aug[1])
                        seg2 = _apply_light_reverb(seg, wet=wet)
                    else:
                        snr_db = float(aug[1])
                        seg2 = _apply_noise(seg, snr_db=snr_db)

                    ctx_wav[s0:e0] = seg2.to(ctx_wav.dtype)

                peak = ctx_wav.abs().max().item() if ctx_wav.numel() > 0 else 0.0
                if peak > 1.2:
                    ctx_wav = ctx_wav / peak

            # ctx_mask：按 latent 长度建
            latent_len = int(wav_cat.shape[0] // latent_hop)
            max_lat = int(hparams.get("max_seq_len", 7500))
            if latent_len > max_lat:
                print(f"latent_len {latent_len} > max_lat {max_lat}")
                continue
            min_lat = int(hparams.get("min_seq_len", 25))
            if latent_len < min_lat:
                print(f"latent_len {latent_len} < min_lat {min_lat}")
                continue

            ctx_latent_len = int(ref_wav_start // latent_hop)
            if latent_len <= 0:
                continue
            ctx_mask = torch.zeros((latent_len, 1), dtype=torch.float32)
            if ctx_latent_len > 0:
                ctx_mask[:min(ctx_latent_len, latent_len)] = 1.0

            item = {
                'wav': wav_cat,
                'text': text_merged,
                'caption': caption_merged,
                'spk_mask': spk_mask_ph.to(torch.long),
                'ctx_wav': ctx_wav,
                'ctx_mask': ctx_mask,
                'ph_token': ph_token.to(torch.long),
                'tone': tone.to(torch.long),
                'mel2ph': mel2ph.to(torch.long),
                'dur': dur.to(torch.long),
                'len': latent_len,
            }
            if mel2ph_sparse is not None:
                item['mel2ph_sparse'] = mel2ph_sparse.to(torch.long)

            yield item
