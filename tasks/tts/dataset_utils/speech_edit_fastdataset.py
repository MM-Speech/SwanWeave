'''
编辑任务所需的FastDataset
参考 tasks/tts/dataset_utils/swan_base_fastdataset.py
'''

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
from utils.text.pinyin_aug import augment_text_with_pinyin_s1s2_safe
from utils.service.file_service import FileQueueClient
from tasks.tts.dataset_utils.base_fastdataset_v2 import BaseShmDataset, valid_item_kv, raw_text_process, shuffle_speaker_ids

DEBUG = False

class SpeechEditTTSShmDataset(BaseShmDataset):
        
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
                    noise_snr=(1.0, 20.0), with_speech=hparams.get('musan_with_speech', False)
                )
            )

        if hparams.get('use_cosyvoice2_text_tokenizer', False) and not hparams.get('online_text_alignment_task', False):
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
                ## not 'text' here
                item_txt = item['txt'].replace('<S1>', '').replace('</S1>', '')
                if item_txt == '':
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
        if hparams.get('online_text_alignment_dataloader', False):
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
                    asr_results = asr_client.wait_result(job_id, poll_s=0.5, timeout_s=300)
                    if asr_results is None:
                        print(f'asr results is None, job_id: {job_id}')
                        texts_aligned = [None] * job_id2batch_size[job_id]
                    else:
                        texts_aligned = asr_results['result']['asr_results']['pause_punct_texts']
                        assert len(texts_aligned) == job_id2batch_size[job_id], f"asr results len {len(texts_aligned)} != batch size {job_id2batch_size[job_id]}"
                except TimeoutError:
                    # traceback.print_exc()
                    texts_aligned = [None] * job_id2batch_size[job_id]
                except:
                    traceback.print_exc()
                    texts_aligned = [None] * job_id2batch_size[job_id]
                texts_aligned_total.extend(texts_aligned)
            assert len(texts_aligned_total) == len(items), f"texts_aligned_total len {len(texts_aligned_total)} != items len {len(items)}"
            items_ = []
            for item_idx in range(len(items)):
                if texts_aligned_total[item_idx] is not None:
                    items[item_idx]['txt'] = texts_aligned_total[item_idx]
                    items_.append(items[item_idx])
            items = items_
            if len(items) == 0:
                return
        
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
                if speech_augmentor is not None:
                    item_tgt['wav'] = speech_augmentor(item_tgt['wav'])
                    # print("[INFO] wav augmented")
            mel_len = len(item_tgt['wav']) // hop_size

            if hparams.get('use_cosyvoice2_text_tokenizer', False) and not hparams.get('online_text_alignment_task', False):
                if hparams.get('mix_text_pinyin', {}).get('enable', False):
                    item_tgt['text'] = augment_text_with_pinyin_s1s2_safe(
                        item_tgt['text'],
                        hparams
                    )
                text_tokens = cosyvoice2_text_tokenizer.encode(item_tgt['text'])
                text_tokens = torch.tensor(text_tokens).long()
                item_tgt['txt_tokens'] = text_tokens
                
            ### ---- 初版 ----
            # if item_tgt.get('ctx_wav') is None:
            #     hop_size = hparams['hop_size']
            #     vae_stride = hparams['vae_stride']

            #     margin_frames = max(int(mel_len * 0.1), 100)
            #     gen_ratio_max = 0.1 # 随机一个数
            #     gen_ratio = random.uniform(0.05, gen_ratio_max)
            #     gen_frames = int(mel_len * gen_ratio)
            #     gen_frames = gen_frames // fm * fm

            #     if margin_frames * 2 >= int(mel_len) or gen_frames <= 0:
            #         min_idx = max(int(mel_len * 0.1), 200)
            #         max_idx = min(int(mel_len * 0.9), int(mel_len) - 200)
            #         if min_idx > max_idx:
            #             min_idx = int(mel_len * 0.4)
            #             max_idx = int(mel_len * 0.6)
                    
            #         rand_length = random.randint(int(mel_len * 0.1), int(mel_len * 0.3))
            #         rand_length = (rand_length // int(fm)) * int(fm)
            #         rand_length = max(int(fm), min(rand_length, int(mel_len)))

            #         gap_start = random.randint(int(mel_len * 0.1), int(mel_len * 0.6))
            #         gap_start = (gap_start // int(fm)) * int(fm)
            #         gap_end = gap_start + rand_length

            #         ctx_mask_mel = torch.ones((int(mel_len), 1), dtype=torch.float32)
            #         ctx_mask_mel[gap_start:gap_end] = 0.0
            #         item_tgt['ctx_mask'] = ctx_mask_mel[::vae_stride]

            #         ctx_wav = item_tgt['wav'].clone()
            #         s = int(gap_start * hop_size)
            #         e = int(gap_end * hop_size)
            #         if e > s:
            #             ctx_wav[s:e] = 0.0
            #         item_tgt['ctx_wav'] = ctx_wav.contiguous()
 
            #     else:
            #         max_gen = int(mel_len) - 2 * margin_frames
            #         gen_frames = max(int(fm), min(gen_frames, max_gen))
            #         gen_frames = (gen_frames // int(fm)) * int(fm)
 
            #         start_min = margin_frames
            #         start_max = int(mel_len) - margin_frames - gen_frames
            #         if start_max < start_min:
            #             start_min = max(0, int(mel_len) // 4)
            #             start_max = max(start_min, int(mel_len) - start_min - gen_frames)
 
            #         gap_start = random.randint(start_min, start_max)
            #         gap_start = (gap_start // int(fm)) * int(fm)
            #         gap_start = max(margin_frames, min(gap_start, int(mel_len) - margin_frames - gen_frames))
            #         gap_end = gap_start + gen_frames
 
            #         ctx_mask_mel = torch.ones((int(mel_len), 1), dtype=torch.float32)
            #         ctx_mask_mel[gap_start:gap_end] = 0.0
            #         item_tgt['ctx_mask'] = ctx_mask_mel[::vae_stride]
 
            #         ctx_wav = item_tgt['wav'].clone()
            #         s = int(gap_start * hop_size)
            #         e = int(gap_end * hop_size)
            #         if e > s:
            #             ctx_wav[s:e] = 0.0
            #         item_tgt['ctx_wav'] = ctx_wav.contiguous()
            
            ### ---- 加窗口，多个mask，完全去掉了margin ----
            # if item_tgt.get('ctx_wav') is None:
            #     hop_size = int(hparams['hop_size'])
            #     vae_stride = int(hparams['vae_stride'])
                
            #     mel_len_i = int(mel_len)

            #     len_min = 52

            #     def _align_to_fm(x):
            #         return (int(x) // int(fm)) * int(fm)

            #     def _single_window_fallback():
            #         if mel_len_i <= len_min + 1:
            #             return None

            #         # margin_frames = max(int(mel_len_i * 0.1), 100)
            #         gap_len_max = min(400, int(mel_len_i * 0.3), int(mel_len_i) - 1)
            #         if gap_len_max <= 0:
            #             return None

            #         gap_len_min = min(len_min, gap_len_max)
            #         gap_len = random.randint(int(gap_len_min), int(gap_len_max))
            #         gap_len = max(int(len_min), int(_align_to_fm(gap_len)))
            #         if gap_len <= 0:
            #             return None

            #         start_min = 0
            #         start_max = int(mel_len_i) - int(gap_len)
            #         if start_max < start_min:
            #             return None

            #         gap_start = random.randint(int(start_min), int(start_max))
            #         gap_start = int(_align_to_fm(gap_start))
            #         gap_start = max(0, min(int(gap_start), int(mel_len_i) - 1))
            #         gap_end = min(int(gap_start) + int(gap_len), int(mel_len_i))
            #         if gap_end <= gap_start:
            #             return None

            #         ctx_mask_mel = torch.ones((mel_len_i, 1), dtype=torch.float32)
            #         ctx_mask_mel[gap_start:gap_end] = 0.0

            #         ctx_wav = item_tgt['wav'].clone()
            #         s = int(gap_start) * int(hop_size)
            #         e = int(gap_end) * int(hop_size)
            #         s = max(0, min(int(s), int(ctx_wav.numel())))
            #         e = max(0, min(int(e), int(ctx_wav.numel())))
            #         if e > s:
            #             ctx_wav[s:e] = 0.0

            #         return ctx_wav.contiguous(), ctx_mask_mel.contiguous()

            #     def _set_identity_ctx():
            #         item_tgt['ctx_wav'] = item_tgt['wav']
            #         item_tgt['ctx_mask'] = torch.ones(
            #             (max(1, (mel_len_i + vae_stride - 1) // vae_stride), 1),
            #             dtype=torch.float32
            #         )

            #     if mel_len_i < 550:
            #         res = _single_window_fallback()
            #         if res is None:
            #             _set_identity_ctx()
            #         else:
            #             ctx_wav, ctx_mask_mel = res
            #             item_tgt['ctx_wav'] = ctx_wav
            #             item_tgt['ctx_mask'] = ctx_mask_mel[::vae_stride].contiguous()
            #     else:
            #         n_windows = random.randint(1, 3)

            #         start_min = 0
            #         start_max = int(mel_len_i) - int(len_min)
            #         if start_max < start_min:
            #             res = _single_window_fallback()
            #             if res is None:
            #                 _set_identity_ctx()
            #             else:
            #                 ctx_wav, ctx_mask_mel = res
            #                 item_tgt['ctx_wav'] = ctx_wav
            #                 item_tgt['ctx_mask'] = ctx_mask_mel[::vae_stride].contiguous()
            #         else:
            #             gen_starts = [random.randint(int(start_min), int(start_max)) for _ in range(int(n_windows))]
            #             gen_starts = sorted(set(int(_align_to_fm(x)) for x in gen_starts))

            #             gen_starts_filtered = []
            #             for s in gen_starts:
            #                 if len(gen_starts_filtered) == 0 or int(s) - int(gen_starts_filtered[-1]) >= 200:
            #                     gen_starts_filtered.append(int(s))
            #             gen_starts = gen_starts_filtered

            #             if len(gen_starts) == 0:
            #                 res = _single_window_fallback()
            #                 if res is None:
            #                     _set_identity_ctx()
            #                 else:
            #                     ctx_wav, ctx_mask_mel = res
            #                     item_tgt['ctx_wav'] = ctx_wav
            #                     item_tgt['ctx_mask'] = ctx_mask_mel[::vae_stride].contiguous()
            #             else:
            #                 ctx_mask_mel = torch.ones((mel_len_i, 1), dtype=torch.float32)
            #                 ctx_wav = item_tgt['wav'].clone()
            #                 valid_cnt = 0
            #                 move_first_to_start = random.random() < 0.02
            #                 move_last_to_end = random.random() < 0.02

            #                 for i, gap_start in enumerate(gen_starts):
            #                     gap_start = int(gap_start)
            #                     is_first_window = i == 0
            #                     is_last_window = i == len(gen_starts) - 1
            #                     next_start = int(gen_starts[i + 1]) if (i + 1) < len(gen_starts) else None

            #                     next_dist = (int(next_start) - int(gap_start)) if next_start is not None else (10 ** 9)
            #                     end_dist = int(mel_len_i) - int(gap_start)

            #                     len_max = min(
            #                         int(next_dist) - 100,
            #                         int(mel_len_i * 0.1),
            #                         400,
            #                         int(end_dist),
            #                     )
            #                     if int(len_max) < int(len_min):
            #                         continue

            #                     gap_len = random.randint(int(len_min), int(len_max))
            #                     gap_len = int(_align_to_fm(gap_len))
            #                     gap_len = max(int(len_min), int(gap_len))
            #                     gap_len = min(int(gap_len), int(len_max))

            #                     gap_end = int(gap_start) + int(gap_len)
            #                     gap_end = min(int(gap_end), int(mel_len_i))
            #                     if is_first_window and move_first_to_start:
            #                         gap_start = 0
            #                         gap_end = min(int(gap_len), int(mel_len_i))
            #                     if is_last_window and move_last_to_end:
            #                         gap_end = int(mel_len_i)
            #                         gap_start = max(0, int(gap_end) - int(gap_len))
            #                     if gap_end <= gap_start:
            #                         continue

            #                     s = int(gap_start) * int(hop_size)
            #                     e = int(gap_end) * int(hop_size)
            #                     if e <= s or e > int(ctx_wav.numel()):
            #                         continue

            #                     ctx_mask_mel[gap_start:gap_end] = 0.0
            #                     ctx_wav[s:e] = 0.0
            #                     valid_cnt += 1

            #                 if valid_cnt == 0:
            #                     res = _single_window_fallback()
            #                     if res is None:
            #                         _set_identity_ctx()
            #                     else:
            #                         ctx_wav, ctx_mask_mel = res
            #                         item_tgt['ctx_wav'] = ctx_wav
            #                         item_tgt['ctx_mask'] = ctx_mask_mel[::vae_stride].contiguous()
            #                 else:
            #                     item_tgt['ctx_wav'] = ctx_wav.contiguous()
            #                     item_tgt['ctx_mask'] = ctx_mask_mel[::vae_stride].contiguous()


            ### ---- 加窗口，多个mask，margin较最开始改小，有0.02的概率把第一段mask放到开头 ----
            '''margin最开始是margin_frames = max(int(mel_len_i * 0.1), 100)，改成margin_frames = max(int(mel_len_i * 0.05), 50)'''
            if item_tgt.get('ctx_wav') is None:
                hop_size = int(hparams['hop_size'])
                vae_stride = int(hparams['vae_stride'])

                mel_len_i = int(mel_len)
                len_min = 52

                def _align_to_fm(x):
                    return (int(x) // int(fm)) * int(fm)

                def _single_window_fallback():
                    if mel_len_i <= len_min + 1:
                        return None

                    margin_frames = max(int(mel_len_i * 0.05), 50)
                    gap_len_max = min(400, int(mel_len_i * 0.3), int(mel_len_i) - 1)
                    if gap_len_max <= 0:
                        return None

                    gap_len_min = min(len_min, gap_len_max)
                    gap_len = random.randint(int(gap_len_min), int(gap_len_max))
                    gap_len = max(int(len_min), int(_align_to_fm(gap_len)))
                    if gap_len <= 0:
                        return None

                    start_min = 0 if mel_len_i < (2 * margin_frames + gap_len) else int(margin_frames)
                    start_max = int(mel_len_i) - int(margin_frames) - int(gap_len)
                    if start_max < start_min:
                        start_min = 0
                        start_max = int(mel_len_i) - int(gap_len)
                    if start_max < start_min:
                        return None

                    gap_start = random.randint(int(start_min), int(start_max))
                    gap_start = int(_align_to_fm(gap_start))
                    gap_start = max(0, min(int(gap_start), int(mel_len_i) - 1))
                    gap_end = min(int(gap_start) + int(gap_len), int(mel_len_i))
                    if gap_end <= gap_start:
                        return None

                    ctx_mask_mel = torch.ones((mel_len_i, 1), dtype=torch.float32)
                    ctx_mask_mel[gap_start:gap_end] = 0.0

                    ctx_wav = item_tgt['wav'].clone()
                    s = int(gap_start) * int(hop_size)
                    e = int(gap_end) * int(hop_size)
                    s = max(0, min(int(s), int(ctx_wav.numel())))
                    e = max(0, min(int(e), int(ctx_wav.numel())))
                    if e > s:
                        ctx_wav[s:e] = 0.0

                    return ctx_wav.contiguous(), ctx_mask_mel.contiguous()

                def _set_identity_ctx():
                    item_tgt['ctx_wav'] = item_tgt['wav']
                    item_tgt['ctx_mask'] = torch.ones(
                        (max(1, (mel_len_i + vae_stride - 1) // vae_stride), 1),
                        dtype=torch.float32
                    )

                if mel_len_i < 550:
                    res = _single_window_fallback()
                    if res is None:
                        _set_identity_ctx()
                    else:
                        ctx_wav, ctx_mask_mel = res
                        item_tgt['ctx_wav'] = ctx_wav
                        item_tgt['ctx_mask'] = ctx_mask_mel[::vae_stride].contiguous()
                else:
                    n_windows = random.randint(1, 3)
                    margin_frames = max(int(mel_len_i * 0.05), 50)

                    start_min = int(margin_frames)
                    start_max = int(mel_len_i) - int(margin_frames) - int(len_min)
                    if start_max < start_min:
                        res = _single_window_fallback()
                        if res is None:
                            _set_identity_ctx()
                        else:
                            ctx_wav, ctx_mask_mel = res
                            item_tgt['ctx_wav'] = ctx_wav
                            item_tgt['ctx_mask'] = ctx_mask_mel[::vae_stride].contiguous()
                    else:
                        gen_starts = [random.randint(int(start_min), int(start_max)) for _ in range(int(n_windows))]
                        gen_starts = sorted(set(int(_align_to_fm(x)) for x in gen_starts))

                        gen_starts_filtered = []
                        for s in gen_starts:
                            if len(gen_starts_filtered) == 0 or int(s) - int(gen_starts_filtered[-1]) >= 200:
                                gen_starts_filtered.append(int(s))
                        gen_starts = gen_starts_filtered

                        if len(gen_starts) == 0:
                            res = _single_window_fallback()
                            if res is None:
                                _set_identity_ctx()
                            else:
                                ctx_wav, ctx_mask_mel = res
                                item_tgt['ctx_wav'] = ctx_wav
                                item_tgt['ctx_mask'] = ctx_mask_mel[::vae_stride].contiguous()
                        else:
                            ctx_mask_mel = torch.ones((mel_len_i, 1), dtype=torch.float32)
                            ctx_wav = item_tgt['wav'].clone()
                            valid_cnt = 0
                            move_first_to_start = random.random() < 0.1

                            for i, gap_start in enumerate(gen_starts):
                                gap_start = int(gap_start)
                                is_first_window = i == 0
                                next_start = int(gen_starts[i + 1]) if (i + 1) < len(gen_starts) else None

                                next_dist = (int(next_start) - int(gap_start)) if next_start is not None else (10 ** 9)
                                end_dist_minus_margin = int(mel_len_i) - int(gap_start) - int(margin_frames)

                                len_max = min(
                                    int(next_dist) - 100,
                                    int(mel_len_i * 0.1),
                                    400,
                                    int(end_dist_minus_margin),
                                )
                                if int(len_max) < int(len_min):
                                    continue

                                gap_len = random.randint(int(len_min), int(len_max))
                                gap_len = int(_align_to_fm(gap_len))
                                gap_len = max(int(len_min), int(gap_len))
                                gap_len = min(int(gap_len), int(len_max))

                                gap_end = int(gap_start) + int(gap_len)
                                gap_end = min(int(gap_end), int(mel_len_i) - int(margin_frames))
                                if is_first_window and move_first_to_start:
                                    gap_start = 0
                                    gap_end = min(int(gap_len), int(mel_len_i))
                                if gap_end <= gap_start:
                                    continue

                                s = int(gap_start) * int(hop_size)
                                e = int(gap_end) * int(hop_size)
                                if e <= s or e > int(ctx_wav.numel()):
                                    continue

                                ctx_mask_mel[gap_start:gap_end] = 0.0
                                ctx_wav[s:e] = 0.0
                                valid_cnt += 1

                            if valid_cnt == 0:
                                res = _single_window_fallback()
                                if res is None:
                                    _set_identity_ctx()
                                else:
                                    ctx_wav, ctx_mask_mel = res
                                    item_tgt['ctx_wav'] = ctx_wav
                                    item_tgt['ctx_mask'] = ctx_mask_mel[::vae_stride].contiguous()
                            else:
                                item_tgt['ctx_wav'] = ctx_wav.contiguous()
                                item_tgt['ctx_mask'] = ctx_mask_mel[::vae_stride].contiguous()
            
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
            
        batch = {
            'item_name': [s['item_name'] for s in samples],
            'nsamples': len(samples),
        }

        if valid_item_kv(samples[0], 'wav'):
            batch['wavs'] = collate_xd([s['wav'] for s in samples], 0.0)
            batch['wav_lengths'] = torch.LongTensor([s['wav'].shape[0] for s in samples])
        if valid_item_kv(samples[0], 'ctx_wav'):
            batch['ctx_wavs'] = collate_xd([s['ctx_wav'] for s in samples], 0.0)
            batch['ctx_wav_lengths'] = torch.LongTensor([s['ctx_wav'].shape[0] for s in samples])
        if valid_item_kv(samples[0], 'mel'):
            batch['mels'] = collate_xd([s['mel'] for s in samples], -6.0)
        if valid_item_kv(samples[0], 'ctx_mask'):
            batch['ctx_mask'] = collate_xd([s['ctx_mask'] for s in samples], 0)
       
        if valid_item_kv(samples[0], 'text'):
            batch['text'] = [s['text'] for s in samples]
        if valid_item_kv(samples[0], 'orig_text'):
            batch['orig_text'] = [s['orig_text'] for s in samples]
        if valid_item_kv(samples[0], 'txt_tokens'):
            batch['txt_tokens'] = collate_xd([s['txt_tokens'] for s in samples], 0)
            batch['txt_lengths'] = torch.LongTensor([s['txt_tokens'].numel() for s in samples])
        
        if valid_item_kv(samples[0], 'caption'):
            batch['caption'] = [s['caption'] for s in samples]
        if valid_item_kv(samples[0], 'global'):
            batch['global'] = [s['global'] for s in samples]
        if valid_item_kv(samples[0], 'local'):
            batch['local'] = [s['local'] for s in samples]
        if valid_item_kv(samples[0], 'caption_audio'):
            batch['caption_audio'] = [s['caption_audio'] for s in samples]
        
        if valid_item_kv(samples[0], 'spk_mask'):
            batch['spk_mask'] = collate_xd([s['spk_mask'] for s in samples], 0)
        if valid_item_kv(samples[0], 'audio_mask'):
            batch['audio_mask'] = collate_xd([s['audio_mask'] for s in samples], 0)
        if valid_item_kv(samples[0], 'bgm_flag'):
            batch['bgm_flag'] = torch.LongTensor([s['bgm_flag'] for s in samples])
        
        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch
    

class _DiscreteShapeBatcher:
    def __init__(self, seq_lens, max_tokens, max_sentences, length_fn):
        self.seq_lens = list(seq_lens)
        self.max_tokens = int(max_tokens)
        self.max_sentences = int(max_sentences)
        self.length_fn = length_fn

        self.buffers = {L: [] for L in self.seq_lens}
        self.bsz_by_len = {}
        for L in self.seq_lens:
            bsz = self.max_tokens // int(L)
            bsz = max(1, bsz)
            bsz = min(bsz, self.max_sentences)
            self.bsz_by_len[int(L)] = int(bsz)

    def collate_batch(self, data_item):
        L = int(self.length_fn(data_item))
        if L not in self.buffers:
            return None

        buf = self.buffers[L]
        buf.append(data_item)

        if len(buf) >= self.bsz_by_len[L]:
            batch = buf
            self.buffers[L] = []
            return batch
        return None

    def clear(self):
        for L in self.buffers:
            self.buffers[L] = []


class SwanTTSBucketShmDataset(SpeechEditTTSShmDataset):
    '''重写了父类的get_batcher, _process_item, collater方法'''
    
    def _get_compile_seq_lens(self, hparams):
        default_seq_lens = [128, 256, 512, 768, 1024, 1536, 2560, 4096]
        seq_lens = hparams.get('compile_seq_lens', default_seq_lens)
        seq_lens = [int(x) for x in seq_lens]
        seq_lens = sorted(set([x for x in seq_lens if x > 0]))

        max_shape_combos = int(hparams.get('compile_max_shape_combos', 8))
        if max_shape_combos > 0 and len(seq_lens) > max_shape_combos:
            seq_lens = seq_lens[:max_shape_combos]

        fm = int(hparams['frames_multiple'])
        max_tokens = int(hparams['max_tokens'])
        seq_lens = [L for L in seq_lens if (L <= max_tokens and (L * 4) % fm == 0)]

        if len(seq_lens) == 0:
            unit = fm // math.gcd(fm, 4)
            L = max(1, min(max_tokens, 512) // unit) * unit
            seq_lens = [int(L)]

        return seq_lens

    def _pick_target_len(self, cur_len, seq_lens):
        cur_len = int(cur_len)
        if cur_len <= seq_lens[0]:
            return int(seq_lens[0])
        if cur_len >= seq_lens[-1]:
            return int(seq_lens[-1])
        idx = bisect.bisect_left(seq_lens, cur_len)
        return int(seq_lens[idx])

    def _pad_or_trim_1d(self, x, target_len, pad_value=0.0):
        target_len = int(target_len)
        if x.numel() == target_len:
            return x
        if x.numel() > target_len:
            return x[:target_len]
        return F.pad(x, (0, target_len - x.numel()), value=float(pad_value))

    def _pad_or_trim_2d_time(self, x, target_t, pad_value=0):
        target_t = int(target_t)
        if x.shape[0] == target_t:
            return x
        if x.shape[0] > target_t:
            return x[:target_t]
        pad_t = target_t - x.shape[0]
        pad = x.new_full((pad_t,) + tuple(x.shape[1:]), pad_value)
        return torch.cat([x, pad], dim=0)

    def _quantize_item_to_target_len(self, item, target_len, hparams):
        hop_size = int(hparams['hop_size'])
        vae_stride = int(hparams.get('vae_stride', 1))

        target_len = int(target_len)
        target_mel_len = target_len * 4
        target_wav_len = target_mel_len * hop_size
        target_ctx_mask_len = (target_mel_len + vae_stride - 1) // vae_stride

        if valid_item_kv(item, 'wav'):
            orig_wav_len = int(item.get('wav_len', item['wav'].shape[0]))
            item['wav'] = self._pad_or_trim_1d(item['wav'], target_wav_len, pad_value=0.0)
            item['wav_len'] = min(orig_wav_len, target_wav_len)

        if valid_item_kv(item, 'ctx_wav'):
            orig_ctx_wav_len = int(item.get('ctx_wav_len', item['ctx_wav'].shape[0]))
            item['ctx_wav'] = self._pad_or_trim_1d(item['ctx_wav'], target_wav_len, pad_value=0.0)
            item['ctx_wav_len'] = min(orig_ctx_wav_len, target_wav_len)

        if valid_item_kv(item, 'ctx_mask'):
            item['ctx_mask'] = self._pad_or_trim_2d_time(item['ctx_mask'], target_ctx_mask_len, pad_value=0)

        item['len'] = target_len
        item['target_len'] = target_len
        return item

    def get_batcher(self, hparams, global_stores):
        seq_lens = self._get_compile_seq_lens(hparams)

        batcher = get_from_global_stores(
            'batcher_discrete_shapes',
            global_stores,
            lambda: _DiscreteShapeBatcher(
                seq_lens=seq_lens,
                max_tokens=hparams['max_tokens'],
                max_sentences=hparams['max_sentences'],
                length_fn=lambda x: x['len'],
            )
        )
        return batcher

    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):
        seq_lens = self._get_compile_seq_lens(hparams)

        for item in super()._process_item(processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):
            if item is None:
                continue
            if 'len' not in item or item['len'] is None:
                continue

            target_len = self._pick_target_len(item['len'], seq_lens)
            item = self._quantize_item_to_target_len(item, target_len, hparams)
            yield item

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

        batch = {
            'item_name': [s['item_name'] for s in samples],
            'nsamples': len(samples),
        }

        if valid_item_kv(samples[0], 'wav'):
            batch['wavs'] = collate_xd([s['wav'] for s in samples], 0.0)
            batch['wav_lengths'] = torch.LongTensor([int(s.get('wav_len', s['wav'].shape[0])) for s in samples])

        if valid_item_kv(samples[0], 'ctx_wav'):
            batch['ctx_wavs'] = collate_xd([s['ctx_wav'] for s in samples], 0.0)
            batch['ctx_wav_lengths'] = torch.LongTensor([int(s.get('ctx_wav_len', s['ctx_wav'].shape[0])) for s in samples])

        if valid_item_kv(samples[0], 'mel'):
            batch['mels'] = collate_xd([s['mel'] for s in samples], -6.0)

        if valid_item_kv(samples[0], 'ctx_mask'):
            batch['ctx_mask'] = collate_xd([s['ctx_mask'] for s in samples], 0)

        if valid_item_kv(samples[0], 'text'):
            batch['text'] = [s['text'] for s in samples]
        if valid_item_kv(samples[0], 'orig_text'):
            batch['orig_text'] = [s['orig_text'] for s in samples]
        if valid_item_kv(samples[0], 'txt_tokens'):
            batch['txt_tokens'] = collate_xd([s['txt_tokens'] for s in samples], 0)
            batch['txt_lengths'] = torch.LongTensor([s['txt_tokens'].numel() for s in samples])

        if valid_item_kv(samples[0], 'caption'):
            batch['caption'] = [s['caption'] for s in samples]
        if valid_item_kv(samples[0], 'global'):
            batch['global'] = [s['global'] for s in samples]
        if valid_item_kv(samples[0], 'local'):
            batch['local'] = [s['local'] for s in samples]
        if valid_item_kv(samples[0], 'caption_audio'):
            batch['caption_audio'] = [s['caption_audio'] for s in samples]

        if valid_item_kv(samples[0], 'spk_mask'):
            batch['spk_mask'] = collate_xd([s['spk_mask'] for s in samples], 0)
        if valid_item_kv(samples[0], 'audio_mask'):
            batch['audio_mask'] = collate_xd([s['audio_mask'] for s in samples], 0)

        if 'len' in samples[0]:
            batch['target_len'] = int(samples[0]['len'])

        if not hasattr(self, 'backup_batch') or self.backup_batch is None or random.random() < 0.001:
            self.backup_batch = batch

        return batch


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
            item['item_name'] = item_['item_name']
            txt = raw_text_process(item_['txt_raw'], wav_len=item['wav_len'])
            if txt is None:
                continue
            item['txt'] = f"<S1>{txt}</S1>"
            if hparams.get('shuffle_spk_ids', True):
                item['txt'] = shuffle_speaker_ids(item['txt'])
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
                    
                    if segment_meta.get('txt_raw') is None:
                        continue
                    txt = segment_meta['txt_raw']
                    item['txt'] = f"<S1>{txt}</S1>"
                    if hparams.get('shuffle_spk_ids', True):
                        item['txt'] = shuffle_speaker_ids(item['txt'])
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
            item['wav'] = torch.cat([item['ctx_wav'], item['wav']], dim=0)
            item['wav_len'] = item['wav'].shape[0]
            ctx_mask = torch.zeros((item['wav'].shape[0] // hparams['hop_size'], 1))
            ctx_mask[:item['ctx_wav'].shape[0] // hparams['hop_size']] = 1.0
            item['ctx_mask'] = ctx_mask[::hparams['vae_stride']]
            
            item['item_name'] = item_['item_name']
            txt = item_['txt_raw']
            item['txt'] = f"<S1>{txt}</S1>"
            if hparams.get('shuffle_spk_ids', True):
                item['txt'] = shuffle_speaker_ids(item['txt'])
            ds_name = item_['ds_name']
            item['spk_name'] = f"{ds_name}#{item_['spk']}"
            item['skip_merge_same_spk'] = True
            items.append(item)
        except:
            continue
    return items

def processer_fn_jsonl(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):

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
                item_name = item_['wav_k']
                vocal_k = item_['wav_k']

                # 读全局音频（和原逻辑一致：先下载到 shm，再 torchaudio.load）
                if hparams.get('load_wav', True):
                    data = tos_client.get_object(vocal_k, verbose=False)
                    if data is None:
                        continue
                    global_wav_path = os.path.join(temp_dir, 'global.m4a')
                    with open(global_wav_path, 'wb') as f:
                        f.write(data)
                    try:
                        global_wav, sr_ = torchaudio.load(global_wav_path)
                        global_wav = global_wav.mean(dim=0).numpy()
                    except:
                        continue
                    if len(global_wav) == 0:
                        continue

                segments = item_.get('segments', None)
                if segments is None:
                    continue

                for segment_idx, segment_meta in enumerate(segments):
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

                    if segment_meta.get('txt_raw') is None:
                        continue
                    txt = segment_meta['txt_raw']
                    item['txt'] = f"<S1>{txt}</S1>"
                    if hparams.get('shuffle_spk_ids', True):
                        item['txt'] = shuffle_speaker_ids(item['txt'])

                    item['spk_name'] = item_name + '#' + segment_meta['spk_name']

                    items.append(item)

            except:
                traceback.print_exc()
                continue

    return items
