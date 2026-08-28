import torch
import torchaudio
import numpy as np
import random

from utils.commons.base_shm_dataset import get_from_global_stores
from utils.commons.dataset_utils import SkipLogger
from utils.audio.vad import run_vad_trim
from utils.text.split_text import get_word_list, remove_spaces_between_chinese
from utils.text import is_chinese
from tasks.tts.dataset_utils.tts_fastdataset_v2 import BaseTTSShmDataset
from tasks.tts.dataset_utils.promptaudio_fastdataset_v2 import _SPACE_RE, _S1S2_TEXT_RE, _I_OPEN_TAG_RE, _I_CLOSE_TAG_RE, _norm_spaces_caption, _normalize_text_field_caption, augment_text_with_pinyin_s1s2_safe, _build_text_from_caption_s1s2, raw_text_process, raw_text_process_s1s2_tagged, _get_sx_token_patterns, build_spk_mask_from_text_tokens, _print_skip


class PromptDpoShmDataset(BaseTTSShmDataset):

    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):
        hop_size = hparams['hop_size']
        fm = hparams['frames_multiple']
        fm_wav = fm * hop_size
        sr = hparams['audio_sample_rate']
        vae_stride = int(hparams.get('vae_stride', 4))

        cosyvoice2_text_tokenizer = None
        if hparams.get('use_cosyvoice2_text_tokenizer', False):
            from utils.text.cosyvoice2_tokenizer import get_tokenizer
            cosyvoice2_text_tokenizer = get_from_global_stores(
                'cosyvoice2_text_tokenizer',
                global_stores,
                lambda: get_tokenizer(multilingual=True, num_languages=100)
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
        if items is None or len(items) == 0:
            _print_skip("processer_fn_returned_none_or_empty", i_worker, n_worker, extra=f"tgt_size={tgt_size}")
            return

        for item_tgt in items:
            try:
                # ======== 1) good / bad / ctx wav 对齐 & 长度统一 & 拼接 ctx ========
                wav_good = item_tgt['wavs_good'].float()
                wav_bad = item_tgt['wavs_bad'].float()
                ctx_wav = item_tgt.get('ctx_wav', None)

                # 保证 1D
                if wav_good.dim() > 1:
                    wav_good = wav_good.mean(dim=0)
                if wav_bad.dim() > 1:
                    wav_bad = wav_bad.mean(dim=0)
                if ctx_wav is not None:
                    if isinstance(ctx_wav, torch.Tensor):
                        ctx_wav = ctx_wav.float()
                        if ctx_wav.dim() > 1:
                            ctx_wav = ctx_wav.mean(dim=0)
                    else:
                        # processer 也可能给 numpy/list
                        ctx_wav = torch.as_tensor(ctx_wav, dtype=torch.float32)
                        if ctx_wav.dim() > 1:
                            ctx_wav = ctx_wav.mean(dim=0)

                # 对齐到 fm_wav
                if fm_wav > 0:
                    wav_good = wav_good[: wav_good.shape[0] // fm_wav * fm_wav]
                    wav_bad = wav_bad[: wav_bad.shape[0] // fm_wav * fm_wav]
                    if ctx_wav is not None:
                        ctx_wav = ctx_wav[: ctx_wav.shape[0] // fm_wav * fm_wav]

                if wav_good.numel() == 0 or wav_bad.numel() == 0:
                    skip_logger.update(1)
                    _print_skip("empty_good_or_bad_after_alignment", i_worker, n_worker, item_name=item_tgt.get('item_name', ''))
                    continue

                # pair 裁到同长（min_len）
                min_len = min(wav_good.shape[0], wav_bad.shape[0])
                if min_len <= 0:
                    skip_logger.update(1)
                    _print_skip("min_len_zero_after_pair_trim", i_worker, n_worker, item_name=item_tgt.get('item_name', ''))
                    continue
                if fm_wav > 0:
                    min_len = (min_len // fm_wav) * fm_wav

                wav_good = wav_good[:min_len].contiguous()
                wav_bad = wav_bad[:min_len].contiguous()

                # ctx_wav 缺失时：按旧逻辑（不拼）
                if ctx_wav is None or ctx_wav.numel() == 0:
                    # 兼容 collater：给一个全 0 ctx_mask 或者直接不放字段都行
                    item_tgt['ctx_wav'] = None
                    item_tgt['ctx_mask'] = None

                    item_tgt['wavs_good'] = wav_good
                    item_tgt['wavs_bad'] = wav_bad
                    item_tgt['wav_lengths'] = int(min_len)
                    item_tgt['wav'] = wav_good
                    item_tgt['wav_len'] = int(min_len)

                else:
                    ctx_wav = ctx_wav.contiguous()
                    ctx_len = int(ctx_wav.shape[0])

                    # 拼接：ctx + good/bad
                    wav_good_full = torch.cat([ctx_wav, wav_good], dim=0).contiguous()
                    wav_bad_full = torch.cat([ctx_wav, wav_bad], dim=0).contiguous()

                    full_len = int(wav_good_full.shape[0])  # good/bad full_len 相同

                    # 构造 ctx_mask（mel 级别 -> latent）
                    mel_len_full = full_len // hop_size
                    mel_len_ctx = ctx_len // hop_size
                    ctx_mask = torch.zeros((mel_len_full, 1), dtype=torch.float32)
                    if mel_len_ctx > 0:
                        ctx_mask[:mel_len_ctx] = 1.0
                    ctx_mask = ctx_mask[::vae_stride]  # latent 级

                    item_tgt['ctx_wav'] = ctx_wav
                    item_tgt['ctx_mask'] = ctx_mask

                    item_tgt['wavs_good'] = wav_good_full
                    item_tgt['wavs_bad'] = wav_bad_full

                    # 给 DPO 任务用：我建议用“拼接后长度”，更一致
                    item_tgt['wav_lengths'] = int(full_len)

                    # 如果你还想保留“主 wav 长度（不含 ctx）”，加这个字段：
                    item_tgt['main_wav_lengths'] = int(min_len)

                    # 后续过滤与 tokenizer 用的 wav/wav_len 也应该是拼接后的
                    item_tgt['wav'] = wav_good_full
                    item_tgt['wav_len'] = int(full_len)

                # ======== 2) 帧长过滤（用拼接后 wav_len） ========
                mel_len_total = item_tgt['wav_len'] // hop_size
                if not (hparams['max_frames'] >= mel_len_total > hparams['min_frames']):
                    skip_logger.update(1)
                    _print_skip(
                        "frames_out_of_range_dpo",
                        i_worker, n_worker,
                        item_name=item_tgt.get('item_name', ''),
                        extra=f"mel_len={mel_len_total}, allowed=({hparams['min_frames']}, {hparams['max_frames']}]"
                    )
                    continue

                # ======== 3) 文本清洗 ========
                if item_tgt.get('use_raw_txt_as_text', False):
                    txt = raw_text_process_s1s2_tagged(item_tgt['txt'], wav_len=item_tgt['wav_len'])
                else:
                    txt = raw_text_process(item_tgt['txt'], wav_len=item_tgt['wav_len'])

                if txt is None:
                    txt = ''
                item_tgt['text'] = txt

                # ======== 4) cosyvoice2 text tokens / spk_mask ========
                if cosyvoice2_text_tokenizer is not None:
                    if hparams.get('mix_text_pinyin', {}).get('enable', False):
                        item_tgt['text'] = augment_text_with_pinyin_s1s2_safe(item_tgt['text'], hparams)

                    text_tokens = cosyvoice2_text_tokenizer.encode(item_tgt['text'])
                    text_tokens = torch.tensor(text_tokens).long()

                    latent_len = int(item_tgt['wav_len'] // hop_size // vae_stride)
                    if latent_len <= 0 or text_tokens.numel() > latent_len:
                        skip_logger.update(1)
                        _print_skip(
                            "txt_tokens_longer_than_latent_dpo",
                            i_worker, n_worker,
                            item_name=item_tgt.get('item_name', ''),
                            extra=f"txt_tokens={text_tokens.numel()}, latent={latent_len}, wav_len={item_tgt['wav_len']}"
                        )
                        continue

                    item_tgt['txt_tokens'] = text_tokens

                    sx_patterns = get_from_global_stores(
                        'cosyvoice2_sx_token_patterns',
                        global_stores,
                        lambda: _get_sx_token_patterns(cosyvoice2_text_tokenizer)
                    )
                    item_tgt['spk_mask'] = build_spk_mask_from_text_tokens(text_tokens, sx_patterns)
                    assert item_tgt['spk_mask'].shape == text_tokens.shape, (
                        item_tgt['spk_mask'].shape, text_tokens.shape
                    )

                # ======== 5) len（latent length） ========
                item_tgt['len'] = item_tgt['wav_len'] // hop_size // vae_stride

                yield item_tgt
                skip_logger.step(1)

            except Exception as e:
                import traceback
                traceback.print_exc()
                _print_skip("process_item_dpo_exception", i_worker, n_worker,
                            item_name=item_tgt.get('item_name', ''),
                            extra=f"err={str(e)}")
                continue

def processer_fn_dpojson(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker):
    fm = hparams['frames_multiple']
    hop_size = hparams['hop_size']
    fm_wav = fm * hop_size
    sr = hparams['audio_sample_rate']

    items = []

    def _load_1d_wav_any(x, target_sr: int):
        """
        x 可以是 wav path(str) 或 (list/np/torch) 波形。
        返回 1D FloatTensor，已重采样到 target_sr（若需要）。
        """
        if x is None:
            return None

        # path
        if isinstance(x, str) and len(x) > 0:
            wav, org_sr = torchaudio.load(x)
            wav = wav.to(torch.float32)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if org_sr != target_sr:
                wav = torchaudio.functional.resample(wav, orig_freq=org_sr, new_freq=target_sr)
            return wav[0].contiguous()

        # array/tensor
        if isinstance(x, torch.Tensor):
            wav = x.to(torch.float32)
            if wav.dim() > 1:
                wav = wav.mean(dim=0)
            return wav.contiguous()

        wav = torch.as_tensor(x, dtype=torch.float32)
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        return wav.contiguous()

    for item_ in raw_item:
        try:
            wav_good_path = item_.get('wav_good_path') or item_.get('wav_path_good')
            wav_bad_path = item_.get('wav_bad_path') or item_.get('wav_path_bad')

            if not wav_good_path or not wav_bad_path:
                _print_skip("missing_good_or_bad_path", i_worker, n_worker,
                            item_name=item_.get('item_name', item_.get('id', '')))
                continue

            # ======== good ========
            try:
                wav_g, sr_g = torchaudio.load(wav_good_path)
                wav_g = wav_g.to(torch.float32)
            except Exception as e:
                _print_skip("load_wav_good_failed", i_worker, n_worker,
                            item_name=item_.get('item_name', wav_good_path),
                            extra=f"path={wav_good_path}, err={str(e)}")
                continue

            if wav_g.shape[0] > 1:
                wav_g = wav_g.mean(dim=0, keepdim=True)
            if sr_g != sr:
                try:
                    wav_g = torchaudio.functional.resample(wav_g, orig_freq=sr_g, new_freq=sr)
                except Exception as e:
                    _print_skip("resample_good_failed", i_worker, n_worker,
                                item_name=item_.get('item_name', wav_good_path),
                                extra=f"from={sr_g} to={sr}, err={str(e)}")
                    continue
            wav_g = wav_g[0]  # [T]

            # ======== bad ========
            try:
                wav_b, sr_b = torchaudio.load(wav_bad_path)
                wav_b = wav_b.to(torch.float32)
            except Exception as e:
                _print_skip("load_wav_bad_failed", i_worker, n_worker,
                            item_name=item_.get('item_name', wav_bad_path),
                            extra=f"path={wav_bad_path}, err={str(e)}")
                continue

            if wav_b.shape[0] > 1:
                wav_b = wav_b.mean(dim=0, keepdim=True)
            if sr_b != sr:
                try:
                    wav_b = torchaudio.functional.resample(wav_b, orig_freq=sr_b, new_freq=sr)
                except Exception as e:
                    _print_skip("resample_bad_failed", i_worker, n_worker,
                                item_name=item_.get('item_name', wav_bad_path),
                                extra=f"from={sr_b} to={sr}, err={str(e)}")
                    continue
            wav_b = wav_b[0]

            # ======== ctx ========

            ctx_src = item_.get('ctx_wav_path', None)

            ctx_wav = None
            if ctx_src is not None:
                ctx_wav = _load_1d_wav_any(ctx_src, target_sr=sr)

            item = {}
            item['wavs_good'] = wav_g
            item['wavs_bad'] = wav_b
            item['ctx_wav'] = ctx_wav  # 可能为 None

            txt = item_.get('text') or item_.get('caption') or item_.get('txt', '')
            item['txt'] = txt

            item['item_name'] = item_.get('item_name', item_.get('utt_id', item_.get('id', wav_good_path)))
            item['spk_name'] = item_.get('spk_name', item_.get('speaker', item['item_name']))
            item['use_raw_txt_as_text'] = True

            items.append(item)
            skip_logger.step(1)

        except Exception as e:
            import traceback
            traceback.print_exc()
            _print_skip("processer_fn_dpojson_exception", i_worker, n_worker,
                        item_name=item_.get('item_name', item_.get('id', '')),
                        extra=f"err={str(e)}")
            continue

    return items
