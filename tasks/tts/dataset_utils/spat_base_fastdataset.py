import collections
import collections.abc
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import argparse
import json
import traceback
from pathlib import Path

import torch
import torchaudio

from utils.commons.base_shm_dataset import get_from_global_stores
from utils.commons.hparams import hparams
from utils.commons.dataset_utils import SkipLogger, collate_xd
from tasks.tts.dataset_utils.swan_base_fastdataset import SwanTTSShmDataset, valid_item_kv


DEBUG = False


def _print_skip(reason: str, i_worker=None, n_worker=None, item_name: str = None, extra: str = ""):
    if DEBUG is False:
        return
    try:
        worker_info = f"{i_worker}/{n_worker}" if i_worker is not None and n_worker is not None else "-"
        name_info = f", item={item_name}" if item_name else ""
        extra_info = f", {extra}" if extra else ""
        print(f"[SPAT_SKIP][{worker_info}] {reason}{name_info}{extra_info}")
    except Exception:
        pass


def _normalize_caption_text(text):
    if text is None:
        return None
    text = str(text).strip()
    if text == "":
        return None
    return text


def _load_spatial_audio_preserve_channels(wav_path: str, target_sr: int) -> torch.Tensor | None:
    try:
        wav, org_sr = torchaudio.load(wav_path)
    except Exception:
        return None
    wav = wav.to(torch.float32)
    if int(org_sr) != int(target_sr):
        wav = torchaudio.functional.resample(wav, orig_freq=int(org_sr), new_freq=int(target_sr))

    # Keep FOA/multichannel audio. Time axis is first so collate_xd pads on time.
    wav = wav.transpose(0, 1).contiguous()
    if wav.numel() == 0:
        return None

    peak = float(torch.max(torch.abs(wav)))
    if peak > 1.0:
        wav = wav / peak
    return wav


def _default_posterior_path(wav_path: str, suffix: str = ".spat_vae_posterior.pt") -> str:
    path = Path(wav_path)
    return str(path.with_name(f"{path.stem}{suffix}"))


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_vae_posterior(path: str) -> dict | None:
    try:
        payload = _torch_load_cpu(path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    mean = payload.get("mean")
    var = payload.get("var")
    if var is None and payload.get("std") is not None:
        var = payload["std"].float().pow(2)
    if not isinstance(mean, torch.Tensor) or not isinstance(var, torch.Tensor):
        return None
    if mean.ndim != 2 or var.shape != mean.shape:
        return None

    mean = mean.float().contiguous()
    var = var.float().clamp_min(0).contiguous()
    latent_len = int(payload.get("latent_len", mean.shape[0]))
    wav_len = int(payload.get("wav_len", latent_len))
    return {
        "mean": mean,
        "var": var,
        "latent_len": min(latent_len, int(mean.shape[0])),
        "wav_len": wav_len,
        "path": str(path),
    }


class SpatBaseShmDataset(SwanTTSShmDataset):
    def _process_item(self, processer_fn, raw_item, tgt_size, hparams, global_stores, i_worker, n_worker):
        hop_size = int(hparams['hop_size'])
        fm_wav = int(hparams['frames_multiple']) * hop_size
        use_precomputed_latents = bool(hparams.get('use_precomputed_latents', False))

        skip_logger: SkipLogger = get_from_global_stores(
            'skip_logger', global_stores,
            lambda: SkipLogger(
                ['no_wav_cnt', 'bad_caption_cnt', 'frames_out_of_range_cnt'],
                interval=1000,
                i_worker=i_worker,
                n_worker=n_worker,
            )
        )

        try:
            items = processer_fn(raw_item, tgt_size, hparams, global_stores, skip_logger, i_worker, n_worker)
        except Exception:
            fn_name = getattr(processer_fn, "__name__", str(processer_fn))
            print(f"| SpatBaseShmDataset: processer_fn crashed worker={i_worker}/{n_worker} fn={fn_name}")
            traceback.print_exc()
            return

        if items is None or len(items) == 0:
            return

        for item_tgt in items:
            has_wav = item_tgt is not None and item_tgt.get('wav') is not None
            has_latent = (
                item_tgt is not None
                and item_tgt.get('latent_mean') is not None
                and item_tgt.get('latent_var') is not None
            )
            if item_tgt is None or (not has_wav and not (use_precomputed_latents and has_latent)):
                skip_logger.update(1)
                continue

            if has_wav and hparams.get('load_wav', True) and fm_wav > 0:
                valid_len = (item_tgt['wav'].shape[0] // fm_wav) * fm_wav
                item_tgt['wav'] = item_tgt['wav'][:valid_len]
                if item_tgt.get('src_wav', None) is not None:
                    item_tgt['src_wav'] = item_tgt['src_wav'][:valid_len]
                    item_tgt['src_wav_len'] = int(item_tgt['src_wav'].shape[0])
                item_tgt['wav_len'] = int(item_tgt['wav'].shape[0])
            elif has_wav:
                item_tgt['wav_len'] = int(item_tgt['wav'].shape[0])
            elif fm_wav > 0:
                item_tgt['wav_len'] = (int(item_tgt['wav_len']) // fm_wav) * fm_wav

            if has_wav:
                if item_tgt['wav'].numel() == 0 or item_tgt['wav'].shape[0] == 0:
                    skip_logger.update(1)
                    _print_skip("empty_wav_after_alignment", i_worker, n_worker, item_name=item_tgt.get('item_name', ''))
                    continue
                item_tgt['wav'] = item_tgt['wav'].contiguous()
            elif int(item_tgt.get('wav_len', 0)) <= 0:
                skip_logger.update(1)
                _print_skip("empty_precomputed_latent", i_worker, n_worker, item_name=item_tgt.get('item_name', ''))
                continue

            mel_len_total = item_tgt['wav_len'] // hop_size
            if not (hparams['max_frames'] >= mel_len_total > hparams['min_frames']):
                skip_logger.update(1)
                _print_skip(
                    "frames_out_of_range",
                    i_worker,
                    n_worker,
                    item_name=item_tgt.get('item_name', ''),
                    extra=f"mel_len={mel_len_total}, allowed=({hparams['min_frames']}, {hparams['max_frames']}]",
                )
                continue

            caption = _normalize_caption_text(item_tgt.get('caption'))
            if caption is None:
                skip_logger.update(1)
                _print_skip("empty_caption", i_worker, n_worker, item_name=item_tgt.get('item_name', ''))
                continue
            item_tgt['caption'] = caption
            item_tgt.setdefault('global', '')
            item_tgt.setdefault('local', caption)

            item_tgt['len'] = mel_len_total // int(hparams.get('vae_stride', 4))
            yield item_tgt
            skip_logger.step(1)

    def collater(self, samples):
        batch = super().collater(samples)
        if not batch:
            return batch

        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]

        if valid_item_kv(samples[0], 'sample_id'):
            batch['sample_id'] = [s['sample_id'] for s in samples]
        if valid_item_kv(samples[0], 'wav_path'):
            batch['wav_path'] = [s['wav_path'] for s in samples]
        if valid_item_kv(samples[0], 'target_wav_path'):
            batch['target_wav_path'] = [s['target_wav_path'] for s in samples]
        if valid_item_kv(samples[0], 'wav_len') and 'wav_lengths' not in batch:
            batch['wav_lengths'] = torch.LongTensor([int(s['wav_len']) for s in samples])
        if valid_item_kv(samples[0], 'src_wav'):
            batch['src_wavs'] = collate_xd([s['src_wav'] for s in samples], 0.0)
            batch['src_wav_lengths'] = torch.LongTensor([int(s.get('src_wav_len', s['src_wav'].shape[0])) for s in samples])
        if valid_item_kv(samples[0], 'latent_mean'):
            batch['latent_means'] = collate_xd([s['latent_mean'] for s in samples], 0.0)
            batch['latent_vars'] = collate_xd([s['latent_var'] for s in samples], 0.0)
            batch['latent_lengths'] = torch.LongTensor([int(s.get('latent_len', s['latent_mean'].shape[0])) for s in samples])
        if valid_item_kv(samples[0], 'src_latent_mean'):
            batch['src_latent_means'] = collate_xd([s['src_latent_mean'] for s in samples], 0.0)
            batch['src_latent_vars'] = collate_xd([s['src_latent_var'] for s in samples], 0.0)
            batch['src_latent_lengths'] = torch.LongTensor([int(s.get('src_latent_len', s['src_latent_mean'].shape[0])) for s in samples])
        if valid_item_kv(samples[0], 'latent_path'):
            batch['latent_path'] = [s['latent_path'] for s in samples]
        if valid_item_kv(samples[0], 'src_latent_path'):
            batch['src_latent_path'] = [s['src_latent_path'] for s in samples]
        if valid_item_kv(samples[0], 'src_wav_path'):
            batch['src_wav_path'] = [s['src_wav_path'] for s in samples]
        if valid_item_kv(samples[0], 'edit_type'):
            batch['edit_type'] = [s['edit_type'] for s in samples]
        if valid_item_kv(samples[0], 'audio_channels'):
            batch['audio_channels'] = torch.LongTensor([int(s['audio_channels']) for s in samples])
        return batch


def processer_fn_jsonl(raw_item, _tgt_size, hparams, _global_stores, skip_logger, i_worker, n_worker):
    sr = int(hparams['audio_sample_rate'])
    items = []
    use_precomputed_latents = bool(hparams.get('use_precomputed_latents', False))
    posterior_suffix = str(hparams.get('precomputed_latent_suffix', '.spat_vae_posterior.pt'))

    for item_ in raw_item:
        try:
            if item_ is None or not isinstance(item_, dict):
                continue

            wav_path = item_.get('wav_path') or item_.get('audio_path')
            src_wav_path = (
                item_.get('src_wav_path')
                or item_.get('source_wav_path')
                or item_.get('orig_wav_path')
                or item_.get('input_wav_path')
            )
            caption = item_.get('caption') or item_.get('text')
            caption = _normalize_caption_text(caption)
            if not wav_path or caption is None:
                skip_logger.update(1)
                _print_skip("missing_wav_or_caption", i_worker, n_worker, extra=f"wav_path={wav_path}")
                continue
            if not bool(hparams.get('train_base', True)) and not src_wav_path:
                skip_logger.update(1)
                _print_skip("missing_src_wav_for_edit_training", i_worker, n_worker, extra=f"wav_path={wav_path}")
                continue

            wav = None
            src_wav = None
            posterior = None
            src_posterior = None
            latent_path = item_.get('latent_path') or item_.get('vae_posterior_path')
            if not latent_path and wav_path:
                latent_path = _default_posterior_path(wav_path, posterior_suffix)
            src_latent_path = item_.get('src_latent_path') or item_.get('src_vae_posterior_path')
            if not src_latent_path and src_wav_path:
                src_latent_path = _default_posterior_path(src_wav_path, posterior_suffix)

            if use_precomputed_latents:
                posterior = _load_vae_posterior(latent_path)
                if posterior is None:
                    skip_logger.update(1)
                    _print_skip("load_latent_failed", i_worker, n_worker, item_name=item_.get('sample_id', ''), extra=str(latent_path))
                    continue
                if src_wav_path:
                    src_posterior = _load_vae_posterior(src_latent_path)
                    if src_posterior is None:
                        skip_logger.update(1)
                        _print_skip("load_src_latent_failed", i_worker, n_worker, item_name=item_.get('sample_id', ''), extra=str(src_latent_path))
                        continue
            else:
                wav = _load_spatial_audio_preserve_channels(wav_path, sr)
                if wav is None:
                    skip_logger.update(1)
                    _print_skip("load_wav_failed", i_worker, n_worker, item_name=item_.get('sample_id', ''))
                    continue
                if src_wav_path:
                    src_wav = _load_spatial_audio_preserve_channels(src_wav_path, sr)
                    if src_wav is None:
                        skip_logger.update(1)
                        _print_skip("load_src_wav_failed", i_worker, n_worker, item_name=item_.get('sample_id', ''))
                        continue

            item = {
                'caption': caption,
                'global': '',
                'local': caption,
                'item_name': str(item_.get('sample_id', item_.get('item_name', wav_path))),
                'sample_id': str(item_.get('sample_id', item_.get('item_name', wav_path))),
                'wav_path': str(wav_path),
                'target_wav_path': str(item_.get('target_wav_path', wav_path)),
                'edit_type': item_.get('edit_type', ''),
            }
            if posterior is not None:
                item.update({
                    'latent_mean': posterior['mean'],
                    'latent_var': posterior['var'],
                    'latent_len': int(posterior['latent_len']),
                    'wav_len': int(posterior['wav_len']),
                    'audio_channels': int(item_.get('audio_channels', 4)),
                    'latent_path': str(latent_path),
                })
            elif wav is not None:
                item.update({
                    'wav': wav,
                    'wav_len': int(wav.shape[0]),
                    'audio_channels': int(wav.shape[1]) if wav.ndim == 2 else 1,
                })
            if src_wav is not None:
                item['src_wav'] = src_wav
                item['src_wav_len'] = int(src_wav.shape[0])
                item['src_wav_path'] = str(src_wav_path)
            if src_posterior is not None:
                item['src_latent_mean'] = src_posterior['mean']
                item['src_latent_var'] = src_posterior['var']
                item['src_latent_len'] = int(src_posterior['latent_len'])
                item['src_wav_len'] = int(src_posterior['wav_len'])
                item['src_latent_path'] = str(src_latent_path)
                item['src_wav_path'] = str(src_wav_path)
            items.append(item)
        except Exception as e:
            _print_skip(
                "processer_fn_jsonl_exception",
                i_worker,
                n_worker,
                item_name=item_.get('sample_id', '') if isinstance(item_, dict) else '',
                extra=f"err={str(e)}",
            )
            continue

    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick load test for spat_base_fastdataset.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/mnt/bn/sa-ag-data/leike/spatial_edit/triplet/metadata_base/spatlibri.jsonl"),
        help="Path to a .json or .jsonl manifest.",
    )
    parser.add_argument("--limit", type=int, default=2, help="Number of samples to load.")
    parser.add_argument("--sample_rate", type=int, default=48000, help="Target sample rate.")
    args = parser.parse_args()

    rows = []
    if args.manifest.suffix == ".json":
        with args.manifest.open("r", encoding="utf-8") as f:
            rows = json.load(f)[: args.limit]
    elif args.manifest.suffix == ".jsonl":
        with args.manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if len(rows) >= args.limit:
                    break
    else:
        raise ValueError(f"Unsupported manifest format: {args.manifest}")

    hparams.clear()
    hparams.update(
        {
            "audio_sample_rate": int(args.sample_rate),
            "hop_size": 480,
            "frames_multiple": 4,
            "min_frames": 1,
            "max_frames": 120000,
            "vae_stride": 4,
            "use_cosyvoice2_text_tokenizer": False,
            "load_wav": True,
        }
    )

    items = processer_fn_jsonl(rows, None, hparams, {}, None, 0, 1)
    dataset = object.__new__(SpatBaseShmDataset)
    items = list(dataset._process_item(lambda *_: items, rows, None, hparams, {}, 0, 1))
    batch = dataset.collater(items) if len(items) > 0 else {}

    print(f"manifest={args.manifest}")
    print(f"loaded_items={len(items)}")
    if len(items) > 0:
        print(f"first_sample_id={items[0].get('sample_id')}")
        print(f"first_wav_shape={tuple(items[0]['wav'].shape)}")
        print(f"first_caption={items[0].get('caption')}")
    if batch:
        print(f"batch_keys={sorted(batch.keys())}")
        if "wavs" in batch:
            print(f"batch_wavs_shape={tuple(batch['wavs'].shape)}")
        if "wav_lengths" in batch:
            print(f"wav_lengths={batch['wav_lengths'].tolist()}")
