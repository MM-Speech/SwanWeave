"""Preference triplets for spatial-audio DPO training."""
import torch

from tasks.dataset_utils.base_dataset import valid_item_kv
from tasks.dataset_utils.spat_base_fastdataset import (
    SpatBaseShmDataset,
    _default_posterior_path,
    _load_spatial_audio_preserve_channels,
    _load_vae_posterior,
    _normalize_caption_text,
)
from utils.commons.dataset_utils import collate_xd


def _align_triplet(item, hparams):
    """Keep the three signals aligned in waveform and latent time."""
    prefixes = ("", "neg_", "src_")
    multiple = int(hparams["frames_multiple"]) * int(hparams["hop_size"])
    wav_len = min(int(item[prefix + "wav_len"]) for prefix in prefixes)
    if multiple > 0:
        wav_len = wav_len // multiple * multiple
    if wav_len <= 0:
        raise ValueError("DPO sample %s has no aligned audio frames" % item["sample_id"])
    if hparams.get("use_precomputed_latents", False):
        latent_len = min(int(item[prefix + "latent_len"]) for prefix in prefixes)
        latent_len = min(latent_len, max(1, wav_len // int(hparams["hop_size"]) //
                                         int(hparams.get("vae_stride", 4))))
        if latent_len <= 0:
            raise ValueError("DPO sample %s has no aligned latent tokens" % item["sample_id"])
        for prefix in prefixes:
            for key in ("latent_mean", "latent_var"):
                item[prefix + key] = item[prefix + key][:latent_len].contiguous()
            item[prefix + "latent_len"] = latent_len
    else:
        for prefix in prefixes:
            item[prefix + "wav"] = item[prefix + "wav"][:wav_len].contiguous()
    for prefix in prefixes:
        item[prefix + "wav_len"] = wav_len


def processer_fn_dpo_jsonl(raw_item, _tgt_size, hparams, _global_stores,
                          _skip_logger, _i_worker, _n_worker):
    """Read canonical source/positive/negative paths and caption (or chosen)."""
    use_latents = bool(hparams.get("use_precomputed_latents", False))
    suffix = str(hparams.get("precomputed_latent_suffix", ".spat_vae_posterior.pt"))
    items = []
    for row in raw_item:
        sample_id = str(row.get("sample_id") or row.get("item_name") or "<unknown>")
        caption = _normalize_caption_text(row.get("caption") or row.get("chosen"))
        paths = {
            "": row.get("pos_wav_path") or row.get("wav_path"),
            "neg_": row.get("neg_wav_path"),
            "src_": row.get("src_wav_path"),
        }
        if caption is None or not all(paths.values()):
            raise ValueError(
                "DPO sample %s requires caption/chosen, src_wav_path, "
                "pos_wav_path/wav_path and neg_wav_path" % sample_id
            )
        item = {
            "sample_id": sample_id, "item_name": sample_id,
            "caption": caption, "global": "", "local": caption,
            "wav_path": str(paths[""]), "target_wav_path": str(paths[""]),
            "src_wav_path": str(paths["src_"]), "neg_wav_path": str(paths["neg_"]),
            "edit_type": row.get("edit_type", ""),
            "dpo_error_mode": row.get("error_mode", row.get("negative_type", "")),
            "audio_channels": int(row.get("audio_channels", 4)),
        }
        for prefix, wav_path in paths.items():
            if use_latents:
                latent_path = row.get(prefix + "latent_path")
                if not prefix:
                    latent_path = row.get("pos_latent_path") or latent_path
                latent_path = latent_path or _default_posterior_path(str(wav_path), suffix)
                posterior = _load_vae_posterior(str(latent_path))
                if posterior is None:
                    raise ValueError("DPO sample %s has unreadable posterior: %s" %
                                     (sample_id, latent_path))
                mean, var = posterior["mean"], posterior["var"]
                if mean.shape[1] != int(hparams.get("latent_dim", 128)):
                    raise ValueError("DPO sample %s has an incompatible latent dimension: %s" %
                                     (sample_id, latent_path))
                if not bool(torch.isfinite(mean).all() and torch.isfinite(var).all()):
                    raise ValueError("DPO sample %s has non-finite posterior values: %s" %
                                     (sample_id, latent_path))
                item.update({
                    prefix + "latent_mean": mean,
                    prefix + "latent_var": var,
                    prefix + "latent_len": int(posterior["latent_len"]),
                    prefix + "wav_len": int(posterior["wav_len"]),
                    prefix + "latent_path": str(latent_path),
                })
            else:
                wav = _load_spatial_audio_preserve_channels(str(wav_path),
                                                            int(hparams["audio_sample_rate"]))
                if wav is None:
                    raise ValueError("DPO sample %s has unreadable audio: %s" %
                                     (sample_id, wav_path))
                if wav.ndim != 2 or wav.shape[1] != 4:
                    raise ValueError("DPO sample %s requires four-channel FOA audio: %s" %
                                     (sample_id, wav_path))
                item[prefix + "wav"] = wav
                item[prefix + "wav_len"] = int(wav.shape[0])
        _align_triplet(item, hparams)
        items.append(item)
    return items


class SpatAudioEditDpoDataset(SpatBaseShmDataset):
    def _process_item(self, processer_fn, raw_item, tgt_size, hparams,
                      global_stores, i_worker, n_worker):
        # Validate triplets outside the base processor's skip-on-error handler.
        items = processer_fn(raw_item, tgt_size, hparams, global_stores,
                             None, i_worker, n_worker)
        yield from super()._process_item(
            lambda *_: items, raw_item, tgt_size, hparams,
            global_stores, i_worker, n_worker,
        )

    def collater(self, samples):
        batch = super().collater(samples)
        if not batch:
            return batch
        if isinstance(samples[0], list):
            samples = samples[0]
        if valid_item_kv(samples[0], "neg_wav"):
            batch["neg_wavs"] = collate_xd([sample["neg_wav"] for sample in samples], 0.0)
            batch["neg_wav_lengths"] = torch.LongTensor([sample["neg_wav_len"] for sample in samples])
        if valid_item_kv(samples[0], "neg_latent_mean"):
            batch["neg_latent_means"] = collate_xd([sample["neg_latent_mean"] for sample in samples], 0.0)
            batch["neg_latent_vars"] = collate_xd([sample["neg_latent_var"] for sample in samples], 0.0)
            batch["neg_latent_lengths"] = torch.LongTensor([sample["neg_latent_len"] for sample in samples])
        for key in ("neg_wav_path", "neg_latent_path", "dpo_error_mode"):
            if valid_item_kv(samples[0], key):
                batch[key] = [sample[key] for sample in samples]
        return batch
