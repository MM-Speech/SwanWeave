import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.commons.base_shm_dataset import get_from_global_stores
from utils.commons.dataset_utils import SkipLogger, collate_xd
from utils.commons.hparams import hparams
from utils.commons.io import print_once
from utils.nn.model_utils import unwrap_model
from utils.nn.seq_utils import sequence_mask

from tasks.tts.dataset_utils.swan_base_fastdataset import valid_item_kv
from tasks.tts.dataset_utils.spat_base_fastdataset import (
    SpatBaseShmDataset,
    _default_posterior_path,
    _load_spatial_audio_preserve_channels,
    _load_vae_posterior,
    _normalize_caption_text,
)
from tasks.tts.spat_audio_edit_task import (
    SpatAudioEditTask,
    encode_stable_audio_parallel,
    merge_foa_pair_latents,
    pad_latent_seq_time,
    sample_precomputed_posterior,
)


def _first_value(item: Dict[str, Any], keys: Iterable[str]):
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def _as_str_path(path) -> Optional[str]:
    if path in (None, ""):
        return None
    return str(path)


def _path_from_base(base_dir: Optional[Path], *parts: str) -> Optional[str]:
    if base_dir is None:
        return None
    return str(base_dir.joinpath(*parts))


def _infer_base_dir(item: Dict[str, Any]) -> Optional[Path]:
    metadata_path = _first_value(item, ("metadata_path", "meta_path", "json_path"))
    if metadata_path:
        return Path(str(metadata_path)).expanduser().resolve().parent

    sample_dir = _first_value(item, ("sample_dir", "dpo_dir", "item_dir"))
    if sample_dir:
        return Path(str(sample_dir)).expanduser().resolve()
    return None


def _resolve_dpo_paths(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    base_dir = _infer_base_dir(item)
    src_wav_path = _first_value(
        item,
        (
            "src_wav_path",
            "source_wav_path",
            "orig_wav_path",
            "input_wav_path",
        ),
    ) or _path_from_base(base_dir, "source", "foa.wav")
    pos_wav_path = _first_value(
        item,
        (
            "pos_wav_path",
            "positive_wav_path",
            "chosen_wav_path",
            "target_wav_path",
            "wav_path",
            "audio_path",
        ),
    ) or _path_from_base(base_dir, "target", "foa.wav")
    neg_wav_path = _first_value(
        item,
        (
            "neg_wav_path",
            "negative_wav_path",
            "rejected_wav_path",
            "bad_wav_path",
        ),
    ) or _path_from_base(base_dir, "negative", "foa.wav")
    return _as_str_path(src_wav_path), _as_str_path(pos_wav_path), _as_str_path(neg_wav_path)


def _resolve_caption(item: Dict[str, Any]) -> Optional[str]:
    captions = item.get("captions")
    first_caption = captions[0] if isinstance(captions, list) and len(captions) > 0 else None
    return _normalize_caption_text(
        _first_value(item, ("caption", "chosen", "positive_caption", "text", "instruction")) or first_caption
    )


def _resolve_latent_path(
    item: Dict[str, Any],
    keys: Iterable[str],
    wav_path: Optional[str],
    suffix: str,
) -> Optional[str]:
    latent_path = _first_value(item, keys)
    if latent_path:
        return str(latent_path)
    if wav_path:
        return _default_posterior_path(wav_path, suffix)
    return None


def _trim_tensor_time(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    target_len = int(target_len)
    if int(tensor.shape[0]) == target_len:
        return tensor.contiguous()
    return tensor[:target_len].contiguous()


def _trim_dpo_item_to_common_length(item: Dict[str, Any], hparams_) -> bool:
    fm_wav = int(hparams_["frames_multiple"]) * int(hparams_["hop_size"])
    use_precomputed_latents = bool(hparams_.get("use_precomputed_latents", False))

    if use_precomputed_latents:
        wav_len = min(
            int(item["wav_len"]),
            int(item["neg_wav_len"]),
            int(item["src_wav_len"]),
        )
        if fm_wav > 0:
            wav_len = (wav_len // fm_wav) * fm_wav
        latent_len = max(1, wav_len // int(hparams_["hop_size"]) // int(hparams_.get("vae_stride", 4)))
        latent_len = min(
            latent_len,
            int(item["latent_mean"].shape[0]),
            int(item["neg_latent_mean"].shape[0]),
            int(item["src_latent_mean"].shape[0]),
        )
        if wav_len <= 0 or latent_len <= 0:
            return False
        for key in ("latent_mean", "latent_var", "neg_latent_mean", "neg_latent_var", "src_latent_mean", "src_latent_var"):
            item[key] = _trim_tensor_time(item[key], latent_len)
        item["wav_len"] = wav_len
        item["neg_wav_len"] = wav_len
        item["src_wav_len"] = wav_len
        item["latent_len"] = latent_len
        item["neg_latent_len"] = latent_len
        item["src_latent_len"] = latent_len
        return True

    wav_len = min(
        int(item["wav"].shape[0]),
        int(item["neg_wav"].shape[0]),
        int(item["src_wav"].shape[0]),
    )
    if fm_wav > 0:
        wav_len = (wav_len // fm_wav) * fm_wav
    if wav_len <= 0:
        return False
    item["wav"] = item["wav"][:wav_len].contiguous()
    item["neg_wav"] = item["neg_wav"][:wav_len].contiguous()
    item["src_wav"] = item["src_wav"][:wav_len].contiguous()
    item["wav_len"] = wav_len
    item["neg_wav_len"] = wav_len
    item["src_wav_len"] = wav_len
    return True


def processer_fn_dpo_jsonl(raw_item, _tgt_size, hparams_, _global_stores, skip_logger, i_worker, n_worker):
    sr = int(hparams_["audio_sample_rate"])
    use_precomputed_latents = bool(hparams_.get("use_precomputed_latents", False))
    posterior_suffix = str(hparams_.get("precomputed_latent_suffix", ".spat_vae_posterior.pt"))
    items = []

    for item_ in raw_item:
        try:
            if item_ is None or not isinstance(item_, dict):
                continue

            src_wav_path, pos_wav_path, neg_wav_path = _resolve_dpo_paths(item_)
            caption = _resolve_caption(item_)
            if not src_wav_path or not pos_wav_path or not neg_wav_path or caption is None:
                skip_logger.update(1)
                continue

            sample_id = str(item_.get("sample_id", item_.get("item_name", Path(pos_wav_path).parent.name)))
            item = {
                "caption": caption,
                "global": "",
                "local": caption,
                "item_name": sample_id,
                "sample_id": sample_id,
                "src_wav_path": str(src_wav_path),
                "wav_path": str(pos_wav_path),
                "target_wav_path": str(pos_wav_path),
                "neg_wav_path": str(neg_wav_path),
                "edit_type": item_.get("edit_type", ""),
                "dpo_error_mode": item_.get("error_mode", item_.get("negative_type", "")),
                "positive_caption": item_.get("chosen", caption),
                "negative_caption": item_.get("rejected", item_.get("negative_caption", "")),
            }

            if use_precomputed_latents:
                pos_latent_path = _resolve_latent_path(
                    item_,
                    ("pos_latent_path", "positive_latent_path", "chosen_latent_path", "latent_path", "vae_posterior_path"),
                    pos_wav_path,
                    posterior_suffix,
                )
                neg_latent_path = _resolve_latent_path(
                    item_,
                    ("neg_latent_path", "negative_latent_path", "rejected_latent_path", "bad_latent_path"),
                    neg_wav_path,
                    posterior_suffix,
                )
                src_latent_path = _resolve_latent_path(
                    item_,
                    ("src_latent_path", "source_latent_path", "src_vae_posterior_path"),
                    src_wav_path,
                    posterior_suffix,
                )
                pos_posterior = _load_vae_posterior(pos_latent_path)
                neg_posterior = _load_vae_posterior(neg_latent_path)
                src_posterior = _load_vae_posterior(src_latent_path)
                if pos_posterior is None or neg_posterior is None or src_posterior is None:
                    skip_logger.update(1)
                    continue
                item.update(
                    {
                        "latent_mean": pos_posterior["mean"],
                        "latent_var": pos_posterior["var"],
                        "latent_len": int(pos_posterior["latent_len"]),
                        "wav_len": int(pos_posterior["wav_len"]),
                        "latent_path": str(pos_latent_path),
                        "neg_latent_mean": neg_posterior["mean"],
                        "neg_latent_var": neg_posterior["var"],
                        "neg_latent_len": int(neg_posterior["latent_len"]),
                        "neg_wav_len": int(neg_posterior["wav_len"]),
                        "neg_latent_path": str(neg_latent_path),
                        "src_latent_mean": src_posterior["mean"],
                        "src_latent_var": src_posterior["var"],
                        "src_latent_len": int(src_posterior["latent_len"]),
                        "src_wav_len": int(src_posterior["wav_len"]),
                        "src_latent_path": str(src_latent_path),
                        "audio_channels": int(item_.get("audio_channels", 4)),
                    }
                )
            else:
                pos_wav = _load_spatial_audio_preserve_channels(pos_wav_path, sr)
                neg_wav = _load_spatial_audio_preserve_channels(neg_wav_path, sr)
                src_wav = _load_spatial_audio_preserve_channels(src_wav_path, sr)
                if pos_wav is None or neg_wav is None or src_wav is None:
                    skip_logger.update(1)
                    continue
                item.update(
                    {
                        "wav": pos_wav,
                        "neg_wav": neg_wav,
                        "src_wav": src_wav,
                        "wav_len": int(pos_wav.shape[0]),
                        "neg_wav_len": int(neg_wav.shape[0]),
                        "src_wav_len": int(src_wav.shape[0]),
                        "audio_channels": int(pos_wav.shape[1]) if pos_wav.ndim == 2 else 1,
                    }
                )

            items.append(item)
        except Exception:
            skip_logger.update(1)
            continue

    return items


class SpatAudioEditDpoShmDataset(SpatBaseShmDataset):
    def _process_item(self, processer_fn, raw_item, tgt_size, hparams_, global_stores, i_worker, n_worker):
        skip_logger: SkipLogger = get_from_global_stores(
            "skip_logger",
            global_stores,
            lambda: SkipLogger(
                ["bad_dpo_item_cnt", "bad_caption_cnt", "frames_out_of_range_cnt"],
                interval=1000,
                i_worker=i_worker,
                n_worker=n_worker,
            ),
        )

        items = processer_fn(raw_item, tgt_size, hparams_, global_stores, skip_logger, i_worker, n_worker)
        if not items:
            return

        hop_size = int(hparams_["hop_size"])
        for item_tgt in items:
            if item_tgt is None:
                skip_logger.update(1)
                continue
            caption = _normalize_caption_text(item_tgt.get("caption"))
            if caption is None:
                skip_logger.update(1)
                continue
            item_tgt["caption"] = caption

            if not _trim_dpo_item_to_common_length(item_tgt, hparams_):
                skip_logger.update(1)
                continue

            mel_len_total = int(item_tgt["wav_len"]) // hop_size
            if not (hparams_["max_frames"] >= mel_len_total > hparams_["min_frames"]):
                skip_logger.update(1)
                continue

            item_tgt["len"] = mel_len_total // int(hparams_.get("vae_stride", 4))
            yield item_tgt
            skip_logger.step(1)

    def collater(self, samples):
        batch = super().collater(samples)
        if not batch:
            return batch
        if len(samples) == 1 and isinstance(samples[0], list):
            samples = samples[0]

        if valid_item_kv(samples[0], "neg_wav"):
            batch["neg_wavs"] = collate_xd([s["neg_wav"] for s in samples], 0.0)
            batch["neg_wav_lengths"] = torch.LongTensor([int(s["neg_wav_len"]) for s in samples])
        if valid_item_kv(samples[0], "neg_latent_mean"):
            batch["neg_latent_means"] = collate_xd([s["neg_latent_mean"] for s in samples], 0.0)
            batch["neg_latent_vars"] = collate_xd([s["neg_latent_var"] for s in samples], 0.0)
            batch["neg_latent_lengths"] = torch.LongTensor(
                [int(s.get("neg_latent_len", s["neg_latent_mean"].shape[0])) for s in samples]
            )
        for key in ("neg_wav_path", "neg_latent_path", "dpo_error_mode", "positive_caption", "negative_caption"):
            if valid_item_kv(samples[0], key):
                batch[key] = [s.get(key, "") for s in samples]
        return batch


class SpatAudioEditDpoTask(SpatAudioEditTask):
    def build_model(self):
        self._build_model()

        if not hparams.get("use_fsdp"):
            cast_result = self.dit.cast_safe_params_to_bf16()
            print_once(
                f"| DiT: Cast {cast_result['bf16_params'] / 1_000_000:.3f} params to bf16, "
                f"remaining {cast_result['fp32_params'] / 1_000_000:.3f} params in fp32"
            )

        self.ref = deepcopy(self.dit)
        self.ref.eval()
        for param in self.ref.parameters():
            param.requires_grad = False
        self.ref.to(self.trainer.device)
        return {"trainable": [self.dit], "others": [self.ref]}

    def load_model(self):
        super().load_model()
        self.ref.load_state_dict(unwrap_model(self.dit).state_dict(), strict=False)
        self.ref.eval()
        for param in self.ref.parameters():
            param.requires_grad = False

    def fsdp_optm2model(self):
        return [self.dit]

    @staticmethod
    def _capture_rng_state(device: torch.device):
        state = {"cpu": torch.random.get_rng_state()}
        if device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(device)
        return state

    @staticmethod
    def _restore_rng_state(state, device: torch.device):
        torch.random.set_rng_state(state["cpu"])
        if device.type == "cuda" and "cuda" in state:
            torch.cuda.set_rng_state(state["cuda"], device)

    def encode_dpo_foa_latents_parallel(
        self,
        pos_wavs: torch.Tensor,
        neg_wavs: torch.Tensor,
        src_wavs: torch.Tensor,
        input_sample_rate: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_a, pos_b = self.prepare_foa_stereo_batches(pos_wavs, input_sample_rate)
        neg_a, neg_b = self.prepare_foa_stereo_batches(neg_wavs, input_sample_rate)
        src_a, src_b = self.prepare_foa_stereo_batches(src_wavs, input_sample_rate)

        chunked = bool(hparams.get("stable_audio_vae_chunked", True))
        chunk_size = int(hparams.get("stable_audio_vae_chunk_size", 128))
        overlap = int(hparams.get("stable_audio_vae_overlap", 32))
        deterministic = bool(hparams.get("vae_deterministic", True))
        pos_lat_a, pos_lat_b, neg_lat_a, neg_lat_b, src_lat_a, src_lat_b = encode_stable_audio_parallel(
            self.vae,
            (pos_a, pos_b, neg_a, neg_b, src_a, src_b),
            chunked=chunked,
            chunk_size=chunk_size,
            overlap=overlap,
            deterministic=deterministic,
        )
        pos_lat = merge_foa_pair_latents(pos_lat_a, pos_lat_b)
        neg_lat = merge_foa_pair_latents(neg_lat_a, neg_lat_b)
        src_lat = merge_foa_pair_latents(src_lat_a, src_lat_b)
        target_len = int(pos_lat.shape[1])
        return pos_lat, pad_latent_seq_time(neg_lat, target_len), pad_latent_seq_time(src_lat, target_len)

    def _prepare_dpo_latents(self, sample, device: torch.device):
        has_precomputed = "latent_means" in sample and "neg_latent_means" in sample
        deterministic = bool(hparams.get("vae_deterministic", True))

        if has_precomputed:
            pos_lat = sample_precomputed_posterior(
                sample["latent_means"].to(device),
                sample["latent_vars"].to(device),
                deterministic,
            ).detach().clone()
            neg_lat = sample_precomputed_posterior(
                sample["neg_latent_means"].to(device),
                sample["neg_latent_vars"].to(device),
                deterministic,
            ).detach().clone()
            src_lat = sample_precomputed_posterior(
                sample["src_latent_means"].to(device),
                sample["src_latent_vars"].to(device),
                deterministic,
            ).detach().clone()
            target_len = int(pos_lat.shape[1])
            neg_lat = pad_latent_seq_time(neg_lat, target_len)
            src_lat = pad_latent_seq_time(src_lat, target_len)
            return pos_lat, neg_lat, src_lat

        pos_wavs = sample["wavs"].to(device).float()
        neg_wavs = sample["neg_wavs"].to(device).float()
        src_wavs = sample["src_wavs"].to(device).float()
        with torch.inference_mode():
            if bool(hparams.get("parallel_foa_vae_encode", True)):
                pos_lat, neg_lat, src_lat = self.encode_dpo_foa_latents_parallel(
                    pos_wavs,
                    neg_wavs,
                    src_wavs,
                    int(hparams["audio_sample_rate"]),
                )
            else:
                pos_lat = self.encode_foa_latent(pos_wavs, int(hparams["audio_sample_rate"]))
                neg_lat = self.encode_foa_latent(neg_wavs, int(hparams["audio_sample_rate"]))
                src_lat = self.encode_foa_latent(src_wavs, int(hparams["audio_sample_rate"]))
        target_len = int(pos_lat.shape[1])
        return (
            pos_lat.detach().clone(),
            pad_latent_seq_time(neg_lat.detach().clone(), target_len),
            pad_latent_seq_time(src_lat.detach().clone(), target_len),
        )

    def _encode_caption(self, sample, batch_size: int, device: torch.device):
        captions = sample.get("caption", None)
        text_embs = None
        caption_text_mark = None
        caption_lens = None
        cap_input_ids = None
        if hparams.get("use_caption", True) and captions is not None:
            with torch.inference_mode(), torch.cuda.amp.autocast(
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                text_embs, caption_text_mark, text_att_mask, cap_input_ids = self.run_goku_text_encoder(captions)
                text_embs = text_embs * text_att_mask[..., None]
                caption_lens = text_att_mask.sum(-1)
            text_embs = text_embs.detach().clone()
            caption_lens = caption_lens.detach().clone()
            if caption_text_mark is not None:
                caption_text_mark = caption_text_mark.detach().clone()
            if cap_input_ids is not None:
                cap_input_ids = cap_input_ids.detach().clone()

            drop_caption_p = float(hparams.get("drop_caption", 0.1))
            if drop_caption_p > 0:
                caption_drop_mask = torch.rand((batch_size,), device=device) < drop_caption_p
                if caption_drop_mask.any():
                    text_embs = text_embs.clone()
                    text_embs[caption_drop_mask] = 0.0
                    if caption_text_mark is not None:
                        caption_text_mark = caption_text_mark.clone()
                        caption_text_mark[caption_drop_mask] = 0
        return text_embs, caption_lens, caption_text_mark, cap_input_ids

    @staticmethod
    def _build_inputs(lat, src_lat, lat_lens, text_embs, caption_lens, caption_text_mark, cap_input_ids):
        return {
            "lat": lat,
            "tgt_lat": lat,
            "src_lat": src_lat,
            "lat_lens": lat_lens,
            "caption_emb": text_embs,
            "caption_lens": caption_lens,
            "caption_text_mark": caption_text_mark,
            "caption_ids": cap_input_ids,
            "bgm_flag": None,
            "quality_flag": None,
        }

    @staticmethod
    def _loss_mask(lat_lens: torch.Tensor, lat: torch.Tensor):
        return sequence_mask(lat_lens, maxlen=int(lat.shape[1]))[:, :, None].to(lat.dtype)

    def _model_diff_losses(self, model, inputs, loss_mask, device: torch.device, rng_state=None):
        if rng_state is not None:
            self._restore_rng_state(rng_state, device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device.type == "cuda"):
            dit_out = model(inputs)

        if isinstance(dit_out, tuple) and len(dit_out) == 3:
            pred, target, moe_aux = dit_out
        elif isinstance(dit_out, tuple) and len(dit_out) == 2:
            pred, target = dit_out
            moe_aux = None
        else:
            raise ValueError("Expected model(inputs) to return (pred, target) or (pred, target, moe_aux)")

        loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        per_sample = (loss * loss_mask).sum(dim=(1, 2))
        per_sample = per_sample / loss_mask.sum(dim=(1, 2)).clamp_min(1.0) / target.shape[-1]
        return per_sample, moe_aux

    def _training_step(self, sample, batch_idx, optimizer_idx):
        if self.trainer.proc_rank_local == 0 and random.random() < 0.0001:
            from utils.commons.os_utils import kill_void

            kill_void()

        device = torch.device(self.trainer.device)
        wav_lengths = sample["wav_lengths"].to(device).long()
        batch_size = int(wav_lengths.shape[0])

        pos_lat, neg_lat, src_lat = self._prepare_dpo_latents(sample, device)
        src_lat = pad_latent_seq_time(src_lat, int(pos_lat.shape[1]))
        neg_lat = pad_latent_seq_time(neg_lat, int(pos_lat.shape[1]))

        vae_wav_lengths = torch.div(
            wav_lengths * int(self.vae_sample_rate),
            int(hparams["audio_sample_rate"]),
            rounding_mode="floor",
        )
        lat_lens = torch.div(
            vae_wav_lengths,
            int(self.vae_downsampling_ratio),
            rounding_mode="floor",
        ).clamp(min=1, max=int(pos_lat.shape[1]))

        text_embs, caption_lens, caption_text_mark, cap_input_ids = self._encode_caption(sample, batch_size, device)

        drop_source_lat_p = float(hparams.get("drop_source_lat", hparams.get("drop_src_lat", 0.1)))
        if drop_source_lat_p > 0:
            source_drop_mask = torch.rand((batch_size,), device=device) < drop_source_lat_p
            if source_drop_mask.any():
                src_lat = src_lat.clone()
                src_lat[source_drop_mask] = 0.0

        pos_inputs = self._build_inputs(
            pos_lat,
            src_lat,
            lat_lens,
            text_embs,
            caption_lens,
            caption_text_mark,
            cap_input_ids,
        )
        neg_inputs = self._build_inputs(
            neg_lat,
            src_lat,
            lat_lens,
            text_embs,
            caption_lens,
            caption_text_mark,
            cap_input_ids,
        )
        pos_loss_mask = self._loss_mask(lat_lens, pos_lat)
        neg_loss_mask = self._loss_mask(lat_lens, neg_lat)

        rng_state = self._capture_rng_state(device) if bool(hparams.get("dpo_share_noise", True)) else None
        loss_pos, moe_pos = self._model_diff_losses(self.dit, pos_inputs, pos_loss_mask, device, rng_state)
        loss_neg, moe_neg = self._model_diff_losses(self.dit, neg_inputs, neg_loss_mask, device, rng_state)
        with torch.no_grad():
            loss_pos_ref, _ = self._model_diff_losses(self.ref, pos_inputs, pos_loss_mask, device, rng_state)
            loss_neg_ref, _ = self._model_diff_losses(self.ref, neg_inputs, neg_loss_mask, device, rng_state)

        beta_target = float(hparams.get("dpo_beta", hparams.get("beta_dpo", 50.0)))
        warmup = int(hparams.get("dpo_beta_warmup_steps", hparams.get("beta_warmup_steps", 2000)))
        beta = beta_target if warmup <= 0 else beta_target * min(1.0, float(self.global_step + 1) / float(warmup))

        logits = beta * ((loss_neg - loss_pos) - (loss_neg_ref - loss_pos_ref))
        dpo_loss = -F.logsigmoid(logits).mean()
        diff_loss = loss_pos.mean()

        moe_aux = None
        moe_terms = [term for term in (moe_pos, moe_neg) if isinstance(term, torch.Tensor)]
        if len(moe_terms) > 0:
            moe_aux = sum(moe_terms) / float(len(moe_terms))

        moe_w = float(hparams.get("moe_aux_loss_weight", 1.0))
        anneal_steps = int(hparams.get("moe_aux_anneal_steps", 200000))
        if anneal_steps > 0 and moe_w > 0:
            ratio = 1.0 - min(self.global_step, anneal_steps) / float(anneal_steps)
            moe_w = moe_w * max(0.2, ratio)

        total_loss = (
            float(hparams.get("dpo_loss_weight", 1.0)) * dpo_loss
            + float(hparams.get("dpo_diff_loss_weight", 1.0)) * diff_loss
        )
        if moe_aux is not None:
            total_loss = total_loss + moe_w * moe_aux

        logs = {
            "diff_loss": diff_loss,
            "dpo_loss": dpo_loss,
            "dpo/beta": torch.tensor(beta, device=device),
            "dpo/acc": (logits.detach() > 0).float().mean(),
            "dpo/logits": logits.detach().mean(),
            "dpo/loss_pos": loss_pos.detach().mean(),
            "dpo/loss_neg": loss_neg.detach().mean(),
            "dpo/loss_pos_ref": loss_pos_ref.detach().mean(),
            "dpo/loss_neg_ref": loss_neg_ref.detach().mean(),
            "dpo/reward_margin": ((loss_neg - loss_pos) - (loss_neg_ref - loss_pos_ref)).detach().mean(),
            "bs": batch_size,
            "ntokens": lat_lens.sum(),
        }
        if moe_aux is not None:
            logs["moe_aux_loss"] = moe_aux

        return total_loss, logs
