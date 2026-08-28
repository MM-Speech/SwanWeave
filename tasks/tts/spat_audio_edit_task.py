import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

from utils.commons.ckpt_utils import load_ckpt, load_ckpt_moe
from utils.commons.base_task import BaseTask
from utils.commons.hparams import hparams
from utils.commons.import_utils import get_class_from_module, import_module_bystr
from utils.commons.io import print_once
from utils.commons.os_utils import kill_void
from utils.commons.tensor_utils import convert_to_np, move_to_cpu
from utils.nn.model_utils import unwrap_model
from utils.nn.seq_utils import sequence_mask

from modules.tts.spat_edit.build_model_utils import DiTBuildModelMixin
from tasks.tts.dataset_utils.dataset_mixin import FastDatasetMixin
from tasks.tts.task_utils.prompttts_task_utils import build_audio_mask_from_ids


def build_stereo_from_foa(wavs: torch.Tensor, channel_pair: Tuple[int, int]) -> torch.Tensor:
    if wavs.ndim != 3:
        raise ValueError(f"Expected wavs shape [B, T, C], got {tuple(wavs.shape)}")
    if wavs.shape[-1] < 4:
        raise ValueError(f"Spatial base task expects 4-channel FOA wavs, got {wavs.shape[-1]} channels")

    left_idx, right_idx = channel_pair
    stereo = wavs[:, :, [left_idx, right_idx]].transpose(1, 2).contiguous()
    return stereo


def pad_latent_time(latent: torch.Tensor, target_len: int) -> torch.Tensor:
    target_len = int(target_len)
    if latent.shape[-1] == target_len:
        return latent
    if latent.shape[-1] > target_len:
        return latent[:, :, :target_len]
    return F.pad(latent, (0, target_len - latent.shape[-1]), mode="constant", value=0)


def pad_latent_seq_time(latent: torch.Tensor, target_len: int) -> torch.Tensor:
    target_len = int(target_len)
    if latent.shape[1] == target_len:
        return latent
    if latent.shape[1] > target_len:
        return latent[:, :target_len, :]
    return F.pad(latent, (0, 0, 0, target_len - latent.shape[1]), mode="constant", value=0)


def merge_foa_pair_latents(lat_a: torch.Tensor, lat_b: torch.Tensor) -> torch.Tensor:
    target_len = min(int(lat_a.shape[-1]), int(lat_b.shape[-1]))
    lat_a = pad_latent_time(lat_a, target_len)
    lat_b = pad_latent_time(lat_b, target_len)
    lat = torch.cat([lat_a, lat_b], dim=1)
    return lat.transpose(1, 2).contiguous()


def encode_stable_audio_safe(
    vae,
    audio: torch.Tensor,
    *,
    chunked: bool,
    chunk_size: int,
    overlap: int,
    deterministic: bool,
) -> torch.Tensor:
    if not hasattr(vae, "encode_audio"):
        encoded = vae.encode(audio)
        latent_dist = getattr(encoded, "latent_dist", None)
        if latent_dist is None:
            return getattr(encoded, "latents", encoded[0] if isinstance(encoded, tuple) else encoded)
        if deterministic and hasattr(latent_dist, "mode"):
            return latent_dist.mode()
        return latent_dist.sample()

    if not chunked:
        return vae.encode_audio(audio, chunked=False)

    samples_per_latent = int(getattr(vae, "downsampling_ratio"))
    chunk_samples = int(chunk_size) * samples_per_latent
    if int(audio.shape[-1]) <= chunk_samples:
        return vae.encode_audio(audio, chunked=False)

    return vae.encode_audio(
        audio,
        chunked=True,
        chunk_size=int(chunk_size),
        overlap=int(overlap),
    )


def _latent_len_from_audio_len(latent_len: int, audio_len: int, max_audio_len: int) -> int:
    if int(audio_len) == int(max_audio_len):
        return int(latent_len)
    return max(1, int(round(float(latent_len) * float(audio_len) / float(max_audio_len))))


def _split_encoded_latents(
    latents: torch.Tensor,
    batch_sizes: Tuple[int, ...],
    audio_lengths: Tuple[int, ...],
    max_audio_len: int,
) -> Tuple[torch.Tensor, ...]:
    split_latents = latents.split(batch_sizes, dim=0)
    max_latent_len = int(latents.shape[-1])
    return tuple(
        pad_latent_time(split_latent, _latent_len_from_audio_len(max_latent_len, audio_len, max_audio_len))
        for split_latent, audio_len in zip(split_latents, audio_lengths)
    )


def _sample_split_latent_dist(
    latent_dist,
    batch_sizes: Tuple[int, ...],
    audio_lengths: Tuple[int, ...],
    max_audio_len: int,
    *,
    deterministic: bool,
) -> Tuple[torch.Tensor, ...]:
    parameters = getattr(latent_dist, "parameters", None)
    if parameters is None:
        latents = latent_dist.mode() if deterministic and hasattr(latent_dist, "mode") else latent_dist.sample()
        return _split_encoded_latents(latents, batch_sizes, audio_lengths, max_audio_len)

    split_parameters = parameters.split(batch_sizes, dim=0)
    max_latent_len = int(parameters.shape[-1])
    outputs = []
    for split_params, audio_len in zip(split_parameters, audio_lengths):
        target_len = _latent_len_from_audio_len(max_latent_len, audio_len, max_audio_len)
        split_params = pad_latent_time(split_params, target_len)
        split_dist = latent_dist.__class__(split_params)
        if deterministic and hasattr(split_dist, "mode"):
            outputs.append(split_dist.mode())
        else:
            outputs.append(split_dist.sample())
    return tuple(outputs)


def encode_stable_audio_parallel(
    vae,
    audio_batches: Tuple[torch.Tensor, ...],
    *,
    chunked: bool,
    chunk_size: int,
    overlap: int,
    deterministic: bool,
) -> Tuple[torch.Tensor, ...]:
    if len(audio_batches) <= 1:
        return tuple(
            encode_stable_audio_safe(
                vae,
                audio,
                chunked=chunked,
                chunk_size=chunk_size,
                overlap=overlap,
                deterministic=deterministic,
            )
            for audio in audio_batches
        )

    batch_sizes = tuple(int(audio.shape[0]) for audio in audio_batches)
    audio_lengths = tuple(int(audio.shape[-1]) for audio in audio_batches)
    max_audio_len = max(audio_lengths)
    padded_batches = tuple(
        audio if int(audio.shape[-1]) == max_audio_len else F.pad(audio, (0, max_audio_len - int(audio.shape[-1])))
        for audio in audio_batches
    )
    audio = torch.cat(padded_batches, dim=0)

    if not hasattr(vae, "encode_audio"):
        encoded = vae.encode(audio)
        latent_dist = getattr(encoded, "latent_dist", None)
        if latent_dist is not None:
            return _sample_split_latent_dist(
                latent_dist,
                batch_sizes,
                audio_lengths,
                max_audio_len,
                deterministic=deterministic,
            )
        latents = getattr(encoded, "latents", encoded[0] if isinstance(encoded, tuple) else encoded)
        return _split_encoded_latents(latents, batch_sizes, audio_lengths, max_audio_len)

    latents = encode_stable_audio_safe(
        vae,
        audio,
        chunked=chunked,
        chunk_size=chunk_size,
        overlap=overlap,
        deterministic=deterministic,
    )
    return _split_encoded_latents(latents, batch_sizes, audio_lengths, max_audio_len)


def preprocess_stable_audio_batch(
    vae,
    audio: torch.Tensor,
    input_sample_rate: int,
    target_sample_rate: int,
    downsampling_ratio: int,
) -> torch.Tensor:
    if hasattr(vae, "preprocess_audio_list_for_encoder"):
        audio_list = [audio[i].float() for i in range(int(audio.shape[0]))]
        return vae.preprocess_audio_list_for_encoder(audio_list, int(input_sample_rate))

    audio = audio.float()
    if int(input_sample_rate) != int(target_sample_rate):
        audio = torchaudio.functional.resample(
            audio,
            orig_freq=int(input_sample_rate),
            new_freq=int(target_sample_rate),
        )
    pad = (-int(audio.shape[-1])) % int(downsampling_ratio)
    if pad:
        audio = F.pad(audio, (0, pad))
    return audio


def sample_precomputed_posterior(mean: torch.Tensor, var: torch.Tensor, deterministic: bool) -> torch.Tensor:
    mean = mean.float()
    var = var.float().clamp_min(0)
    if deterministic:
        return mean
    if mean.ndim == 3 and mean.shape[-1] % 2 == 0:
        half = int(mean.shape[-1]) // 2
        mean_a = mean[:, :, :half].transpose(1, 2).contiguous()
        mean_b = mean[:, :, half:].transpose(1, 2).contiguous()
        var_a = var[:, :, :half].transpose(1, 2).contiguous()
        var_b = var[:, :, half:].transpose(1, 2).contiguous()
        lat_a = mean_a + torch.sqrt(var_a) * torch.randn_like(mean_a)
        lat_b = mean_b + torch.sqrt(var_b) * torch.randn_like(mean_b)
        return torch.cat([lat_a, lat_b], dim=1).transpose(1, 2).contiguous()
    return mean + torch.sqrt(var) * torch.randn_like(mean)


class SpatAudioEditBaseTask(FastDatasetMixin, BaseTask):
    def __init__(self):
        self.dataset_cls = import_module_bystr(hparams["dataset_cls"])
        self.train_dataloader = FastDatasetMixin.train_dataloader.__get__(self)
        self.test_dataloader = FastDatasetMixin.test_dataloader.__get__(self)
        self.val_dataloader = FastDatasetMixin.val_dataloader.__get__(self)
        self.hparams = hparams
        self.config = AttrDict(hparams)
        super().__init__()

    def fsdp_wrap_policy(self):
        import modules.commons.hf.transformer
        import modules.commons.hf.transformer_dit
        import modules.commons.hf.transformer_dit_moe
        import modules.commons.hf.transformer_moe
        import modules.tts.swanaudio.swan_dit_ecmoe
        import modules.tts.swanaudio.swan_dit_moe
        import modules.tts.spat_edit.spat_dit_moe

        qwen_layer = get_class_from_module(
            "transformers.models.qwen3.modeling_qwen3",
            "Qwen3DecoderLayer",
        )

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            model_blocks = (
                modules.commons.hf.transformer.EncoderLayer,
                modules.commons.hf.transformer.DecoderLayer,
                modules.commons.hf.transformer_moe.EncoderLayer,
                modules.commons.hf.transformer_moe.DecoderLayer,
                modules.commons.hf.transformer_dit.EncoderLayer,
                modules.commons.hf.transformer_dit_moe.EncoderLayer,
                modules.tts.spat_edit.spat_dit_moe.EncoderLayer,
                modules.tts.swanaudio.swan_dit_moe.EncoderLayer,
                modules.tts.swanaudio.swan_dit_ecmoe.EncoderLayer,
                qwen_layer,
            )
            return recurse or isinstance(module, model_blocks)

        return custom_auto_wrap_policy

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        super().on_after_optimization(epoch, batch_idx, optimizer, optimizer_idx)
        if hparams.get("use_moe_ffn", False) and hparams.get("moe_use_bias_balance", False):
            model = unwrap_model(self.dit)
            if hasattr(model, "update_moe_bias_balance"):
                model.update_moe_bias_balance()
        if hparams.get("use_ema", False):
            self.ema_update(self.ema_model, self.dit, self.config.ema_decay)

    @torch.no_grad()
    def ema_update(self, ema_model, model, decay):
        ema_params = dict(unwrap_model(ema_model).named_parameters())
        for name, param in unwrap_model(model).named_parameters():
            param_ema = ema_params[name]
            src = param.detach()
            if src.dtype != torch.float32:
                src = src.float()
            if param_ema.dtype != torch.float32:
                param_ema.data = param_ema.data.float()
            param_ema.mul_(decay).add_(src.to(param_ema.device), alpha=1.0 - decay)

    def load_model(self):
        ckpt_path = hparams.get("load_ckpt", "")
        ckpt_audio = hparams.get("load_ckpt_audio", "")
        if ckpt_path == "":
            return

        if hparams.get("use_moe_ffn", False):
            load_ckpt_moe(
                self.dit,
                ckpt_base_dir=ckpt_path,
                ckpt_audio_base_dir=ckpt_audio,
                model_name="dit",
                strict=False,
                mmap=True,
            )
            if hparams.get("use_ema", False):
                self.ema_model.load_state_dict(self.dit.state_dict(), strict=False)
        else:
            load_ckpt(self.dit, ckpt_path, "dit", strict=False, mmap=True)
            if hparams.get("use_ema", False):
                load_ckpt(self.ema_model, ckpt_path, "dit", strict=False, mmap=True)

        if (
            not hparams.get("train_base", True)
            and hparams.get("init_edit_lat_proj_from_t2a", True)
        ):
            initialized = unwrap_model(self.dit).init_edit_lat_proj_from_lat_proj()
            if initialized:
                print_once("| initialized edit_lat_proj from lat_proj for t2a -> edit finetuning")
            if hparams.get("use_ema", False):
                unwrap_model(self.ema_model).init_edit_lat_proj_from_lat_proj()

    def build_optimizer(self):
        model = unwrap_model(self.dit)
        disable_wd = hparams.get("disable_weight_decay_on_bias_and_norm_and_embed", False)

        if hparams.get("optimizer_type", "adamw") == "muon":
            from modules.optimizers.muon_aux import build_muon_aux_adam

            extra_kw = hparams.get("muon_extra_adam_keywords", None)
            optimizer, n_muon, n_decay, n_no_decay = build_muon_aux_adam(
                model,
                muon_cfg=hparams.get("muon", None),
                adamw_cfg={**dict(self.config.optimizer), **hparams.get("muon_aux_adam", {})},
                extra_adam_keywords=tuple(extra_kw) if extra_kw is not None else None,
                disable_wd_on_bias_norm_embed=disable_wd,
            )
            print_once(f"| Muon: {n_muon} muon, {n_decay} adamw(decay), {n_no_decay} adamw(no-decay)")
            return optimizer

        from modules.optimizers.adamw_aux import build_adamw

        optimizer, n_decay, n_no_decay = build_adamw(
            model,
            optimizer_cfg=dict(self.config.optimizer),
            disable_wd_on_bias_norm_embed=disable_wd,
        )
        print_once(f"| AdamW: {n_decay} decay, {n_no_decay} no-decay")
        return optimizer

    def fsdp_optm2model(self):
        return [self.dit]


class SpatAudioEditTask(DiTBuildModelMixin, SpatAudioEditBaseTask):
    def prepare_foa_stereo_batches(self, wavs: torch.Tensor, input_sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_a = tuple(hparams.get("stable_audio_foa_pair_a", [0, 1]))
        pair_b = tuple(hparams.get("stable_audio_foa_pair_b", [2, 3]))
        audio_a = build_stereo_from_foa(wavs, pair_a)
        audio_b = build_stereo_from_foa(wavs, pair_b)
        audio_a = preprocess_stable_audio_batch(
            self.vae,
            audio_a,
            input_sample_rate,
            self.vae_sample_rate,
            self.vae_downsampling_ratio,
        )
        audio_b = preprocess_stable_audio_batch(
            self.vae,
            audio_b,
            input_sample_rate,
            self.vae_sample_rate,
            self.vae_downsampling_ratio,
        )
        return audio_a, audio_b

    def encode_preprocessed_foa_stereo_batches(
        self,
        audio_a: torch.Tensor,
        audio_b: torch.Tensor,
        *,
        parallel: bool,
    ) -> torch.Tensor:
        chunked = bool(hparams.get("stable_audio_vae_chunked", True))
        chunk_size = int(hparams.get("stable_audio_vae_chunk_size", 128))
        overlap = int(hparams.get("stable_audio_vae_overlap", 32))
        deterministic = bool(hparams.get("vae_deterministic", True))
        if parallel:
            lat_a, lat_b = encode_stable_audio_parallel(
                self.vae,
                (audio_a, audio_b),
                chunked=chunked,
                chunk_size=chunk_size,
                overlap=overlap,
                deterministic=deterministic,
            )
        else:
            lat_a = encode_stable_audio_safe(
                self.vae,
                audio_a,
                chunked=chunked,
                chunk_size=chunk_size,
                overlap=overlap,
                deterministic=deterministic,
            )
            lat_b = encode_stable_audio_safe(
                self.vae,
                audio_b,
                chunked=chunked,
                chunk_size=chunk_size,
                overlap=overlap,
                deterministic=deterministic,
            )
        return merge_foa_pair_latents(lat_a, lat_b)

    @torch.no_grad()
    def encode_foa_latent(self, wavs: torch.Tensor, input_sample_rate: int) -> torch.Tensor:
        audio_a, audio_b = self.prepare_foa_stereo_batches(wavs, input_sample_rate)
        parallel = bool(hparams.get("parallel_foa_vae_encode", False))
        return self.encode_preprocessed_foa_stereo_batches(audio_a, audio_b, parallel=parallel)

    @torch.no_grad()
    def encode_edit_foa_latents_parallel(
        self,
        wavs: torch.Tensor,
        src_wavs: torch.Tensor,
        input_sample_rate: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt_a, tgt_b = self.prepare_foa_stereo_batches(wavs, input_sample_rate)
        src_a, src_b = self.prepare_foa_stereo_batches(src_wavs, input_sample_rate)

        chunked = bool(hparams.get("stable_audio_vae_chunked", True))
        chunk_size = int(hparams.get("stable_audio_vae_chunk_size", 128))
        overlap = int(hparams.get("stable_audio_vae_overlap", 32))
        deterministic = bool(hparams.get("vae_deterministic", True))
        tgt_lat_a, tgt_lat_b, src_lat_a, src_lat_b = encode_stable_audio_parallel(
            self.vae,
            (tgt_a, tgt_b, src_a, src_b),
            chunked=chunked,
            chunk_size=chunk_size,
            overlap=overlap,
            deterministic=deterministic,
        )
        lat = merge_foa_pair_latents(tgt_lat_a, tgt_lat_b)
        src_lat = merge_foa_pair_latents(src_lat_a, src_lat_b)
        src_lat = pad_latent_seq_time(src_lat, int(lat.shape[1]))
        return lat, src_lat

    def run_goku_text_encoder(self, captions: list):
        self.goku_tokenizer.padding_side = "left"
        inputs = self.goku_tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=hparams.get("text_max_token_length", 256),
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.trainer.device)
        attention_mask = inputs.attention_mask.to(self.trainer.device)
        encoder_hidden_states = self.goku_text_encoder(
            input_ids,
            return_dict=False,
            attention_mask=attention_mask,
        )[0]

        if hparams.get("use_caption_text_mark", False):
            caption_text_mark = build_audio_mask_from_ids(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=self.goku_tokenizer,
            )
        else:
            caption_text_mark = None
        return encoder_hidden_states, caption_text_mark, attention_mask, input_ids

    def _training_step(self, sample, batch_idx, optimizer_idx):
        if self.trainer.proc_rank_local == 0 and random.random() < 0.0001:
            kill_void()
        loss_output, model_out = self.run_model(sample)

        moe_w = hparams.get("moe_aux_loss_weight", 1.0)
        anneal_steps = hparams.get("moe_aux_anneal_steps", 200000)
        if anneal_steps > 0 and moe_w > 0:
            ratio = 1.0 - min(self.global_step, anneal_steps) / float(anneal_steps)
            moe_w = moe_w * max(0.2, ratio)

        loss_weights = {"diff_loss": 1.0, "moe_aux_loss": moe_w}
        total_loss = sum(
            loss_weights.get(k, 1.0) * v
            for k, v in loss_output.items()
            if isinstance(v, torch.Tensor) and v.requires_grad
        )

        if (
            self.trainer.proc_rank == 0
            and self.global_step < hparams.get("debug_save_batches", 0)
            and "wavs" in sample
        ):
            save_dir = f'{hparams["work_dir"]}/sample_batches/step_{self.global_step}'
            os.makedirs(save_dir, exist_ok=True)
            sample_cpu = convert_to_np(move_to_cpu(sample))
            for i in range(sample_cpu["nsamples"]):
                wav = sample_cpu["wavs"][i, : sample_cpu["wav_lengths"][i]]
                sf.write(f"{save_dir}/{i}.wav", wav, hparams["audio_sample_rate"], "PCM_16")
            np.save(f"{save_dir}/batch.npy", sample_cpu, allow_pickle=True)

        return total_loss, loss_output

    def run_model(self, sample, infer=False, infer_steps=None):
        model_out: Dict[str, Any] = {}
        losses_out: Dict[str, Any] = {}
        has_precomputed_latents = "latent_means" in sample and "latent_vars" in sample
        if infer or ("wavs" not in sample and not has_precomputed_latents):
            return losses_out, model_out

        device = torch.device(self.trainer.device)
        wav_lengths = sample["wav_lengths"].to(device).long()
        batch_size = int(wav_lengths.shape[0])

        train_base = hparams.get("train_base", True)
        src_lat = None
        if has_precomputed_latents:
            deterministic = bool(hparams.get("vae_deterministic", True))
            lat = sample_precomputed_posterior(
                sample["latent_means"].to(device),
                sample["latent_vars"].to(device),
                deterministic,
            ).detach().clone()
            if not train_base:
                if "src_latent_means" not in sample or "src_latent_vars" not in sample:
                    raise KeyError(
                        "train_base=False with precomputed latents requires sample['src_latent_means'] and sample['src_latent_vars']"
                    )
                src_lat = sample_precomputed_posterior(
                    sample["src_latent_means"].to(device),
                    sample["src_latent_vars"].to(device),
                    deterministic,
                )
                src_lat = pad_latent_seq_time(src_lat.detach().clone(), int(lat.shape[1]))
        else:
            wavs = sample["wavs"].to(device).float()
            batch_size = int(wavs.shape[0])
            if not train_base:
                src_wavs = sample.get("src_wavs", None)
                if src_wavs is None:
                    raise KeyError(
                        "train_base=False requires source audio in sample['src_wavs']"
                    )
                src_wavs = src_wavs.to(device).float()
                if bool(hparams.get("parallel_foa_vae_encode", False)):
                    with torch.inference_mode():
                        lat, src_lat = self.encode_edit_foa_latents_parallel(
                            wavs,
                            src_wavs,
                            int(hparams["audio_sample_rate"]),
                        )
                    lat = lat.detach().clone()
                    src_lat = src_lat.detach().clone()
                else:
                    with torch.inference_mode():
                        lat = self.encode_foa_latent(wavs, int(hparams["audio_sample_rate"]))
                    lat = lat.detach().clone()
                    with torch.inference_mode():
                        src_lat = self.encode_foa_latent(src_wavs, int(hparams["audio_sample_rate"]))
                    src_lat = pad_latent_seq_time(src_lat.detach().clone(), int(lat.shape[1]))
            else:
                with torch.inference_mode():
                    lat = self.encode_foa_latent(wavs, int(hparams["audio_sample_rate"]))
                lat = lat.detach().clone()

        if not train_base and src_lat is None:
            raise KeyError("train_base=False requires source audio or source precomputed latents")

        vae_wav_lengths = torch.div(
            wav_lengths * int(self.vae_sample_rate),
            int(hparams["audio_sample_rate"]),
            rounding_mode="floor",
        )
        lat_lens = torch.div(
            vae_wav_lengths,
            int(self.vae_downsampling_ratio),
            rounding_mode="floor",
        ).clamp(min=1, max=int(lat.shape[1]))

        loss_mask = sequence_mask(lat_lens, maxlen=int(lat.shape[1]))[:, :, None].to(lat.dtype)

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
            text_att_mask = text_att_mask.detach().clone()
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

        if (not train_base) and src_lat is not None:
            drop_source_lat_p = float(hparams.get("drop_source_lat", hparams.get("drop_src_lat", 0.1)))
            if drop_source_lat_p > 0:
                source_drop_mask = torch.rand((batch_size,), device=device) < drop_source_lat_p
                if source_drop_mask.any():
                    src_lat = src_lat.clone()
                    src_lat[source_drop_mask] = 0.0

        inputs = {
            "lat": lat,
            "lat_lens": lat_lens,
            "caption_emb": text_embs,
            "caption_lens": caption_lens,
            "caption_text_mark": caption_text_mark,
            "caption_ids": cap_input_ids,
            "bgm_flag": None,
            "quality_flag": None,
        }
        if not train_base:
            inputs["tgt_lat"] = lat
            inputs["src_lat"] = src_lat

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device.type == "cuda"):
            dit_out = self.dit(inputs)

        if isinstance(dit_out, tuple) and len(dit_out) == 3:
            model_out, target, moe_aux = dit_out
        elif isinstance(dit_out, tuple) and len(dit_out) == 2:
            model_out, target = dit_out
            moe_aux = None
        else:
            raise ValueError("Expected self.dit(inputs) to return (model_out, target) or (model_out, target, moe_aux)")

        loss = F.mse_loss(model_out.float(), target.float(), reduction="none")
        loss = (loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0) / target.shape[-1]
        losses_out["diff_loss"] = loss
        if moe_aux is not None:
            losses_out["moe_aux_loss"] = moe_aux
        losses_out["bs"] = batch_size
        losses_out["ntokens"] = lat_lens.sum()
        return losses_out, model_out
