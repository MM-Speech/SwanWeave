"""Diffusion preference training with a fixed reference editor."""
from copy import deepcopy
from pathlib import Path

import yaml

import torch
import torch.nn.functional as F

from tasks.spat_audio_edit_task import (
    SpatAudioEditTask,
    encode_stable_audio_parallel,
    merge_foa_pair_latents,
    sample_precomputed_posterior,
)
from utils.commons.base_task import BaseTask
from utils.commons.ckpt_utils import get_all_ckpts, load_ckpt
from utils.commons.hparams import hparams
from utils.nn.model_utils import unwrap_model
from utils.nn.seq_utils import sequence_mask


def _checkpoint_file(location, steps=None):
    """Resolve exactly the checkpoint that the trainer would load."""
    path = Path(location).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    model_only = path / "model_only_last.ckpt"
    if model_only.is_file():
        return model_only
    candidates = get_all_ckpts(str(path), steps)
    if candidates:
        return Path(candidates[0])
    raise FileNotFoundError(f"No model checkpoint found in {path}")


def resolve_dpo_checkpoint_config(config):
    """Use checkpoint architecture, without importing its data/optimizer recipe.

    An existing DPO experiment restores its saved policy AND fixed reference.
    A fresh experiment initializes both from load_ckpt. Explicit dense/moe
    selection validates the checkpoint; it does not convert architectures.
    """
    resume = None
    for directory in (config.get("work_dir"), config.get("resume_from")):
        if directory and Path(directory).is_dir():
            candidates = get_all_ckpts(str(directory), config.get("resume_from_checkpoint", 0))
            if candidates:
                resume = Path(candidates[0]).resolve()
                break
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", mmap=True, weights_only=True)
        state = saved.get("state_dict", saved)
        if "dit" not in state or "ref" not in state:
            raise ValueError("A DPO resume checkpoint must contain dit and ref; "
                             "initialize SFT weights via load_ckpt in a NEW exp_name")
        del state, saved
        checkpoint = resume
    else:
        if not config.get("load_ckpt"):
            raise ValueError("DPO requires load_ckpt from a trained spatial editor")
        checkpoint = _checkpoint_file(config["load_ckpt"])
        config["load_ckpt"] = str(checkpoint)
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Expected config.yaml next to checkpoint: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        model_config = yaml.safe_load(stream)
    if not isinstance(model_config, dict):
        raise ValueError(f"Invalid checkpoint configuration: {config_path}")
    is_moe = model_config.get("use_moe_ffn", False)
    if not isinstance(is_moe, bool):
        raise ValueError(f"use_moe_ffn must be a boolean in {config_path}")
    model_type = "moe" if is_moe else "dense"
    requested = str(config.get("dpo_model_type", "auto")).lower()
    if requested not in ("auto", "dense", "moe"):
        raise ValueError("dpo_model_type must be auto, dense or moe")
    if requested != "auto" and requested != model_type:
        raise ValueError(f"Requested {requested}, but {config_path} specifies {model_type}; "
                         "choose a matching checkpoint")
    if model_config.get("train_base", False):
        raise ValueError("DPO requires an editing checkpoint with train_base=false")
    model_keys = {
        "model_size", "latent_dim", "caption_encoder_dim", "encoder_ffn_mult",
        "use_caption", "use_caption_text_mark", "use_caption_pool_in_adaln",
        "use_dynamic_cross_gate", "use_gated_attention", "use_spk_mask",
        "use_bgm_flag", "use_quality_flag", "use_cosyvoice2_text_tokenizer",
        "use_engram", "decoder_sparse_step", "mlp_only_layers", "output_router_logits",
        "audio_sample_rate", "text_max_token_length", "latent_norm_mode",
        "stable_audio_foa_pair_a", "stable_audio_foa_pair_b",
    }
    # These control DPO optimization, not the architecture of the loaded editor.
    dpo_settings = {
        "moe_aux_loss_weight", "moe_aux_anneal_steps", "moe_expert_dropout",
        "moe_use_gumbel", "moe_bias_update_rate", "moe_load_balance_loss_coef",
        "moe_router_z_loss_coef", "moe_null_loss_coef", "moe_task_balance_loss_coef",
    }
    for key, value in model_config.items():
        is_model_key = key in model_keys or key.startswith(("moe_", "ec_", "engram_"))
        if is_model_key and key not in dpo_settings:
            config[key] = value
    config["train_base"] = False
    config["use_moe_ffn"] = is_moe
    config["dpo_resume_checkpoint"] = str(resume) if resume is not None else ""
    config["dpo_checkpoint_config"] = str(config_path)
    config["preserve_module_dtypes"] = sorted(set(config.get("preserve_module_dtypes", [])) | {"ref"})
    print(f"| DPO model: {model_type}; architecture from {config_path}")


class SpatAudioEditDpoTask(SpatAudioEditTask):
    def __init__(self):
        resolve_dpo_checkpoint_config(hparams)
        if "ref" in hparams.get("not_save_modules", []):
            raise ValueError("DPO checkpoints must save ref to preserve the fixed reference")
        if hparams.get("train_base", False):
            raise ValueError("DPO requires an editing checkpoint with train_base=false")
        hparams["use_ema"] = False
        hparams["init_edit_lat_proj_from_t2a"] = False
        # Preference scores must not differ just because the reference is eval.
        # Preserve learned routing biases, but keep routing randomness disabled.
        hparams["moe_expert_dropout"] = 0.0
        hparams["moe_use_gumbel"] = False
        self._dpo_models_built = False
        super().__init__()

    def build_model(self):
        self._build_model()
        if not hparams.get("use_fsdp", False):
            self.dit.cast_safe_params_to_bf16()
        if not hparams.get("dpo_resume_checkpoint"):
            initial_checkpoint = hparams.get("load_ckpt", "")
            if not initial_checkpoint:
                raise ValueError("Fresh DPO training requires load_ckpt from a trained editor")
            load_ckpt(self.dit, initial_checkpoint, "dit", strict=True, mmap=True)
        # On resume Trainer restores both dit and ref from the DPO checkpoint.
        # No original SFT checkpoint is needed in that case.
        self.ref = deepcopy(self.dit)
        self.ref.to(self.trainer.device)
        self._dpo_models_built = True
        self._freeze_reference_and_routing()
        return {"trainable": [self.dit], "others": [self.ref]}

    def load_model(self):
        # build_model already initializes a fresh run. Trainer handles resume.
        if not self._dpo_models_built:
            raise RuntimeError("Build DPO models before loading their weights")
        self._freeze_reference_and_routing()

    def train(self, mode=True):
        super().train(mode)
        self._freeze_reference_and_routing()
        return self

    def _freeze_reference_and_routing(self):
        reference = getattr(self, "ref", None)
        if reference is not None:
            reference.eval().requires_grad_(False)
        policy = getattr(self, "dit", None)
        if policy is not None:
            # Keep learned routing_bias active, without changing its running load
            # statistics or enabling training-only stochastic expert selection.
            for module in unwrap_model(policy).modules():
                if hasattr(module, "routing_bias"):
                    module.eval()

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        # Advance the scheduler; the CFM hook also changes MoE routing biases,
        # which would move preference scores relative to the fixed reference.
        BaseTask.on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx)

    @staticmethod
    def _capture_rng_state(device):
        state = {"cpu": torch.random.get_rng_state()}
        if device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(device)
        return state

    @torch.no_grad()
    def _prepare_dpo_latents(self, sample, device):
        deterministic = bool(hparams.get("vae_deterministic", False))
        posterior_keys = (
            ("latent_means", "latent_vars"),
            ("neg_latent_means", "neg_latent_vars"),
            ("src_latent_means", "src_latent_vars"),
        )
        if "latent_means" in sample:
            missing = [key for pair in posterior_keys for key in pair if key not in sample]
            if missing:
                raise KeyError("DPO posterior batch is missing: " + ", ".join(missing))
            latents = tuple(
                sample_precomputed_posterior(
                    sample[mean].to(device), sample[var].to(device), deterministic,
                )
                for mean, var in posterior_keys
            )
        else:
            waveform_keys = ("wavs", "neg_wavs", "src_wavs")
            missing = [key for key in waveform_keys if key not in sample]
            if missing:
                raise KeyError("DPO audio batch is missing: " + ", ".join(missing))
            waveforms = tuple(sample[key].to(device).float() for key in waveform_keys)
            with torch.inference_mode():
                if hparams.get("parallel_foa_vae_encode", True):
                    stereo_pairs = tuple(
                        pair
                        for wav in waveforms
                        for pair in self.prepare_foa_stereo_batches(wav, int(hparams["audio_sample_rate"]))
                    )
                    encoded = encode_stable_audio_parallel(
                        self.vae, stereo_pairs,
                        chunked=bool(hparams.get("stable_audio_vae_chunked", True)),
                        chunk_size=int(hparams.get("stable_audio_vae_chunk_size", 128)),
                        overlap=int(hparams.get("stable_audio_vae_overlap", 32)),
                        deterministic=deterministic,
                    )
                    latents = tuple(
                        merge_foa_pair_latents(encoded[index], encoded[index + 1])
                        for index in range(0, len(encoded), 2)
                    )
                else:
                    latents = tuple(
                        self.encode_foa_latent(wav, int(hparams["audio_sample_rate"]))
                        for wav in waveforms
                    )
        target_length = min(int(latent.shape[1]) for latent in latents)
        if target_length < 1:
            raise ValueError("DPO inputs must contain at least one latent frame")
        # Clone outside inference_mode so DiT can save these tensors for backward.
        return tuple(latent[:, :target_length].detach().clone() for latent in latents)

    def _latent_lengths(self, sample, latent, device):
        lengths = []
        for key in ("wav_lengths", "neg_wav_lengths", "src_wav_lengths"):
            if key in sample:
                vae_samples = torch.div(
                    sample[key].to(device).long() * int(self.vae_sample_rate),
                    int(hparams["audio_sample_rate"]), rounding_mode="floor",
                )
                lengths.append(torch.div(vae_samples, int(self.vae_downsampling_ratio), rounding_mode="floor"))
        for key in ("latent_lengths", "neg_latent_lengths", "src_latent_lengths"):
            if key in sample:
                lengths.append(sample[key].to(device).long())
        if not lengths:
            raise KeyError("DPO batch needs wav_lengths or latent_lengths")
        return torch.stack(lengths).amin(dim=0).clamp(min=1, max=int(latent.shape[1]))

    def _caption_inputs(self, sample, device, apply_dropout):
        if not hparams.get("use_caption", True):
            return {"caption_emb": None, "caption_lens": None}
        if "caption" not in sample:
            raise KeyError("DPO batch requires caption instructions")
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
        ):
            embeddings, _, attention, input_ids = self.run_goku_text_encoder(sample["caption"])
            embeddings = embeddings * attention[..., None]
            lengths = attention.sum(-1)
        embeddings = embeddings.detach().clone()
        lengths = lengths.detach().clone()
        if apply_dropout:
            probability = float(hparams.get("drop_caption", 0.0))
            mask = torch.rand(embeddings.shape[0], device=device) < probability
            embeddings[mask] = 0.0
        return {
            "caption_emb": embeddings,
            "caption_lens": lengths,
            "caption_ids": input_ids.detach().clone(),
        }

    @staticmethod
    def _per_sample_loss(output, loss_mask):
        prediction, target = output[:2]
        error = F.mse_loss(prediction.float(), target.float(), reduction="none")
        losses = (error * loss_mask).sum(dim=(1, 2))
        losses = losses / loss_mask.sum(dim=(1, 2)).clamp_min(1.0) / target.shape[-1]
        return losses, output[2] if len(output) == 3 else None

    def _pair_losses(self, model, positive, negative, mask, device, rng_state):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            positive_output, negative_output = model(
                {"positive": positive, "negative": negative, "rng_state": rng_state},
                task="dpo_pair",
            )
        return self._per_sample_loss(positive_output, mask), self._per_sample_loss(negative_output, mask)

    def _total_loss(self, losses):
        total = (
            float(hparams.get("dpo_loss_weight", 1.0)) * losses["dpo_loss"]
            + float(hparams.get("dpo_diff_loss_weight", 1.0)) * losses["diff_loss"]
        )
        if "moe_aux_loss" in losses:
            weight = float(hparams.get("moe_aux_loss_weight", 1.0))
            anneal_steps = int(hparams.get("moe_aux_anneal_steps", 200000))
            if anneal_steps > 0:
                weight *= max(0.2, 1.0 - min(self.global_step, anneal_steps) / float(anneal_steps))
            total = total + weight * losses["moe_aux_loss"]
        return total

    def run_model(self, sample, infer=False, infer_steps=None, *, condition_dropout=None):
        if infer:
            raise NotImplementedError("Use the spatial inference entry for audio generation")
        device = torch.device(self.trainer.device)
        positive_latent, negative_latent, source_latent = self._prepare_dpo_latents(sample, device)
        lengths = self._latent_lengths(sample, positive_latent, device)
        batch_size = int(positive_latent.shape[0])
        apply_dropout = self.training if condition_dropout is None else bool(condition_dropout)
        caption = self._caption_inputs(sample, device, apply_dropout)
        if apply_dropout:
            probability = float(hparams.get("drop_source_lat", hparams.get("drop_src_lat", 0.0)))
            mask = torch.rand(batch_size, device=device) < probability
            source_latent[mask] = 0.0
        common = {"src_lat": source_latent, "lat_lens": lengths, **caption}
        positive = {**common, "lat": positive_latent, "tgt_lat": positive_latent}
        negative = {**common, "lat": negative_latent, "tgt_lat": negative_latent}
        mask = sequence_mask(lengths, maxlen=int(positive_latent.shape[1]))[:, :, None].float()
        rng_state = self._capture_rng_state(device) if hparams.get("dpo_share_noise", True) else None
        (loss_positive, aux_positive), (loss_negative, aux_negative) = self._pair_losses(
            self.dit, positive, negative, mask, device, rng_state,
        )
        with torch.no_grad():
            (reference_positive, _), (reference_negative, _) = self._pair_losses(
                self.ref, positive, negative, mask, device, rng_state,
            )
        beta_target = float(hparams.get("dpo_beta", 50.0))
        warmup = int(hparams.get("dpo_beta_warmup_steps", 2000))
        beta = beta_target if warmup <= 0 else beta_target * min(1.0, (self.global_step + 1) / float(warmup))
        margin = (loss_negative - loss_positive) - (reference_negative - reference_positive)
        logits = beta * margin
        logs = {
            "diff_loss": loss_positive.mean(),
            "dpo_loss": -F.logsigmoid(logits).mean(),
            "dpo/beta": logits.new_tensor(beta),
            "dpo/acc": (logits.detach() > 0).float().mean(),
            "dpo/logits": logits.detach().mean(),
            "dpo/reward_margin": margin.detach().mean(),
            "dpo/loss_pos": loss_positive.detach().mean(),
            "dpo/loss_neg": loss_negative.detach().mean(),
            "dpo/loss_pos_ref": reference_positive.mean(),
            "dpo/loss_neg_ref": reference_negative.mean(),
            "bs": batch_size,
            "ntokens": lengths.sum(),
        }
        auxiliary = [term for term in (aux_positive, aux_negative) if isinstance(term, torch.Tensor)]
        if auxiliary:
            logs["moe_aux_loss"] = sum(auxiliary) / len(auxiliary)
        return logs, {}

    @torch.no_grad()
    def validation_step(self, sample, batch_idx):
        losses, _ = self.run_model(sample, condition_dropout=False)
        return {
            "losses": {key: value for key, value in losses.items() if key not in ("bs", "ntokens")},
            "total_loss": self._total_loss(losses),
            "nsamples": losses["bs"],
        }
