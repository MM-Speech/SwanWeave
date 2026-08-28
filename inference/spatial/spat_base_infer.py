import argparse
import collections
import collections.abc
import os
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

from modules.tts.spat_edit.build_model_utils import DiTBuildModelMixin
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams, set_hparams


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def resolve_model_config_path(dit_ckpt: str, config_path: str = "") -> str:
    if config_path:
        return config_path
    if dit_ckpt.endswith((".ckpt", ".pt", ".pth", ".safetensors")):
        return os.path.join(os.path.dirname(dit_ckpt), "config.yaml")
    return os.path.join(dit_ckpt, "config.yaml")


def decode_stable_audio_safe(
    vae,
    latents: torch.Tensor,
    *,
    chunked: bool,
    chunk_size: int,
    overlap: int,
) -> torch.Tensor:
    if not hasattr(vae, "decode_audio"):
        if (not chunked) or int(latents.shape[-1]) <= int(chunk_size):
            decoded = vae.decode(latents)
            return getattr(decoded, "sample", decoded[0] if isinstance(decoded, tuple) else decoded)

        hop_size = int(chunk_size) - int(overlap)
        if hop_size <= 0:
            raise ValueError("stable_audio_vae_overlap must be smaller than stable_audio_vae_chunk_size")

        total_latents = int(latents.shape[-1])
        downsampling_ratio = 1
        for ratio in list(getattr(vae.config, "downsampling_ratios", [2, 4, 4, 8, 8])):
            downsampling_ratio *= int(ratio)
        total_samples = total_latents * downsampling_ratio

        starts = list(range(0, total_latents - int(chunk_size) + 1, hop_size))
        final_start = total_latents - int(chunk_size)
        if starts[-1] != final_start:
            starts.append(final_start)

        decoded_chunks = []
        for start in starts:
            decoded = vae.decode(latents[:, :, start : start + int(chunk_size)])
            decoded_chunks.append(getattr(decoded, "sample", decoded[0] if isinstance(decoded, tuple) else decoded))

        audio = decoded_chunks[0].new_zeros((latents.shape[0], decoded_chunks[0].shape[1], total_samples))
        overlap_samples = (int(overlap) // 2) * downsampling_ratio
        for index, (start, chunk_audio) in enumerate(zip(starts, decoded_chunks)):
            if index == len(starts) - 1:
                sample_end = total_samples
                sample_start = sample_end - int(chunk_audio.shape[-1])
            else:
                sample_start = int(start) * downsampling_ratio
                sample_end = sample_start + int(chunk_audio.shape[-1])

            crop_start = 0
            crop_end = int(chunk_audio.shape[-1])
            if index > 0:
                sample_start += overlap_samples
                crop_start += overlap_samples
            if index < len(starts) - 1:
                sample_end -= overlap_samples
                crop_end -= overlap_samples
            audio[:, :, sample_start:sample_end] = chunk_audio[:, :, crop_start:crop_end]
        return audio

    if (not chunked) or int(latents.shape[-1]) <= int(chunk_size):
        return vae.decode_audio(latents, chunked=False)
    return vae.decode_audio(
        latents,
        chunked=True,
        chunk_size=int(chunk_size),
        overlap=int(overlap),
    )


class SpatBaseInfer(DiTBuildModelMixin):
    def __init__(
        self,
        device: str,
        dit_ckpt: str,
        config_path: str = "",
        precision: str = "bf16",
        use_ema: bool = False,
    ):
        self.device = torch.device(device)
        self.use_ema = bool(use_ema)
        if precision == "fp16":
            self.precision = torch.float16
        elif precision == "fp32":
            self.precision = torch.float32
        else:
            self.precision = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.build_model(dit_ckpt=dit_ckpt, config_path=config_path)

    def build_model(self, dit_ckpt: str, config_path: str = ""):
        config_path = resolve_model_config_path(dit_ckpt, config_path)
        set_hparams(config=config_path, print_hparams=False, global_hparams=True)
        hparams["exp_name"] = "infer"
        hparams["use_fsdp"] = False
        self.config = AttrDict(hparams)
        self.trainer = SimpleNamespace(device=self.device)

        attn_implementation = hparams.get("attn_implementation", "sdpa")
        self._build_model(attn_implementation=attn_implementation)

        model_name = "ema_model" if self.use_ema and hparams.get("use_ema", False) else "dit"
        load_ckpt(self.dit, dit_ckpt, model_name, strict=False)

        self.vae.eval().to(self.device)
        self.vae_dtype = next(self.vae.parameters()).dtype
        self.dit.eval().to(self.device, dtype=self.precision)
        if getattr(self, "goku_text_encoder", None) is not None:
            self.goku_text_encoder.eval().to(self.device, dtype=self.precision)

    @torch.no_grad()
    def run_goku_text_encoder(self, captions):
        if getattr(self, "goku_text_encoder", None) is None:
            raise RuntimeError("Caption encoder is not built. Check use_caption / model config.")
        self.goku_tokenizer.padding_side = "left"
        hidden_states = []
        attention_masks = []
        max_len = 0

        for caption in captions:
            inputs = self.goku_tokenizer(
                [caption],
                padding=False,
                truncation=True,
                max_length=hparams.get("text_max_token_length", 256),
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.to(device=self.device, dtype=torch.long)
            attention_mask = inputs.attention_mask.to(device=self.device, dtype=torch.long)
            encoder_hidden_state = self.goku_text_encoder(
                input_ids=input_ids,
                return_dict=False,
                attention_mask=attention_mask,
            )[0]
            hidden_states.append(encoder_hidden_state)
            attention_masks.append(attention_mask)
            max_len = max(max_len, int(attention_mask.shape[1]))

        padded_hidden_states = []
        padded_attention_masks = []
        for encoder_hidden_state, attention_mask in zip(hidden_states, attention_masks):
            pad_len = max_len - int(attention_mask.shape[1])
            if pad_len > 0:
                encoder_hidden_state = F.pad(encoder_hidden_state, (0, 0, pad_len, 0), value=0.0)
                attention_mask = F.pad(attention_mask, (pad_len, 0), value=0)
            padded_hidden_states.append(encoder_hidden_state)
            padded_attention_masks.append(attention_mask)

        return torch.cat(padded_hidden_states, dim=0), torch.cat(padded_attention_masks, dim=0)

    @torch.no_grad()
    def build_caption_inputs(
        self,
        caption: str,
        target_latent_len: int,
    ):
        all_embs, all_att = self.run_goku_text_encoder([caption])
        pos_emb = all_embs[0:1] * all_att[0:1][..., None]
        pos_lens = all_att[0:1].sum(-1).long()
        neg_emb = torch.zeros_like(pos_emb)
        neg_lens = pos_lens.clone()

        caption_emb = torch.cat([pos_emb, neg_emb], dim=0).detach().clone()
        caption_lens = torch.cat([pos_lens, neg_lens], dim=0).detach().clone()
        tgt_len = torch.full(
            (2,),
            int(target_latent_len),
            dtype=torch.long,
            device=self.device,
        )
        return {
            "caption_emb": caption_emb,
            "caption_lens": caption_lens,
            "tgt_len": tgt_len,
        }

    @torch.no_grad()
    def sample_latent(
        self,
        caption: str,
        target_latent_len: int,
        num_steps: int = 20,
        cfg_w: float = 3.0,
        timestep_annealing_w=(0.6, 0.6, 1.0),
        use_amo_sampler: bool = False,
        use_sway: bool = True,
    ) -> torch.Tensor:
        inputs = self.build_caption_inputs(
            caption=caption,
            target_latent_len=target_latent_len,
        )
        autocast_enabled = self.device.type == "cuda" and self.precision in (torch.float16, torch.bfloat16)
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, dtype=self.precision, enabled=autocast_enabled):
                lat = self.dit.inference(
                    inputs,
                    timesteps=int(num_steps),
                    seq_cfg_w=float(cfg_w),
                    timestep_annealing_w=tuple(float(x) for x in timestep_annealing_w),
                    use_amo_sampler=bool(use_amo_sampler),
                    use_sway=bool(use_sway),
                )
        return lat.detach().to(torch.float32)

    @torch.no_grad()
    def decode_foa_latent(self, lat: torch.Tensor) -> torch.Tensor:
        if lat.ndim != 3 or int(lat.shape[-1]) != 128:
            raise ValueError(f"Expected latent shape [B, L, 128], got {tuple(lat.shape)}")

        chunked = bool(hparams.get("stable_audio_vae_chunked", True))
        chunk_size = int(hparams.get("stable_audio_vae_chunk_size", 128))
        overlap = int(hparams.get("stable_audio_vae_overlap", 32))

        lat_a = lat[:, :, :64].transpose(1, 2).contiguous().to(self.device, dtype=self.vae_dtype)
        lat_b = lat[:, :, 64:].transpose(1, 2).contiguous().to(self.device, dtype=self.vae_dtype)

        with torch.inference_mode():
            wav_a = decode_stable_audio_safe(
                self.vae,
                lat_a,
                chunked=chunked,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            wav_b = decode_stable_audio_safe(
                self.vae,
                lat_b,
                chunked=chunked,
                chunk_size=chunk_size,
                overlap=overlap,
            )

        wav_a = wav_a.to(torch.float32)
        wav_b = wav_b.to(torch.float32)
        wav_len = min(int(wav_a.shape[-1]), int(wav_b.shape[-1]))
        wav_a = wav_a[:, :, :wav_len]
        wav_b = wav_b[:, :, :wav_len]
        wav_foa = torch.cat([wav_a, wav_b], dim=1)
        return wav_foa.transpose(1, 2).contiguous()

    def seconds_to_latent_len(self, seconds: float) -> int:
        return max(
            1,
            int(round(float(seconds) * float(self.vae_sample_rate) / float(self.vae_downsampling_ratio))),
        )

    @torch.no_grad()
    def infer(
        self,
        caption: str,
        out_path: str,
        seconds: float = 10.0,
        target_latent_len: int = 0,
        num_steps: int = 20,
        cfg: float = 3.0,
        use_amo_sampler: bool = False,
        use_sway: bool = True,
    ):
        if target_latent_len <= 0:
            target_latent_len = self.seconds_to_latent_len(seconds)
        lat = self.sample_latent(
            caption=caption,
            target_latent_len=target_latent_len,
            num_steps=num_steps,
            cfg_w=cfg,
            use_amo_sampler=use_amo_sampler,
            use_sway=use_sway,
        )
        wav = self.decode_foa_latent(lat)[0].cpu().numpy()
        if not np.isfinite(wav).all():
            raise ValueError("Inference produced NaN or Inf waveform before saving")
        peak = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
        if peak > 1.0:
            wav = wav / peak * 0.99

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        sf.write(out_path, wav, int(self.vae_sample_rate), subtype="PCM_16")
        return {
            "out_path": out_path,
            "sample_rate": int(self.vae_sample_rate),
            "num_samples": int(wav.shape[0]),
            "num_channels": int(wav.shape[1]),
            "duration_sec": float(wav.shape[0]) / float(self.vae_sample_rate),
            "target_latent_len": int(target_latent_len),
        }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit_ckpt", type=str, required=True, help="Checkpoint file or experiment dir")
    parser.add_argument("--config", type=str, default="", help="Optional config.yaml path")
    parser.add_argument("--caption", type=str, required=True, help="Positive caption")
    parser.add_argument("--out_path", type=str, required=True, help="Output 4-channel wav path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cuda:0 / cpu")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seconds", type=float, default=10.0, help="Target duration in seconds")
    parser.add_argument("--target_latent_len", type=int, default=0, help="Override target latent length")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of ODE sampling steps")
    parser.add_argument("--cfg", type=float, default=3.0, help="Single CFG weight for caption conditioning")
    parser.add_argument("--use_amo_sampler", action="store_true")
    parser.add_argument("--no_sway", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    infer = SpatBaseInfer(
        device=args.device,
        dit_ckpt=args.dit_ckpt,
        config_path=args.config,
        precision=args.precision,
        use_ema=args.use_ema,
    )
    result = infer.infer(
        caption=args.caption,
        out_path=args.out_path,
        seconds=args.seconds,
        target_latent_len=args.target_latent_len,
        num_steps=args.num_steps,
        cfg=args.cfg,
        use_amo_sampler=args.use_amo_sampler,
        use_sway=not args.no_sway,
    )
    print(result)


if __name__ == "__main__":
    main()

# python inference/tts/spat_base_infer.py \
#   --dit_ckpt checkpoints/260516_spat_base_base \
#   --config checkpoints/260516_spat_base_base/config.yaml \
#   --caption "Toilet flush to your left" \
#   --out_path /mnt/bn/sa-ag-data/leike/spatial_edit/ScriptSpeech/users/infer_out/spat_base_toilet_left.wav \
#   --device cuda \
#   --precision bf16 \
#   --seconds 8 \
#   --num_steps 20 \
#   --cfg 3.0