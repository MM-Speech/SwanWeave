import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from inference.tts.spat_base_infer import SpatBaseInfer
from utils.commons.hparams import hparams


def build_stereo_from_foa(wavs: torch.Tensor, channel_pair) -> torch.Tensor:
    if wavs.ndim != 3:
        raise ValueError(f"Expected wavs shape [B, T, C], got {tuple(wavs.shape)}")
    if wavs.shape[-1] < 4:
        raise ValueError(f"Spatial edit inference expects 4-channel FOA wavs, got {wavs.shape[-1]} channels")
    left_idx, right_idx = channel_pair
    return wavs[:, :, [left_idx, right_idx]].transpose(1, 2).contiguous()


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
        import torchaudio

        audio = torchaudio.functional.resample(
            audio,
            orig_freq=int(input_sample_rate),
            new_freq=int(target_sample_rate),
        )
    pad = (-int(audio.shape[-1])) % int(downsampling_ratio)
    if pad:
        audio = F.pad(audio, (0, pad))
    return audio


def load_foa_wav(wav_path: str, target_sr: int) -> torch.Tensor:
    import torchaudio

    wav, src_sr = torchaudio.load(wav_path)
    wav = wav.to(torch.float32)
    if int(src_sr) != int(target_sr):
        wav = torchaudio.functional.resample(wav, orig_freq=int(src_sr), new_freq=int(target_sr))
    wav = wav.transpose(0, 1).contiguous()
    if wav.ndim != 2 or int(wav.shape[1]) < 4:
        raise ValueError(f"Expected FOA wav with at least 4 channels, got shape {tuple(wav.shape)} from {wav_path}")
    if int(wav.shape[1]) > 4:
        wav = wav[:, :4]
    peak = float(torch.max(torch.abs(wav))) if wav.numel() > 0 else 0.0
    if peak > 1.0:
        wav = wav / peak
    return wav


def save_foa_wav(out_path: str, wav: np.ndarray, sample_rate: int):
    if not np.isfinite(wav).all():
        raise ValueError("Inference produced NaN or Inf waveform before saving")
    peak = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
    if peak > 1.0:
        wav = wav / peak * 0.99
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(out_path, wav, int(sample_rate), subtype="PCM_16")


def resolve_cfg_weights(cfg=None, caption_cfg=None, source_cfg=None):
    if cfg is not None and (caption_cfg is not None or source_cfg is not None):
        raise ValueError("--cfg cannot be combined with --caption_cfg/--source_cfg")
    if (caption_cfg is None) != (source_cfg is None):
        raise ValueError("--caption_cfg and --source_cfg must be provided together")
    if caption_cfg is None:
        value = 3.0 if cfg is None else float(cfg)
        return value, value
    return float(source_cfg), float(caption_cfg)


class SpatEditInfer(SpatBaseInfer):
    def build_model(self, dit_ckpt: str, config_path: str = ""):
        super().build_model(dit_ckpt=dit_ckpt, config_path=config_path)
        if bool(hparams.get("train_base", True)):
            raise ValueError(
                "spat_edit_infer.py requires an edit model config with train_base=false. "
                "Check --dit_ckpt/--config."
            )

    @torch.no_grad()
    def encode_foa_latent(self, wavs: torch.Tensor, input_sample_rate: int) -> torch.Tensor:
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

        chunked = bool(hparams.get("stable_audio_vae_chunked", True))
        chunk_size = int(hparams.get("stable_audio_vae_chunk_size", 128))
        overlap = int(hparams.get("stable_audio_vae_overlap", 32))
        deterministic = bool(hparams.get("vae_deterministic", True))
        lat_a = encode_stable_audio_safe(
            self.vae,
            audio_a.to(device=self.device, dtype=self.vae_dtype),
            chunked=chunked,
            chunk_size=chunk_size,
            overlap=overlap,
            deterministic=deterministic,
        )
        lat_b = encode_stable_audio_safe(
            self.vae,
            audio_b.to(device=self.device, dtype=self.vae_dtype),
            chunked=chunked,
            chunk_size=chunk_size,
            overlap=overlap,
            deterministic=deterministic,
        )
        target_len = min(int(lat_a.shape[-1]), int(lat_b.shape[-1]))
        lat_a = pad_latent_time(lat_a, target_len)
        lat_b = pad_latent_time(lat_b, target_len)
        lat = torch.cat([lat_a, lat_b], dim=1)
        return lat.transpose(1, 2).contiguous().to(torch.float32)

    @torch.no_grad()
    def build_edit_inputs(
        self,
        *,
        caption: str,
        src_lat: torch.Tensor,
        target_latent_len: int,
    ):
        all_embs, all_att = self.run_goku_text_encoder([caption])
        pos_emb = all_embs[0:1] * all_att[0:1][..., None]
        pos_lens = all_att[0:1].sum(-1).long()
        neg_emb = torch.zeros_like(pos_emb)
        neg_lens = pos_lens.clone()

        src_lat = pad_latent_seq_time(src_lat, int(target_latent_len))
        src_lat = src_lat.to(device=self.device, dtype=torch.float32)
        zero_src_lat = torch.zeros_like(src_lat)
        src_lat_cfg = torch.cat([src_lat, src_lat, zero_src_lat], dim=0).detach().clone()

        return {
            "caption_emb": torch.cat([pos_emb, neg_emb, neg_emb], dim=0).detach().clone(),
            "caption_lens": torch.cat([pos_lens, neg_lens, neg_lens], dim=0).detach().clone(),
            "tgt_len": torch.full((3,), int(target_latent_len), dtype=torch.long, device=self.device),
            "src_lat": src_lat_cfg,
        }

    @torch.no_grad()
    def sample_edit_latent(
        self,
        *,
        caption: str,
        src_lat: torch.Tensor,
        target_latent_len: int,
        num_steps: int = 20,
        source_cfg_w: float = None,
        caption_cfg_w: float = None,
        timestep_annealing_w=(0.6, 0.6, 1.0),
        use_amo_sampler: bool = False,
        use_sway: bool = True,
    ) -> torch.Tensor:
        inputs = self.build_edit_inputs(
            caption=caption,
            src_lat=src_lat,
            target_latent_len=target_latent_len,
        )
        autocast_enabled = self.device.type == "cuda" and self.precision in (torch.float16, torch.bfloat16)
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, dtype=self.precision, enabled=autocast_enabled):
                lat = self.dit.inference(
                    inputs,
                    timesteps=int(num_steps),
                    seq_cfg_w=(source_cfg_w, caption_cfg_w),
                    timestep_annealing_w=tuple(float(x) for x in timestep_annealing_w),
                    use_amo_sampler=bool(use_amo_sampler),
                    use_sway=bool(use_sway),
                )
        return lat.detach().to(torch.float32)

    @torch.no_grad()
    def infer_edit(
        self,
        *,
        src_wav_path: str,
        caption: str,
        out_path: str,
        target_latent_len: int = 0,
        num_steps: int = 20,
        cfg: float = None,
        caption_cfg: float = None,
        source_cfg: float = None,
        use_amo_sampler: bool = False,
        use_sway: bool = True,
        match_src_duration: bool = True,
    ):
        input_sample_rate = int(hparams["audio_sample_rate"])
        src_wav = load_foa_wav(src_wav_path, input_sample_rate)
        src_duration_sec = float(src_wav.shape[0]) / float(input_sample_rate)
        src_wavs = src_wav.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            src_lat = self.encode_foa_latent(src_wavs, input_sample_rate)
        if target_latent_len <= 0:
            target_latent_len = int(src_lat.shape[1])
        else:
            src_lat = pad_latent_seq_time(src_lat, int(target_latent_len))

        source_cfg_w, caption_cfg_w = resolve_cfg_weights(cfg, caption_cfg, source_cfg)
        edited_lat = self.sample_edit_latent(
            caption=caption,
            src_lat=src_lat,
            target_latent_len=int(target_latent_len),
            num_steps=num_steps,
            source_cfg_w=source_cfg_w,
            caption_cfg_w=caption_cfg_w,
            use_amo_sampler=use_amo_sampler,
            use_sway=use_sway,
        )
        wav = self.decode_foa_latent(edited_lat)[0].cpu().numpy()
        if match_src_duration:
            target_samples = int(round(src_duration_sec * float(self.vae_sample_rate)))
            wav = wav[:target_samples]
        save_foa_wav(out_path, wav, int(self.vae_sample_rate))

        return {
            "src_wav_path": src_wav_path,
            "caption": caption,
            "out_path": out_path,
            "sample_rate": int(self.vae_sample_rate),
            "num_samples": int(wav.shape[0]),
            "num_channels": int(wav.shape[1]),
            "duration_sec": float(wav.shape[0]) / float(self.vae_sample_rate),
            "src_duration_sec": src_duration_sec,
            "target_latent_len": int(target_latent_len),
            "caption_cfg": float(caption_cfg_w),
            "source_cfg": float(source_cfg_w),
        }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit_ckpt", type=str, required=True, help="Edit checkpoint file or experiment dir")
    parser.add_argument("--config", type=str, default="", help="Optional config.yaml path")
    parser.add_argument("--src_wav", type=str, required=True, help="Original/source 4-channel FOA wav path")
    parser.add_argument("--caption", type=str, required=True, help="Edit instruction caption")
    parser.add_argument("--out_path", type=str, required=True, help="Output edited 4-channel wav path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cuda:0 / cpu")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--target_latent_len", type=int, default=0, help="Override output latent length")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of ODE sampling steps")
    parser.add_argument("--cfg", type=float, default=None, help="Shared CFG weight; defaults to 3.0")
    parser.add_argument("--caption_cfg", type=float, default=None, help="Instruction/caption CFG weight")
    parser.add_argument("--source_cfg", type=float, default=None, help="Source-preservation CFG weight")
    parser.add_argument("--use_amo_sampler", action="store_true")
    parser.add_argument("--no_sway", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--no_match_src_duration", action="store_true")
    parser.add_argument("--summary_path", type=str, default="", help="Optional json path for inference metadata")
    return parser


def main():
    args = build_parser().parse_args()
    resolve_cfg_weights(args.cfg, args.caption_cfg, args.source_cfg)
    infer = SpatEditInfer(
        device=args.device,
        dit_ckpt=args.dit_ckpt,
        config_path=args.config,
        precision=args.precision,
        use_ema=args.use_ema,
    )
    result = infer.infer_edit(
        src_wav_path=args.src_wav,
        caption=args.caption,
        out_path=args.out_path,
        target_latent_len=args.target_latent_len,
        num_steps=args.num_steps,
        cfg=args.cfg,
        caption_cfg=args.caption_cfg,
        source_cfg=args.source_cfg,
        use_amo_sampler=args.use_amo_sampler,
        use_sway=not args.no_sway,
        match_src_duration=not args.no_match_src_duration,
    )
    print(result)
    if args.summary_path:
        summary_dir = os.path.dirname(args.summary_path)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        with open(args.summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


# python inference/tts/spat_edit_infer.py \
#   --dit_ckpt checkpoints/260517_edit_base_test \
#   --src_wav /mnt/bn/sa-ag-data/leike/spatial_edit/triplet/audio_distance_motion_pyroom/00000000/origin/foa.wav \
#   --caption "The Gurgling moves farther away from directly to your right over its duration." \
#   --out_path users/infer_out/edit3.wav \
#   --device cuda \
#   --precision bf16 \
#   --num_steps 20 \
#   --caption_cfg 3.0 \
#   --source_cfg 3.0
