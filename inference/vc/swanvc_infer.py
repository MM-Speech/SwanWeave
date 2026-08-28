import argparse
import collections
import collections.abc
import json
import math
import os
import random
import re
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from attrdictionary import AttrDict

from utils.commons.ckpt_utils import get_all_ckpts, load_ckpt
from utils.commons.dataset_utils import collate_xd
from utils.commons.hparams import hparams, set_hparams
from utils.commons.upload_tos_utils import send_file_to_tos
from utils.nn.seq_utils import sequence_mask

from modules.vc.swanvc.build_model_utils import DiTBuildModelMixin


def _safe_fname(x: Optional[str]) -> str:
    if x is None:
        return "none"
    x = str(x)
    x = x.replace("/", "_").replace("\\", "_").replace(os.sep, "_")
    x = re.sub(r"\s+", "_", x).strip("_")
    return x or "none"


def flatten_grouped_samples(samples_cfg):
    if isinstance(samples_cfg, list):
        samples = []
        for i, sample in enumerate(samples_cfg):
            s = dict(sample or {})
            s.setdefault("__group__", "default")
            s.setdefault("__group_idx__", i)
            samples.append(s)
        return samples

    if not isinstance(samples_cfg, dict):
        return []

    flat = []
    for group_name, group_list in samples_cfg.items():
        if not isinstance(group_list, (list, tuple)):
            continue
        for gi, sample in enumerate(group_list):
            s = dict(sample or {})
            s["__group__"] = group_name
            s["__group_idx__"] = gi
            flat.append(s)
    return flat


def _parse_ckpt_step_from_path(ckpt_path: Optional[str]) -> Optional[int]:
    if not isinstance(ckpt_path, str) or not ckpt_path:
        return None
    m = re.findall(r".*steps_(\d+)(?:\.ckpt|_backbone\.ckpt)?", ckpt_path)
    if not m:
        return None
    try:
        return int(m[0])
    except Exception:
        return None


def _infer_exp_name_from_ckpt_arg(ckpt_arg: Optional[str]) -> str:
    if not ckpt_arg:
        return "exp"
    p = os.path.expanduser(str(ckpt_arg)).strip()
    if os.path.isfile(p):
        return os.path.basename(os.path.dirname(os.path.realpath(p))) or "exp"
    if os.path.isdir(p):
        return os.path.basename(os.path.realpath(p)) or "exp"
    return os.path.basename(p.rstrip("/").rstrip("\\")) or "exp"


def _choose_step_for_dir(ckpt_arg: Optional[str]) -> Optional[int]:
    if not ckpt_arg:
        return None
    p = os.path.expanduser(str(ckpt_arg))
    if os.path.isdir(p):
        ckpts = get_all_ckpts(p, steps=None)
        if ckpts:
            return _parse_ckpt_step_from_path(ckpts[0])
    return _parse_ckpt_step_from_path(p)


def _ckpt_report_line(name: str, ckpt_arg: Optional[str]) -> str:
    if not ckpt_arg:
        return f"{name}: <none>"
    p = os.path.realpath(os.path.expanduser(str(ckpt_arg)))
    step = _parse_ckpt_step_from_path(p)
    if step is None:
        return f"{name}: {p}"
    return f"{name}: {p} (steps={step})"


def _to_file_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def _json_ready(obj: Any):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _get_tos_bucket() -> str:
    cluster = os.environ.get("CLUSTER", "").lower()
    return "sa-ag-sg-research-sg" if cluster == "va" else "humanaigc-ads"


def upload_group_audios_to_tos(group_infos: Dict[str, List[Dict[str, Any]]], run_tag: str):
    sub_dir = f"swanvc/{run_tag}"
    bucket = _get_tos_bucket()
    for infos in group_infos.values():
        for info in infos:
            info["content_tos_url"] = send_file_to_tos(info["content_ref"], sub_dir=sub_dir, bucket=bucket)
            info["timbre_tos_url"] = send_file_to_tos(info["timbre_ref"], sub_dir=sub_dir, bucket=bucket)
            info["output_tos_url"] = send_file_to_tos(info["output_wav"], sub_dir=sub_dir, bucket=bucket)
    return sub_dir, bucket


def splice_silence(
    wav: np.ndarray,
    sr: int = 24000,
    sil_sec: float = 0.0,
    mode: str = "both",
) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if sil_sec <= 0 or wav.ndim != 1 or wav.size == 0:
        return wav

    n = int(sr * float(sil_sec))
    if n <= 0:
        return wav

    zeros = np.zeros((n,), dtype=wav.dtype)
    head = zeros if mode in ("in", "both") else np.zeros((0,), dtype=wav.dtype)
    tail = zeros if mode in ("out", "both") else np.zeros((0,), dtype=wav.dtype)
    return np.concatenate([head, wav, tail], axis=0)


def gen_local_audio_html(
    group_infos: Dict[str, List[Dict[str, Any]]],
    output_fp: str,
    title_name: Optional[str] = None,
    extra_desc: Optional[str] = None,
):
    os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    with open(output_fp, "w", encoding="utf-8") as f:
        print("<html lang='zh'>", file=f)
        print("<head>", file=f)
        print("<meta charset='UTF-8'>", file=f)
        print("<meta name='viewport' content='width=device-width, initial-scale=1.0'>", file=f)
        print(f"<title>{title_name or 'SwanVC Report'}</title>", file=f)
        print(
            """
<style>
body { margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #fafafa; }
.container { max-width: 1600px; margin: 0 auto; }
h1 { margin: 0 0 8px 0; }
p.description { white-space: pre-wrap; background: #fff; padding: 12px; border-radius: 8px; border: 1px solid #ddd; }
h2 { margin-top: 28px; }
table { width: 100%; border-collapse: collapse; background: #fff; }
th, td { border: 1px solid #ddd; padding: 10px; vertical-align: top; }
th { background: #f0f4ff; }
audio { width: 320px; max-width: 100%; }
pre { white-space: pre-wrap; word-break: break-word; margin: 0; font-size: 12px; }
.path { font-size: 12px; color: #666; word-break: break-all; margin-top: 6px; }
</style>
""",
            file=f,
        )
        print("</head><body><div class='container'>", file=f)
        print(f"<h1>{title_name or 'SwanVC Report'}</h1>", file=f)
        if extra_desc:
            safe_desc = (
                str(extra_desc)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            print(f"<p class='description'>{safe_desc}</p>", file=f)

        for group_name, infos in group_infos.items():
            print(f"<h2>{group_name}</h2>", file=f)
            print("<table>", file=f)
            print(
                "<tr>"
                "<th>Sample</th>"
                "<th>Content Ref</th>"
                "<th>Timbre Ref</th>"
                "<th>Output</th>"
                "<th>Metadata</th>"
                "</tr>",
                file=f,
            )
            for info in infos:
                sample_id = info.get("sample_id", "")
                content_uri = info.get("content_tos_url") or _to_file_uri(info["content_ref"])
                timbre_uri = info.get("timbre_tos_url") or _to_file_uri(info["timbre_ref"])
                output_uri = info.get("output_tos_url") or _to_file_uri(info["output_wav"])
                meta_text = json.dumps(_json_ready(info.get("meta", {})), ensure_ascii=False, indent=2)

                print("<tr>", file=f)
                print(f"<td>{sample_id}</td>", file=f)
                print(
                    f"<td><audio controls preload='none'><source src='{content_uri}' type='audio/wav'></audio>"
                    f"<div class='path'>{info['content_ref']}</div></td>",
                    file=f,
                )
                print(
                    f"<td><audio controls preload='none'><source src='{timbre_uri}' type='audio/wav'></audio>"
                    f"<div class='path'>{info['timbre_ref']}</div></td>",
                    file=f,
                )
                print(
                    f"<td><audio controls preload='none'><source src='{output_uri}' type='audio/wav'></audio>"
                    f"<div class='path'>{info['output_wav']}</div></td>",
                    file=f,
                )
                safe_meta = (
                    meta_text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                print(f"<td><pre>{safe_meta}</pre></td>", file=f)
                print("</tr>", file=f)
            print("</table>", file=f)

        print("</div></body></html>", file=f)
    return output_fp


class SwanVCInfer(DiTBuildModelMixin):
    def __init__(self, device: str, dit_ckpt: str, vae_ckpt: Optional[str] = None):
        self.device = torch.device(device)
        self.trainer = AttrDict(device=self.device)
        self.precision = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.resamplers: Dict[int, torchaudio.transforms.Resample] = {}
        self.build_model(dit_ckpt=dit_ckpt, vae_ckpt=vae_ckpt)

    def _amp_ctx(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.precision)
        return nullcontext()

    def build_model(self, dit_ckpt: str, vae_ckpt: Optional[str] = None):
        if dit_ckpt.endswith(".ckpt"):
            config_path = os.path.join(Path(dit_ckpt).parent, "config.yaml")
        else:
            config_path = os.path.join(dit_ckpt, "config.yaml")

        set_hparams(config=config_path, print_hparams=False, global_hparams=True)
        hparams["exp_name"] = "swanvc_infer"
        if vae_ckpt is not None:
            hparams["vae_ckpt"] = vae_ckpt
        self.config = AttrDict(hparams)

        self._build_model(attn_implementation="flash_attention_2")
        load_ckpt(self.dit, dit_ckpt, "dit", strict=False)

        self.vae.eval().to(self.device, dtype=self.precision)
        self.dit.eval().to(self.device, dtype=self.precision)

        if hasattr(self.dit, "cast_safe_params_to_bf16") and self.precision == torch.bfloat16 and not hparams.get("use_fsdp", False):
            self.dit.cast_safe_params_to_bf16()

        self.sr = int(getattr(self.vae, "sample_rate", hparams.get("audio_sample_rate", 24000)))
        hop_len = getattr(self.vae, "hop_length", None)
        if hop_len is None:
            hop_size = int(self.hp_vae.get("hop_size", hparams.get("hop_size", 960)))
            vae_stride = int(self.hp_vae.get("vae_stride", hparams.get("vae_stride", 1)))
            hop_len = hop_size * vae_stride
        self.hop_length = int(hop_len)

        print(
            f"[INFO] SwanVCInfer ready | sr={self.sr}, hop_length={self.hop_length}, "
            f"semantic_token_type={hparams.get('semantic_token_type', 'content_style')}, "
            f"device={self.device}, precision={self.precision}"
        )

    def _load_audio(self, path: str, max_sec: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        wav, _ = librosa.load(path, sr=self.sr, mono=True)
        raw_len = int(wav.shape[0])

        if max_sec is not None and max_sec > 0:
            max_len = int(max_sec * self.sr)
            wav = wav[:max_len]

        wav = wav.astype(np.float32, copy=False)
        peak = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
        if peak > 1.0:
            wav = wav / peak

        info = {
            "path": os.path.realpath(path),
            "samples": int(wav.shape[0]),
            "seconds": float(wav.shape[0]) / float(self.sr),
            "truncated": bool(wav.shape[0] != raw_len),
        }
        return wav, info

    def _get_resampler(self, target_sr: int):
        if target_sr not in self.resamplers:
            self.resamplers[target_sr] = torchaudio.transforms.Resample(
                orig_freq=self.sr, new_freq=target_sr
            ).to(self.device)
        return self.resamplers[target_sr]

    def _wav_np_to_tensor(self, wav: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        wav_t = torch.from_numpy(np.asarray(wav, dtype=np.float32)).to(self.device)[None, :]
        wav_len = torch.LongTensor([wav_t.shape[1]]).to(self.device)
        return wav_t, wav_len

    def _vae_encode_latent(self, wav: np.ndarray) -> Tuple[torch.Tensor, int]:
        wav_t, wav_len = self._wav_np_to_tensor(wav)
        kwargs = dict(
            chunk_sec=hparams.get("vae_encode_chunk_sec", 20),
            max_batch_size=hparams.get("vae_encode_max_batch_size", 128),
            deterministic=hparams.get("vae_deterministic", False),
        )
        with torch.inference_mode(), self._amp_ctx():
            try:
                out = self.vae.encode_latent(wav_t, wav_len, **kwargs)
            except TypeError:
                out = self.vae.encode_latent(wav_t, wav_len)

        if isinstance(out, (tuple, list)):
            lat = out[0]
            lat_len = int(out[1][0].item()) if len(out) > 1 and isinstance(out[1], torch.Tensor) else int(lat.shape[1])
        else:
            lat = out
            lat_len = int(lat.shape[1])

        lat = lat[:, :lat_len]
        return lat, lat_len

    def _vae_decode_to_wav(self, lat: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), self._amp_ctx():
            out = self.vae.decode(lat)
        if isinstance(out, (tuple, list)):
            out = out[0]

        if out.ndim == 3:
            wav = out[0, 0]
        elif out.ndim == 2:
            wav = out[0]
        else:
            raise RuntimeError(f"Unexpected vae.decode output shape: {tuple(out.shape)}")
        return wav.to(torch.float32)

    def _extract_semantic_tokens(self, wav: np.ndarray) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        wav_t, wav_len = self._wav_np_to_tensor(wav)

        token_sr = int(self.semantic_tokenizer.token_sample_rate)
        wav_16k = self._get_resampler(token_sr)(wav_t)
        wav_16k_len = wav_len * token_sr // self.sr

        semantic_type = hparams.get("semantic_token_type", "content_style")
        if semantic_type == "content_style":
            vector_type = "content_style"
        elif semantic_type == "content":
            vector_type = "content"
        else:
            raise ValueError(f"Unsupported semantic_token_type={semantic_type}")

        with torch.inference_mode():
            results = self.semantic_tokenizer.extract_from_16k_batch_chunked(
                wavs=wav_16k,
                wav_lengths=wav_16k_len,
                batch_size=256,
                vector_type=vector_type,
                reduce_content=False,
                reduce_content_style=False,
            )

        result = results[0]
        if semantic_type == "content_style":
            token_ids = torch.from_numpy(result.content_style_ids).long()
        else:
            token_ids = torch.from_numpy(result.content_ids).long()

        if token_ids.numel() < 2:
            raise RuntimeError("Semantic tokens too short, cannot run VC inference.")

        if token_ids.numel() % 2 == 1:
            token_ids = token_ids[:-1]

        meta = {
            "semantic_type": semantic_type,
            "token_sample_rate": token_sr,
            "semantic_tokens": int(token_ids.numel()),
            "latent_frames_from_semantic": int(token_ids.numel() // 2),
        }
        return token_ids[None, :].to(self.device), int(token_ids.numel()), meta

    @staticmethod
    def _pad_or_trim_latent_time(latent: torch.Tensor, target_t: int) -> torch.Tensor:
        if latent.shape[1] == target_t:
            return latent
        if latent.shape[1] > target_t:
            return latent[:, :target_t]
        return torch.nn.functional.pad(latent, (0, 0, 0, target_t - latent.shape[1]), mode="constant", value=0.0)

    @torch.no_grad()
    def convert(
        self,
        content_ref: str,
        timbre_ref: str,
        num_step: int = 40,
        cfg_w: Tuple[float, float] = (1.5, 3.0),
        timestep_annealing_w: Tuple[float, float, float] = (0.6, 0.6, 1.0),
        use_amo_sampler: bool = False,
        use_sway: bool = True,
        max_timbre_sec: float = 15.0,
        pad_silence_sec: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        content_wav, content_audio_info = self._load_audio(content_ref, max_sec=None)
        timbre_wav, timbre_audio_info = self._load_audio(timbre_ref, max_sec=max_timbre_sec)

        prompt_lat, prompt_lat_len_raw = self._vae_encode_latent(timbre_wav)
        prompt_sem, prompt_sem_len, prompt_sem_meta = self._extract_semantic_tokens(timbre_wav)
        content_sem, content_sem_len, content_sem_meta = self._extract_semantic_tokens(content_wav)

        prompt_lat_len = prompt_sem_len // 2
        content_lat_len = content_sem_len // 2
        if content_lat_len <= 0:
            raise RuntimeError("Content semantic length is zero after alignment.")
        if prompt_lat_len <= 0:
            raise RuntimeError("Timbre semantic length is zero after alignment.")

        prompt_lat = self._pad_or_trim_latent_time(prompt_lat, prompt_lat_len)
        prompt_lat = prompt_lat.to(dtype=self.precision)

        semantic_tokens = torch.cat([prompt_sem, content_sem], dim=1)
        semantic_lens = torch.LongTensor([semantic_tokens.shape[1]]).to(self.device)
        semantic_mask = sequence_mask(semantic_lens, maxlen=semantic_tokens.shape[1]).to(self.device)

        total_lat_len = prompt_lat_len + content_lat_len
        latent_dim = int(prompt_lat.shape[-1])

        lat_ctx_cond = torch.zeros((1, total_lat_len, latent_dim), device=self.device, dtype=self.precision)
        ctx_mask_cond = torch.zeros((1, total_lat_len, 1), device=self.device, dtype=self.precision)
        lat_ctx_cond[:, :prompt_lat_len] = prompt_lat
        ctx_mask_cond[:, :prompt_lat_len] = 1.0

        lat_ctx_zero = torch.zeros_like(lat_ctx_cond)
        ctx_mask_zero = torch.zeros_like(ctx_mask_cond)
        semantic_uncond = torch.full_like(semantic_tokens, self.cfg_mask_token)

        inputs = {
            "semantic_tokens": torch.cat([semantic_tokens, semantic_tokens, semantic_uncond], dim=0).long(),
            "semantic_lens": torch.cat([semantic_lens, semantic_lens, semantic_lens], dim=0),
            "semantic_mask": torch.cat([semantic_mask, semantic_mask, semantic_mask], dim=0),
            "lat_ctx": torch.cat([lat_ctx_cond, lat_ctx_zero, lat_ctx_zero], dim=0),
            "ctx_mask": torch.cat([ctx_mask_cond, ctx_mask_zero, ctx_mask_zero], dim=0),
            "tgt_len": torch.LongTensor([total_lat_len, total_lat_len, total_lat_len]).to(self.device),
        }

        with self._amp_ctx():
            pred_lat = self.dit.inference(
                inputs,
                timesteps=num_step,
                seq_cfg_w=cfg_w,
                timestep_annealing_w=timestep_annealing_w,
                use_amo_sampler=use_amo_sampler,
                use_sway=use_sway,
            )

        pred_lat = pred_lat[0:1]
        pred_lat[:, :prompt_lat_len] = prompt_lat

        wav_dec = self._vae_decode_to_wav(pred_lat)
        drop_wav = int(prompt_lat_len * self.hop_length)
        wav_pred = wav_dec[drop_wav:]

        if wav_pred.numel() == 0:
            raise RuntimeError("Decoded waveform is empty after dropping prompt region.")

        peak = float(wav_pred.abs().max().item())
        if peak > 1.0:
            wav_pred = wav_pred / peak

        wav_np = wav_pred.cpu().numpy()
        wav_np = splice_silence(wav_np, sr=self.sr, sil_sec=pad_silence_sec, mode="both")

        meta = {
            "content_audio": content_audio_info,
            "timbre_audio": timbre_audio_info,
            "prompt_lat_len_raw": int(prompt_lat_len_raw),
            "prompt_lat_len_used": int(prompt_lat_len),
            "content_lat_len_used": int(content_lat_len),
            "total_lat_len": int(total_lat_len),
            "drop_prompt_samples": int(drop_wav),
            "output_seconds": float(len(wav_np)) / float(self.sr),
            "num_step": int(num_step),
            "cfg_w": list(cfg_w),
            "timestep_annealing_w": list(timestep_annealing_w),
            "use_amo_sampler": bool(use_amo_sampler),
            "use_sway": bool(use_sway),
            "pad_silence_sec": float(pad_silence_sec),
            "semantic": {
                "prompt": prompt_sem_meta,
                "content": content_sem_meta,
                "concat_tokens": int(semantic_tokens.shape[1]),
            },
            "model": {
                "sample_rate": int(self.sr),
                "hop_length": int(self.hop_length),
                "semantic_token_type": hparams.get("semantic_token_type", "content_style"),
                "cfg_mask_token": int(self.cfg_mask_token),
            },
        }
        return wav_np, meta


def resolve_cli_or_cfg(cli_value, cfg: Dict[str, Any], key: str, default=None):
    if cli_value is not None:
        return cli_value
    if isinstance(cfg, dict) and key in cfg:
        return cfg[key]
    return default


def require_value(value, key: str):
    if value is None:
        raise ValueError(f"Missing required argument/config: {key}")
    return value


def load_samples(args) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    has_cli_sample = (args.content_ref is not None) or (args.timbre_ref is not None) or (args.sample_id is not None)
    if has_cli_sample:
        if args.content_ref is None or args.timbre_ref is None:
            raise ValueError("When overriding sample from CLI, both `--content_ref` and `--timbre_ref` are required.")
        samples = [{
            "id": args.sample_id or "sample_0000",
            "content_ref": args.content_ref,
            "timbre_ref": args.timbre_ref,
            "__group__": "default",
            "__group_idx__": 0,
        }]
    else:
        samples = flatten_grouped_samples(cfg.get("samples", []))
        if not samples:
            raise ValueError("No samples found from CLI or config.")

    for i, s in enumerate(samples):
        if not s.get("content_ref"):
            raise ValueError(f"Sample #{i} missing `content_ref`.")
        if not s.get("timbre_ref"):
            raise ValueError(f"Sample #{i} missing `timbre_ref`.")
    return cfg, samples


def make_output_paths(args, cfg):
    exp_name = _infer_exp_name_from_ckpt_arg(args.dit_ckpt)
    step = _choose_step_for_dir(args.dit_ckpt)
    step_dirname = f"steps_{step}" if step is not None else "steps_unknown"
    bench = cfg.get("bench", "vc")
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg_w = resolve_cli_or_cfg(args.cfg_w, cfg, "cfg_w", [1.5, 3.0])
    timestep_annealing_w = resolve_cli_or_cfg(args.timestep_annealing_w, cfg, "timestep_annealing_w", [0.6, 0.6, 1.0])
    num_step = resolve_cli_or_cfg(args.num_step, cfg, "num_step", 40)
    use_amo_sampler = resolve_cli_or_cfg(args.use_amo_sampler, cfg, "use_amo_sampler", False)
    use_sway = resolve_cli_or_cfg(args.use_sway, cfg, "use_sway", True)

    setting_dir = (
        f"cfg{tuple(cfg_w)}"
        f"_timew{tuple(timestep_annealing_w)}"
        f"_amo{use_amo_sampler}"
        f"_sway{use_sway}"
        f"_step{num_step}"
    )
    out_dir = os.path.join(
        args.out_path,
        exp_name,
        step_dirname,
        _safe_fname(bench),
        _safe_fname(setting_dir),
        run_tag,
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, exp_name, step_dirname, bench, run_tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dit_ckpt", type=str, default=None)
    parser.add_argument("--vae_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=None)

    parser.add_argument("--content_ref", type=str, default=None)
    parser.add_argument("--timbre_ref", type=str, default=None)
    parser.add_argument("--sample_id", type=str, default=None)

    parser.add_argument("--num_step", type=int, default=None)
    parser.add_argument("--cfg_w", nargs=2, type=float, default=None)
    parser.add_argument("--timestep_annealing_w", nargs=3, type=float, default=None)
    parser.add_argument("--use_amo_sampler", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_sway", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max_timbre_sec", type=float, default=None)
    parser.add_argument("--pad_silence_sec", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg, samples = load_samples(args)
    dit_ckpt = require_value(resolve_cli_or_cfg(args.dit_ckpt, cfg, "dit_ckpt"), "dit_ckpt")
    vae_ckpt = resolve_cli_or_cfg(args.vae_ckpt, cfg, "vae_ckpt")
    device = resolve_cli_or_cfg(args.device, cfg, "device", "cuda:0")
    out_path = resolve_cli_or_cfg(args.out_path, cfg, "out_path", "infer_vc_outputs")
    seed = int(resolve_cli_or_cfg(args.seed, cfg, "seed", 1234))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    args.dit_ckpt, args.vae_ckpt, args.out_path = dit_ckpt, vae_ckpt, out_path
    out_dir, exp_name, step_dirname, bench, run_tag = make_output_paths(args, cfg)

    num_step = int(resolve_cli_or_cfg(args.num_step, cfg, "num_step", 40))
    cfg_w = tuple(float(x) for x in resolve_cli_or_cfg(args.cfg_w, cfg, "cfg_w", [1.5, 3.0]))
    timestep_annealing_w = tuple(float(x) for x in resolve_cli_or_cfg(args.timestep_annealing_w, cfg, "timestep_annealing_w", [0.6, 0.6, 1.0]))
    use_amo_sampler = bool(resolve_cli_or_cfg(args.use_amo_sampler, cfg, "use_amo_sampler", False))
    use_sway = bool(resolve_cli_or_cfg(args.use_sway, cfg, "use_sway", True))
    max_timbre_sec = float(resolve_cli_or_cfg(args.max_timbre_sec, cfg, "max_timbre_sec", 15.0))
    pad_silence_sec = float(resolve_cli_or_cfg(args.pad_silence_sec, cfg, "pad_silence_sec", 0.0))

    infer = SwanVCInfer(device=device, dit_ckpt=dit_ckpt, vae_ckpt=vae_ckpt)

    group_infos: Dict[str, List[Dict[str, Any]]] = {}
    all_results = []

    for i, sample in enumerate(samples):
        group = sample.get("__group__", "default")
        group_infos.setdefault(group, [])

        sample_id = (
            sample.get("id")
            or sample.get("name")
            or sample.get("utt_id")
            or f"{group}_{int(sample.get('__group_idx__', i)):04d}"
        )
        content_ref = os.path.realpath(os.path.expanduser(sample["content_ref"]))
        timbre_ref = os.path.realpath(os.path.expanduser(sample["timbre_ref"]))

        print(f"[{i+1}/{len(samples)}] sample_id={sample_id}")
        print(f"  content_ref: {content_ref}")
        print(f"  timbre_ref : {timbre_ref}")

        wav, meta = infer.convert(
            content_ref=content_ref,
            timbre_ref=timbre_ref,
            num_step=int(sample.get("num_step", num_step)),
            cfg_w=tuple(sample.get("cfg_w", cfg_w)),
            timestep_annealing_w=tuple(sample.get("timestep_annealing_w", timestep_annealing_w)),
            use_amo_sampler=bool(sample.get("use_amo_sampler", use_amo_sampler)),
            use_sway=bool(sample.get("use_sway", use_sway)),
            max_timbre_sec=float(sample.get("max_timbre_sec", max_timbre_sec)),
            pad_silence_sec=float(sample.get("pad_silence_sec", pad_silence_sec)),
        )

        out_name = f"{i:04d}_{_safe_fname(sample_id)}.wav"
        out_wav = os.path.join(out_dir, out_name)
        sf.write(out_wav, wav, infer.sr, "PCM_16")

        meta["sample_id"] = sample_id
        meta["content_ref"] = content_ref
        meta["timbre_ref"] = timbre_ref
        meta["output_wav"] = out_wav
        meta["group"] = group
        meta["group_idx"] = int(sample.get("__group_idx__", i))

        meta_fp = os.path.join(out_dir, f"{i:04d}_{_safe_fname(sample_id)}.json")
        with open(meta_fp, "w", encoding="utf-8") as f:
            json.dump(_json_ready(meta), f, ensure_ascii=False, indent=2)

        info = {
            "sample_id": sample_id,
            "content_ref": content_ref,
            "timbre_ref": timbre_ref,
            "output_wav": out_wav,
            "meta": meta,
        }
        group_infos[group].append(info)
        all_results.append(info)

        print(f"  output_wav : {out_wav}")
        print(f"  meta_json  : {meta_fp}")

    sub_dir, bucket = upload_group_audios_to_tos(group_infos, run_tag)

    desc_lines = [
        f"Run time: {run_tag}",
        f"Output dir: {os.path.realpath(out_dir)}",
        f"Samples: {len(samples)}",
        f"Seed: {seed}",
        f"cfg_w: {cfg_w}",
        f"timestep_annealing_w: {timestep_annealing_w}",
        f"use_amo_sampler: {use_amo_sampler}",
        f"use_sway: {use_sway}",
        f"num_step: {num_step}",
        f"max_timbre_sec: {max_timbre_sec}",
        f"pad_silence_sec: {pad_silence_sec}",
        _ckpt_report_line("DiT", args.dit_ckpt),
        _ckpt_report_line("VAE", args.vae_ckpt or hparams.get("vae_ckpt")),
        f"semantic_token_type: {hparams.get('semantic_token_type', 'content_style')}",
    ]
    desc = "\n".join(desc_lines)

    title_name = "/".join([exp_name, step_dirname, str(bench)])
    html_fp = os.path.join(out_dir, f"report_{_safe_fname(bench)}.html")
    gen_local_audio_html(group_infos=group_infos, output_fp=html_fp, title_name=title_name, extra_desc=desc)
    html_tos = send_file_to_tos(html_fp, sub_dir=sub_dir, bucket=bucket)

    results_fp = os.path.join(out_dir, "results.json")
    with open(results_fp, "w", encoding="utf-8") as f:
        json.dump(_json_ready(all_results), f, ensure_ascii=False, indent=2)

    print(f"[DONE] html report: {html_fp}")
    print(f"[DONE] html tos   : {html_tos}")
    print(f"[DONE] results json: {results_fp}")
    print(f"[DONE] output dir   : {out_dir}")


if __name__ == "__main__":
    main()


# python inference/vc/swanvc_infer.py \
#   --dit_ckpt checkpoints/260410_swanvc_vevo \
#   --config egs/vc/inference/swanvc_vevo_infer.yaml

