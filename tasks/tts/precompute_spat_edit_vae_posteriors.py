#!/usr/bin/env python3
"""Precompute FOA Stable Audio VAE posterior stats for spat edit training."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from tempfile import NamedTemporaryFile
from typing import Iterable

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.tts.spat_edit.build_model_utils import build_stable_audio_vae
from tasks.tts.dataset_utils.spat_base_fastdataset import (
    _default_posterior_path,
    _load_spatial_audio_preserve_channels,
)
from tasks.tts.spat_audio_edit_task import (
    build_stereo_from_foa,
    pad_latent_time,
    preprocess_stable_audio_batch,
)
from utils.commons.hparams import hparams, set_hparams


DEFAULT_MANIFEST = Path(
    "/mnt/bn/sa-ag-data/leike/spatial_edit/triplet/metadata_training/audio_edit_train.jsonl"
)
DEFAULT_CONFIG = "egs/tts/spat_audio_edit.yaml"
DEFAULT_SUFFIX = ".spat_vae_posterior.pt"


@dataclass(frozen=True)
class WavJob:
    wav_path: str
    posterior_path: str


TARGET_AUDIO_KEYS = (
    "wav_path",
    "target_wav_path",
    "pos_wav_path",
    "positive_wav_path",
    "chosen_wav_path",
    "target_audio",
    "target_path",
    "audio_path",
)
SOURCE_AUDIO_KEYS = (
    "src_wav_path",
    "source_wav_path",
    "orig_wav_path",
    "input_wav_path",
    "source_audio",
    "source_path",
)
NEGATIVE_AUDIO_KEYS = (
    "neg_wav_path",
    "negative_wav_path",
    "rejected_wav_path",
    "bad_wav_path",
    "negative_audio",
    "negative_path",
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def collect_jobs(rows: list[dict], suffix: str, limit: int | None = None) -> list[WavJob]:
    seen = set()
    jobs = []
    for row in rows:
        for keys in (TARGET_AUDIO_KEYS, SOURCE_AUDIO_KEYS, NEGATIVE_AUDIO_KEYS):
            wav_path = None
            for key in keys:
                wav_path = row.get(key)
                if wav_path:
                    break
            if not wav_path or wav_path in seen:
                continue
            seen.add(wav_path)
            jobs.append(WavJob(wav_path=str(wav_path), posterior_path=_default_posterior_path(wav_path, suffix)))
            if limit is not None and len(jobs) >= limit:
                return jobs
    return jobs


def with_latent_paths(rows: list[dict], suffix: str) -> list[dict]:
    out_rows = []
    for row in rows:
        out = dict(row)
        target_wav_path = None
        src_wav_path = None
        neg_wav_path = None
        for key in TARGET_AUDIO_KEYS:
            target_wav_path = out.get(key)
            if target_wav_path:
                break
        for key in SOURCE_AUDIO_KEYS:
            src_wav_path = out.get(key)
            if src_wav_path:
                break
        for key in NEGATIVE_AUDIO_KEYS:
            neg_wav_path = out.get(key)
            if neg_wav_path:
                break

        if target_wav_path:
            out["wav_path"] = str(target_wav_path)
            out["target_wav_path"] = str(target_wav_path)
            out["pos_wav_path"] = str(target_wav_path)
            target_latent_path = _default_posterior_path(target_wav_path, suffix)
            out["latent_path"] = target_latent_path
            out["pos_latent_path"] = target_latent_path
            out["positive_latent_path"] = target_latent_path
            out["chosen_latent_path"] = target_latent_path
            out["target_latent_path"] = target_latent_path
        if src_wav_path:
            out["src_wav_path"] = str(src_wav_path)
            out["source_wav_path"] = str(src_wav_path)
            src_latent_path = _default_posterior_path(src_wav_path, suffix)
            out["src_latent_path"] = src_latent_path
            out["source_latent_path"] = src_latent_path
            out["src_vae_posterior_path"] = src_latent_path
        if neg_wav_path:
            out["neg_wav_path"] = str(neg_wav_path)
            out["negative_wav_path"] = str(neg_wav_path)
            neg_latent_path = _default_posterior_path(neg_wav_path, suffix)
            out["neg_latent_path"] = neg_latent_path
            out["negative_latent_path"] = neg_latent_path
            out["rejected_latent_path"] = neg_latent_path
            out["bad_latent_path"] = neg_latent_path
        out_rows.append(out)
    return out_rows


def _latent_dist_from_vae(vae, audio: torch.Tensor):
    encoded = vae.encode(audio)
    latent_dist = getattr(encoded, "latent_dist", None)
    if latent_dist is None:
        raise RuntimeError("The configured VAE does not expose latent_dist; posterior precompute requires diffusers AutoencoderOobleck.")
    if not hasattr(latent_dist, "mean") or not hasattr(latent_dist, "var"):
        raise RuntimeError("latent_dist must expose mean and var.")
    return latent_dist


@torch.inference_mode()
def encode_foa_posterior(vae, wav: torch.Tensor, input_sample_rate: int, vae_sample_rate: int, downsampling_ratio: int) -> dict:
    wav = wav.unsqueeze(0)
    audio_a = build_stereo_from_foa(wav, tuple(hparams.get("stable_audio_foa_pair_a", [0, 1])))
    audio_b = build_stereo_from_foa(wav, tuple(hparams.get("stable_audio_foa_pair_b", [2, 3])))
    audio_a = preprocess_stable_audio_batch(vae, audio_a, input_sample_rate, vae_sample_rate, downsampling_ratio)
    audio_b = preprocess_stable_audio_batch(vae, audio_b, input_sample_rate, vae_sample_rate, downsampling_ratio)
    audio = torch.cat([audio_a, audio_b], dim=0)

    latent_dist = _latent_dist_from_vae(vae, audio)
    mean_a, mean_b = latent_dist.mean.split(1, dim=0)
    var_a, var_b = latent_dist.var.split(1, dim=0)

    target_len = min(int(mean_a.shape[-1]), int(mean_b.shape[-1]))
    mean = torch.cat([pad_latent_time(mean_a, target_len), pad_latent_time(mean_b, target_len)], dim=1)
    var = torch.cat([pad_latent_time(var_a, target_len), pad_latent_time(var_b, target_len)], dim=1)
    return {
        "mean": mean.squeeze(0).transpose(0, 1).contiguous().cpu().float(),
        "var": var.squeeze(0).transpose(0, 1).contiguous().cpu().float(),
    }


def save_atomic(path: str, payload: dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path_obj.parent, prefix=f".{path_obj.name}.", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path_obj)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def precompute_one(job: WavJob, vae, vae_sample_rate: int, downsampling_ratio: int, overwrite: bool) -> str:
    if not overwrite and Path(job.posterior_path).exists():
        return "skipped"

    wav = _load_spatial_audio_preserve_channels(job.wav_path, int(hparams["audio_sample_rate"]))
    if wav is None:
        raise RuntimeError(f"failed to load wav: {job.wav_path}")

    fm_wav = int(hparams["frames_multiple"]) * int(hparams["hop_size"])
    if fm_wav > 0:
        valid_len = (int(wav.shape[0]) // fm_wav) * fm_wav
        wav = wav[:valid_len]
    if wav.numel() == 0 or int(wav.shape[0]) <= 0:
        raise RuntimeError(f"empty wav after trimming: {job.wav_path}")

    device = next(vae.parameters()).device
    posterior = encode_foa_posterior(
        vae,
        wav.to(device=device, dtype=torch.float32),
        int(hparams["audio_sample_rate"]),
        int(vae_sample_rate),
        int(downsampling_ratio),
    )
    payload = {
        "version": 1,
        "format": "spat_edit_foa_vae_posterior",
        "wav_path": job.wav_path,
        "sample_rate": int(hparams["audio_sample_rate"]),
        "vae_sample_rate": int(vae_sample_rate),
        "downsampling_ratio": int(downsampling_ratio),
        "wav_len": int(wav.shape[0]),
        "latent_len": int(posterior["mean"].shape[0]),
        "latent_dim": int(posterior["mean"].shape[1]),
        "mean": posterior["mean"],
        "var": posterior["var"],
    }
    save_atomic(job.posterior_path, payload)
    return "done"


def worker(rank: int, args: argparse.Namespace, jobs: list[WavJob], progress_queue=None) -> None:
    set_hparams(config=args.config, hparams_str=args.hparams, print_hparams=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-GPU VAE precompute.")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    vae, vae_info = build_stable_audio_vae(device=device)
    vae.eval().requires_grad_(False)

    shard = jobs[rank:: args.num_gpus]
    counts = {"done": 0, "skipped": 0, "failed": 0}
    for idx, job in enumerate(shard, start=1):
        status = "failed"
        error = ""
        try:
            status = precompute_one(
                job,
                vae,
                int(vae_info["sample_rate"]),
                int(vae_info["downsampling_ratio"]),
                bool(args.overwrite),
            )
            counts[status] = counts.get(status, 0) + 1
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            counts["failed"] += 1
            if progress_queue is not None:
                progress_queue.put((status, rank, job.wav_path, job.posterior_path, error))
            if args.strict:
                raise
            continue

        if progress_queue is not None:
            progress_queue.put((status, rank, job.wav_path, job.posterior_path, error))
        elif idx == 1 or idx % int(args.log_interval) == 0 or idx == len(shard):
            print(
                f"[rank {rank}] {idx}/{len(shard)} done={counts['done']} skipped={counts['skipped']} failed={counts['failed']}",
                flush=True,
            )


def run_workers(args: argparse.Namespace, jobs: list[WavJob]) -> dict[str, int]:
    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes = [
        ctx.Process(target=worker, args=(rank, args, jobs, progress_queue))
        for rank in range(int(args.num_gpus))
    ]
    for process in processes:
        process.start()

    counts = {"done": 0, "skipped": 0, "failed": 0}
    completed = 0
    with tqdm(total=len(jobs), desc="precompute VAE posterior", unit="wav", dynamic_ncols=True) as pbar:
        while completed < len(jobs):
            try:
                status, rank, wav_path, posterior_path, error = progress_queue.get(timeout=1.0)
            except Empty:
                if not any(process.is_alive() for process in processes):
                    break
                continue

            completed += 1
            counts[status] = counts.get(status, 0) + 1
            pbar.update(1)
            pbar.set_postfix(done=counts["done"], skipped=counts["skipped"], failed=counts["failed"])
            if status == "failed":
                tqdm.write(f"[rank {rank}] failed: {wav_path} -> {posterior_path}: {error}")

    for process in processes:
        process.join()

    failed_processes = [(idx, process.exitcode) for idx, process in enumerate(processes) if process.exitcode != 0]
    if failed_processes:
        raise RuntimeError(f"worker process failed: {failed_processes}")
    if completed != len(jobs):
        raise RuntimeError(f"progress incomplete: completed={completed}, expected={len(jobs)}")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute spat edit FOA VAE posterior mean/var beside each wav.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--hparams", "-hp", default="")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    jobs = collect_jobs(rows, args.suffix, args.limit)
    output_manifest = args.output_manifest
    if output_manifest is None:
        output_manifest = args.manifest.with_name(f"{args.manifest.stem}_latent{args.manifest.suffix}")

    print(f"manifest={args.manifest}")
    print(f"rows={len(rows)} unique_wavs={len(jobs)} num_gpus={args.num_gpus}")
    print(f"posterior_suffix={args.suffix}")
    print(f"output_manifest={output_manifest}")

    if args.dry_run:
        for job in jobs[:20]:
            print(f"{job.wav_path} -> {job.posterior_path}")
        return 0

    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be >= 1")
    counts = run_workers(args, jobs)

    write_jsonl(output_manifest, with_latent_paths(rows, args.suffix))
    print(f"summary: done={counts['done']} skipped={counts['skipped']} failed={counts['failed']}")
    print(f"wrote {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
