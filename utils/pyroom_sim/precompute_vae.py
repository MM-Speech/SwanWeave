#!/usr/bin/env python3
"""Save FOA VAE mean/std beside audio referenced by rendered metadata.

Run from the SwanWeave root:
    python utils/pyroom_sim/precompute_vae.py
    python utils/pyroom_sim/precompute_vae.py --metadata /data/rendered_metadata/edit_train.jsonl
Only posterior PT files are written; input metadata is never changed.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.spatial.build_model_utils import build_stable_audio_vae
from tasks.dataset_utils.spat_base_fastdataset import (
    _default_posterior_path,
    _load_spatial_audio_preserve_channels,
    _load_vae_posterior,
)
from tasks.spat_audio_edit_task import (
    build_stereo_from_foa,
    preprocess_stable_audio_batch,
)
from utils.commons.hparams import hparams, set_hparams
from utils.spatial_audio_editing.data_paths import DATA_ROOT
from utils.spatial_audio_editing.src.progress import Progress


MANIFEST_NAMES = (
    "edit_train.jsonl", "edit_test.jsonl",
    "gen_train.jsonl", "gen_test.jsonl",
    "dpo_train.jsonl", "dpo_test.jsonl",
)
AUDIO_KEYS = ("wav_path", "pos_wav_path", "src_wav_path", "neg_wav_path")


def collect_audio(metadata, data_root, limit, progress):
    """Deduplicate audio shared by instruction variants and different manifests."""
    paths = {}
    for manifest in metadata:
        with manifest.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                audio = [row[key] for key in AUDIO_KEYS if row.get(key)]
                if not audio:
                    raise ValueError(f"{manifest}:{line_number}: no FOA audio paths")
                for value in audio:
                    path = Path(value).expanduser()
                    if not path.is_absolute():
                        path = data_root / path
                    path = path.resolve()
                    if not path.is_file():
                        raise FileNotFoundError(f"{manifest}:{line_number}: {path}")
                    paths.setdefault(path, None)
                    if limit and len(paths) >= limit:
                        return list(paths)
                progress.update()
    return list(paths)


@torch.inference_mode()
def encode_posterior(vae, vae_info, path, device):
    sample_rate = int(hparams["audio_sample_rate"])
    wav = _load_spatial_audio_preserve_channels(str(path), sample_rate)
    if wav is None or wav.ndim != 2 or wav.shape[1] != 4:
        raise ValueError(f"Expected readable four-channel FOA: {path}")
    # Match the training dataset's waveform alignment before VAE preprocessing.
    multiple = int(hparams["frames_multiple"]) * int(hparams["hop_size"])
    if multiple > 0:
        wav = wav[:len(wav) // multiple * multiple]
    if len(wav) == 0:
        raise ValueError(f"Empty audio after alignment: {path}")
    wav_len = len(wav)
    wav = wav.unsqueeze(0).to(device=device, dtype=next(vae.parameters()).dtype)
    pairs = (
        hparams.get("stable_audio_foa_pair_a", [0, 1]),
        hparams.get("stable_audio_foa_pair_b", [2, 3]),
    )
    audio = torch.cat([
        preprocess_stable_audio_batch(
            vae, build_stereo_from_foa(wav, tuple(pair)), sample_rate,
            int(vae_info["sample_rate"]), int(vae_info["downsampling_ratio"]),
        )
        for pair in pairs
    ], dim=0)
    # Keep the distribution so training can draw a fresh latent on each visit.
    distribution = getattr(vae.encode(audio), "latent_dist", None)
    if distribution is None:
        raise ValueError("Posterior extraction requires a VAE exposing latent_dist "
                         "(the released diffusers AutoencoderOobleck).")
    mean = torch.cat(distribution.mean.split(1, dim=0), dim=1)
    std = torch.cat(distribution.var.clamp_min(0).sqrt().split(1, dim=0), dim=1)
    mean = mean.squeeze(0).transpose(0, 1).float().contiguous().cpu()
    std = std.squeeze(0).transpose(0, 1).float().contiguous().cpu()
    if mean.ndim != 2 or mean.shape[1] != int(hparams["latent_dim"]):
        raise ValueError(f"Incompatible latent shape {tuple(mean.shape)}: {path}")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError(f"Non-finite posterior: {path}")
    stat = path.stat()
    return {
        "mean": mean, "std": std,
        "latent_len": int(mean.shape[0]), "latent_dim": int(mean.shape[1]),
        "wav_len": wav_len, "sample_rate": sample_rate,
        "vae_sample_rate": int(vae_info["sample_rate"]),
        "downsampling_ratio": int(vae_info["downsampling_ratio"]),
        "wav_path": str(path),
        "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
        "channel_pairs": [list(pair) for pair in pairs],
    }


def save_posterior(destination, payload):
    """Publish a complete PT file; interrupted writes never look complete."""
    with NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                            delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, nargs="+",
                        help="JSONL files; defaults to the six rendered training lists.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--config", default="egs/spat_audio_edit.yaml")
    parser.add_argument("--hparams", default="")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, help="Maximum number of unique audio files.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main():
    args = parse_args()
    set_hparams(config=args.config, hparams_str=args.hparams, print_hparams=False)
    root = args.data_root.expanduser().resolve()
    metadata = args.metadata
    if metadata is None:
        metadata = [root / "rendered_metadata" / name for name in MANIFEST_NAMES]
        metadata = [path for path in metadata if path.is_file()]
    if not metadata:
        raise FileNotFoundError(f"No training lists found in {root / 'rendered_metadata'}")
    suffix = str(hparams.get("precomputed_latent_suffix", ".spat_vae_posterior.pt"))
    with Progress(desc="Reading rendered metadata", unit="rows") as progress:
        audio_paths = collect_audio(metadata, root, args.limit, progress)
        progress.reset(total=len(audio_paths), desc="Encoding FOA VAE posteriors", unit="audio")
        vae = None
        counts = {"encoded": 0, "skipped": 0}
        for path in audio_paths:
            destination = Path(_default_posterior_path(str(path), suffix))
            if destination.exists() and not args.overwrite:
                if _load_vae_posterior(str(destination)) is None:
                    raise ValueError(f"Unreadable posterior; regenerate with --overwrite: {destination}")
                counts["skipped"] += 1
            else:
                if vae is None:
                    vae, info = build_stable_audio_vae(torch.device(args.device))
                save_posterior(destination, encode_posterior(vae, info, path, args.device))
                counts["encoded"] += 1
            progress.update()
            progress.set_postfix(**counts)
        progress.write(f"Unique audio: {len(audio_paths)}; encoded: {counts['encoded']}; "
                       f"skipped: {counts['skipped']}")


if __name__ == "__main__":
    main()
