#!/usr/bin/env python3
"""
Spatial Audio Editing — single-operation triplet generation.

Generates (caption, original_scene, target_scene) triplets WITHOUT rendering audio.
Scene events carry relative audio_path so downstream renderers can locate the clips.

Usage:
  python pipeline.py                                       # configs/audio_only.json
  python pipeline.py --config configs/speech_audio.json
  python pipeline.py --config configs/audio_only.json --test-set
  python pipeline.py --config configs/speech_audio.json --test-set --test-size 100
"""

import argparse
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.spatial_audio_editing.src.assets import REPO_ROOT, load_assets, load_config, stable_int_hash
from utils.spatial_audio_editing.src.caption import CaptionGenerator, scene_to_text
from utils.spatial_audio_editing.src.scene import DYNAMIC_EDIT_TYPES, fit_scene_duration
from utils.spatial_audio_editing.src.progress import Progress
from utils.spatial_audio_editing.src.writer import append_result, append_text_result, load_completed_tasks

DEFAULT_CONFIG = REPO_ROOT / "configs" / "audio_only.json"


# ============================================================================
# Config
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate spatial audio editing triplets")
    parser.add_argument("config_path", nargs="?", help="Config JSON path")
    parser.add_argument("--config", dest="config_flag", help="Config JSON path")
    parser.add_argument("--output-dir", help="Override cfg['output']['dir']")
    parser.add_argument("--samples-per-type", type=int, default=20,
                        help="Override cfg['samples_per_type'] with one default count")
    parser.add_argument("--seed", type=int, help="Override cfg['generation']['seed']")
    parser.add_argument("--num-workers", type=int, help="Override num_workers")
    parser.add_argument("--test-set", action="store_true",
                        help="Generate test scenes under the test paths in data_paths.py")
    parser.add_argument("--test-size", type=int, default=50,
                        help="Samples per edit type for --test-set (default: 50)")
    parser.add_argument("--test-output-root", default=None,
                        help="Override the central test-scene root (one domain directory is appended)")
    return parser.parse_args()


def infer_domain(cfg: dict, config_path: str = None) -> str:
    if cfg.get("domain"):
        return cfg["domain"]
    output_dir = cfg.get("output", {}).get("dir", "")
    if output_dir:
        return Path(output_dir).name
    if config_path:
        name = Path(config_path).stem
        return name[:-5] if name.endswith("_test") else name
    return "data"


def apply_cli_overrides(cfg: dict, args, config_path: str = None):
    cfg.setdefault("generation", {})
    cfg.setdefault("output", {})

    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    if args.test_set:
        domain = infer_domain(cfg, config_path)
        if args.test_output_root is not None:
            cfg["output"]["dir"] = str(Path(args.test_output_root) / domain)
        cfg["samples_per_type"] = {"default": args.samples_per_type or args.test_size}
        if args.seed is None:
            # Vary scenes; raw-clip separation requires separately split source data.
            cfg["generation"]["seed"] = cfg["generation"].get(
                "test_seed", cfg["generation"].get("seed", 42) + 100000)

    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir
    if args.samples_per_type is not None and not args.test_set:
        cfg["samples_per_type"] = {"default": args.samples_per_type}
    if args.seed is not None:
        cfg["generation"]["seed"] = args.seed


def clear_output_jsonl(cfg: dict, progress=None):
    out_dir = Path(cfg["output"]["dir"])
    if not out_dir.exists():
        return
    removed = 0
    for jsonl_file in out_dir.glob("*.jsonl"):
        jsonl_file.unlink()
        removed += 1
    if removed:
        message = f"  Cleared {removed} existing JSONL files from {out_dir}"
        (progress.write if progress is not None else print)(message)


def samples_per_type(cfg: dict) -> dict:
    spt = cfg["samples_per_type"]
    if isinstance(spt, int):
        return {et: spt for et in cfg["edit_types"]}
    return {et: spt.get(et, spt.get("default", 20)) for et in cfg["edit_types"]}


# ============================================================================
# Worker (one SceneGenerator / CaptionGenerator per process)
# ============================================================================

_CTX = {}


def _init_worker(cfg: dict, assets):
    _CTX["scene_gen"] = assets.scene_generator(cfg)
    _CTX["caption_gen"] = CaptionGenerator(cfg)


def _worker(task: Tuple[str, int, int]) -> Optional[dict]:
    edit_type, sample_idx, seed = task
    random.seed(seed)
    np.random.seed(seed)

    scene_gen = _CTX["scene_gen"]
    # room_change samples start from a random room; all other edits use the default room.
    if edit_type == "room_change":
        scene_gen.set_sample_room(random.choice(scene_gen.room_names))
    else:
        scene_gen.set_sample_room(scene_gen.base_room)

    dynamic = edit_type in DYNAMIC_EDIT_TYPES
    if dynamic:
        duration = round(random.uniform(5.5, scene_gen.dur_max), 1)
        n_events = random.randint(1, 3)
    else:
        duration = round(random.uniform(scene_gen.dur_min, scene_gen.dur_max), 1)
        n_events = random.randint(scene_gen.evt_min, scene_gen.evt_max)
    if edit_type == "remove_event":
        n_events = max(2, n_events)

    for _ in range(5):
        original = scene_gen.generate_scene(duration, n_events, use_long=dynamic)
        if len(original) >= 2 or edit_type != "remove_event":
            break
    else:
        return None

    room = scene_gen.default_room
    room_dims = scene_gen.room_sizes[room]
    target, edit_info = scene_gen.apply_edit(original, edit_type, duration)
    if edit_type == "room_change":
        # Keep source/target geometry identical inside both rooms, with a wall margin.
        axes = ("x", "y", "z")
        common_bounds = {
            axis: min(room_dims[axis], edit_info["new_dims"][axis])
            for axis in axes
        }
        limits = {axis: common_bounds[axis] / 2.0 - 0.1 for axis in axes}
        target_events = {event["event_id"]: event for event in target}
        for event in original:
            if all(abs(event["position"][axis]) <= limits[axis] for axis in axes):
                continue
            for _ in range(200):
                position = dict(zip(axes, scene_gen._random_position(
                    room_bounds=common_bounds, y_range=(0.0, 0.0),
                )))
                if all(abs(position[axis]) <= limits[axis] for axis in axes):
                    break
            else:
                return None
            event["position"] = position
            target_events[event["event_id"]]["position"] = position.copy()
    # NB: captions and scene_to_text both draw from `random`; keep this order for reproducibility.
    captions = _CTX["caption_gen"].generate(original, target, edit_info, room=room, room_dims=room_dims)

    result = {
        "sample_id": f"{sample_idx:08d}",
        "edit_type": edit_type,
        "duration": duration,
        "room": room,
        "room_dims": room_dims,
        "captions": captions,
        "original_scene": original,
        "target_scene": target,
        "edit_info": edit_info,
        "original_text": scene_to_text(original),
        "target_text": scene_to_text(target),
    }
    if edit_type == "room_change":
        result["target_room"] = edit_info["new_room"]
        result["target_room_dims"] = edit_info["new_dims"]
    # Fit last, after caption RNG calls; preserve all event/edit-info timestamps.
    result["duration"] = fit_scene_duration(duration, original, target, edit_info)
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    config_path = (args.config_flag or args.config_path
                   or os.environ.get("PIPELINE_CONFIG", str(DEFAULT_CONFIG)))
    cfg = load_config(config_path, split="test" if args.test_set else "train")
    apply_cli_overrides(cfg, args, config_path)

    seed = cfg["generation"].get("seed", 42)
    n_workers = cfg.get("num_workers", min(8, mp.cpu_count()))

    print("=" * 60)
    print("Spatial Audio Editing — Triplet Generation")
    print(f"  Config: {config_path}")
    print(f"  Output: {cfg['output']['dir']}")
    if args.test_set:
        print(f"  Mode: test set ({cfg['samples_per_type']['default']} samples/type)")
    print(f"  Caption mode: {cfg['api'].get('caption_mode', 'api')}")
    print(f"  Workers: {n_workers}")
    print("=" * 60)

    with Progress(desc="Loading assets", unit="clips") as progress:
        if args.test_set:
            clear_output_jsonl(cfg, progress=progress)

        assets = load_assets(cfg, progress=progress)

        # Build task list (resumes from existing output).
        per_type = samples_per_type(cfg)
        progress.reset(desc="Checking completed tasks", unit="files")
        completed_tasks = load_completed_tasks(cfg, progress=progress)
        base_seed = seed * 10000
        tasks = [
            (etype, i, base_seed + stable_int_hash(etype, 100000) + i)
            for etype, n in per_type.items()
            for i in range(n)
            if (etype, i) not in completed_tasks
        ]

        total_all = sum(per_type.values())
        skipped = total_all - len(tasks)
        if skipped:
            progress.write(f"Resuming: {skipped}/{total_all} already done, {len(tasks)} remaining")
        else:
            progress.write(f"Building {total_all} tasks over {len(per_type)} edit types")

        progress.reset(total=len(tasks), desc="Generating triplets", unit="tasks")
        completed = failed = 0
        progress.set_postfix(generated=completed, failed=failed, resumed=skipped)
        if not tasks:
            progress.write("All tasks already completed.")
            return

        with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(cfg, assets)) as pool:
            for result in pool.imap_unordered(_worker, tasks):
                if result is None:
                    failed += 1
                else:
                    append_result(result, cfg)
                    append_text_result(result, cfg)
                    completed += 1
                progress.update(1)
                progress.set_postfix(generated=completed, failed=failed, resumed=skipped)

        progress.write(f"Generated {completed} triplets")
        if failed:
            progress.write(f"Skipped {failed} tasks after max attempts")

    print("\nDone.")


if __name__ == "__main__":
    main()
