#!/usr/bin/env python3
"""
Generate multi-operation spatial audio editing triplets.

Reuses the single-operation scene/edit primitives, but applies 2/3/4 edits
sequentially to one original scene and writes positive examples.
DPO negatives are generated separately with generate_dpo_negatives.py.

Usage:
  python generate_multi_operations.py --config configs/audio_only.json
  python generate_multi_operations.py --config configs/speech_audio.json
  python generate_multi_operations.py --config configs/audio_only.json --counts 2:100,3:50,4:25
"""

import argparse
import copy
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.spatial_audio_editing import data_paths

from utils.spatial_audio_editing.src.assets import load_assets, load_config, stable_int_hash
from utils.spatial_audio_editing.src.caption import _fallback_caption, join_clauses, scene_to_text
from utils.spatial_audio_editing.src.scene import DYNAMIC_EDIT_TYPES, EDIT_TYPES, fit_scene_duration
from utils.spatial_audio_editing.src.progress import Progress
from utils.spatial_audio_editing.src.writer import append_result, append_text_result, load_completed_tasks

# Global edits (extract_event, room_change) are excluded by default.
DEFAULT_ALLOWED_EDIT_TYPES = [t for t in EDIT_TYPES if t not in ("extract_event", "room_change")]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate multi-operation triplets")
    parser.add_argument("--config", required=True, help="Base single-operation config JSON")
    parser.add_argument("--output-dir", help="Override output directory (default: <output.dir>_multi)")
    parser.add_argument("--counts", default="2:200000,3:100000,4:50000",
                        help="Operation-count targets, e.g. 2:200000,3:100000,4:50000")
    parser.add_argument("--allowed-edit-types",
                        help="Comma-separated edit types. Defaults to all non-global edits.")
    parser.add_argument("--seed", type=int, help="Override cfg['generation']['seed']")
    parser.add_argument("--test-set", action="store_true",
                        help="Use central test paths; set --counts for the desired test sizes")
    parser.add_argument("--num-workers", type=int, help="Override num_workers")
    parser.add_argument("--max-attempts", type=int, default=80,
                        help="Attempts per sample before skipping (default: 80)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Remove existing multi_* JSONL files for requested counts first")
    return parser.parse_args()


def parse_counts(raw: str) -> dict:
    counts = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, value = part.split(":", 1)
        op_count = int(key.strip())
        if op_count < 2:
            raise ValueError("Multi-operation counts must be >= 2")
        counts[op_count] = int(value.strip())
    if not counts:
        raise ValueError("No valid counts provided")
    return counts


def apply_overrides(cfg: dict, args, counts: dict):
    cfg.setdefault("generation", {})
    cfg.setdefault("output", {})
    split = "test" if getattr(args, "test_set", False) else "train"
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir
    else:
        domain = cfg.get("domain") or Path(cfg["output"]["dir"]).name.removesuffix("_multi")
        if domain not in data_paths.MULTI_SCENE_DIRS:
            raise ValueError("Cannot identify the scene domain; provide --output-dir")
        cfg["output"]["dir"] = str(data_paths.MULTI_SCENE_DIRS[domain][split])
    if args.seed is not None:
        cfg["generation"]["seed"] = args.seed
    elif split == "test":
        cfg["generation"]["seed"] = cfg["generation"].get(
            "test_seed", cfg["generation"].get("seed", 42) + 100000)
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    cfg["samples_per_type"] = {f"multi_{n}op": count for n, count in counts.items()}


def clear_outputs(output_dir: Path, counts: dict, progress=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    for op_count in counts:
        stem = f"multi_{op_count}op"
        for path in (output_dir / f"{stem}.jsonl", output_dir / f"{stem}_text.jsonl"):
            if path.exists():
                path.unlink()
                (progress.write if progress is not None else print)(f"  Removed {path}")


def generate_multi_captions(edit_infos: List[dict]) -> List[str]:
    """Three captions, each joining one template caption per edit in a different style."""
    captions = []
    for variant in range(3):
        clauses = []
        for info in edit_infos:
            choices = _fallback_caption(info)
            clauses.append(choices[variant % len(choices)])
        captions.append(join_clauses(clauses, variant))
    return captions


def scenes_equal(a, b) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ============================================================================
# Worker (one SceneGenerator per process)
# ============================================================================

_CTX = {}


def _init_worker(cfg: dict, assets, allowed_edit_types: List[str], max_attempts: int):
    _CTX["scene_gen"] = assets.scene_generator(cfg)
    _CTX["allowed_edit_types"] = list(allowed_edit_types)
    _CTX["max_attempts"] = max_attempts


def choose_edit_types(operation_count: int) -> List[str]:
    allowed = _CTX["allowed_edit_types"]
    if len(allowed) >= operation_count:
        return random.sample(allowed, operation_count)
    return random.choices(allowed, k=operation_count)


def _worker(task: Tuple[int, int, int]) -> Optional[dict]:
    operation_count, sample_idx, seed = task
    random.seed(seed)
    np.random.seed(seed)
    scene_gen = _CTX["scene_gen"]

    for _ in range(_CTX["max_attempts"]):
        edit_types = choose_edit_types(operation_count)
        has_dynamic = any(t in DYNAMIC_EDIT_TYPES for t in edit_types)
        if has_dynamic:
            duration = round(random.uniform(5.5, scene_gen.dur_max), 1)
        else:
            duration = round(random.uniform(scene_gen.dur_min, scene_gen.dur_max), 1)
        n_events = random.randint(max(2, scene_gen.evt_min), scene_gen.evt_max)

        original = scene_gen.generate_scene(duration, n_events, use_long=has_dynamic)
        if len(original) < 2 and "remove_event" in edit_types:
            continue

        # Apply edits in sequence; reject the attempt if any edit falls back to
        # a different type or is a no-op.
        current = copy.deepcopy(original)
        edit_infos = []
        for edit_type in edit_types:
            try:
                target, info = scene_gen.apply_edit(current, edit_type, duration)
            except Exception:
                break
            if info.get("type") != edit_type or scenes_equal(current, target):
                break
            edit_infos.append(info)
            current = target
        else:
            if scenes_equal(original, current):
                continue
            # NB: scene_to_text draws from `random`; keep this call order for reproducibility.
            original_text = scene_to_text(original)
            target_text = scene_to_text(current)
            result = {
                "sample_id": f"{sample_idx:08d}",
                "edit_type": f"multi_{operation_count}op",
                "operation_count": operation_count,
                "edit_types": edit_types,
                "duration": duration,
                "room": scene_gen.default_room,
                "room_dims": scene_gen.room_sizes[scene_gen.default_room],
                "captions": generate_multi_captions(edit_infos),
                "original_scene": original,
                "target_scene": current,
                "edit_info": edit_infos,
                "original_text": original_text,
                "target_text": target_text,
            }
            result["duration"] = fit_scene_duration(duration, original, current, edit_infos)
            return result

    return None


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    counts = parse_counts(args.counts)
    cfg = load_config(args.config, split="test" if args.test_set else "train")
    apply_overrides(cfg, args, counts)

    allowed_edit_types = (
        [s.strip() for s in args.allowed_edit_types.split(",") if s.strip()]
        if args.allowed_edit_types
        else DEFAULT_ALLOWED_EDIT_TYPES
    )

    seed = cfg["generation"].get("seed", 42)
    n_workers = cfg.get("num_workers", min(8, mp.cpu_count()))
    output_dir = Path(cfg["output"]["dir"])

    print("=" * 60)
    print("Spatial Audio Editing - Multi-Operation Generation")
    print(f"  Config: {args.config}")
    print(f"  Output: {output_dir}")
    print(f"  Counts: {counts}")
    print(f"  Allowed edit types: {allowed_edit_types}")
    print(f"  Workers: {n_workers}")
    print("=" * 60)

    with Progress(desc="Loading assets", unit="clips") as progress:
        if args.overwrite:
            clear_outputs(output_dir, counts, progress=progress)

        assets = load_assets(cfg, progress=progress)

        progress.reset(desc="Checking completed tasks", unit="files")
        completed_tasks = load_completed_tasks(cfg, progress=progress)
        base_seed = seed * 10000 + stable_int_hash("multi_operation", 100000)
        tasks = []
        for operation_count, target_count in counts.items():
            edit_type = f"multi_{operation_count}op"
            type_seed = base_seed + stable_int_hash(edit_type, 1000000)
            tasks += [(operation_count, i, type_seed + i)
                      for i in range(target_count) if (edit_type, i) not in completed_tasks]

        total_all = sum(counts.values())
        skipped = total_all - len(tasks)
        if skipped:
            progress.write(f"Resuming: {skipped}/{total_all} already done, {len(tasks)} remaining")
        else:
            progress.write(f"Building {total_all} multi-operation tasks")

        progress.reset(total=len(tasks), desc="Generating multi-operation triplets", unit="tasks")
        completed = failed = 0
        progress.set_postfix(generated=completed, failed=failed, resumed=skipped)
        if not tasks:
            progress.write("All tasks already completed.")
            return

        init_args = (cfg, assets, allowed_edit_types, args.max_attempts)
        with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=init_args) as pool:
            for result in pool.imap_unordered(_worker, tasks):
                if result is None:
                    failed += 1
                else:
                    append_result(result, cfg)
                    append_text_result(result, cfg)
                    completed += 1
                progress.update(1)
                progress.set_postfix(generated=completed, failed=failed, resumed=skipped)

        progress.write(f"Generated {completed} multi-operation triplets")
        if failed:
            progress.write(f"Skipped {failed} tasks after max attempts")

    print("\nDone.")


if __name__ == "__main__":
    main()
