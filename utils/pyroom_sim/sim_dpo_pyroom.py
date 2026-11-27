#!/usr/bin/env python3
"""Render source/positive/negative FOA triplets and export DPO training JSONL."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.pyroom_sim import sim_pairs_pyroom as render
from utils.pyroom_sim import render_config as config
from utils.pyroom_sim import sim_static_pyroom as static
from utils.pyroom_sim import sim_dynamic_pyroom as dynamic
from utils.spatial_audio_editing.src.progress import Progress


IMPLEMENTATION_HASH = hashlib.sha256(
    Path(__file__).read_bytes() + render.IMPLEMENTATION_HASH.encode()
).hexdigest()
PATH_FIELDS = {"source": "src_wav_path", "positive": "pos_wav_path", "negative": "neg_wav_path"}


def sample_identity(row, domain, split):
    raw_id = row.get("source_sample_id", row.get("sample_id"))
    if raw_id is None:
        raise ValueError("DPO record requires source_sample_id or sample_id")
    raw_id = render.component(raw_id, "source_sample_id")
    task = render.component(row["edit_type"], "edit_type")
    mode = render.component(row["error_mode"], "error_mode")
    return f"dpo/{domain}/{split}/{task}/{raw_id}/{mode}"


def negative_record(row, source_room):
    """Compile rejected edits in their own room, rather than the positive room."""
    info = deepcopy(row.get("negative_edit_info") or [])
    operations = info if isinstance(info, list) else [info]
    for operation in operations:
        if operation.get("type") != "angle_motion":
            continue
        start = operation.get("start_position", operation.get("old_position"))
        end = operation.get("end_position", operation.get("event", {}).get("position"))
        if start is not None and end is not None and abs(start["y"]) < 1e-8 and abs(end["y"]) < 1e-8:
            # Wrong-position negatives may retain the positive rotation_deg.
            # Derive the actual horizontal turn from the negative endpoints.
            angle = math.atan2(end["x"], -end["z"]) - math.atan2(start["x"], -start["z"])
            operation["rotation_deg"] = math.degrees(math.atan2(math.sin(angle), math.cos(angle)))
    dimensions = dict(zip(static.AXES, source_room.tolist()))
    return {"room_dims": dimensions,
            "target_room_dims": row.get("negative_room_dims", dimensions),
            "target_scene": row["negative_scene"], "edit_info": info}


def render_triplet(row, *, domain, split="train", data_root=None, settings=None,
                   overwrite=False, input_metadata=None):
    """Render all three scenes with one shared scale; commit metadata last."""
    data_root = Path(data_root or config.DATA_ROOT).expanduser().absolute()
    if domain not in config.SINGLE_RENDER_DIRS or split not in ("train", "test"):
        raise ValueError("domain must be audio_only/speech_audio; split must be train/test")
    identity = sample_identity(row, domain, split)
    _, _, _, task, raw_id, mode = identity.split("/")
    directory = render.rooted(config.DPO_RENDER_ROOTS[split], data_root) / task / f"{domain}_{raw_id}" / mode
    paths = {"source": directory / "source/foa.flac", "positive": directory / "target/foa.flac",
             "negative": directory / "negative/foa.flac"}
    render.relative_path(directory, data_root)
    completion = directory / "metadata.json"
    if not overwrite:
        saved = render.completed_metadata(completion, paths.values())
        if saved is not None:
            return saved, "skipped"
    settings = render.validated_settings(settings)
    caption = row.get("chosen", row.get("caption"))
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("DPO chosen/caption must contain the correct editing instruction")
    duration = static.finite(row["duration"], "duration")
    frames = round(duration * settings["sample_rate"])
    if duration <= 0 or frames < 1:
        raise ValueError("duration must contain at least one audio frame")
    source_room, positive_room = render.scene_rooms(row)
    negative = negative_record(row, source_room)
    _, negative_room = render.scene_rooms(negative)
    scenes = {"source": row["original_scene"], "positive": dynamic.target_events(row),
              "negative": dynamic.target_events(negative)}
    rooms = {"source": source_room, "positive": positive_room, "negative": negative_room}
    room_settings = render.room_render_settings(rooms, settings)
    sources = {}
    for side, events in scenes.items():
        if not isinstance(events, list):
            raise ValueError(f"{side} scene must be an event list")
        ids = [event["event_id"] for event in events]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate event_id in {side} scene")
        for event in events:
            try:
                dynamic.validate_event_geometry(event, rooms[side], settings)
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError(f"{side} event_id={event.get('event_id')}: {error}") from error
            path = render.resolve_audio(event.get("audio_path", event.get("source_path")), data_root)
            sources[str(path)] = render.audio_signature(path)
    fingerprint = hashlib.sha256(render.canonical({
        "row": row, "settings": settings, "room_settings": room_settings,
        "sources": sources, "domain": domain, "split": split,
        "input_metadata": input_metadata, "implementation": IMPLEMENTATION_HASH,
    }).encode()).hexdigest()
    renderer = render.PairRenderer(data_root, settings)
    audio = {side: renderer.scene(events, duration, rooms[side], side) for side, events in scenes.items()}
    peaks = {side: float(np.max(np.abs(samples))) for side, samples in audio.items()}
    if min(peaks.values()) <= 0:
        raise ValueError("Source, positive or negative audio is silent")
    if np.array_equal(audio["positive"], audio["negative"]):
        raise ValueError("Positive and negative audio are identical")
    scale = 10 ** (settings["peak_dbfs"] / 20) / max(peaks.values())
    metadata = {
        "sample_id": identity, "source_sample_id": raw_id, "domain": domain, "split": split,
        "edit_type": task, "error_mode": mode, "caption": caption.strip(),
        **{PATH_FIELDS[side]: render.relative_path(path, data_root) for side, path in paths.items()},
        "original_text": row.get("original_text", ""), "target_text": row.get("target_text", ""),
        "negative_text": row.get("negative_text", ""), "rejected": row.get("rejected", ""),
        "duration": frames / settings["sample_rate"], "sample_rate": settings["sample_rate"],
        "channels": 4, "channel_order": "WYZX", "w_gain": 1 / math.sqrt(2),
        "room_dims": {side: dict(zip(static.AXES, room.tolist())) for side, room in rooms.items()},
        "render": {"settings": settings, "room_settings": room_settings,
                   "save_scale": scale, "peaks_before_scale": peaks,
                   "motions": renderer.motion_info},
    }
    if input_metadata is not None:
        metadata["input_metadata"] = input_metadata
    for key in ("operation_count", "edit_types"):
        if key in row:
            metadata[key] = row[key]
    signatures = {}
    for side, path in paths.items():
        signatures[side] = render.atomic_flac(path, audio[side] * scale, settings["sample_rate"], settings["subtype"])
    render.atomic_json(completion, {**metadata, "_cache": {"fingerprint": fingerprint, **signatures}})
    return metadata, "rendered"


def input_files(args, data_root):
    rendered_root = render.rooted(config.DPO_RENDERED_METADATA_ROOT, data_root)
    if args.metadata:
        paths = list(dict.fromkeys(path.expanduser().absolute() for path in args.metadata))
    else:
        paths = []
        for domain in ([args.domain] if args.domain else config.DPO_SCENE_DIRS):
            for split in ([args.split] if args.split else ("train", "test")):
                root = render.rooted(config.DPO_SCENE_DIRS[domain][split], data_root)
                paths.extend(sorted(path for path in root.glob("*_negative.jsonl")
                                    if args.kind == "all" or path.stem.startswith("multi_") == (args.kind == "multi")))
    contexts, destinations = [], set()
    for path in paths:
        if not path.is_file():
            print(f"[SKIP] Metadata file not found: {path}")
            continue
        inferred = next((parent.name.removesuffix("_multi") for parent in path.parents
                         if parent.name.removesuffix("_multi") in config.DPO_SCENE_DIRS), None)
        domain = args.domain or inferred
        if domain not in config.DPO_SCENE_DIRS or (args.domain and inferred and args.domain != inferred):
            raise ValueError(f"Cannot infer matching domain for {path}; specify --domain")
        split = args.split or ("test" if "test" in path.parts or "test_set" in path.parts else "train")
        destination = rendered_root / domain / split / (path.stem.removesuffix("_negative") + ".jsonl")
        if destination in destinations:
            raise ValueError(f"Input files collide at {destination}")
        destinations.add(destination)
        contexts.append((path, domain, split, destination))
    return contexts


def tasks(contexts, args, data_root, settings):
    seen = set()
    for context in contexts:
        count = 0
        with context[0].open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                if args.limit is not None and count >= args.limit:
                    break
                row, error = None, None
                try:
                    row = json.loads(line)
                    identity = sample_identity(row, context[1], context[2])
                    if identity in seen:
                        raise ValueError(f"Duplicate DPO sample: {identity}")
                    seen.add(identity)
                except (ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
                yield context, number, row, error, data_root, settings, args.overwrite
                count += 1


def process_task(task):
    context, number, row, error, data_root, settings, overwrite = task
    identity = None
    try:
        if error:
            raise ValueError(error)
        identity = sample_identity(row, context[1], context[2])
        record, status = render_triplet(
            row, domain=context[1], split=context[2], data_root=data_root, settings=settings,
            overwrite=overwrite, input_metadata={"path": os.path.relpath(context[0], data_root), "line": number})
        return context[3], identity, record, status, None
    except Exception as error:
        return context[3], identity, None, "failed", {
            "input": f"{context[0]}:{number}", "sample_id": identity, "error": str(error)}


def write_training_manifests(data_root, progress):
    """Export the canonical paths/instruction expected by spat_dpo_dataset.py."""
    data_root = Path(data_root).expanduser().absolute()
    root = render.rooted(config.DPO_RENDERED_METADATA_ROOT, data_root)
    output = render.rooted(config.RENDERED_METADATA_ROOT, data_root)
    counts = {}
    progress.reset(desc="Writing DPO training manifests", unit="triplets")
    for split in ("train", "test"):
        seen, count = set(), 0
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as spool:
            for domain in config.DPO_SCENE_DIRS:
                for path in sorted((root / domain / split).glob("*.jsonl")):
                    if path.name.startswith(".") or path.stem.endswith(".errors"):
                        continue
                    for line in path.read_text().splitlines():
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if row["sample_id"] in seen or row["domain"] != domain or row["split"] != split:
                            raise ValueError(f"Duplicate or mismatched DPO sample in {path}: {row['sample_id']}")
                        seen.add(row["sample_id"])
                        record = {key: row[key] for key in ("sample_id", "source_sample_id", "caption", "domain",
                                  "split", "edit_type", "error_mode", "duration", "sample_rate", "channel_order")}
                        record["audio_channels"] = row["channels"]
                        record.update({key: str((data_root / row[key]).absolute()) for key in PATH_FIELDS.values()})
                        spool.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        count += 1
                        progress.update()
            destination = output / f"dpo_{split}.jsonl"
            render.merge_manifest(destination, spool, set())
            progress.write(f"{destination}: {count} rows")
        counts[split] = count
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", "--metadata-jsonl", type=Path, nargs="+")
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    parser.add_argument("--domain", choices=tuple(config.DPO_SCENE_DIRS))
    parser.add_argument("--split", choices=("train", "test"), help="Default: both splits")
    parser.add_argument("--kind", choices=("single", "multi", "all"), default="all")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--limit", type=int, help="Maximum nonempty records PER input file")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--motion-step-seconds", type=float)
    args = parser.parse_args()
    if args.num_workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("num-workers and limit must be positive")
    data_root = args.data_root.expanduser().absolute()
    if args.export_only:
        with Progress(desc="Writing DPO training manifests", unit="triplets") as progress:
            write_training_manifests(data_root, progress)
        return
    settings = render.validated_settings({key: value for key, value in {
        "sample_rate": args.sample_rate, "motion_step_seconds": args.motion_step_seconds}.items() if value is not None})
    contexts = input_files(args, data_root)
    if not contexts:
        print("No DPO scene metadata found; rebuilding manifests from existing rendered metadata.")
        with Progress(desc="Writing DPO training manifests", unit="triplets") as progress:
            write_training_manifests(data_root, progress)
        return
    static.load_pyroom()
    render.render_files(contexts, args, data_root, settings, processor=process_task,
                        task_generator=tasks, exporter=write_training_manifests, unit="triplets")


if __name__ == "__main__":
    main()
