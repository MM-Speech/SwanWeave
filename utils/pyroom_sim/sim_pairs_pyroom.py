#!/usr/bin/env python3
"""Render editing FOA pairs and DPO triplets, then export all training manifests."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

import numpy as np
import soundfile as sf

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.pyroom_sim import render_config as config
from utils.pyroom_sim import sim_static_pyroom as static
from utils.pyroom_sim import sim_dynamic_pyroom as dynamic
from utils.spatial_audio_editing.src.progress import Progress


AXES = static.AXES
# Cache invalidation covers all three implementation files.
IMPLEMENTATION_HASH = hashlib.sha256(b"\0".join(
    path.read_bytes() for path in (Path(__file__), Path(static.__file__), Path(dynamic.__file__))
)).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def component(value, name):
    value = str(value)
    if value in ("", ".", "..") or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"{name} must be a single safe path component: {value!r}")
    return value


def rooted(path, data_root):
    """Relocate configured dataset paths when --data-root is supplied."""
    path = Path(path).expanduser().absolute()
    try:
        return data_root / path.relative_to(Path(config.DATA_ROOT).expanduser().absolute())
    except ValueError:
        return path  # Explicit source roots outside DATA_ROOT remain unchanged.


def relative_path(path, data_root):
    return Path(path).absolute().relative_to(data_root).as_posix()


def resolve_audio(value, data_root):
    path = Path(value)
    if path.is_absolute():
        result = path
    elif path.parts and path.parts[0] in config.SOURCE_ROOTS:
        result = rooted(config.SOURCE_ROOTS[path.parts[0]], data_root).joinpath(*path.parts[1:])
    else:
        result = data_root / path
    if not result.is_file():
        raise FileNotFoundError(f"Source audio not found: {result}")
    return result


def validated_settings(overrides=None):
    settings = dict(config.SETTINGS)
    if overrides:
        unknown = set(overrides) - settings.keys()
        if unknown:
            raise ValueError(f"Unknown render settings: {sorted(unknown)}")
        settings.update(overrides)
    for key in ("sample_rate", "max_order", "surround_points"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int):
            raise ValueError(f"{key} must be an integer")
    if settings["sample_rate"] < 1 or settings["max_order"] < 0:
        raise ValueError("sample_rate must be positive; max_order must be nonnegative")
    if settings["surround_points"] < 4 or settings["surround_points"] % 2:
        raise ValueError("surround_points must be even and at least four")
    for key in ("peak_dbfs", "absorption", "sound_speed", "ir_seconds", "ir_trim_ms",
                "late_reverb_start_ms", "late_reverb_gain_db", "motion_step_seconds", "min_source_distance"):
        settings[key] = static.finite(settings[key], key)
    if not 0 <= settings["absorption"] <= 1 or settings["sound_speed"] <= 0:
        raise ValueError("Invalid absorption or sound_speed")
    if settings["peak_dbfs"] >= 0 or min(settings["ir_seconds"], settings["ir_trim_ms"], settings["motion_step_seconds"]) <= 0:
        raise ValueError("peak_dbfs must be negative; IR lengths and motion step must be positive")
    if settings["late_reverb_start_ms"] < 0 or settings["late_reverb_gain_db"] > 0:
        raise ValueError("Late-reverb start must be nonnegative and gain must not be positive")
    if settings["min_source_distance"] <= 0:
        raise ValueError("min_source_distance must be positive")
    if settings["angle_radius"] not in ("min", "start"):
        raise ValueError("angle_radius must be min or start")
    if not sf.check_format("FLAC", settings["subtype"]):
        raise ValueError(f"Unsupported FLAC subtype: {settings['subtype']}")
    return settings


def room_render_settings(rooms, settings):
    """Record validated effective settings for each source/target/negative room."""
    result = {}
    for side, dimensions in rooms.items():
        name, effective = config.settings_for_room(dimensions, settings)
        result[side] = {"preset": name, "settings": validated_settings(effective)}
    return result


class PairRenderer:
    """Per-pair caches reuse unchanged events/RIRs without leaking state across samples."""

    def __init__(self, data_root, settings):
        self.data_root, self.settings = data_root, settings
        self.signals, self.rirs, self.events = {}, {}, {}
        self.motion_info = []

    def signal(self, event, duration):
        start = static.finite(event.get("start_time", event.get("start")), "event.start_time")
        end = static.finite(event.get("end_time", event.get("end")), "event.end_time")
        if not 0 <= start < end <= duration + 1e-6:
            raise ValueError(f"Invalid event interval [{start}, {end}) for {duration}s")
        repeat = event.get("repeat", 1)
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ValueError("repeat must be a positive integer")
        path = resolve_audio(event.get("audio_path", event.get("source_path")), self.data_root)
        key = (str(path), start, end, repeat)
        if key not in self.signals:
            self.signals[key] = static.read_event_audio(path, start, end, repeat, self.settings["sample_rate"])
        return self.signals[key], start

    def static(self, signal, position, dimensions):
        return static.render_at_position(signal, position, dimensions, self.settings, self.rirs)

    def spatial(self, signal, event, dimensions, side):
        audio, details = dynamic.render_event(signal, event, dimensions, self.settings, self.rirs, side)
        self.motion_info.extend(details)
        return audio

    def scene(self, events, duration, dimensions, side):
        sample_rate = self.settings["sample_rate"]
        mix = np.zeros((round(duration * sample_rate), 4), dtype=np.float32)
        for event in events:
            key = canonical({"event": event, "room": dimensions.tolist(),
                             "side": side if event.get("motion") else None})
            if key not in self.events:
                signal, start = self.signal(event, duration)
                signal, gain = dynamic.apply_gain_envelope(signal, event)
                self.events[key] = (self.spatial(signal, event, dimensions, side), start, gain)
            rendered, start, gain = self.events[key]
            offset = round(start * sample_rate)
            count = min(len(rendered), len(mix) - offset)
            if count > 0:
                mix[offset:offset + count] += rendered[:count] * gain
        if not np.all(np.isfinite(mix)):
            raise ValueError("Non-finite rendered mixture")
        return mix


def atomic_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_flac(path, audio, sample_rate, subtype):
    """Encode and hash in memory, then atomically save the same FLAC bytes."""
    with io.BytesIO() as buffer:
        sf.write(buffer, audio, sample_rate, format="FLAC", subtype=subtype)
        encoded_audio = buffer.getvalue()
    signature = [len(encoded_audio), hashlib.sha256(encoded_audio).hexdigest()]

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(encoded_audio)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return signature


def audio_signature(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def completed_metadata(path, audio_paths):
    """Reuse saved metadata when all output audio files exist and are nonempty."""
    try:
        for audio_path in audio_paths:
            info = audio_path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                return None
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or "sample_id" not in saved:
            return None
        return {key: value for key, value in saved.items() if key != "_cache"}
    except (OSError, ValueError):
        return None


def pair_directory(row, domain, split, data_root):
    task = component(row["edit_type"], "edit_type")
    identity = component(row["sample_id"], "sample_id")
    multi = re.fullmatch(r"multi_([0-9]+)op", task)
    if multi and int(multi[1]) >= 2:
        group = rooted(config.MULTI_RENDER_ROOTS[split], data_root) / task
        return group / f"{domain}_{identity}"
    if task not in config.SINGLE_RENDER_DIRS[domain][split]:
        raise ValueError(f"Unknown edit_type: {task}")
    return rooted(config.SINGLE_RENDER_DIRS[domain][split][task], data_root) / identity


def scene_rooms(row):
    """Resolve source/final room dimensions, including ordered room edits."""
    operations = row.get("edit_info") or []
    operations = operations if isinstance(operations, list) else [operations]
    source_dims = row["room_dims"]
    target_dims = row.get("target_room_dims", source_dims)
    found_room_change = False
    for operation in operations:
        if operation.get("type") == "room_change":
            if not found_room_change:
                source_dims = operation.get("old_dims", source_dims)
                found_room_change = True
            target_dims = operation.get("new_dims", target_dims)
    return static.room_dimensions(source_dims), static.room_dimensions(target_dims)


def render_pair(row, *, domain, split="train", data_root=None, settings=None,
                overwrite=False, input_metadata=None):
    """Public API: one scene-edit record -> one audio pair and one metadata record."""
    data_root = Path(data_root or config.DATA_ROOT).expanduser().absolute()
    if domain not in config.SINGLE_RENDER_DIRS or split not in ("train", "test"):
        raise ValueError("domain must be audio_only/speech_audio; split must be train/test")
    directory = pair_directory(row, domain, split, data_root)
    source_path, target_path = directory / "origin/foa.flac", directory / "target/foa.flac"
    relative_path(directory, data_root)  # Enforce the relative-path export contract.
    completion = directory / "metadata.json"
    if not overwrite:
        saved = completed_metadata(completion, (source_path, target_path))
        if saved is not None:
            return saved, "skipped"
    settings = validated_settings(settings)
    for key in ("original_scene", "target_scene"):
        if not isinstance(row.get(key), list):
            raise ValueError(f"{key} must be an event list")
        ids = [event["event_id"] for event in row[key]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate event_id in {key}")
    for key in ("original_text", "target_text"):
        if not isinstance(row.get(key), str):
            raise ValueError(f"{key} must contain the scene caption")
    captions = row.get("captions", [row["caption"]] if "caption" in row else None)
    if not isinstance(captions, list) or not captions or any(not isinstance(text, str) or not text.strip() for text in captions):
        raise ValueError("captions must contain one or more editing instructions")
    duration = static.finite(row["duration"], "duration")
    frames = round(duration * settings["sample_rate"])
    if duration <= 0 or frames < 1:
        raise ValueError("duration must contain at least one audio frame")
    source_room, target_room = scene_rooms(row)
    room_settings = room_render_settings({"source": source_room, "target": target_room}, settings)
    target = dynamic.target_events(row)
    for side, events, room in (("source", row["original_scene"], source_room),
                               ("target", target, target_room)):
        for event in events:
            try:
                dynamic.validate_event_geometry(event, room, settings)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{side} event_id={event.get('event_id')}: {error}") from error
    source_stats = {}
    for event in row["original_scene"] + target:
        path = resolve_audio(event.get("audio_path", event.get("source_path")), data_root)
        source_stats[str(path)] = audio_signature(path)
    fingerprint = hashlib.sha256(canonical({"row": row, "settings": settings,
                                            "room_settings": room_settings, "sources": source_stats,
                                            "input_metadata": input_metadata,
                                            "domain": domain, "split": split, "implementation": IMPLEMENTATION_HASH}).encode()).hexdigest()
    renderer = PairRenderer(data_root, settings)
    source_audio = renderer.scene(row["original_scene"], duration, source_room, "source")
    target_audio = renderer.scene(target, duration, target_room, "target")
    source_peak = float(np.max(np.abs(source_audio)))
    target_peak = float(np.max(np.abs(target_audio)))
    peak = max(source_peak, target_peak)
    if min(source_peak, target_peak) <= 0:
        raise ValueError("Source or target audio is silent")
    scale = 10 ** (settings["peak_dbfs"] / 20) / peak
    metadata = {
        "sample_id": f"{domain}/{split}/{row['edit_type']}/{row['sample_id']}",
        "source_sample_id": str(row["sample_id"]), "domain": domain, "split": split,
        "edit_type": row["edit_type"],
        "src_wav_path": relative_path(source_path, data_root),
        "wav_path": relative_path(target_path, data_root),
        "original_text": row["original_text"], "target_text": row["target_text"],
        "captions": captions, "duration": frames / settings["sample_rate"],
        "sample_rate": settings["sample_rate"], "channels": 4, "channel_order": "WYZX",
        "w_gain": 1 / math.sqrt(2),
        "room_dims": dict(zip(AXES, source_room.tolist())),
        "target_room_dims": dict(zip(AXES, target_room.tolist())),
        "render": {"settings": settings, "room_settings": room_settings, "save_scale": scale,
                   "source_peak_before_scale": source_peak, "target_peak_before_scale": target_peak,
                   "motions": renderer.motion_info},
    }
    if input_metadata is not None:
        metadata["input_metadata"] = input_metadata
    for key in ("operation_count", "edit_types"):
        if key in row:
            metadata[key] = row[key]
    source_signature = atomic_flac(source_path, source_audio * scale, settings["sample_rate"], settings["subtype"])
    target_signature = atomic_flac(target_path, target_audio * scale, settings["sample_rate"], settings["subtype"])
    atomic_json(completion, {**metadata, "_cache": {"fingerprint": fingerprint,
                "source": source_signature, "target": target_signature}})
    return metadata, "rendered"


def process_task(task):
    context, line, row, error, data_root, settings, overwrite = task
    input_path, domain, split, manifest = context
    label = f"{input_path}:{line}"
    sample_key = None
    try:
        if error:
            raise ValueError(error)
        sample_key = f"{domain}/{split}/{row['edit_type']}/{row['sample_id']}"
        origin = {"path": os.path.relpath(input_path, data_root), "line": line}
        metadata, status = render_pair(row, domain=domain, split=split, data_root=data_root,
                                       settings=settings, overwrite=overwrite, input_metadata=origin)
        return manifest, sample_key, metadata, status, None
    except Exception as error:
        return manifest, sample_key, None, "failed", {"input": label, "sample_id": sample_key, "error": str(error)}


def input_files(args, data_root):
    metadata_root = rooted(config.METADATA_ROOT, data_root)
    rendered_root = rooted(config.RENDERED_METADATA_ROOT, data_root)
    contexts = []
    if args.metadata:
        paths = list(dict.fromkeys(path.expanduser().absolute() for path in args.metadata))
    else:
        paths = []
        domains = [args.domain] if args.domain else list(config.SINGLE_RENDER_DIRS)
        splits = [args.split] if args.split else ["train", "test"]
        for domain in domains:
            for split in splits:
                parent = metadata_root if split == "train" else metadata_root / "test_set"
                for kind in (("single", "multi") if args.kind == "all" else (args.kind,)):
                    directory = parent / (domain if kind == "single" else f"{domain}_multi")
                    paths.extend(sorted(path for path in directory.glob("*.jsonl")
                                        if not path.stem.endswith(("_text", "_negative", ".errors"))))
    destinations = set()
    for path in paths:
        if not path.is_file():
            print(f"[SKIP] Metadata file not found: {path}")
            continue
        inferred_domain = path.parent.name.removesuffix("_multi")
        domain = args.domain or inferred_domain
        if args.domain and inferred_domain in config.SINGLE_RENDER_DIRS and args.domain != inferred_domain:
            raise ValueError(f"Input directory {path.parent} conflicts with --domain {args.domain}")
        if domain not in config.SINGLE_RENDER_DIRS:
            raise ValueError(f"Cannot infer domain for {path}; specify --domain")
        split = args.split or ("test" if "test_set" in path.parts else "train")
        destination = rendered_root / domain / split / path.name
        if destination in destinations:
            raise ValueError(f"Input files collide at {destination}")
        destinations.add(destination)
        contexts.append((path, domain, split, destination))
    return contexts


def count_rows(contexts, limit, progress):
    total = 0
    for path, *_ in contexts:
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                count += 1
                total += 1
                progress.update()
                if limit is not None and count >= limit:
                    break
    return total


def tasks(contexts, args, data_root, settings):
    seen = set()
    for context in contexts:
        count = 0
        with context[0].open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                if args.limit is not None and count >= args.limit:
                    break
                row, error = None, None
                try:
                    row = json.loads(line)
                    key = (context[1], context[2], row["edit_type"], str(row["sample_id"]))
                    if key in seen:
                        raise ValueError(f"Duplicate editing sample: {key}")
                    seen.add(key)
                except (ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
                yield context, line_number, row, error, data_root, settings, args.overwrite
                count += 1


def merge_manifest(destination, spool, changed, preserve_existing=False):
    """Upsert attempted IDs, preserving previous records outside a --limit run."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            if preserve_existing and destination.is_file():
                with destination.open(encoding="utf-8") as previous:
                    for line in previous:
                        if line.strip() and json.loads(line)["sample_id"] not in changed:
                            output.write(line if line.endswith("\n") else line + "\n")
            spool.seek(0)
            for line in spool:
                output.write(line)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)



def write_training_manifests(data_root, progress):
    """Rebuild edit/gen train/test lists from all committed rendered metadata."""
    data_root = Path(data_root).expanduser().absolute()
    rendered_root = rooted(config.RENDERED_METADATA_ROOT, data_root)
    spools = {(kind, split): tempfile.TemporaryFile(mode="w+", encoding="utf-8")
              for kind in ("edit", "gen") for split in ("train", "test")}
    counts = {key: 0 for key in spools}
    seen = set()
    progress.reset(desc="Writing training manifests", unit="pairs")

    def write(kind, split, row):
        spools[kind, split].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        counts[kind, split] += 1

    try:
        for split in ("train", "test"):
            for domain in config.SINGLE_RENDER_DIRS:
                for source in sorted((rendered_root / domain / split).glob("*.jsonl")):
                    if source.name.startswith(".") or source.stem.endswith((".errors", "_text", "_negative")):
                        continue
                    with source.open(encoding="utf-8") as stream:
                        for line in stream:
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            identity = record["sample_id"]
                            if identity in seen:
                                raise ValueError(f"Duplicate rendered sample_id in {source}: {identity}")
                            seen.add(identity)
                            if record["domain"] != domain or record["split"] != split:
                                raise ValueError(f"Rendered domain/split mismatch in {source}: {identity}")
                            source_path = str((data_root / record["src_wav_path"]).absolute())
                            target_path = str((data_root / record["wav_path"]).absolute())
                            common = {key: record[key] for key in (
                                "source_sample_id", "domain", "split", "edit_type", "duration",
                                "sample_rate", "channel_order") if key in record}
                            common["audio_channels"] = record.get("channels", 4)
                            captions = list(dict.fromkeys(text.strip() for text in record["captions"] if text.strip()))
                            if not captions:
                                raise ValueError(f"Missing editing instructions in {source}: {identity}")
                            for index, caption in enumerate(captions):
                                write("edit", split, {
                                    **common, "sample_id": f"{identity}/caption_{index}",
                                    "caption": caption, "src_wav_path": source_path, "wav_path": target_path,
                                })
                            for side, text_key, wav_path in (
                                ("source", "original_text", source_path), ("target", "target_text", target_path),
                            ):
                                caption = record[text_key].strip()
                                if not caption:
                                    raise ValueError(f"Missing {text_key} in {source}: {identity}")
                                write("gen", split, {
                                    **common, "sample_id": f"{identity}/{side}",
                                    "caption": caption, "wav_path": wav_path,
                                })
                            progress.update()
        for (kind, split), spool in spools.items():
            destination = rendered_root / f"{kind}_{split}.jsonl"
            merge_manifest(destination, spool, set())
            progress.write(f"{destination}: {counts[kind, split]} rows")
    finally:
        for spool in spools.values():
            spool.close()
    return counts


def render_files(contexts, args, data_root, settings, *, processor=process_task,
                 task_generator=tasks, exporter=write_training_manifests, unit="pairs"):
    """Shared worker/progress/manifest loop for editing pairs and DPO triplets."""
    counts = {"rendered": 0, "skipped": 0, "failed": 0}
    spools = {context[3]: tempfile.TemporaryFile(mode="w+", encoding="utf-8") for context in contexts}
    changed = {destination: set() for destination in spools}
    errors = {context[3]: [] for context in contexts}
    pool = None
    try:
        with Progress(desc="Counting scene records", unit=unit) as progress:
            total = count_rows(contexts, args.limit, progress)
            progress.reset(total=total, desc=f"Rendering FOA {unit}", unit=unit)
            jobs = task_generator(contexts, args, data_root, settings)
            if args.num_workers == 1:
                results = map(processor, jobs)
            else:
                pool = mp.get_context("spawn").Pool(args.num_workers)
                results = pool.imap_unordered(processor, jobs, chunksize=1)
            for destination, identity, metadata, status, error in results:
                counts[status] += 1
                if identity is not None:
                    changed[destination].add(identity)
                if metadata is not None:
                    spools[destination].write(json.dumps(metadata, ensure_ascii=False, allow_nan=False) + "\n")
                if error is not None:
                    errors[destination].append(error)
                    progress.write(f"[ERROR] {error['input']}: {error['error']}")
                progress.update()
                progress.set_postfix(**counts)
            if pool is not None:
                pool.close()
                pool.join()
                pool = None
            progress.reset(total=len(spools), desc="Writing rendered metadata", unit="files")
            for destination, spool in spools.items():
                merge_manifest(destination, spool, changed[destination], preserve_existing=args.limit is not None)
                error_path = destination.with_suffix(".errors.jsonl")
                error_path.parent.mkdir(parents=True, exist_ok=True)
                with error_path.open("w", encoding="utf-8") as stream:
                    for error in errors[destination]:
                        stream.write(json.dumps(error, ensure_ascii=False) + "\n")
                progress.update()
            exporter(data_root, progress)
            progress.write(f"Rendered={counts['rendered']}, skipped={counts['skipped']}, failed={counts['failed']}")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        for spool in spools.values():
            spool.close()
    if counts["failed"]:
        raise SystemExit(1)
    return counts


def all_input_files(args, data_root):
    """Discover both ordinary scenes and *_negative.jsonl DPO scenes."""
    from utils.pyroom_sim import sim_dpo_pyroom as dpo

    if not args.metadata:
        return input_files(args, data_root) + dpo.input_files(args, data_root)
    contexts = []
    for negative, discover in ((False, input_files), (True, dpo.input_files)):
        paths = [path for path in args.metadata if path.stem.endswith("_negative") == negative]
        if paths:
            selected = argparse.Namespace(**{**vars(args), "metadata": paths})
            contexts.extend(discover(selected, data_root))
    return contexts


def is_dpo_context(context, data_root):
    return context[3].is_relative_to(rooted(config.DPO_RENDERED_METADATA_ROOT, data_root))


def all_tasks(contexts, args, data_root, settings):
    from utils.pyroom_sim import sim_dpo_pyroom as dpo

    pairs = [context for context in contexts if not is_dpo_context(context, data_root)]
    triplets = [context for context in contexts if is_dpo_context(context, data_root)]
    yield from tasks(pairs, args, data_root, settings)
    yield from dpo.tasks(triplets, args, data_root, settings)


def process_scene_task(task):
    if is_dpo_context(task[0], task[4]):
        from utils.pyroom_sim import sim_dpo_pyroom as dpo
        return dpo.process_task(task)
    return process_task(task)


def write_all_training_manifests(data_root, progress):
    from utils.pyroom_sim import sim_dpo_pyroom as dpo

    write_training_manifests(data_root, progress)
    dpo.write_training_manifests(data_root, progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", "--metadata-jsonl", type=Path, nargs="+")
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    parser.add_argument("--domain", choices=tuple(config.SINGLE_RENDER_DIRS))
    parser.add_argument("--split", choices=("train", "test"), help="Filter discovery to one split; default: both")
    parser.add_argument("--kind", choices=("single", "multi", "all"), default="all")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--limit", type=int, help="Maximum nonempty records PER input file")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate complete pairs too")
    parser.add_argument("--export-only", action="store_true", help="Rebuild all six training JSONL files without rendering audio")
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--motion-step-seconds", type=float)
    args = parser.parse_args()
    if args.num_workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("num-workers and limit must be positive")
    data_root = args.data_root.expanduser().absolute()
    if args.export_only:
        with Progress(desc="Writing training manifests", unit="pairs") as progress:
            write_all_training_manifests(data_root, progress)
        return
    overrides = {key: value for key, value in {"sample_rate": args.sample_rate,
                 "motion_step_seconds": args.motion_step_seconds}.items() if value is not None}
    try:
        settings = validated_settings(overrides)
        contexts = all_input_files(args, data_root)
        if not contexts:
            print("No scene metadata found; rebuilding manifests from existing rendered metadata.")
            with Progress(desc="Writing training manifests", unit="pairs") as progress:
                write_all_training_manifests(data_root, progress)
            return
        static.load_pyroom()
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.error(str(error))
    render_files(contexts, args, data_root, settings, processor=process_scene_task,
                 task_generator=all_tasks, exporter=write_all_training_manifests, unit="records")


if __name__ == "__main__":
    main()
