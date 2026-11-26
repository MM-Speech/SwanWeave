#!/usr/bin/env python3
"""
Generate DPO negative data from positive spatial audio editing triplets.

For each positive sample, generates one DPO pair whose rejected caption describes
an incorrect operation (wrong event, no operation, wrong direction, wrong
position, ...), together with the scene that incorrect operation would produce.
Error-mode counts are sampling weights, not the number of output rows. Each
output record keeps the original scene fields from the positive sample so
downstream code can read it like a normal spatial-audio editing example.

Usage:
  python generate_dpo_negatives.py                                 # DATA_ROOT/metadata/{audio_only,speech_audio}
  python generate_dpo_negatives.py --domain audio_only             # single domain
  python generate_dpo_negatives.py --domain audio_only --max 100   # quick subset
  python generate_dpo_negatives.py --domain audio_only --multi
  python generate_dpo_negatives.py --dry-run                       # print stats only
"""

import argparse
import hashlib
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.spatial_audio_editing import data_paths

from utils.spatial_audio_editing.src.caption import _fallback_caption, join_clauses, scene_to_text
from utils.spatial_audio_editing.src.progress import Progress
from utils.spatial_audio_editing.src.scene import (
    ROOM_SIZES, random_fade_gains, fit_scene_duration,
    _polar_position, snap_cardinal_position,
)


# Fields copied from the positive sample into each negative record.
NORMAL_SAMPLE_FIELDS = (
    "duration",
    "room",
    "room_dims",
    "captions",
    "original_scene",
    "target_scene",
    "edit_info",
    "original_text",
    "target_text",
    "target_room",
    "target_room_dims",
    "operation_count",
    "edit_types",
)

DEFAULT_DOMAINS = ["audio_only", "speech_audio"]


# ============================================================================
# Error mode configuration ("count" = sampling weight)
# ============================================================================

EDIT_ERROR_MODES = {
    "add_event": {
        "wrong_event":    {"count": 2},
        "no_operation":   {"count": 1},
    },
    "remove_event": {
        "wrong_event":    {"count": 2},
        "no_operation":   {"count": 1},
    },
    "replace_event": {
        "wrong_replace":  {"count": 2},
        "no_operation":   {"count": 1},
    },
    "volume_change": {
        "wrong_event":     {"count": 2},
        "wrong_direction": {"count": 2},
        "no_operation":    {"count": 1},
    },
    "angle_change": {
        "wrong_event":  {"count": 2},
        "wrong_angle":  {"count": 2},
        "no_operation": {"count": 1},
    },
    "angle_change_plane": {
        "wrong_event":  {"count": 2},
        "wrong_angle":  {"count": 2},
        "no_operation": {"count": 1},
    },
    "distance_change": {
        "wrong_event":     {"count": 2},
        "wrong_direction": {"count": 2},
        "no_operation":    {"count": 1},
    },
    "relocation": {
        "wrong_event":    {"count": 2},
        "wrong_position": {"count": 2},
        "no_operation":   {"count": 1},
    },
    "relocation_plane": {
        "wrong_event":    {"count": 2},
        "wrong_position": {"count": 2},
        "no_operation":   {"count": 1},
    },
    "room_change": {
        "wrong_room":   {"count": 2},
        "no_operation": {"count": 1},
    },
    "extract_event": {
        "wrong_class":  {"count": 2},
        "no_operation": {"count": 1},
    },
    # ---- dynamic ----
    "volume_fade": {
        "wrong_direction": {"count": 2},
        "wrong_event":     {"count": 2},
        "no_operation":    {"count": 1},
    },
    "angle_motion": {
        "wrong_position": {"count": 2},
        "no_operation":   {"count": 1},
    },
    "distance_motion": {
        "wrong_position": {"count": 2},
        "no_operation":   {"count": 1},
    },
}

MULTI_ERROR_MODES = {
    "wrong_operation": {"count": 3},
    "omit_operation": {"count": 2},
    "no_operation": {"count": 1},
}

NO_OPERATION_CAPTIONS = [
    "Keep the scene as is.",
    "Don't change anything.",
    "Leave everything unchanged.",
    "Do nothing to the audio.",
    "Keep the audio exactly the same.",
    "Leave the scene alone.",
    "Make no changes.",
    "Don't alter anything.",
    "Keep it as it is.",
    "Preserve the current scene.",
    "Don't touch anything.",
    "Leave the audio as it currently is.",
    "No changes needed.",
    "Just keep everything the way it is.",
    "Keep all sounds where they are.",
    "Don't modify the scene.",
    "Let the audio remain unchanged.",
    "Keep the original scene intact.",
    "Leave it exactly as is.",
    "Stay with the current arrangement.",
]


def stable_int_seed(*parts) -> int:
    """Return a deterministic seed that does not depend on Python hash salts."""
    text = "\n".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _weighted_choice(mode_cfg: dict) -> str:
    mode_names = list(mode_cfg.keys())
    weights = [mode_cfg[name].get("count", 1) for name in mode_names]
    return random.choices(mode_names, weights=weights, k=1)[0]


# ============================================================================
# Scene helpers
# ============================================================================

def _pick_other_event(scene, exclude_event_id):
    """Pick a random event from the scene that is NOT the excluded one."""
    others = [e for e in scene if e["event_id"] != exclude_event_id]
    return random.choice(others) if others else None


def _next_event_id(scene):
    return max((e.get("event_id", -1) for e in scene), default=-1) + 1


def _event_index(scene, event_id):
    for i, event in enumerate(scene):
        if event.get("event_id") == event_id:
            return i
    return None


def _copy_sound_identity(dst, src):
    for key in ("class", "audio_path", "audio_duration", "domain", "transcript", "speaker_id"):
        if key in src:
            dst[key] = deepcopy(src[key])
        elif key in dst:
            dst.pop(key, None)


def _position_opposite(pos, jitter=2.0):
    return snap_cardinal_position({
        "x": round(-pos.get("x", 0) + random.uniform(-jitter, jitter), 2),
        "y": round(pos.get("y", 0), 2),
        "z": round(-pos.get("z", 0) + random.uniform(-jitter, jitter), 2),
    })


def _opposite_distance_position(old_pos, actual_direction):
    azimuth = np.arctan2(old_pos["x"], -old_pos["z"])
    if actual_direction in ("farther", "receding"):
        new_dist = round(random.uniform(0.0, 2.0), 2)
    else:
        new_dist = round(random.uniform(8.0, 10.0), 2)
    return _polar_position(new_dist, azimuth)


def _apply_edit_info(scene, info):
    """Apply an edit_info dict to a scene copy and return the edited scene."""
    target = deepcopy(scene)
    t = info.get("type")

    if t == "add_event":
        target.append(deepcopy(info["added"]))

    elif t == "remove_event":
        removed_id = info["removed"]["event_id"]
        target = [e for e in target if e.get("event_id") != removed_id]

    elif t == "replace_event":
        idx = _event_index(target, info["old"]["event_id"])
        if idx is not None:
            target[idx] = deepcopy(info["new"])

    elif t in ("volume_change", "volume_fade"):
        idx = _event_index(target, info["event"]["event_id"])
        if idx is not None:
            target[idx]["gain"] = info.get("new_gain", info.get("end_gain", target[idx].get("gain", 1.0)))

    elif t in ("relocation", "relocation_plane", "distance_change", "angle_change", "angle_change_plane"):
        idx = _event_index(target, info["event"]["event_id"])
        if idx is not None:
            target[idx]["position"] = deepcopy(info["new_position"])

    elif t in ("distance_motion", "angle_motion"):
        idx = _event_index(target, info["event"]["event_id"])
        if idx is not None:
            target[idx]["position"] = deepcopy(info["end_position"])

    elif t == "extract_event":
        target = [e for e in target if e.get("class") == info["kept_class"]]

    # room_change only changes room metadata, not the events.

    target.sort(key=lambda e: e.get("start_time", 0))
    return target


# ============================================================================
# Negative edits
# ============================================================================

def _caption_from_info(info):
    return random.choice(_fallback_caption(info))


def _no_op(scene):
    return deepcopy(scene), {"type": "no_operation"}, random.choice(NO_OPERATION_CAPTIONS), {}


def _negative(scene, info, extras=None):
    """(negative_scene, negative_edit_info, rejected_caption, extra_fields) for a wrong edit."""
    return _apply_edit_info(scene, info), info, _caption_from_info(info), extras or {}


def _make_negative_edit(original_scene, edit_info, error_mode):
    """Create an incorrect edit_info plus the negative scene it would produce."""
    t = edit_info.get("type")

    if error_mode == "no_operation":
        return _no_op(original_scene)

    # add_event: add the wrong sound, copied from another existing event.
    if t == "add_event" and error_mode == "wrong_event":
        actual = edit_info["added"]
        other = _pick_other_event(original_scene, actual["event_id"])
        if other is None:
            return _no_op(original_scene)
        wrong_added = deepcopy(actual)
        wrong_added["event_id"] = _next_event_id(original_scene)
        _copy_sound_identity(wrong_added, other)
        wrong_added["position"] = deepcopy(other["position"])
        return _negative(original_scene, {"type": "add_event", "added": wrong_added})

    # remove_event: remove a different existing event.
    if t == "remove_event" and error_mode == "wrong_event":
        other = _pick_other_event(original_scene, edit_info["removed"]["event_id"])
        if other is None:
            return _no_op(original_scene)
        return _negative(original_scene, {"type": "remove_event", "removed": deepcopy(other)})

    # replace_event: replace the intended event with the wrong class.
    if t == "replace_event" and error_mode == "wrong_replace":
        old = edit_info["old"]
        other = _pick_other_event(original_scene, old["event_id"])
        if other is None:
            return _no_op(original_scene)
        wrong_new = deepcopy(edit_info["new"])
        _copy_sound_identity(wrong_new, other)
        for key in ("event_id", "position", "start_time", "end_time", "gain"):
            wrong_new[key] = deepcopy(old[key])
        return _negative(original_scene, {"type": "replace_event", "old": deepcopy(old), "new": wrong_new})

    # volume_change: either modify the wrong event, or invert the direction.
    if t == "volume_change" and error_mode in ("wrong_event", "wrong_direction"):
        if error_mode == "wrong_event":
            other = _pick_other_event(original_scene, edit_info["event"]["event_id"])
            if other is None:
                return _no_op(original_scene)
            event = deepcopy(other)
            old_gain = other.get("gain", 1.0)
            direction = edit_info.get("direction", "up")
        else:
            event = deepcopy(edit_info["event"])
            old_gain = edit_info.get("old_gain", event.get("gain", 1.0))
            direction = "down" if edit_info.get("direction") == "up" else "up"
        event["gain"] = round(old_gain * (2.0 if direction == "up" else 0.5), 2)
        return _negative(original_scene, {
            "type": "volume_change",
            "event": event,
            "old_gain": old_gain,
            "new_gain": event["gain"],
            "direction": direction,
        })

    # Static spatial edits applied to the wrong event.
    if t in ("relocation", "relocation_plane", "angle_change", "angle_change_plane",
             "distance_change") and error_mode == "wrong_event":
        other = _pick_other_event(original_scene, edit_info["event"]["event_id"])
        if other is None:
            return _no_op(original_scene)
        info = deepcopy(edit_info)
        info["event"] = deepcopy(other)
        info["old_position"] = deepcopy(other["position"])
        return _negative(original_scene, info)

    # Static spatial edits with a wrong (roughly opposite) target position.
    if ((t in ("relocation", "relocation_plane") and error_mode == "wrong_position")
            or (t in ("angle_change", "angle_change_plane") and error_mode == "wrong_angle")):
        info = deepcopy(edit_info)
        wrong = _position_opposite(info["new_position"])
        if t.endswith("_plane"):
            wrong["y"] = 0
        info["new_position"] = wrong
        info["event"]["position"] = deepcopy(wrong)
        if t in ("angle_change", "angle_change_plane"):
            info["new_azimuth_deg"] = round(
                np.degrees(np.arctan2(wrong["x"], -wrong["z"])), 1)
        return _negative(original_scene, info)

    if t == "distance_change" and error_mode == "wrong_direction":
        info = deepcopy(edit_info)
        wrong = _opposite_distance_position(info["old_position"], info.get("direction", "farther"))
        info["new_position"] = wrong
        info["direction"] = "closer" if info.get("direction") == "farther" else "farther"
        info["event"]["position"] = deepcopy(wrong)
        return _negative(original_scene, info)

    # Dynamic edits.
    if t == "volume_fade" and error_mode == "wrong_event":
        other = _pick_other_event(original_scene, edit_info["event"]["event_id"])
        if other is None:
            return _no_op(original_scene)
        # NB: the default is evaluated eagerly; this random draw is kept for reproducibility.
        direction = edit_info.get("direction", random.choice(["in", "out"]))
        start_gain, end_gain = random_fade_gains(direction)
        event = deepcopy(other)
        event["gain"] = end_gain
        return _negative(original_scene, {
            "type": "volume_fade",
            "event": event,
            "old_gain": other.get("gain", 1.0),
            "start_gain": start_gain,
            "end_gain": end_gain,
            "direction": direction,
        })

    if t == "volume_fade" and error_mode == "wrong_direction":
        info = deepcopy(edit_info)
        direction = "out" if info.get("direction") == "in" else "in"
        start_gain, end_gain = random_fade_gains(direction)
        info["direction"] = direction
        info["start_gain"] = start_gain
        info["end_gain"] = end_gain
        info["event"]["gain"] = end_gain
        return _negative(original_scene, info)

    if t == "angle_motion" and error_mode == "wrong_position":
        info = deepcopy(edit_info)
        wrong = _position_opposite(info["end_position"])
        wrong["y"] = 0
        info["end_position"] = wrong
        info["event"]["position"] = deepcopy(wrong)
        start = info["start_position"]
        old_azimuth = np.arctan2(start["x"], -start["z"])
        new_azimuth = np.arctan2(wrong["x"], -wrong["z"])
        rotation = np.arctan2(np.sin(new_azimuth - old_azimuth),
                             np.cos(new_azimuth - old_azimuth))
        info["old_azimuth_deg"] = round(np.degrees(old_azimuth), 1)
        info["new_azimuth_deg"] = round(np.degrees(new_azimuth), 1)
        info["rotation_deg"] = round(np.degrees(rotation), 1)
        return _negative(original_scene, info)

    if t == "distance_motion" and error_mode == "wrong_position":
        info = deepcopy(edit_info)
        wrong = _opposite_distance_position(info["start_position"], info.get("direction", "receding"))
        info["direction"] = "approaching" if info.get("direction") == "receding" else "receding"
        info["end_position"] = wrong
        info["event"]["position"] = deepcopy(wrong)
        return _negative(original_scene, info)

    # extract_event: keep the wrong class.
    if t == "extract_event" and error_mode == "wrong_class":
        actual_kept = edit_info["kept_class"]
        other_classes = sorted({e["class"] for e in original_scene if e.get("class") != actual_kept})
        if not other_classes:
            return _no_op(original_scene)
        kept_class = random.choice(other_classes)
        return _negative(original_scene, {
            "type": "extract_event",
            "kept_class": kept_class,
            "kept_events": [deepcopy(e) for e in original_scene if e.get("class") == kept_class],
            "removed_events": [deepcopy(e) for e in original_scene if e.get("class") != kept_class],
        })

    # room_change: switch to a room other than the old and the correct new one.
    if t == "room_change" and error_mode == "wrong_room":
        old_room = edit_info.get("old_room", "large")
        actual_new = edit_info.get("new_room", "medium")
        other_rooms = ([r for r in ROOM_SIZES if r not in (old_room, actual_new)]
                       or [r for r in ROOM_SIZES if r != actual_new])
        wrong_room = random.choice(other_rooms)
        info = {
            "type": "room_change",
            "old_room": old_room,
            "new_room": wrong_room,
            "old_dims": edit_info.get("old_dims", ROOM_SIZES.get(old_room, ROOM_SIZES["large"])),
            "new_dims": ROOM_SIZES[wrong_room],
        }
        extras = {"negative_room": wrong_room, "negative_room_dims": ROOM_SIZES[wrong_room]}
        return deepcopy(original_scene), info, _caption_from_info(info), extras

    return _no_op(original_scene)


def _make_multi_negative(data, error_mode):
    """Negative for a multi-op sample: drop one step, corrupt one step, or do nothing."""
    edit_infos = data.get("edit_info", [])
    original_scene = data.get("original_scene", [])
    if (not edit_infos or error_mode not in ("omit_operation", "wrong_operation")
            or (error_mode == "omit_operation" and len(edit_infos) <= 1)):
        return deepcopy(original_scene), [], random.choice(NO_OPERATION_CAPTIONS), {}

    bad_idx = random.randrange(len(edit_infos))
    current = deepcopy(original_scene)
    negative_infos, clauses, extras = [], [], {}
    for i, info in enumerate(edit_infos):
        if i == bad_idx and error_mode == "omit_operation":
            continue
        if i == bad_idx and error_mode == "wrong_operation":
            modes = EDIT_ERROR_MODES.get(info["type"], {})
            step_modes = {k: v for k, v in modes.items() if k != "no_operation"} or modes
            step_mode = _weighted_choice(step_modes) if step_modes else "no_operation"
            current, negative_info, clause, step_extras = _make_negative_edit(current, info, step_mode)
            extras.update(step_extras)
        else:
            current = _apply_edit_info(current, info)
            negative_info, clause = deepcopy(info), _caption_from_info(info)
        negative_infos.append(negative_info)
        clauses.append(clause)
    return current, negative_infos, join_clauses(clauses), extras


# ============================================================================
# Per-sample entry point
# ============================================================================

def generate_negative(data: dict, seed: int) -> Optional[dict]:
    """Generate one negative DPO record for a positive sample (single- or multi-op)."""
    is_multi = isinstance(data.get("edit_info"), list)
    modes = MULTI_ERROR_MODES if is_multi else EDIT_ERROR_MODES.get(data["edit_type"])
    captions = data.get("captions", [])
    if not modes or not captions:
        return None

    random.seed(seed)
    np.random.seed(seed)
    error_mode = _weighted_choice(modes)
    chosen = random.choice(captions)

    def make():
        if is_multi:
            return _make_multi_negative(data, error_mode)
        return _make_negative_edit(data["original_scene"], data["edit_info"], error_mode)

    negative_scene, negative_info, rejected, extras = make()
    for _ in range(10):  # ensure rejected differs from chosen
        if rejected != chosen:
            break
        negative_scene, negative_info, rejected, extras = make()

    record = {k: data[k] for k in NORMAL_SAMPLE_FIELDS if k in data}
    record.update({
        "source_sample_id": data["sample_id"],
        "edit_type": data["edit_type"],
        "error_mode": error_mode,
        "prompt": data["original_text"],
        "chosen": chosen,
        "rejected": rejected,
        "negative_scene": negative_scene,
        "negative_text": scene_to_text(negative_scene),
        "negative_edit_info": negative_info,
    })
    record.update(extras)
    record["duration"] = fit_scene_duration(
        data["duration"], record["original_scene"], record["target_scene"], negative_scene,
        record["edit_info"], negative_info,
    )
    return record


# ============================================================================
# File processing
# ============================================================================

def positive_jsonl_files(domain_dir: Path, edit_type: str = None):
    files = [f for f in sorted(domain_dir.glob("*.jsonl"))
             if not f.stem.endswith(("_text", "_negative"))]
    if edit_type:
        files = [f for f in files if f.stem == edit_type]
    return files


def process_file(input_path, output_path, max_samples=None, dry_run=False,
                 progress=None, progress_counts=None):
    """Read positive lines from one JSONL file, write negative DPO pairs. Returns (n_in, n_out)."""
    num_input = 0
    num_output = 0
    if progress_counts is None:
        progress_counts = {"input": 0, "output": 0}
    fout = None if dry_run else open(output_path, "w", encoding="utf-8")
    try:
        with open(input_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                num_input += 1
                written = 0
                if not dry_run:
                    data = json.loads(line)
                    # Seed from edit_type/sample_id so results are reproducible.
                    seed = stable_int_seed(data.get("edit_type", ""), data["sample_id"])
                    record = generate_negative(data, seed)
                    if record is not None:
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        num_output += 1
                        written = 1
                progress_counts["input"] += 1
                progress_counts["output"] += written
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(**progress_counts)
                if max_samples and num_input >= max_samples:
                    break
    finally:
        if fout:
            fout.close()
    return num_input, num_output


def _known_domain(name: str) -> Optional[str]:
    domain = name.removesuffix("_multi")
    return domain if domain in DEFAULT_DOMAINS else None


def generate_dpo_files(base_dir: Path, domains, max_samples=None, dry_run=False,
                       edit_type=None, output_dir: Optional[Path] = None, split="train",
                       progress=None, progress_counts=None):
    """Read positive scenes and write negatives to central or explicit paths.

    base_dir may hold JSONL files directly, or hold domain subdirectories.
    An explicit output_dir is final for one input directory, and receives domain
    subdirectories when processing a parent with several domains.
    """
    write = progress.write if progress is not None else print
    if progress_counts is None:
        progress_counts = {"input": 0, "output": 0}
    base_dir = Path(base_dir)
    output_dir = Path(output_dir) if output_dir is not None else None
    if split not in data_paths.SPLITS:
        raise ValueError("split must be train or test")
    domains = list(domains)
    requested = {_known_domain(name) for name in domains}
    requested.discard(None)
    if not base_dir.is_dir():
        write(f"[SKIP] {base_dir} does not exist")
        return 0, 0

    known = _known_domain(base_dir.name)
    if known is not None or positive_jsonl_files(base_dir, edit_type=edit_type):
        if known is not None and requested and known not in requested:
            raise ValueError(f"Input directory belongs to {known}, outside the selected domains")
        domain = known or (next(iter(requested)) if len(requested) == 1 else None)
        if domain is None and output_dir is None:
            raise ValueError("A custom JSONL directory requires --domain or --negative-output-dir")
        domain_dirs = [(domain or base_dir.name, base_dir)]
    else:
        domain_dirs = []
        for name in domains:
            domain = _known_domain(name)
            if domain is None:
                raise ValueError(f"Unknown domain: {name}")
            directory = base_dir / name
            if not directory.is_dir() and not name.endswith("_multi"):
                directory = base_dir / f"{domain}_multi"
            domain_dirs.append((domain, directory))

    total_in = total_out = 0
    for domain, domain_dir in domain_dirs:
        if not domain_dir.is_dir():
            write(f"[SKIP] {domain_dir} does not exist")
            continue
        if output_dir is None:
            destination = Path(data_paths.DPO_SCENE_DIRS[domain][split])
        else:
            destination = output_dir / domain if len(domain_dirs) > 1 else output_dir
        jsonl_files = positive_jsonl_files(domain_dir, edit_type=edit_type)
        write(f"Domain: {domain} ({len(jsonl_files)} edit types); output: {destination}")
        if not dry_run and jsonl_files:
            destination.mkdir(parents=True, exist_ok=True)
        for jf in jsonl_files:
            out_path = destination / f"{jf.stem}_negative.jsonl"
            n_in, n_out = process_file(
                jf, out_path, max_samples=max_samples, dry_run=dry_run,
                progress=progress, progress_counts=progress_counts,
            )
            total_in += n_in
            total_out += n_out
            if dry_run:
                write(f"  {jf.stem}: {n_in} positive lines (dry-run)")
            else:
                write(f"  {jf.stem}: {n_in} in -> {n_out} negative pairs")

    if dry_run:
        write(f"Total: {total_in} positive lines")
    else:
        write(f"Total: {total_in} positive -> {total_out} negative DPO pairs")
    return total_in, total_out


def main():
    parser = argparse.ArgumentParser(description="Generate DPO negative data")
    parser.add_argument("--domain", choices=DEFAULT_DOMAINS, help="Process only one domain")
    parser.add_argument("--split", choices=data_paths.SPLITS, default="train")
    parser.add_argument("--multi", action="store_true", help="Use multi-operation positive scenes")
    parser.add_argument("--max", type=int, default=10,
                        help="Max samples per edit type (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only count lines, don't generate")
    parser.add_argument("--edit-type", type=str, default=None,
                        help="Process only one edit type")
    parser.add_argument("--output-dir", "--input-dir", dest="output_dir", type=Path,
                        default=None, help="Positive JSONL input directory (legacy --output-dir meaning)")
    parser.add_argument("--negative-output-dir", type=Path, default=None,
                        help="Override the negative destination; multiple domains get subdirectories")
    args = parser.parse_args()
    domains = [args.domain] if args.domain else DEFAULT_DOMAINS
    with Progress(desc="Generating DPO negatives", total=None, unit="samples") as progress:
        progress_counts = {"input": 0, "output": 0}
        progress.set_postfix(**progress_counts)
        options = dict(max_samples=args.max, dry_run=args.dry_run, edit_type=args.edit_type,
                       split=args.split, progress=progress, progress_counts=progress_counts)
        if args.output_dir is not None:
            names = [f"{domain}_multi" for domain in domains] if args.multi else domains
            generate_dpo_files(args.output_dir, names, output_dir=args.negative_output_dir, **options)
        else:
            inputs = data_paths.MULTI_SCENE_DIRS if args.multi else data_paths.SINGLE_SCENE_DIRS
            for domain in domains:
                destination = args.negative_output_dir
                if destination is not None and len(domains) > 1:
                    destination = destination / domain
                generate_dpo_files(inputs[domain][args.split], [domain], output_dir=destination, **options)


if __name__ == "__main__":
    main()
