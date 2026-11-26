"""Spatial audio scene generation and edit operations.

Coordinate convention: the listener sits at the origin facing -z;
x is left(-)/right(+), y is down(-)/up(+), z is front(-)/back(+).
"""

import math
import random
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from utils.spatial_audio_editing.src.caption import _direction_index


ROOM_SIZES = {
    "small":  {"x": 2,   "y": 2,   "z": 2},
    "medium": {"x": 6,   "y": 4,   "z": 6},
    "large":  {"x": 21,  "y": 10,  "z": 21},
    "huge":   {"x": 100, "y": 100, "z": 100},
}

EDIT_TYPES = [
    "add_event",
    "remove_event",
    "replace_event",
    "volume_change",
    "relocation",
    "relocation_plane",
    "distance_change",
    "angle_change",
    "angle_change_plane",
    "distance_motion",
    "angle_motion",
    "volume_fade",
    "extract_event",
    "room_change",
]

# Edits whose effect evolves over time; they need scenes/clips of >= 5 s.
DYNAMIC_EDIT_TYPES = {"distance_motion", "angle_motion", "volume_fade"}

NEAR_RANGE = (0.0, 2.0)
FAR_RANGE = (8.0, 10.0)


# ---- geometry helpers ----

def _azimuth_and_distance(pos: dict) -> Tuple[float, float]:
    """Horizontal azimuth (0 = front, +pi/2 = right) and horizontal distance."""
    return np.arctan2(pos["x"], -pos["z"]), np.sqrt(pos["x"] ** 2 + pos["z"] ** 2)


def snap_cardinal_position(position: dict, horizontal_distance: float = None) -> dict:
    """Align front/back/left/right to their axes, preserving height and diagonals.

    Use the caption sectors so coordinates and direction text agree. Ordinary
    sampling only removes the lateral component; polar edits can retain their
    required horizontal distance. Positions described as nearby are unchanged.
    """
    result = dict(position)
    if math.hypot(position["x"], position["z"]) < 0.3:
        return result
    direction = _direction_index(position["x"], position["z"])
    if direction in (0, 4):
        result["x"] = 0.0
        if horizontal_distance is not None:
            result["z"] = round(math.copysign(horizontal_distance, position["z"]), 2)
    elif direction in (2, 6):
        result["z"] = 0.0
        if horizontal_distance is not None:
            result["x"] = round(math.copysign(horizontal_distance, position["x"]), 2)
    return result


def _polar_position(dist: float, azimuth: float) -> dict:
    """Horizontal point with cardinal directions aligned at the requested radius."""
    return snap_cardinal_position({
        "x": round(dist * np.sin(azimuth), 2),
        "y": 0,
        "z": round(-dist * np.cos(azimuth), 2),
    }, horizontal_distance=dist)


def _random_rotation() -> float:
    """Random azimuth rotation of 60-180 degrees, clockwise or counter-clockwise."""
    return random.choice([
        random.uniform(np.radians(60), np.radians(180)),
        random.uniform(np.radians(-180), np.radians(-60)),
    ])


def _random_rotated_position(dist: float, old_azimuth: float) -> tuple:
    """Sample a 60-180 degree rotation, measured after cardinal alignment."""
    for _ in range(200):
        position = _polar_position(dist, old_azimuth + _random_rotation())
        azimuth, _ = _azimuth_and_distance(position)
        rotation = np.arctan2(np.sin(azimuth - old_azimuth),
                             np.cos(azimuth - old_azimuth))
        if abs(rotation) >= np.radians(60):
            return position, azimuth, rotation
    raise RuntimeError("Could not sample an aligned rotation of at least 60 degrees")


def random_fade_gains(direction: str) -> Tuple[float, float]:
    """(start_gain, end_gain) for a fade-in ("in") or fade-out ("out")."""
    quiet_range, loud_range = (0.1, 0.3), (0.7, 1.2)
    start_range, end_range = (quiet_range, loud_range) if direction == "in" else (loud_range, quiet_range)
    start_gain = round(random.uniform(*start_range), 2)
    end_gain = round(random.uniform(*end_range), 2)
    return start_gain, end_gain


def _longest_event_index(scene: List[dict]) -> int:
    return max(range(len(scene)), key=lambda i: scene[i]["end_time"] - scene[i]["start_time"])


def fit_scene_duration(duration: float, *scenes_and_edits) -> float:
    """Keep 0.4 s after all recorded events plus 0-40% of the excess tail.

    Include edit-info snapshots so intermediate events also fit the common
    timeline. Event intervals themselves are preserved. Round the fixed margin
    up and the retained excess down to centiseconds, without cutting the margin.
    """
    def event_ends(value):
        if isinstance(value, dict):
            if "start_time" in value and "end_time" in value:
                yield value["end_time"]
            for child in value.values():
                yield from event_ends(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from event_ends(child)

    last_end = max(event_ends(scenes_and_edits), default=0.0)
    required = math.ceil((last_end + 0.4) * 100 - 1e-9) / 100
    excess = max(0.0, duration - required)
    retained = math.floor(random.uniform(0.0, 0.4) * excess * 100 + 1e-9) / 100
    return round(required + retained, 2)


class SceneGenerator:
    """Generates random spatial audio scenes and applies the edit operations in EDIT_TYPES."""

    def __init__(self, cfg: dict, classes: List[str], class_index: defaultdict, dur_cache: dict,
                 speech_meta: dict = None, speech_classes: List[str] = None):
        g = cfg["generation"]
        self.dur_min = g["duration_min"]
        self.dur_max = g["duration_max"]
        self.evt_min = g["num_events_min"]
        self.evt_max = g["num_events_max"]
        self.gain_range = tuple(g["gain_range"])
        self.classes = classes
        self.class_index = class_index
        self.dur_cache = dur_cache
        self.speech_meta = speech_meta or {}
        self.speech_classes = speech_classes or []
        self.audio_classes = [c for c in classes if c not in self.speech_classes]
        self._mixed_mode = bool(self.speech_classes and self.audio_classes)

        # Pre-filter clips >= 5s for dynamic edits
        self.long_class_index = defaultdict(list)
        for cls, paths in class_index.items():
            long_paths = [p for p in paths if dur_cache.get(p, 0) >= 5.0]
            if long_paths:
                self.long_class_index[cls] = long_paths

        self.room_sizes = cfg.get("room_sizes", ROOM_SIZES)
        self.room_names = list(self.room_sizes.keys())
        self.base_room = cfg.get("default_room", "large")
        self.set_sample_room(self.base_room)

    def set_sample_room(self, room_name: str):
        """Set the room used for the current sample (room_change picks a random one)."""
        self.room_bounds = self.room_sizes[room_name]
        self.default_room = room_name

    # ---- helpers ----

    def _random_clip(self, cls: str, use_long: bool = False) -> Optional[Tuple[str, float]]:
        idx = self.long_class_index if use_long else self.class_index
        paths = idx.get(cls, [])
        if not paths:
            return None
        path = random.choice(paths)
        dur = self.dur_cache.get(path)
        if dur is None or dur < 0.5:
            return None
        return path, dur

    def _random_position(self, room_bounds: dict = None, y_range: tuple = None,
                         free_plane: bool = False) -> tuple:
        if room_bounds is None:
            room_bounds = self.room_bounds
        half_x = room_bounds["x"] / 2.0
        half_z = room_bounds["z"] / 2.0
        half_y = room_bounds["y"] / 2.0

        max_h_dist = min(10.0, min(half_x, half_z))
        h_buckets = [(0.8, 1.0), (1.0, 3.0), (3.0, 5.0), (5.0, max_h_dist)]
        h_buckets = [(a, min(b, max_h_dist)) for a, b in h_buckets if a < max_h_dist and a < b]

        if y_range is not None:
            y_buckets = [tuple(y_range)]
        else:
            level = min(1.0, half_y * 0.2)
            max_abs_y = half_y * 0.9
            y_buckets = [
                (-max_abs_y, -level * 2),
                (-level * 2, -level),
                (-level, level),
                (level, level * 2),
                (level * 2, max_abs_y),
            ]
            y_buckets = [(a, b) for a, b in y_buckets if a < b]

        for _ in range(200):
            azimuth = random.uniform(-np.pi, np.pi)
            h_dist = round(random.uniform(*random.choice(h_buckets)), 2)
            x = round(h_dist * np.sin(azimuth), 2)
            z = round(-h_dist * np.cos(azimuth), 2)
            y = round(random.uniform(*random.choice(y_buckets)), 2)
            if free_plane or abs(x) < 0.2 or abs(y) < 0.2 or abs(z) < 0.2:
                break
        position = snap_cardinal_position({"x": x, "y": y, "z": z})
        return position["x"], position["y"], position["z"]

    # ---- event / scene generation ----

    def generate_event(self, cls: str, scene_duration: float, event_id: int, use_long: bool = False) -> Optional[dict]:
        clip = self._random_clip(cls, use_long)
        if clip is None:
            return None

        rel_path, audio_dur = clip
        natural_dur = min(audio_dur, scene_duration)
        if use_long:
            # Ensure event duration >= 5.0s in the scene
            latest_start = scene_duration - 5.05
            if latest_start <= 0:
                return None
            max_start = max(0.0, min(natural_dur, scene_duration - natural_dur * 0.5, latest_start))
        else:
            max_start = max(0.0, min(natural_dur, scene_duration - natural_dur * 0.5))
        start = round(random.uniform(0.0, max_start), 2)
        # The initial scene duration guides placement; the whole group is fitted
        # after editing, instead of truncating this planned event at the boundary.
        end = round(start + natural_dur, 2)

        x, y, z = self._random_position(y_range=(0.0, 0.0))
        event = {
            "event_id": event_id,
            "class": cls,
            "audio_path": rel_path,
            "audio_duration": round(audio_dur, 2),
            "position": {"x": x, "y": y, "z": z},
            "start_time": start,
            "end_time": round(end, 2),
            "gain": round(random.uniform(*self.gain_range), 2),
        }

        if cls in self.speech_classes:
            event["domain"] = "speech"
            meta = self.speech_meta.get(rel_path, {})
            event["transcript"] = meta.get("transcript", "")
            event["speaker_id"] = meta.get("speaker_id", "")
        else:
            event["domain"] = "audio"

        return event

    def _fill_scene(self, scene: List[dict], class_pool: List[str], n: int, duration: float,
                    use_long: bool, tries_per_event: int = 50):
        """Append up to n events drawn from class_pool (each with a few retries)."""
        for _ in range(n):
            for __ in range(tries_per_event):
                evt = self.generate_event(random.choice(class_pool), duration, len(scene), use_long)
                if evt:
                    scene.append(evt)
                    break

    def generate_scene(self, duration: float, num_events: int, use_long: bool = False) -> List[dict]:
        if not self._mixed_mode:
            return self._generate_scene_any(duration, num_events, use_long)

        # Mixed mode: ensure at least 1 speech + 1 audio event
        if num_events >= 2:
            n_speech = random.randint(1, num_events - 1)
        else:
            n_speech = random.randint(0, 1)
        n_audio = num_events - n_speech

        scene = []
        self._fill_scene(scene, self.speech_classes, n_speech, duration, use_long)
        self._fill_scene(scene, self.audio_classes, n_audio, duration, use_long)

        # Fill any shortfall with random domain
        for _ in range(100):
            if len(scene) >= num_events:
                break
            evt = self.generate_event(random.choice(self.classes), duration, len(scene), use_long)
            if evt:
                scene.append(evt)

        scene.sort(key=lambda e: e["start_time"])
        return scene

    def _generate_scene_any(self, duration: float, num_events: int, use_long: bool = False) -> List[dict]:
        """Scene generation without domain constraints (audio-only mode)."""
        available = list(self.classes)
        random.shuffle(available)
        scene = []
        for cls in available:
            if len(scene) >= num_events:
                break
            evt = self.generate_event(cls, duration, len(scene), use_long)
            if evt:
                scene.append(evt)
        for _ in range(200):
            if len(scene) >= num_events:
                break
            evt = self.generate_event(random.choice(self.classes), duration, len(scene), use_long)
            if evt:
                scene.append(evt)
        scene.sort(key=lambda e: e["start_time"])
        return scene

    # ---- edit application ----

    def apply_edit(self, original: List[dict], edit_type: str, scene_duration: float,
                   _depth: int = 0) -> Tuple[List[dict], dict]:
        """Apply one edit to a copy of `original`; returns (target_scene, edit_info).

        If the edit cannot be applied to this scene it is retried (or falls back
        to add_event for remove/extract on too-small scenes).
        """
        if _depth >= 50:
            raise RuntimeError(f"apply_edit recursion limit reached for {edit_type}")

        def retry(fallback_type: str = edit_type):
            return self.apply_edit(original, fallback_type, scene_duration, _depth + 1)

        target = deepcopy(original)

        if edit_type == "add_event":
            next_id = max((e["event_id"] for e in target), default=-1) + 1
            if self._mixed_mode and random.random() < 0.5:
                cls = random.choice(self.speech_classes)
            elif self._mixed_mode:
                cls = random.choice(self.audio_classes)
            else:
                cls = random.choice(self.classes)
            evt = self.generate_event(cls, scene_duration, next_id)
            if evt is None:
                return retry()
            target.append(evt)
            info = {"type": "add_event", "added": evt}

        elif edit_type == "remove_event":
            if len(target) < 2:
                return retry("add_event")
            removed = target.pop(random.randrange(len(target)))
            info = {"type": "remove_event", "removed": removed}

        elif edit_type == "replace_event":
            cross_domain = None
            if self._mixed_mode:
                # Cross-domain replace: speech -> audio or audio -> speech
                if random.random() < 0.5:
                    src_domain, new_classes, cross_domain = "speech", self.audio_classes, "speech_to_audio"
                else:
                    src_domain, new_classes, cross_domain = "audio", self.speech_classes, "audio_to_speech"
                candidates = [i for i, e in enumerate(target) if e.get("domain") == src_domain]
                if not candidates:
                    return retry()
                idx = random.choice(candidates)
                old = target[idx]
                cls = random.choice(new_classes)
            else:
                idx = random.randrange(len(target))
                old = target[idx]
                cls = random.choice([c for c in self.classes if c != old["class"]])
            evt = self.generate_event(cls, scene_duration, old["event_id"])
            if evt is None:
                return retry()
            evt["position"] = deepcopy(old["position"])
            for key in ("start_time", "end_time", "gain"):
                evt[key] = old[key]
            target[idx] = evt
            info = {"type": "replace_event", "old": old, "new": evt}
            if cross_domain:
                info["cross_domain"] = cross_domain

        elif edit_type == "volume_change":
            idx = random.randrange(len(target))
            factor = random.choice([2.0, 0.5])
            old_gain = target[idx]["gain"]
            target[idx]["gain"] = round(old_gain * factor, 2)
            info = {
                "type": "volume_change",
                "event": target[idx],
                "old_gain": old_gain,
                "new_gain": target[idx]["gain"],
                "direction": "up" if factor > 1 else "down",
            }

        elif edit_type in ("relocation", "relocation_plane"):
            # New position: >= 60 deg azimuth change and >= 2 m distance change.
            idx = random.randrange(len(target))
            old_pos = dict(target[idx]["position"])
            old_azimuth, old_dist = _azimuth_and_distance(old_pos)

            for _ in range(200):
                new_x, new_y, new_z = self._random_position(free_plane=True)
                new_az, new_dist = _azimuth_and_distance({"x": new_x, "z": new_z})
                az_diff = abs(np.arctan2(np.sin(new_az - old_azimuth),
                                         np.cos(new_az - old_azimuth)))
                if az_diff >= np.radians(60) and abs(new_dist - old_dist) >= 2.0:
                    break
            if edit_type == "relocation_plane":
                new_y = 0

            target[idx]["position"] = {"x": new_x, "y": new_y, "z": new_z}
            info = {
                "type": edit_type,
                "event": target[idx],
                "old_position": old_pos,
                "new_position": dict(target[idx]["position"]),
            }

        elif edit_type == "distance_change":
            idx = random.randrange(len(target))
            old_pos = dict(target[idx]["position"])
            azimuth, old_dist = _azimuth_and_distance(old_pos)
            new_dist = round(random.uniform(*FAR_RANGE if old_dist < 4.0 else NEAR_RANGE), 2)

            target[idx]["position"] = _polar_position(new_dist, azimuth)
            info = {
                "type": "distance_change",
                "event": target[idx],
                "old_position": old_pos,
                "new_position": dict(target[idx]["position"]),
                "direction": "farther" if new_dist > old_dist else "closer",
            }

        elif edit_type in ("angle_change", "angle_change_plane"):
            # Rotate around the listener at the same horizontal distance;
            # angle_change additionally picks a new height.
            idx = random.randrange(len(target))
            old_pos = dict(target[idx]["position"])
            old_azimuth, dist = _azimuth_and_distance(old_pos)
            new_pos, new_azimuth, _ = _random_rotated_position(dist, old_azimuth)
            if edit_type == "angle_change":
                _, new_pos["y"], _ = self._random_position(free_plane=True)
            target[idx]["position"] = new_pos
            info = {
                "type": edit_type,
                "event": target[idx],
                "old_position": old_pos,
                "new_position": dict(new_pos),
                "old_azimuth_deg": round(np.degrees(old_azimuth), 1),
                "new_azimuth_deg": round(np.degrees(new_azimuth), 1),
                "keep_distance": round(dist, 2),
            }

        elif edit_type == "distance_motion":
            evt = target[_longest_event_index(target)]
            old_pos = dict(evt["position"])
            azimuth, old_dist = _azimuth_and_distance(old_pos)
            if old_dist < 4.0:
                d_new = round(random.uniform(*FAR_RANGE), 2)
                direction = "receding"
            else:
                d_new = round(random.uniform(*NEAR_RANGE), 2)
                direction = "approaching"

            evt["position"] = _polar_position(d_new, azimuth)
            info = {
                "type": "distance_motion",
                "event": evt,
                "old_position": old_pos,
                "start_position": dict(old_pos),
                "end_position": dict(evt["position"]),
                "direction": direction,
            }

        elif edit_type == "angle_motion":
            evt = target[_longest_event_index(target)]
            old_pos = dict(evt["position"])
            old_azimuth, dist = _azimuth_and_distance(old_pos)
            new_pos, new_azimuth, rotation = _random_rotated_position(dist, old_azimuth)

            evt["position"] = new_pos
            info = {
                "type": "angle_motion",
                "event": evt,
                "old_position": old_pos,
                "start_position": dict(old_pos),
                "end_position": dict(evt["position"]),
                "old_azimuth_deg": round(np.degrees(old_azimuth), 1),
                "new_azimuth_deg": round(np.degrees(new_azimuth), 1),
                "rotation_deg": round(np.degrees(rotation), 1),
                "keep_distance": round(dist, 2),
            }

        elif edit_type == "volume_fade":
            evt = target[_longest_event_index(target)]
            old_gain = evt["gain"]
            direction = random.choice(["in", "out"])
            start_gain, end_gain = random_fade_gains(direction)

            evt["gain"] = end_gain
            info = {
                "type": "volume_fade",
                "event": evt,
                "old_gain": old_gain,
                "start_gain": start_gain,
                "end_gain": end_gain,
                "direction": direction,
            }

        elif edit_type == "extract_event":
            if len(target) < 2 or len({e["class"] for e in target}) < 2:
                return retry("add_event")
            kept_class = target[random.randrange(len(target))]["class"]
            kept = [e for e in target if e["class"] == kept_class]
            removed = [e for e in target if e["class"] != kept_class]
            target[:] = kept
            info = {
                "type": "extract_event",
                "kept_class": kept_class,
                "kept_events": kept,
                "removed_events": removed,
            }

        elif edit_type == "room_change":
            new_room = random.choice([r for r in self.room_names if r != self.default_room])
            info = {
                "type": "room_change",
                "old_room": self.default_room,
                "new_room": new_room,
                "old_dims": self.room_sizes[self.default_room],
                "new_dims": self.room_sizes[new_room],
            }

        else:
            raise ValueError(f"Unknown edit type: {edit_type}")

        target.sort(key=lambda e: e["start_time"])
        return target, info
