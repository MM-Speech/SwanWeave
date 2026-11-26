"""Motion trajectories, gain envelopes and ordered-edit compilation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np

from utils.pyroom_sim import sim_static_pyroom as static
from utils.pyroom_sim.sim_static_pyroom import finite, vector, validate_position

WALL_MARGIN = 0.01
POSITION_TOLERANCE = 0.02

def arc_extrema(start, tangent, angle):
    theta = [0.0, angle]
    for a, b in zip(start, tangent):
        base = math.atan2(float(b), float(a))
        for k in range(-1, 3):
            candidate = base + k * math.pi
            if 0 < candidate < angle:
                theta.append(candidate)
    theta = np.array(theta)
    return np.cos(theta)[:, None] * start + np.sin(theta)[:, None] * tangent


@dataclass
class MotionPath:
    kind: str
    start: np.ndarray
    end: np.ndarray
    tangent: np.ndarray | None = None
    angle: float = 0.0
    radius: float = 0.0
    endpoint_snapped: bool = False

    def positions(self, fraction):
        fraction = np.asarray(fraction, dtype=np.float64)
        if self.kind == "distance":
            return self.start + fraction[..., None] * (self.end - self.start)
        theta = fraction * self.angle
        return self.radius * (np.cos(theta)[..., None] * self.start
                              + np.sin(theta)[..., None] * self.tangent)


def motion_path(motion, dimensions, radius_mode, min_distance=0.0):
    """Reference straight-line / shortest great-circle path, including 180-degree ties."""
    kind = motion["kind"]
    start, end = vector(motion["from"]), vector(motion["to"])
    validate_position(start, dimensions, min_distance)
    validate_position(end, dimensions, min_distance)
    if kind == "distance":
        if "via" in motion:
            raise ValueError("Distance motion must be a straight line without via")
        delta = end - start
        squared = float(delta @ delta)
        closest = np.clip(-float(start @ delta) / squared, 0, 1) if squared else 0
        closest_distance = float(np.linalg.norm(start + closest * delta))
        if closest_distance + 1e-12 < min_distance:
            raise ValueError(f"Motion distance {closest_distance:.6g} m is below minimum {min_distance:g} m")
        if closest_distance < 1e-4:
            raise ValueError("Distance motion passes through the listener")
        return MotionPath(kind, start, end)
    if kind != "angle":
        raise ValueError(f"Unknown motion kind: {kind}")
    radii = [float(np.linalg.norm(start)), float(np.linalg.norm(end))]
    unit_start, unit_end = start / radii[0], end / radii[1]
    dot = float(np.clip(unit_start @ unit_end, -1, 1))
    residual = unit_end - dot * unit_start
    sine = float(np.linalg.norm(residual))
    angle = math.atan2(sine, dot)
    via = vector(motion["via"]) if "via" in motion else None
    if via is not None:
        validate_position(via, dimensions, min_distance)
        radii.append(float(np.linalg.norm(via)))
        via = via / radii[-1]
    tolerance = POSITION_TOLERANCE / min(radii[:2])
    opposite = (dot < 0 and sine < 1e-10) or (via is not None and np.linalg.norm(unit_start + unit_end) <= tolerance)
    snapped = False
    if opposite:
        guide = via if via is not None else np.eye(3)[np.argmin(np.abs(unit_start))]
        tangent = guide - float(guide @ unit_start) * unit_start
        length = float(np.linalg.norm(tangent))
        if length < 1e-8:
            raise ValueError("Antipodal via must be off the endpoint axis")
        tangent /= length
        angle = math.pi
        snapped = bool(np.linalg.norm(unit_start + unit_end) > 1e-10)
        unit_end = -unit_start
    elif sine < 1e-10:
        tangent = np.zeros(3)
        angle = 0.0
        if via is not None and np.linalg.norm(via - unit_start) > tolerance:
            raise ValueError("via conflicts with a zero-angle path")
    else:
        tangent = residual / sine
        if via is not None:
            via_angle = math.atan2(float(via @ tangent), float(via @ unit_start))
            projected = (via @ unit_start) * unit_start + (via @ tangent) * tangent
            if np.linalg.norm(via - projected) > tolerance or not -tolerance <= via_angle <= angle + tolerance:
                raise ValueError("via is not on the shortest arc")
    maxima = np.max(np.abs(arc_extrema(unit_start, tangent, angle)), axis=0)
    occupied = maxima > 1e-12
    limit = float(np.min((dimensions[occupied] / 2 - WALL_MARGIN) / maxima[occupied]))
    radius = min(radii) if radius_mode == "min" else radii[0]
    if radius <= 0 or radius > limit + 1e-8:
        raise ValueError("Angular path does not fit the room at its requested radius")
    if radius + 1e-12 < min_distance:
        raise ValueError(f"Motion radius {radius:.6g} m is below minimum {min_distance:g} m")
    return MotionPath(kind, unit_start, unit_end, tangent, angle, radius, snapped)


def motion_windows(samples, sample_rate, step_seconds):
    hop = max(1, round(step_seconds * sample_rate))
    nodes = np.append(np.arange(0, samples, hop, dtype=np.int64), samples)
    for index, center in enumerate(nodes):
        left = int(nodes[index - 1]) if index else 0
        right = int(nodes[index + 1]) if index + 1 < len(nodes) else samples
        weights = np.empty(right - left, dtype=np.float32)
        rising, falling = int(center) - left, right - int(center)
        if rising:
            weights[:rising] = np.arange(rising, dtype=np.float64) / rising
        if falling:
            weights[rising:] = 1 - np.arange(falling, dtype=np.float64) / falling
        yield float(center) / samples, left, weights


def target_events(row):
    """Compile ordered edits into final event motion and absolute gain envelopes."""
    events = deepcopy(row["target_scene"])
    info = row.get("edit_info") or []
    operations = info if isinstance(info, list) else [info]
    motions, gains = {}, {}
    cancel_motion = {"relocation", "relocation_plane", "distance_change", "angle_change", "angle_change_plane", "remove_event", "replace_event"}
    for operation in operations:
        kind = operation.get("type")
        event = next((operation[key] for key in ("event", "new", "old", "added", "removed")
                      if isinstance(operation.get(key), dict)), None)
        if event is None:
            continue
        key = event["event_id"]
        if kind in cancel_motion:
            motions.pop(key, None)
        if kind in ("remove_event", "replace_event", "volume_change"):
            gains.pop(key, None)
        if kind in ("distance_motion", "angle_motion"):
            component_kind = "distance" if kind == "distance_motion" else "angle"
            motions.setdefault(key, {})[component_kind] = {
                "kind": component_kind,
                "from": operation.get("start_position", operation.get("old_position")),
                "to": operation.get("end_position", event.get("position")),
            }
            # Generated angle_motion is horizontal. Preserve its plane and signed
            # turn at the 180-degree tie, rather than selecting a vertical arc.
            motion = motions[key][component_kind]
            if (component_kind == "angle" and "rotation_deg" in operation
                    and abs(motion["from"]["y"]) < 1e-8 and abs(motion["to"]["y"]) < 1e-8):
                start = vector(motion["from"])
                middle = math.atan2(start[0], -start[2]) + math.radians(operation["rotation_deg"]) / 2
                radius = np.linalg.norm(start)
                motion["via"] = {"x": float(radius * math.sin(middle)), "y": 0.0,
                                 "z": float(-radius * math.cos(middle))}
        elif kind == "volume_fade":
            gains[key] = {"from": operation["start_gain"], "to": operation["end_gain"]}
    for event in events:
        key = event["event_id"]
        if key in motions:
            components = motions[key]
            event["motion"] = (next(iter(components.values())) if len(components) == 1
                               else {"kind": "combined", **components})
        if key in gains:
            event["gain_envelope"] = gains[key]
    return events


def event_trajectory(event, dimensions, settings):
    """Return a trajectory and its actual endpoints without computing RIRs."""
    motion = event["motion"]
    if motion["kind"] == "combined":
        angular = motion_path(motion["angle"], dimensions, settings["angle_radius"], settings["min_source_distance"])
        radial = motion_path(motion["distance"], dimensions, settings["angle_radius"], settings["min_source_distance"])
        maximum_radius = max(np.linalg.norm(radial.start), np.linalg.norm(radial.end))
        extremes = arc_extrema(angular.start, angular.tangent, angular.angle)
        if np.any(np.max(np.abs(extremes), axis=0) * maximum_radius >= dimensions / 2 - WALL_MARGIN):
            raise ValueError("Combined distance/angle motion does not fit the room")
        def position_at(fraction):
            direction = angular.positions(fraction) / angular.radius
            return direction * np.linalg.norm(radial.positions(fraction))
    else:
        path = motion_path(motion, dimensions, settings["angle_radius"], settings["min_source_distance"])
        position_at = path.positions
    actual_start, actual_end = position_at(0), position_at(1)
    if motion["kind"] == "combined" and np.linalg.norm(actual_end - vector(event["position"])) > POSITION_TOLERANCE:
        raise ValueError("Conflicting motion edits: final trajectory does not match target position")
    return position_at, actual_start, actual_end


def validate_event_geometry(event, dimensions, settings):
    """Check positions and the continuous path before either side is rendered."""
    for position in static.event_positions(event, settings):
        validate_position(position, dimensions, settings["min_source_distance"])
    if event.get("motion") is not None:
        if event.get("is_surround", False):
            raise ValueError("An event cannot be both moving and surround")
        event_trajectory(event, dimensions, settings)


def render_event(signal, event, dimensions, settings, rir_cache=None, side="target"):
    """Render one static/moving event and return its motion diagnostics."""
    motion = event.get("motion")
    if not isinstance(event.get("is_surround", False), bool):
        raise ValueError("is_surround must be a boolean")
    if motion is None:
        return static.render_static_event(signal, event, dimensions, settings, rir_cache), []
    if event.get("is_surround", False):
        raise ValueError("An event cannot be both moving and surround")
    position_at, actual_start, actual_end = event_trajectory(event, dimensions, settings)
    diagnostics = [{"side": side, "event_id": event.get("event_id"),
                    "kind": motion["kind"], "start": actual_start.tolist(),
                    "end": actual_end.tolist()}]
    if np.array_equal(actual_start, actual_end):
        return static.render_at_position(signal, actual_start, dimensions, settings, rir_cache), diagnostics
    sample_rate = settings["sample_rate"]
    output = np.zeros((len(signal) + math.ceil(settings["ir_seconds"] * sample_rate) + 1, 4), dtype=np.float32)
    used = len(signal)
    for fraction, offset, weights in motion_windows(len(signal), sample_rate, settings["motion_step_seconds"]):
        chunk = signal[offset:offset + len(weights)] * weights
        if not np.any(chunk):
            continue
        rendered = static.render_at_position(chunk, position_at(fraction), dimensions, settings, rir_cache)
        end = offset + len(rendered)
        if end > len(output):
            output = np.pad(output, ((0, end - len(output)), (0, 0)))
        output[offset:end] += rendered
        used = max(used, end)
    return output[:used], diagnostics


def apply_gain_envelope(signal, event):
    """Envelopes use absolute gains, so final gain must not be multiplied twice."""
    envelope = event.get("gain_envelope")
    gain = finite(event.get("gain", 1), "event.gain")
    if gain < 0:
        raise ValueError("event.gain must be nonnegative")
    if envelope is not None:
        start_gain, end_gain = finite(envelope["from"], "start_gain"), finite(envelope["to"], "end_gain")
        if min(start_gain, end_gain) < 0:
            raise ValueError("Gain envelope must be nonnegative")
        signal = signal * np.linspace(start_gain, end_gain, len(signal), dtype=np.float32)
        gain = 1.0  # Envelope stores absolute gain, not a factor on final gain.
    return signal, gain
