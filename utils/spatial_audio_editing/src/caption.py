"""Caption generation (rule-based templates or LLM) and spatial-to-English text helpers."""

import bisect
import os
import random
import time
from typing import List

import numpy as np

from utils.spatial_audio_editing.src.progress import write


# ============================================================================
# Cartesian position → English direction text
# ============================================================================

# Eight 45° sectors, clockwise from the front (listener faces -z, +x is right).
_SECTOR_BOUNDS = [k * (np.pi / 8) for k in (-7, -5, -3, -1, 1, 3, 5, 7)]
_BRACKET_DIRECTIONS = (
    "directly in front", "front-right", "directly to your right", "behind-right",
    "directly behind", "behind-left", "directly to your left", "front-left",
)
_NATURAL_DIRECTIONS = (
    "directly in front", "ahead on the right", "directly to your right", "behind on the right",
    "directly behind", "behind on the left", "directly to your left", "ahead on the left",
)


def _direction_index(x: float, z: float) -> int:
    """Index into the *_DIRECTIONS tuples for a horizontal position."""
    azimuth = np.arctan2(x, -z)
    return (bisect.bisect_right(_SECTOR_BOUNDS, azimuth) + 4) % 8


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def _snippet(text: str, max_len: int) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def position_text(x: float, z: float, y: float = None, include_distance: bool = True,
                   include_height: bool = True) -> str:
    """Cartesian (x=L/R, z=F/B, listener facing z-) → structured hint for LLM.

    Height is only appended when include_height=True AND |y| >= 1m.
    include_distance=False → omit distance (relocation).
    include_height=False  → omit height (all non-relocation, non-angle_change edits).
    """
    h_dist = np.sqrt(x**2 + z**2)

    if h_dist < 1.0 and y is not None and abs(y) >= 1.0 and include_height:
        return "[directly above]" if y > 0 else "[directly below]"
    if h_dist < 0.3:
        return "[right around you, ≈0m]"

    direction = _BRACKET_DIRECTIONS[_direction_index(x, z)]

    if include_height and y is not None and abs(y) >= 1.0:
        height = "above" if y > 0 else "below"
        direction = f"{direction}, {height}"

    if include_distance:
        return f"[{direction}, ≈{h_dist:.1f}m]"
    else:
        return f"[{direction}]"


def _natural_direction(x: float, z: float, include_distance: float = 1.0) -> str:
    """Cartesian position → natural language direction (no brackets, no numbers).

    include_distance: 0.0=never, 1.0=always.
    """
    h_dist = np.sqrt(x**2 + z**2)
    if h_dist < 0.3:
        return "nearby" if include_distance > 0 else "right around you"

    dir_str = _NATURAL_DIRECTIONS[_direction_index(x, z)]

    if include_distance <= 0:
        return dir_str

    if random.random() < include_distance:
        if h_dist < 1.0:
            dist = "nearby"
        elif h_dist < 3.0:
            dist = "a short distance away"
        elif h_dist < 5.0:
            dist = "a moderate distance away"
        else:
            dist = "a fair distance away"
        return f"{dir_str}, {dist}"

    return dir_str


def _natural_position(pos: dict, include_distance: float = 1.0,
                      include_height: bool = False) -> str:
    """Event position dict → natural language spatial description (no brackets)."""
    result = _natural_direction(pos["x"], pos["z"], include_distance=include_distance)
    if include_height:
        y = pos.get("y", 0)
        if abs(y) >= 1.0:
            height = "above" if y > 0 else "below"
            result = f"{result}, {height}"
    return result


def scene_to_text(scene: List[dict]) -> str:
    """Convert a scene to a concise spatial text description for text-to-audio.

    Format: "class1 in direction1, distance1; class2 in direction2, distance2"
    Speech events include a short transcript snippet.
    """
    parts = []
    for evt in scene:
        cls = _pretty(evt["class"])
        pos_desc = _natural_position(evt["position"], include_distance=1.0)

        if evt.get("domain") == "speech" and evt.get("transcript"):
            parts.append(f"{cls} '{_snippet(evt['transcript'], 50)}' in {pos_desc}")
        else:
            parts.append(f"{cls} in {pos_desc}")

    return "; ".join(parts)


def scene_to_lines(scene: List[dict]) -> List[str]:
    """Render a scene as human-readable lines for the LLM prompt, with A/B/C labels."""
    lines = []
    for i, evt in enumerate(scene):
        letter = chr(ord("A") + i)
        cls = _pretty(evt["class"])
        pos_start = position_text(evt["position"]["x"], evt["position"]["z"], include_height=False)
        time_str = f"{evt['start_time']:.1f}s–{evt['end_time']:.1f}s"
        gain_str = f", gain {evt['gain']:.1f}" if abs(evt["gain"] - 1.0) > 0.15 else ""

        if evt.get("domain") == "speech" and evt.get("transcript"):
            label = f'{cls} "{_snippet(evt["transcript"], 80)}"'
        else:
            label = cls

        lines.append(f"  {letter}. {label}  {pos_start}  {time_str}{gain_str}")
    return lines


def _event_letter(scene: List[dict], event_id: int) -> str:
    for i, e in enumerate(scene):
        if e["event_id"] == event_id:
            return chr(ord("A") + i)
    return "?"


def summarise_edit(info: dict, original_scene: List[dict] = None) -> str:
    """One-line structured description of an edit, used in the LLM prompt."""
    t = info["type"]

    def letter_of(e: dict) -> str:
        return _event_letter(original_scene, e["event_id"]) if original_scene else ""

    if t == "add_event":
        e = info["added"]
        pos = position_text(e["position"]["x"], e["position"]["z"], include_height=False)
        return f'[ADD] "{_pretty(e["class"])}" → at {pos}'

    if t == "remove_event":
        e = info["removed"]
        return f'[REMOVE] {letter_of(e)} "{_pretty(e["class"])}"'

    if t == "replace_event":
        old, new = info["old"], info["new"]
        letter = letter_of(old)
        old_lbl = f'speech "{_snippet(old["transcript"], 60)}"' if old.get("transcript") else _pretty(old["class"])
        new_lbl = f'speech "{_snippet(new["transcript"], 60)}"' if new.get("transcript") else _pretty(new["class"])

        cross = info.get("cross_domain", "")
        if cross == "speech_to_audio":
            return f'[REPLACE speech→audio] {letter} {old_lbl} → "{new_lbl}" (same position)'
        if cross == "audio_to_speech":
            return f'[REPLACE audio→speech] {letter} "{old_lbl}" → {new_lbl} (same position)'
        return f'[REPLACE] {letter} "{old_lbl}" → "{new_lbl}" (same position)'

    if t == "extract_event":
        n_removed = len(info["removed_events"])
        return f'[EXTRACT] Keep only "{_pretty(info["kept_class"])}", remove {n_removed} other event(s)'

    if t == "room_change":
        od, nd = info["old_dims"], info["new_dims"]
        return (f'[ROOM] {info["old_room"]} ({od["x"]}x{od["y"]}x{od["z"]}m) → '
                f'{info["new_room"]} ({nd["x"]}x{nd["y"]}x{nd["z"]}m)')

    if "event" not in info:
        return str(info)

    # Remaining edits modify a single existing event.
    e = info["event"]
    target = f'{letter_of(e)} "{_pretty(e["class"])}"'

    if t == "volume_change":
        return f'[VOLUME {info["direction"].upper()}] {target} (gain {info["old_gain"]}→{info["new_gain"]})'

    if t == "volume_fade":
        return f'[VOLUME FADE {info["direction"].upper()}] {target} (gain {info["start_gain"]}→{info["end_gain"]})'

    if t in ("relocation", "relocation_plane", "angle_change", "angle_change_plane"):
        with_height = t in ("relocation", "angle_change")
        old = position_text(**info["old_position"], include_distance=False, include_height=with_height)
        new = position_text(**info["new_position"], include_distance=False, include_height=with_height)
        tag = {"relocation": "RELOCATE", "relocation_plane": "RELOCATE PLANE",
               "angle_change": "ANGLE", "angle_change_plane": "ANGLE PLANE"}[t]
        suffix = f' (same distance ≈{info["keep_distance"]}m)' if t.startswith("angle") else ""
        return f'[{tag}] {target} from {old} → {new}{suffix}'

    if t == "distance_change":
        old = position_text(**info["old_position"], include_height=False)
        new = position_text(**info["new_position"], include_height=False)
        return f'[DISTANCE {info["direction"].upper()}] {target} from {old} → {new}'

    if t in ("distance_motion", "angle_motion"):
        start = position_text(**info["start_position"], include_height=False)
        end = position_text(**info["end_position"], include_height=False)
        if t == "distance_motion":
            return f'[DISTANCE MOTION {info["direction"].upper()}] {target} moves from {start} → {end}'
        return (f'[ANGLE MOTION] {target} arcs from {start} → {end} '
                f'(≈{info["keep_distance"]}m, {abs(info["rotation_deg"]):.0f}°)')

    return str(info)


# ============================================================================
# LLM prompts
# ============================================================================

CAPTION_SYSTEM_PROMPT = (
    "You are a precise spatial audio editor. Given a specific edit operation, "
    "write a short, natural English spoken instruction. "
    "Be accurate: the verb in your caption MUST match the edit type.\n\n"
    "=== STATIC edits (sound stays at a fixed position/volume): ===\n"
    '- ADD: "add", "insert", "place", "put"\n'
    '- REMOVE: "remove", "delete", "take out", "cut"\n'
    '- REPLACE (within-domain): "replace", "swap", "change X to Y" (same position, different sound class)\n'
    '- REPLACE (cross-domain speech→audio): "replace the speech X with Y", "swap the spoken words X for Y sound" (speech event replaced by sound effect)\n'
    '- REPLACE (cross-domain audio→speech): "replace the X with someone saying Y", "swap the X for a voice saying Y" (sound effect replaced by speech)\n'
    '- VOLUME CHANGE: "turn up/down", "boost", "lower", "raise" (static gain adjustment)\n'
    '- DISTANCE CHANGE: "move closer", "push farther" (static, same direction, changed distance)\n'
    '- ANGLE CHANGE: "shift to the left/right", "move to the side" (static, same distance, changed direction and height — do NOT mention distance)\n'
    '- ANGLE CHANGE PLANE: "shift to the left/right", "move to the side" (static, same distance, changed direction on the horizontal plane ONLY — do NOT mention height or distance)\n'
    '- RELOCATION: "move", "shift", "relocate", "reposition" (static, change direction and height — do NOT mention distance)\n'
    '- RELOCATION PLANE: "move", "shift", "relocate", "reposition" (static, change direction on the horizontal plane ONLY — do NOT mention height or distance)\n\n'
    "=== DYNAMIC edits (sound property CHANGES CONTINUOUSLY during playback): ===\n"
    '- DISTANCE MOTION: "approaches", "recedes", "draws closer", "moves away" — sound moves along a radial line\n'
    '- ANGLE MOTION: "circles around", "rotates around", "arcs", "pans around you" — sound moves along an arc, same distance\n'
    '- VOLUME FADE: "fades in", "fades out", "gradually gets louder/quieter" — gain changes over time\n\n'
    "=== FILTER edit (keeps one sound class, removes all others): ===\n"
    '- EXTRACT: "keep only", "extract", "isolate", "leave only", "solo", "remove everything except" (one sound class remains, all other events are removed)\n\n'
    "=== ENVIRONMENT edit (changes the whole acoustic space): ===\n"
    '- ROOM CHANGE: "make the room larger/smaller", "expand/shrink the space", "switch to a small/large/huge room" (all sound events remain the same, only the room size changes)\n\n'
    "CRITICAL: For DYNAMIC edits, the caption MUST convey continuous motion/change, not a static state.\n"
    "For STATIC edits, describe the resulting state, not motion.\n"
    "For ROOM CHANGE, describe the change of room size.\n\n"
    "Bracket values like [front-right, ≈3.2m] are spatial hints; rephrase them qualitatively "
    "but PRESERVE the full directional precision:\n"
    '  "directly in front" → "directly in front"  |  "front-right" → "ahead on the right"\n'
    '  "directly behind" → "directly behind"      |  "behind-left" → "behind on the left"\n'
    '  "directly to your right/left" → keep as-is |  "behind-right" → "behind on the right"  |  "front-left" → "ahead on the left"\n'
    "For distance (when included):  ≈0.5m → nearby,  ≈2m → a short distance away,  "
    "≈3.5m → a moderate distance away,  ≈5m+ → a fair distance away.\n\n"
    "HEIGHT RULES (CRITICAL):\n"
    "- ONLY RELOCATION and ANGLE CHANGE edits may mention above/below/height in the caption.\n"
    "- For ALL other edit types, strictly NEVER mention above, below, height, or elevation.\n"
    "- RELOCATION and ANGLE CHANGE brackets do NOT include distance — only describe direction and height.\n\n"
    "Constraints:\n"
    "- For RELOCATION, RELOCATION PLANE, DISTANCE CHANGE, ANGLE CHANGE, and ANGLE CHANGE PLANE: randomly make 1 or 2 of the 3 captions "
    "detailed (\"from X to Y\" including both source and target positions), and the rest simple "
    "(only target position, e.g. \"to Y\").\n"
    "- Do NOT drop the left/right part of the direction.\n"
    "- Do NOT copy bracket text verbatim and do NOT mention exact distances or numbers.\n"
    "- Do NOT invent effects that are not described.\n"
    "- Each caption must be 10-25 words, one sentence.\n"
    "- Vary your phrasing and spatial descriptions across samples.\n"
    "- Output exactly 3 lines, one caption per line, no numbering, no quotes."
)

CAPTION_USER_TEMPLATE = """Room: {room}
Scene before editing:
{original}

Edit: {edit_detail}

Write 3 different English instructions for this exact edit."""


# ============================================================================
# Fallback (template-based, no LLM)
# ============================================================================

def _detailed_and_simple(detailed: list, simple: list) -> list:
    """3 captions: 1-2 detailed ("from X to Y") plus simple ("to Y") ones."""
    n_detailed = random.choice([1, 2])
    return random.sample(detailed, n_detailed) + random.sample(simple, 3 - n_detailed)


def _fallback_caption(info: dict) -> list:
    """Rule-based captions: 3 phrasings sampled from per-edit-type template pools."""
    t = info["type"]

    # -- add_event ----------------------------------------------------------
    if t == "add_event":
        cls = _pretty(info["added"]["class"])
        pos = _natural_position(info["added"]["position"])
        pool = [
            f"Add a {cls} sound {pos}.",
            f"Place {cls} {pos}.",
            f"Insert {cls} {pos}.",
            f"Add {cls} positioned {pos}.",
            f"Place a {cls} sound {pos}.",
            f"Insert a {cls} effect {pos}.",
            f"Put a {cls} sound {pos}.",
            f"Add {cls} audio {pos}.",
            f"add a {cls} sound {pos}.",
            f"place {cls} {pos}.",
            f"insert {cls} {pos}.",
            f"Add a {cls} noise {pos}.",
            f"Insert the {cls} sound {pos}.",
            f"Place the {cls} sound {pos}.",
            f"Add the sound of {cls} {pos}.",
            f"Insert a {cls} sound {pos}.",
            f"Add a {cls} {pos}.",
            f"Place a {cls} {pos}.",
            f"Put {cls} {pos}.",
            f"Add the {cls} sound {pos}.",
        ]
        return random.sample(pool, 3)

    # -- remove_event -------------------------------------------------------
    if t == "remove_event":
        cls = _pretty(info["removed"]["class"])
        pool = [
            f"Remove the {cls} sound.",
            f"Delete the {cls}.",
            f"Take out the {cls}.",
            f"Cut the {cls} from the scene.",
            f"Remove the {cls} from the scene.",
            f"Delete the {cls} audio.",
            f"Take the {cls} out.",
            f"Strip the {cls} sound.",
            f"remove the {cls} sound.",
            f"delete the {cls}.",
            f"take out the {cls}.",
            f"cut out the {cls}.",
            f"Remove the {cls} sound event.",
            f"Delete the {cls} from the scene.",
            f"Take out the {cls} sound event.",
            f"Cut out the {cls} from the scene.",
            f"Remove the {cls} completely.",
            f"Delete the {cls} sound event entirely.",
            f"Drop the {cls} from the scene.",
            f"Eliminate the {cls} sound.",
        ]
        return random.sample(pool, 3)

    # -- replace_event ------------------------------------------------------
    if t == "replace_event":
        cross = info.get("cross_domain", "")
        old_transcript = info["old"].get("transcript", "")
        new_transcript = info["new"].get("transcript", "")
        old_cls = _pretty(info["old"]["class"])
        new_cls = _pretty(info["new"]["class"])

        if cross == "speech_to_audio":
            # Speech event replaced by audio
            snippet = _snippet(old_transcript, 50)
            pool = [
                f'Replace the speech "{snippet}" with {new_cls}.',
                f'Swap the speech "{snippet}" for {new_cls}.',
                f'Change the speech "{snippet}" to {new_cls}.',
                f'Replace the spoken words "{snippet}" with a {new_cls} sound.',
                f'Swap out the speech "{snippet}" and put in {new_cls}.',
                f'Change "{snippet}" to {new_cls}, keeping the same location.',
                f'Substitute {new_cls} for the speech "{snippet}".',
                f'Replace "{snippet}" with {new_cls} at the same spot.',
                f'Take out the speech "{snippet}" and add {new_cls}.',
                f'Swap the spoken "{snippet}" for a {new_cls} sound.',
                f'Change "{snippet}" to the sound of {new_cls}.',
                f'Replace the voice saying "{snippet}" with {new_cls}.',
                f'Drop the speech "{snippet}" and put {new_cls} in its place.',
                f'Remove the spoken words "{snippet}" and insert {new_cls}.',
                f'Swap out the voice for a {new_cls} sound.',
                f'Replace "{snippet}" with the sound of {new_cls} at the same location.',
                f'Instead of the speech "{snippet}", play a {new_cls} sound.',
                f'Change the speech "{snippet}" into a {new_cls} sound.',
                f'Remove "{snippet}" and substitute a {new_cls} in its place.',
                f'Swap the speech "{snippet}" for the sound of {new_cls}, same position.',
            ]
        elif cross == "audio_to_speech":
            # Audio event replaced by speech
            snippet = _snippet(new_transcript, 50)
            pool = [
                f'Replace the {old_cls} with someone saying "{snippet}".',
                f'Swap the {old_cls} for speech: "{snippet}".',
                f'Change the {old_cls} to a voice saying "{snippet}".',
                f'Replace the {old_cls} sound with the spoken words "{snippet}".',
                f'Swap out the {old_cls} and put in speech: "{snippet}".',
                f'Change the {old_cls} to someone saying "{snippet}", keeping the same location.',
                f'Substitute speech "{snippet}" for the {old_cls}.',
                f'Replace the {old_cls} with a {new_cls} voice saying "{snippet}".',
                f'Take out the {old_cls} and add spoken words: "{snippet}".',
                f'Swap the {old_cls} for a voice saying "{snippet}".',
                f'Change the {old_cls} to the sound of someone saying "{snippet}".',
                f'Replace the {old_cls} sound with a voice: "{snippet}".',
                f'Drop the {old_cls} and put in a {new_cls} voice saying "{snippet}".',
                f'Remove the {old_cls} and insert speech: "{snippet}".',
                f'Instead of the {old_cls}, play someone saying "{snippet}".',
                f'Swap the {old_cls} sound for spoken words: "{snippet}".',
                f'Change the {old_cls} into a voice saying "{snippet}" at the same spot.',
                f'Substitute the {old_cls} with a {new_cls} speaker saying "{snippet}".',
                f'Take the {old_cls} out and replace it with the speech "{snippet}".',
                f'Replace the {old_cls} with someone speaking: "{snippet}", same position.',
            ]
        else:
            # Within-domain replace
            old, new = old_cls, new_cls
            pool = [
                f"Replace the {old} with {new}.",
                f"Swap the {old} for {new}.",
                f"Change the {old} to {new}.",
                f"Replace the {old} with {new} at the same position.",
                f"Swap out the {old} and put in {new}.",
                f"Change the {old} to {new}, keeping the same location.",
                f"Substitute {new} for the {old}.",
                f"Replace the {old} sound with {new} at the same spot.",
                f"replace the {old} with {new} at the same position.",
                f"swap the {old} for {new} at the same position.",
                f"change the {old} to {new} at the same position.",
                f"Swap the {old} with {new}.",
                f"Change the {old} to {new} without moving it.",
                f"Swap out the {old} for {new}.",
                f"Replace the {old} with {new}, keeping it in place.",
                f"Change the {old} to {new} at the same spot.",
                f"Drop the {old} and put in {new} instead.",
                f"Remove the {old} and insert {new} at the same position.",
                f"Swap in {new} for the {old}, same location.",
                f"Replace the {old} with {new} without moving it.",
            ]
        return random.sample(pool, 3)

    # -- volume_change ------------------------------------------------------
    if t == "volume_change":
        cls = _pretty(info["event"]["class"])
        d = info["direction"]
        if d == "up":
            pool = [
                f"Turn up the {cls}.",
                f"Boost the volume of the {cls}.",
                f"Make the {cls} louder.",
                f"Increase the volume of the {cls}.",
                f"Raise the {cls}'s gain.",
                f"Make the {cls} sound louder.",
                f"Turn up the volume of the {cls}.",
                f"Boost the {cls} to a higher level.",
                f"turn up the {cls}.",
                f"boost the volume of the {cls}.",
                f"make the {cls} louder.",
                f"Increase the gain of the {cls}.",
                f"Raise the volume of the {cls}.",
                f"Boost the {cls}.",
                f"Make the {cls} significantly louder.",
                f"Turn up the {cls} sound.",
                f"Boost the {cls}'s volume.",
                f"Increase the loudness of the {cls}.",
                f"Make the {cls} much louder.",
                f"Amplify the {cls} sound.",
            ]
        else:
            pool = [
                f"Turn down the {cls}.",
                f"Lower the volume of the {cls}.",
                f"Make the {cls} quieter.",
                f"Reduce the {cls}'s gain.",
                f"Decrease the volume of the {cls}.",
                f"Make the {cls} sound quieter.",
                f"Turn down the volume of the {cls}.",
                f"Lower the {cls} to a softer level.",
                f"turn down the {cls}.",
                f"lower the volume of the {cls}.",
                f"make the {cls} quieter.",
                f"Reduce the volume of the {cls}.",
                f"Lower the gain of the {cls}.",
                f"Reduce the {cls}.",
                f"Make the {cls} much quieter.",
                f"Turn down the {cls} sound.",
                f"Lower the {cls}'s volume.",
                f"Reduce the loudness of the {cls}.",
                f"Make the {cls} very quiet.",
                f"Attenuate the {cls} sound.",
            ]
        return random.sample(pool, 3)

    # -- relocation / relocation_plane (*_plane never mentions height) ---
    if t in ("relocation", "relocation_plane"):
        cls = _pretty(info["event"]["class"])
        with_height = t == "relocation"
        old = _natural_position(info["old_position"], include_distance=0.0, include_height=with_height)
        new = _natural_position(info["new_position"], include_distance=0.0, include_height=with_height)
        detailed = [
            f"Move the {cls} from {old} to {new}.",
            f"Relocate the {cls} from {old} to {new}.",
            f"Shift the {cls} from {old} to {new}.",
            f"Move the {cls} sound from {old} to {new}.",
            f"Relocate the {cls} sound from {old} to {new}.",
            f"Shift the {cls} sound from {old} to {new}.",
            f"Move the {cls} from {old} over to {new}.",
            f"Relocate the {cls} from {old} over to {new}.",
            f"Shift the {cls} from {old} to a position {new}.",
            f"Take the {cls} and move it from {old} to {new}.",
            f"Reposition the {cls} from {old} to {new}.",
            f"Move the {cls} from its current spot at {old} over to {new}.",
            f"Change the {cls}'s location from {old} to {new}.",
            f"Take the {cls} sound and relocate it from {old} to {new}.",
            f"Pick up the {cls} from {old} and place it {new}.",
            f"Shift the {cls}'s position from {old} to {new}.",
            f"Move the {cls} away from {old} and put it {new}.",
            f"Reposition the {cls} sound from {old} to {new}.",
            f"Transfer the {cls} from {old} to a new spot {new}.",
            f"Move the {cls} from {old} so it ends up {new}.",
        ]
        simple = [
            f"Move the {cls} to {new}.",
            f"Shift the {cls} to {new}.",
            f"Relocate the {cls} to {new}.",
            f"Move the {cls} sound to {new}.",
            f"Shift the {cls} to a position {new}.",
            f"Reposition the {cls} to {new}.",
            f"Position the {cls} to {new}.",
            f"Move the {cls} over to {new}.",
            f"Shift the {cls} sound to {new}.",
            f"Relocate the {cls} sound to {new}.",
            f"Place the {cls} at {new}.",
            f"Move the {cls} so it is {new}.",
            f"Put the {cls} {new}.",
            f"Reposition the {cls} sound to {new}.",
            f"Set the {cls} to a new position {new}.",
            f"Shift the {cls} over so it's {new}.",
            f"Move the {cls}'s location to {new}.",
            f"Place the {cls} sound {new}.",
            f"Change the {cls}'s position to {new}.",
            f"Put the {cls} at a new spot {new}.",
        ]
        return _detailed_and_simple(detailed, simple)

    # -- distance_change ----------------------------------------------------
    if t == "distance_change":
        cls = _pretty(info["event"]["class"])
        old = _natural_direction(info["old_position"]["x"], info["old_position"]["z"], include_distance=1.0)
        new = _natural_direction(info["new_position"]["x"], info["new_position"]["z"], include_distance=1.0)
        d = info["direction"]
        if d == "farther":
            detailed = [
                f"Move the {cls} farther, from {old} to {new}.",
                f"Push the {cls} from {old} to {new}.",
                f"Shift the {cls} farther from {old} to {new}.",
                f"Move the {cls} sound farther away, from {old} to {new}.",
                f"Relocate the {cls} from {old} to {new}, farther away.",
                f"Push the {cls} farther away, from {old} to {new}.",
                f"Move the {cls} from {old} to {new}, making it farther away.",
                f"Extend the {cls} from {old} to {new}.",
                f"Shift the {cls} sound farther from {old} to {new}.",
                f"Relocate the {cls} from {old} to {new}, pushing it farther out.",
                f"Move the {cls} from {old} out to {new}, farther from you.",
                f"Push the {cls} sound from {old} to a more distant {new}.",
                f"Take the {cls} from {old} and push it farther to {new}.",
                f"Reposition the {cls} from {old} to {new}, increasing the distance.",
                f"Shift the {cls} from {old} farther outward to {new}.",
                f"Move the {cls} away from you, from {old} to {new}.",
                f"Extend the {cls} sound from {old} out to {new}.",
                f"Push the {cls} from {old} so it ends up farther at {new}.",
                f"Relocate the {cls} from {old} to a farther position {new}.",
                f"Move the {cls} from {old} farther out so it's {new}.",
            ]
            simple = [
                f"Push the {cls} farther to {new}.",
                f"Move the {cls} to a greater distance {new}.",
                f"Make the {cls} farther away at {new}.",
                f"Shift the {cls} farther away to {new}.",
                f"Push the {cls} sound farther away to {new}.",
                f"Relocate the {cls} to a more distant {new}.",
                f"Extend the {cls} to {new}.",
                f"Move the {cls} farther out to {new}.",
                f"Push the {cls} away to {new}.",
                f"Push the {cls} to a greater distance at {new}.",
                f"Move the {cls} farther so it's {new}.",
                f"Make the {cls} more distant at {new}.",
                f"Push the {cls} out to {new}.",
                f"Set the {cls} farther away at {new}.",
                f"Place the {cls} farther out at {new}.",
                f"Move the {cls} away so it ends up {new}.",
                f"Relocate the {cls} farther to {new}.",
                f"Shift the {cls} to a more distant spot {new}.",
                f"Push the {cls} farther out so it's {new}.",
                f"Move the {cls} to {new}, farther away from you.",
            ]
        else:
            detailed = [
                f"Move the {cls} closer, from {old} to {new}.",
                f"Bring the {cls} from {old} to {new}.",
                f"Shift the {cls} closer from {old} to {new}.",
                f"Move the {cls} sound nearer, from {old} to {new}.",
                f"Relocate the {cls} from {old} to {new}, closer to you.",
                f"Bring the {cls} closer, from {old} to {new}.",
                f"Move the {cls} from {old} to {new}, bringing it closer.",
                f"Pull the {cls} nearer from {old} to {new}.",
                f"Bring the {cls} nearer from {old} to {new}.",
                f"Shift the {cls} nearer from {old} to {new}.",
                f"Move the {cls} from {old} in closer to {new}.",
                f"Bring the {cls} sound from {old} to a nearer {new}.",
                f"Take the {cls} from {old} and bring it closer to {new}.",
                f"Reposition the {cls} from {old} to {new}, reducing the distance.",
                f"Shift the {cls} from {old} inward to {new}.",
                f"Move the {cls} toward you, from {old} to {new}.",
                f"Pull the {cls} sound from {old} closer to {new}.",
                f"Bring the {cls} from {old} so it ends up nearer at {new}.",
                f"Relocate the {cls} from {old} to a closer position {new}.",
                f"Move the {cls} from {old} nearer so it's {new}.",
            ]
            simple = [
                f"Bring the {cls} closer to {new}.",
                f"Move the {cls} to a nearer position {new}.",
                f"Make the {cls} closer at {new}.",
                f"Shift the {cls} nearer to {new}.",
                f"Bring the {cls} sound closer to {new}.",
                f"Relocate the {cls} to a closer {new}.",
                f"Bring the {cls} nearer to {new}.",
                f"Pull the {cls} closer to {new}.",
                f"Move the {cls} nearer to {new}.",
                f"Bring the {cls} in closer to {new}.",
                f"Move the {cls} closer so it's {new}.",
                f"Make the {cls} nearer at {new}.",
                f"Pull the {cls} in to {new}.",
                f"Set the {cls} closer at {new}.",
                f"Place the {cls} nearer at {new}.",
                f"Move the {cls} toward you so it's {new}.",
                f"Relocate the {cls} closer to {new}.",
                f"Shift the {cls} to a nearer spot {new}.",
                f"Bring the {cls} in so it's {new}.",
                f"Move the {cls} to {new}, closer to you.",
            ]
        return _detailed_and_simple(detailed, simple)

    # -- angle_change / angle_change_plane (*_plane never mentions height) ---
    if t in ("angle_change", "angle_change_plane"):
        cls = _pretty(info["event"]["class"])
        with_height = t == "angle_change"
        old = _natural_position(info["old_position"], include_distance=0.0, include_height=with_height)
        new = _natural_position(info["new_position"], include_distance=0.0, include_height=with_height)
        detailed = [
            f"Move the {cls} from {old} to {new}, at the same distance.",
            f"Change the direction of the {cls} from {old} to {new}.",
            f"Shift the {cls} from {old} to {new}, keeping the same distance.",
            f"Reposition the {cls} from {old} to {new}, without changing how far away it is.",
            f"Move the {cls} sound from {old} to {new}, maintaining its distance.",
            f"Relocate the {cls} from {old} to {new}, keeping its distance unchanged.",
            f"Move the {cls} around you from {old} to {new}.",
            f"Shift the {cls} from {old} to {new}, while keeping the same distance.",
            f"Relocate the {cls} from {old} to {new}, without changing distance.",
            f"Move the {cls} sound around you from {old} to {new}.",
            f"Rotate the {cls} from {old} to {new}, same distance away.",
            f"Shift the {cls} sideways from {old} to {new} at the same distance.",
            f"Move the {cls} from {old} around to {new}, maintaining distance.",
            f"Change the {cls}'s angle from {old} to {new}, keeping its distance.",
            f"Pan the {cls} from {old} over to {new}, without altering how far it is.",
            f"Take the {cls} from {old} and shift it to {new}, same distance.",
            f"Move the {cls} laterally from {old} to {new}, keeping the same distance.",
            f"Reposition the {cls} from {old} to {new}, maintaining its distance from you.",
            f"Swing the {cls} around from {old} to {new} at the same distance.",
            f"Shift the {cls}'s direction from {old} to {new}, distance unchanged.",
        ]
        simple = [
            f"Shift the {cls} to {new}.",
            f"Move the {cls} to {new} at the same distance.",
            f"Change the {cls}'s direction to {new}.",
            f"Reposition the {cls} to {new}.",
            f"Shift the {cls} sideways to {new}.",
            f"Move the {cls} sound to {new}.",
            f"Change the position of the {cls} to {new}, maintaining the distance.",
            f"Relocate the {cls} to {new}, keeping it at the same distance.",
            f"Move the {cls} to {new}, while keeping the same distance.",
            f"Shift the {cls} to {new}, without changing how far away it is.",
            f"Rotate the {cls} to {new}, same distance.",
            f"Pan the {cls} over to {new}.",
            f"Move the {cls} sideways so it's {new}.",
            f"Change the {cls} to face {new}, at the same distance.",
            f"Shift the {cls} around so it's {new}.",
            f"Move the {cls} to {new}, keeping the distance unchanged.",
            f"Swing the {cls} over to {new}.",
            f"Reposition the {cls} so it's {new}, same distance.",
            f"Move the {cls} laterally to {new}.",
            f"Shift the {cls} so it ends up {new}, maintaining distance.",
        ]
        return _detailed_and_simple(detailed, simple)

    # -- distance_motion ----------------------------------------------------
    if t == "distance_motion":
        cls = _pretty(info["event"]["class"])
        start = _natural_position(info["start_position"], include_distance=0.0)
        if info["direction"] == "approaching":
            pool = [
                f"The {cls} gradually draws closer from {start}.",
                f"The {cls} approaches from {start}, getting nearer over time.",
                f"The {cls} moves closer from {start} as it plays.",
                f"The sound of {cls} comes closer from {start}.",
                f"The {cls} draws closer from {start}, approaching over its duration.",
                f"Have the {cls} approach from {start}.",
                f"Make the {cls} move closer from {start} over time.",
                f"The {cls} comes toward you from {start}.",
                f"Let the {cls} approach from {start}, drawing nearer as it plays.",
                f"The {cls} advances from {start} toward you.",
                f"Make the {cls} gradually approach from {start}.",
                f"The {cls} moves in your direction from {start}.",
                f"Have the {cls} sound draw closer from {start}.",
                f"The {cls} travels toward you from {start} over its duration.",
                f"Let the {cls} come nearer from {start} over time.",
                f"The {cls} closes in from {start} as it plays.",
                f"Make the {cls} sound approach from {start} gradually.",
                f"The {cls} moves toward the listener from {start}.",
                f"Have the {cls} get closer and closer from {start}.",
                f"The {cls} nears you from {start} over its duration.",
            ]
        else:
            pool = [
                f"The {cls} gradually recedes from {start}.",
                f"The {cls} moves away from {start}, getting farther over time.",
                f"The {cls} recedes from {start} as it plays.",
                f"The sound of {cls} fades into the distance from {start}.",
                f"The {cls} moves farther away from {start} over its duration.",
                f"Have the {cls} recede from {start}.",
                f"Make the {cls} move farther from {start} over time.",
                f"The {cls} retreats from {start} into the distance.",
                f"Let the {cls} recede from {start}, growing more distant as it plays.",
                f"The {cls} pulls away from you starting at {start}.",
                f"Make the {cls} gradually recede from {start}.",
                f"The {cls} moves away into the distance from {start}.",
                f"Have the {cls} sound grow more distant from {start}.",
                f"The {cls} travels away from you from {start} over its duration.",
                f"Let the {cls} move farther away from {start} over time.",
                f"The {cls} withdraws from {start} as it plays.",
                f"Make the {cls} sound recede from {start} gradually.",
                f"The {cls} moves away from the listener starting at {start}.",
                f"Have the {cls} get farther and farther from {start}.",
                f"The {cls} distances itself from you starting at {start}.",
            ]
        return random.sample(pool, 3)

    # -- angle_motion -------------------------------------------------------
    if t == "angle_motion":
        cls = _pretty(info["event"]["class"])
        start = _natural_position(info["start_position"], include_distance=0.0)
        end = _natural_position(info["end_position"], include_distance=0.0)
        pool = [
            f"The {cls} arcs around you from {start} to {end}.",
            f"Make the {cls} arc from {start} to {end} at a constant distance.",
            f"The {cls} circles around you from {start} to {end}.",
            f"Arc the {cls} from {start} to {end}.",
            f"Make the {cls} move in an arc from {start} to {end}.",
            f"Pan the {cls} around from {start} to {end}.",
            f"The {cls} rotates around you from {start} to {end}.",
            f"Let the {cls} travel along an arc from {start} to {end}.",
            f"Make the {cls} circle around you from {start} to {end}.",
            f"The {cls} sweeps around you from {start} to {end}.",
            f"Move the {cls} in an arc from {start} to {end}.",
            f"Have the {cls} arc around you from {start} to {end}.",
            f"Let the {cls} sweep from {start} to {end} around you.",
            f"The {cls} glides around you from {start} to {end}.",
            f"Make the {cls} travel in a circle from {start} to {end}.",
            f"The {cls} orbits around you from {start} to {end}.",
            f"Have the {cls} pan around from {start} to {end} at a fixed distance.",
            f"Let the {cls} revolve around you from {start} to {end}.",
            f"Make the {cls} swing around from {start} to {end}.",
            f"The {cls} traces an arc from {start} to {end} around you.",
        ]
        return random.sample(pool, 3)

    # -- volume_fade --------------------------------------------------------
    if t == "volume_fade":
        cls = _pretty(info["event"]["class"])
        d = info["direction"]
        if d == "in":
            pool = [
                f"Fade in the {cls} over its duration.",
                f"Make the {cls} gradually get louder.",
                f"Bring up the {cls} volume over time.",
                f"Gradually increase the volume of the {cls}.",
                f"Slowly fade in the {cls} sound.",
                f"Make the {cls} grow louder as it plays.",
                f"Bring the {cls} in gradually from quiet to audible.",
                f"Let the {cls} swell in volume over time.",
                f"Gradually bring up the {cls}.",
                f"Fade in the {cls} sound.",
                f"Raise the {cls} volume over time.",
                f"Increase the {cls} volume slowly until it reaches normal level.",
                f"Start the {cls} quietly and let it build up over time.",
                f"Make the {cls} fade in from silence.",
                f"Let the {cls} emerge gradually, growing louder.",
                f"Bring the {cls} from quiet to full volume over its duration.",
                f"Have the {cls} ramp up in volume as it plays.",
                f"Gradually raise the {cls} from a low level to normal.",
                f"Make the {cls} sound swell from soft to loud.",
                f"Let the {cls} fade up from quiet over time.",
            ]
        else:
            pool = [
                f"Fade out the {cls} over its duration.",
                f"Make the {cls} gradually get quieter.",
                f"Dial down the {cls} volume over time.",
                f"Gradually reduce the volume of the {cls}.",
                f"Slowly fade out the {cls} sound.",
                f"Make the {cls} grow quieter as it plays.",
                f"Let the {cls} fade away gradually.",
                f"Reduce the {cls} until it fades to silence.",
                f"Gradually fade out the {cls}.",
                f"Fade out the {cls} sound.",
                f"Lower the {cls} volume over time.",
                f"Decrease the gain of the {cls} until it becomes very quiet.",
                f"Start the {cls} at normal volume and let it trail off.",
                f"Make the {cls} fade out to silence.",
                f"Let the {cls} diminish gradually, growing quieter.",
                f"Bring the {cls} from full volume down to quiet over its duration.",
                f"Have the {cls} ramp down in volume as it plays.",
                f"Gradually lower the {cls} from normal level to near silence.",
                f"Make the {cls} sound fade from loud to soft.",
                f"Let the {cls} fade down to quiet over time.",
            ]
        return random.sample(pool, 3)

    # -- extract_event -------------------------------------------------------
    if t == "extract_event":
        cls = _pretty(info["kept_class"])
        n_removed = len(info["removed_events"])
        pool = [
            f"Keep only the {cls}, removing all other sounds.",
            f"Extract the {cls} sound, dropping everything else.",
            f"Isolate the {cls} from the scene.",
            f"Leave only the {cls} in the scene.",
            f"Remove all sounds except the {cls}.",
            f"Keep the {cls} and remove all other audio.",
            f"Extract only the {cls}, discarding the rest.",
            f"Strip away everything but the {cls}.",
            f"Keep just the {cls} sound.",
            f"Isolate the {cls} sound, removing other events.",
            f"Extract the {cls} from the mix.",
            f"Leave the {cls} and take out everything else.",
            f"Keep the {cls} only, dropping {n_removed} other sound(s).",
            f"Preserve the {cls} and remove the other sounds.",
            f"Extract and keep only the {cls}.",
            f"Solo the {cls}, muting all other sounds.",
            f"Keep the {cls} and discard everything else.",
            f"Remove every sound except the {cls}.",
            f"Let only the {cls} remain, cutting the rest.",
            f"Mute everything but the {cls}.",
        ]
        return random.sample(pool, 3)

    # -- room_change --------------------------------------------------------
    if t == "room_change":
        old_r = info["old_room"]
        new_r = info["new_room"]
        size_order = {"small": 0, "medium": 1, "large": 2, "huge": 3}
        shrinking = size_order.get(new_r, 0) < size_order.get(old_r, 0)
        if shrinking:
            pool = [
                f"Make the room {new_r}.",
                f"Shrink the space to a {new_r} room.",
                f"Switch to a {new_r} room.",
                f"Make the room smaller, to a {new_r} space.",
                f"Shrink the acoustic space to {new_r}.",
                f"Reduce the room size to {new_r}.",
                f"Transform the space into a {new_r} room.",
                f"Make the space more confined, switching to {new_r}.",
                f"Shrink the room to a {new_r} space.",
                f"Make the room {new_r} while keeping all sounds in their positions.",
                f"Switch from a {old_r} room to a {new_r} room.",
                f"Shrink the room to {new_r}, making the space more intimate.",
                f"Change the room to a {new_r} space.",
                f"Convert the {old_r} room to a {new_r} space.",
                f"Tighten the space to a {new_r} room.",
                f"Make the environment more compact, switching to a {new_r} room.",
                f"Reduce the acoustic environment to a {new_r} space.",
                f"Scale the room down to {new_r}.",
                f"Bring the walls in to create a {new_r} room.",
                f"Shrink the environment to a {new_r} space.",
            ]
        else:
            pool = [
                f"Make the room {new_r}.",
                f"Expand the space to a {new_r} room.",
                f"Switch to a {new_r} room.",
                f"Make the room much larger, to a {new_r} space.",
                f"Expand the acoustic space to {new_r}.",
                f"Open up the room to {new_r}.",
                f"Transform the space into a {new_r} room.",
                f"Make the space vast and spacious, switching to {new_r}.",
                f"Expand the room to a {new_r} space.",
                f"Make the room {new_r} while keeping all sounds in their positions.",
                f"Switch from an {old_r} room to a {new_r} room.",
                f"Expand the room to {new_r}, creating a more open space.",
                f"Change the room to a {new_r} space.",
                f"Convert the {old_r} room to a {new_r} space.",
                f"Widen the space to a {new_r} room.",
                f"Make the environment more expansive, switching to a {new_r} room.",
                f"Enlarge the acoustic environment to a {new_r} space.",
                f"Scale the room up to {new_r}.",
                f"Push the walls out to create a {new_r} room.",
                f"Expand the environment to a {new_r} space.",
            ]
        return random.sample(pool, 3)

    return ["Edit the audio scene."] * 3


# ============================================================================
# Multi-operation captions: join single-edit clauses into one instruction
# ============================================================================

def _strip_sentence(text: str) -> str:
    return text.strip().strip('"').strip("'").rstrip(".").strip()


def _lower_first(text: str) -> str:
    text = _strip_sentence(text)
    return text[0].lower() + text[1:] if text else text


def _capitalize_first(text: str) -> str:
    return text[0].upper() + text[1:]


def join_clauses(clauses: List[str], variant: int = 0) -> str:
    """Join per-edit captions into one sequential instruction.

    variant 0: "First A, then B, and finally C."
    variant 1: "A; then B; then C."           (2 clauses: "A, and then B.")
    variant 2: "A, B, and then C."            (2 clauses: "A; then B.")
    """
    clauses = [_lower_first(c) for c in clauses if _strip_sentence(c)]
    if not clauses:
        return "Edit the audio scene."
    if len(clauses) == 1:
        return _capitalize_first(clauses[0]) + "."
    if len(clauses) == 2:
        if variant == 0:
            return f"First {clauses[0]}, then {clauses[1]}."
        if variant == 1:
            return f"{_capitalize_first(clauses[0])}, and then {clauses[1]}."
        return f"{_capitalize_first(clauses[0])}; then {clauses[1]}."

    if variant == 0:
        middle = ", then ".join(clauses[1:-1])
        return f"First {clauses[0]}, then {middle}, and finally {clauses[-1]}."
    if variant == 1:
        return _capitalize_first("; then ".join(clauses)) + "."
    return f"{_capitalize_first(', '.join(clauses[:-1]))}, and then {clauses[-1]}."


# ============================================================================
# Caption generator: rule-based templates or LLM (zhipu / deepseek / openai)
# ============================================================================

API_KEY_ENV_VARS = {"zhipu": "ZHIPUAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY"}


class CaptionGenerator:
    """Rule-based captions (caption_mode="rule") or an LLM client (caption_mode="api").

    API keys are read from the environment (see API_KEY_ENV_VARS); a "key" field
    in the config profile is also accepted but should not be committed.
    """

    def __init__(self, cfg: dict):
        api_cfg = cfg["api"]
        self.temperature = api_cfg.get("temperature", 0.9)
        self.max_tokens = api_cfg.get("max_tokens", 384)
        self.caption_mode = api_cfg.get("caption_mode", "api")
        self._client = None

        if self.caption_mode == "rule":
            return

        if self.caption_mode != "api":
            raise ValueError(f"Unknown caption_mode '{self.caption_mode}': must be 'api' or 'rule'")

        profiles = api_cfg.get("profiles", {})
        active = api_cfg.get("active", "glm4")
        profile = profiles.get(active, {})
        self.provider = profile.get("provider", "zhipu")
        self.model = profile.get("model", "glm-4")

        env_var = API_KEY_ENV_VARS.get(self.provider, "API_KEY")
        key = profile.get("key") or os.environ.get(env_var, "")
        if not key:
            raise ValueError(
                f"No API key for profile '{active}' (provider={self.provider}). "
                f"Set the {env_var} environment variable, "
                f"or use caption_mode='rule' for template-based captions."
            )

        try:
            if self.provider == "zhipu":
                from zhipuai import ZhipuAI
                self._client = ZhipuAI(api_key=key)
            else:
                from openai import OpenAI
                base_url = profile.get("base_url") or (
                    "https://api.deepseek.com" if self.provider == "deepseek" else None
                )
                self._client = OpenAI(api_key=key, base_url=base_url)
        except Exception as e:
            raise RuntimeError(f"Failed to init LLM client for '{active}': {e}")

    def generate(self, original: List[dict], target: List[dict], edit_info: dict,
                 room: str = "large", room_dims: dict = None, retries: int = 3) -> list:
        if self.caption_mode == "rule":
            return _fallback_caption(edit_info)

        room_str = f"{room.capitalize()} room"
        if room_dims:
            room_str += f" ({room_dims['x']}x{room_dims['y']}x{room_dims['z']}m)"
        prompt = CAPTION_USER_TEMPLATE.format(
            room=room_str,
            original="\n".join(scene_to_lines(original)),
            edit_detail=summarise_edit(edit_info, original_scene=original),
        )
        messages = [
            {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=min(self.temperature + attempt * 0.05, 1.2),
                    max_tokens=self.max_tokens,
                )
                raw = resp.choices[0].message.content.strip()
                lines = [l.strip().lstrip("123. ").strip('"').strip("'") for l in raw.split("\n")]
                captions = [l for l in lines if l and len(l) > 5]
                if len(captions) >= 3:
                    return captions[:3]
                elif len(captions) >= 1 and attempt == retries - 1:
                    return captions
            except Exception as e:
                write(f"    LLM retry {attempt+1}: {e}")
                time.sleep(1.5 ** attempt)

        return _fallback_caption(edit_info)
