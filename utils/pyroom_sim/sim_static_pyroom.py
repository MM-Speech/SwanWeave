"""Static PyRoom RIR, FOA encoding, source preparation and multi-event primitives."""

from __future__ import annotations

import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from utils.pyroom_sim import render_config as config

AXES = ("x", "y", "z")
_PRA = None

def finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def vector(value):
    result = np.array([finite(value[axis], f"position.{axis}") for axis in AXES], dtype=np.float64)
    if np.linalg.norm(result) < 1e-6:
        raise ValueError("A directional source cannot coincide with the listener")
    return result


def room_dimensions(value):
    result = np.array([finite(value[axis], f"room_dims.{axis}") for axis in AXES])
    if np.any(result <= 0):
        raise ValueError("Room dimensions must be positive")
    return result


def validate_position(position, dimensions, min_distance=0.0):
    if not np.all(np.isfinite(position)):
        raise ValueError("Source position must be finite")
    if np.any(np.abs(position) >= dimensions / 2):
        raise ValueError(f"Source outside room: position={position.tolist()}, room={dimensions.tolist()}")
    distance = float(np.linalg.norm(position))
    if distance + 1e-12 < min_distance:
        raise ValueError(f"Source distance {distance:.6g} m is below minimum {min_distance:g} m")


def load_pyroom():
    global _PRA
    if _PRA is None:
        import pyroomacoustics
        _PRA = pyroomacoustics
    return _PRA


def center_rir(position, dimensions, settings):
    """Reference shoebox RIR, with propagation delay retained during tail shaping."""
    _, settings = config.settings_for_room(dimensions, settings)
    pra = load_pyroom()
    sample_rate = settings["sample_rate"]
    room_dim = dimensions[[0, 2, 1]].astype(np.float32)
    origin = room_dim / 2
    source = origin + position[[0, 2, 1]].astype(np.float32)
    if np.any(source <= 0) or np.any(source >= room_dim):
        raise ValueError("Source is on/outside a wall after coordinate conversion")
    pra.constants.set("c", settings["sound_speed"])
    room = pra.ShoeBox(room_dim.tolist(), fs=sample_rate,
                       materials=pra.Material(energy_absorption=settings["absorption"]),
                       max_order=settings["max_order"])
    room.add_source(source.astype(float).tolist())
    room.add_microphone_array(pra.MicrophoneArray(origin.reshape(3, 1), fs=sample_rate))
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=np.float32)
    peak = float(np.max(np.abs(rir)))
    if peak <= 0 or not np.all(np.isfinite(rir)):
        raise ValueError("Pyroom produced a silent/non-finite RIR")
    hits = np.flatnonzero(np.abs(rir) >= max(peak * 1e-3, 1e-8))
    onset = int(hits[0]) if hits.size else 0
    end = min(len(rir), onset + max(1, round(sample_rate * settings["ir_trim_ms"] / 1000)),
              max(onset + 1, round(sample_rate * settings["ir_seconds"])))
    rir = rir[:end].copy()
    late = onset + max(0, round(sample_rate * settings["late_reverb_start_ms"] / 1000))
    if late < len(rir):
        gain = 10 ** (settings["late_reverb_gain_db"] / 20)
        fade_end = min(len(rir), late + max(1, round(sample_rate * 0.01)))
        rir[late:fade_end] *= np.linspace(1, gain, fade_end - late, dtype=np.float32)
        rir[fade_end:] *= gain
    return rir


def foa_rirs(rir, position):
    unit = position / np.linalg.norm(position)
    # Project coordinates: +x right, +y up, -z front. Output: W,Y,Z,X.
    gains = np.array([1 / math.sqrt(2), -unit[0], unit[1], -unit[2]], dtype=np.float32)
    return rir[None, :] * gains[:, None]


def convolve(signal, rirs):
    length = len(signal) + rirs.shape[1] - 1
    fft_length = 1 << math.ceil(math.log2(length))
    signal_fft = np.fft.rfft(signal, n=fft_length)
    result = np.fft.irfft(np.fft.rfft(rirs, n=fft_length, axis=1) * signal_fft,
                         n=fft_length, axis=1)
    return result[:, :length].T.astype(np.float32)


def read_event_audio(source, start, end, repeat, sample_rate):
    """Prepare a validated interval; crop/pad each repeat and preserve 0.1 s gaps."""
    audio, input_rate = sf.read(source, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not audio.size or not np.all(np.isfinite(audio)) or not np.any(audio):
        raise ValueError(f"Empty, silent or non-finite source audio: {source}")
    if input_rate != sample_rate:
        divisor = math.gcd(input_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, input_rate // divisor).astype(np.float32)
    play_duration = (end - start - (repeat - 1) * 0.1) / repeat
    length = round(play_duration * sample_rate)
    if length < 1:
        raise ValueError("Event interval is too short for repeat and 0.1s gaps")
    clip = np.zeros(length, dtype=np.float32)
    clip[:min(length, len(audio))] = audio[:length]
    signal = np.zeros(round((end - start) * sample_rate), dtype=np.float32)
    for index in range(repeat):
        offset = round(index * (play_duration + 0.1) * sample_rate)
        count = min(len(clip), len(signal) - offset)
        if count > 0:
            signal[offset:offset + count] = clip[:count]
    if not np.any(signal):
        raise ValueError(f"Silent event after cropping: {source}")
    return signal


def render_at_position(signal, position, dimensions, settings, rir_cache=None):
    if rir_cache is None:
        rir_cache = {}
    validate_position(position, dimensions, settings["min_source_distance"])
    key = tuple(position) + tuple(dimensions)
    if key not in rir_cache:
        rir_cache[key] = foa_rirs(center_rir(position, dimensions, settings), position)
    return convolve(signal, rir_cache[key])


def event_positions(event, settings):
    """Actual static source locations; surround uses a ring rather than its placeholder."""
    if not isinstance(event.get("is_surround", False), bool):
        raise ValueError("is_surround must be a boolean")
    if not event.get("is_surround", False):
        return [vector(event["position"])]
    count = settings["surround_points"]
    return [np.array([2 * math.cos(a), 0, 2 * math.sin(a)])
            for a in np.arange(count) * 2 * math.pi / count]


def render_static_event(signal, event, dimensions, settings, rir_cache=None):
    positions = event_positions(event, settings)
    if not event.get("is_surround", False):
        return render_at_position(signal, positions[0], dimensions, settings, rir_cache)
    count = settings["surround_points"]
    rirs = []
    for position in positions:
        validate_position(position, dimensions, settings["min_source_distance"])
        rirs.append(foa_rirs(center_rir(position, dimensions, settings), position))
    combined = np.zeros((4, max(rir.shape[1] for rir in rirs)), dtype=np.float32)
    for rir in rirs:
        combined[:, :rir.shape[1]] += rir / count
    return convolve(signal, combined)

