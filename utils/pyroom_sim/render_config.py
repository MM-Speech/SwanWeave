"""Paths and acoustic defaults for sim_pairs_pyroom.py. Edit this file as needed."""

import math

from utils.spatial_audio_editing import data_paths


# All relative paths in the rendered metadata are relative to DATA_ROOT.
DATA_ROOT = data_paths.DATA_ROOT
METADATA_ROOT = data_paths.SCENE_ROOT
RENDERED_METADATA_ROOT = DATA_ROOT / "rendered_metadata"
NUM_WORKERS = 8

# Logical prefixes used by the scene generator's event.audio_path.
SOURCE_ROOTS = {
    "FSDKaggle2019.audio_train_curated": data_paths.FSD_TRAIN_DIR,
    "FSDKaggle2019.audio_test": data_paths.FSD_TEST_DIR,
    "pico_sliced": data_paths.PICO_SLICED_DIR,
    "LibriSpeech": data_paths.DATASET_DIRS["librispeech"],
}

# Keep the established single/multi audio directory names.
SINGLE_RENDER_DIRS = data_paths.SINGLE_RENDER_DIRS
MULTI_RENDER_ROOTS = data_paths.MULTI_RENDER_ROOTS
DPO_SCENE_DIRS = data_paths.DPO_SCENE_DIRS
DPO_RENDER_ROOTS = {"train": DATA_ROOT / "triplet_dpo", "test": DATA_ROOT / "triplet_dpo_test"}
DPO_RENDERED_METADATA_ROOT = RENDERED_METADATA_ROOT / "dpo"

SETTINGS = {
    "sample_rate": 44100,
    "subtype": "PCM_16",
    "peak_dbfs": -1.0,
    "absorption": 0.8,
    "max_order": 3,
    "sound_speed": 343.0,
    "ir_seconds": 0.35,
    "ir_trim_ms": 350.0,
    "late_reverb_start_ms": 20.0,
    "late_reverb_gain_db": -35.0,
    "motion_step_seconds": 0.05,
    "angle_radius": "min",  # Smaller endpoint radius; "start" keeps the start radius.
    "min_source_distance": 0.2,  # Metres from the listener, including the whole trajectory.
    "surround_points": 16,
}


SMALL_SETTINGS = {
    "absorption": 0.5,
    "max_order": 7,
    "ir_seconds": 0.35,
    "ir_trim_ms": 350.0,
    "late_reverb_start_ms": 5.0,
    "late_reverb_gain_db": -9.0,
}

MEDIUM_SETTINGS = {
    "absorption": 0.7,
    "max_order": 5,
    "ir_seconds": 0.35,
    "ir_trim_ms": 350.0,
    "late_reverb_start_ms": 5.0,
    "late_reverb_gain_db": -18.0,
}

# Match actual listener-relative dimensions in x, y, z order.
ROOM_PRESETS = (
    ("small", (2.0, 2.0, 2.0), SMALL_SETTINGS),
    ("medium", (6.0, 4.0, 6.0), MEDIUM_SETTINGS),
)


def settings_for_room(dimensions, settings):
    """Select acoustic overrides; unmatched room sizes keep the supplied settings."""
    result = dict(settings)
    for name, expected, overrides in ROOM_PRESETS:
        if len(dimensions) == 3 and all(
            math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-6)
            for actual, wanted in zip(dimensions, expected)
        ):
            result.update(overrides)
            return name, result
    return "default", result
