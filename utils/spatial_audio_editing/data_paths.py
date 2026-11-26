"""Central paths for spatial-audio dataset preparation.

Set DATA_ROOT to your dataset directory, independently of the project code.
The slicing and scene-generation scripts use these paths by default; explicit
command-line paths take precedence. Importing this module does not create
directories, load data, or split clips.

Single-operation audio: triplet/<name>_pyroom/<sample_id>/.
Multi-operation audio: triplet_multi/multi_Nop/<sample_id>/.
Test audio uses the corresponding triplet_test and triplet_multi_test roots.
"""

from pathlib import Path


# Shared root for source datasets and processing outputs. Set your own path.
DATA_ROOT = Path('your own directory')

# Source datasets. Make these directories and put the datasets in them. 
# These defaults match the layouts in configs/*.json.
DATASET_DIRS = {
    "fsdkaggle2019": DATA_ROOT / "FSDKaggle2019",
    "picoaudio": DATA_ROOT / "PicoAudio",
    "librispeech": DATA_ROOT / "LibriSpeech" / "LibriSpeech",
}
FSD_TRAIN_DIR = DATASET_DIRS["fsdkaggle2019"] / "FSDKaggle2019.audio_train_curated"
FSD_TEST_DIR = DATASET_DIRS["fsdkaggle2019"] / "FSDKaggle2019.audio_test"
FSD_LABELS_FILE = (DATASET_DIRS["fsdkaggle2019"] / "FSDKaggle2019.meta"
                   / "train_curated_post_competition.csv")

PICO_AUDIO_DIR = DATASET_DIRS["picoaudio"] / "data" / "multi_event_train"
PICO_METADATA_FILE = DATASET_DIRS["picoaudio"] / "meta_data" / "train.json"

# Slices and generated labels use their own subdirectory under DATA_ROOT.
PICO_PROCESSED_ROOT = DATASET_DIRS["picoaudio"]
PICO_SLICED_DIR = PICO_PROCESSED_ROOT / "sliced"
PICO_LABELS_FILE = PICO_PROCESSED_ROOT / "sliced_labels.csv"
# This root must contain SPEAKERS.TXT and the train-clean-* directories.
LIBRISPEECH_SPEAKERS_FILE = DATASET_DIRS["librispeech"] / "SPEAKERS.TXT"
LIBRISPEECH_SUBSET_DIRS = {
    subset: DATASET_DIRS["librispeech"] / subset
    for subset in ("train-clean-100", "train-clean-360")
}

# Source combinations used by the two generation domains.
DATASET_COMBINATIONS = {
    "audio_only": ("fsdkaggle2019", "picoaudio"),
    "speech_audio": ("fsdkaggle2019", "picoaudio", "librispeech"),
}
DOMAINS = tuple(DATASET_COMBINATIONS)
SPLITS = ("train", "test")
EDIT_TYPES = (
    "add_event", "remove_event", "replace_event", "volume_change",
    "relocation", "relocation_plane", "distance_change", "angle_change",
    "angle_change_plane", "distance_motion", "angle_motion", "volume_fade",
    "extract_event", "room_change",
)
# Multi-operation samples randomly select edits; groups describe counts only.
# The generator excludes extract_event and room_change from its default pool.
MULTI_OPERATION_COUNTS = (2, 3, 4)
MULTI_GROUPS = tuple(f"multi_{count}op" for count in MULTI_OPERATION_COUNTS)

# Rendered audio follows the existing single/multi root conventions.
# No train/ or test/ directory is automatically appended by the renderers.
TRIPLET_ROOT = DATA_ROOT / "triplet"
TRIPLET_MULTI_ROOT = DATA_ROOT / "triplet_multi"
SINGLE_RENDER_ROOTS = {
    "train": TRIPLET_ROOT,
    "test": DATA_ROOT / "triplet_test",
}
MULTI_RENDER_ROOTS = {
    "train": TRIPLET_MULTI_ROOT,
    "test": DATA_ROOT / "triplet_multi_test",
}

# Most folder stems match the task name; distance, volume and room use the
# short names in the renderer defaults/scripts. Keep full/plane tasks distinct.
SINGLE_RENDER_STEMS = {
    "add_event": "add_event",
    "remove_event": "remove_event",
    "replace_event": "replace_event",
    "volume_change": "volume",
    "relocation": "relocation",
    "relocation_plane": "relocation_plane",
    "distance_change": "distance",
    "angle_change": "angle_change",
    "angle_change_plane": "angle_change_plane",
    "distance_motion": "distance_motion",
    "angle_motion": "angle_motion",
    "volume_fade": "volume_fade",
    "extract_event": "extract_event",
    "room_change": "room",
}
# The existing mixed speech/audio rendering scripts use the joint_ prefix.
SINGLE_RENDER_PREFIXES = {"audio_only": "audio", "speech_audio": "joint"}
SINGLE_RENDER_DIRS = {
    domain: {
        split: {
            task: root / f"{prefix}_{stem}_pyroom"
            for task, stem in SINGLE_RENDER_STEMS.items()
        }
        for split, root in SINGLE_RENDER_ROOTS.items()
    }
    for domain, prefix in SINGLE_RENDER_PREFIXES.items()
}
MULTI_RENDER_DIRS = {
    split: {group: root / group for group in MULTI_GROUPS}
    for split, root in MULTI_RENDER_ROOTS.items()
}
# Single-operation renderer --output_dir: SINGLE_RENDER_DIRS[domain][split][task].
# Multi-operation renderer --output_dir: MULTI_RENDER_ROOTS[split]; it appends
# multi_Nop itself. MULTI_RENDER_DIRS contains final group paths for readers.
CFM_AUDIO_SUBDIRS = {"source": "origin", "target": "target"}
DPO_AUDIO_SUBDIRS = {"source": "source", "target": "target", "negative": "negative"}
# All generated audio uses FLAC; sample rate, channels and processing stay unchanged.
AUDIO_FORMAT = "FLAC"
AUDIO_SUFFIX = ".flac"
FOA_FILENAME = f"foa{AUDIO_SUFFIX}"
POSTERIOR_SUFFIX = ".spat_vae_posterior.pt"
POSTERIOR_FILENAME = f"foa{POSTERIOR_SUFFIX}"
POSTERIOR_FIELDS = ("mean", "std")
# One PT file beside each audio file: foa.flac -> foa.spat_vae_posterior.pt.
# Existing precompute code writes mean/var; save sqrt(var) as std when integrated.
# Training already accepts mean/std. Keep length and sample-rate metadata too.

# Scene descriptions are JSONL files, separate from the rendered samples.
SCENE_ROOT = DATA_ROOT / "metadata"
SCENE_SPLIT_ROOTS = {
    "train": SCENE_ROOT,
    "test": SCENE_ROOT / "test_set",
}
SINGLE_SCENE_DIRS = {
    domain: {split: root / domain for split, root in SCENE_SPLIT_ROOTS.items()}
    for domain in DOMAINS
}
# generate_multi_operations.py defaults to <single output.dir>_multi.
MULTI_SCENE_DIRS = {
    domain: {
        split: path.with_name(f"{path.name}_multi")
        for split, path in split_dirs.items()
    }
    for domain, split_dirs in SINGLE_SCENE_DIRS.items()
}

DURATION_CACHE_ROOT = DATA_ROOT / "duration_cache"
DPO_SCENE_ROOT = DATA_ROOT / "dpo_scenes"
MANIFEST_ROOT = DATA_ROOT / "manifests"
FIGURE_ROOT = DATA_ROOT / "figures"


def _split_dirs(root: Path) -> dict:
    return {
        domain: {split: root / domain / split for split in SPLITS}
        for domain in DOMAINS
    }


DPO_SCENE_DIRS = _split_dirs(DPO_SCENE_ROOT)
DURATION_CACHE_FILES = {
    domain: {split: DURATION_CACHE_ROOT / domain / f"{split}.json.gz"
             for split in SPLITS}
    for domain in DOMAINS
}
MANIFEST_DIRS = _split_dirs(MANIFEST_ROOT)
CFM_MANIFESTS = {
    domain: {split: MANIFEST_DIRS[domain][split] / "cfm.jsonl" for split in SPLITS}
    for domain in DOMAINS
}
DPO_MANIFESTS = {
    domain: {split: MANIFEST_DIRS[domain][split] / "dpo.jsonl" for split in SPLITS}
    for domain in DOMAINS
}
