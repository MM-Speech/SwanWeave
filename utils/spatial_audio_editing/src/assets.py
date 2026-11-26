"""Shared setup for the generation entry points (pipeline.py, generate_multi_operations.py)."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from utils.spatial_audio_editing import data_paths

from utils.spatial_audio_editing.src.loader import FSDLoader, LibriSpeechLoader, precompute_durations
from utils.spatial_audio_editing.src.scene import SceneGenerator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_PATH = data_paths.DURATION_CACHE_FILES["audio_only"]["train"]


def _path_string(path) -> str:
    return str(Path(path).expanduser().absolute())


def _domain(cfg: dict) -> str:
    domain = cfg.get("domain")
    if domain is None:
        domain = "speech_audio" if cfg.get("data", {}).get("speech_base") else "audio_only"
    if domain not in data_paths.DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}; expected one of {data_paths.DOMAINS}")
    return domain


def _cache_path(cfg: dict) -> str:
    split = cfg.get("split", "train")
    if split not in data_paths.SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {data_paths.SPLITS}")
    return _path_string(data_paths.DURATION_CACHE_FILES[_domain(cfg)][split])


def load_config(path, split: str = "train") -> dict:
    """Read generation settings, then take all dataset/output paths from data_paths."""
    if split not in data_paths.SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {data_paths.SPLITS}")
    config_path = Path(path).expanduser()
    if not config_path.is_absolute() and not config_path.is_file():
        config_path = REPO_ROOT / config_path
    with config_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    domain = _domain(cfg)
    cfg["domain"], cfg["split"] = domain, split
    data = cfg.setdefault("data", {})
    fsd_base = _path_string(data_paths.DATASET_DIRS["fsdkaggle2019"])
    labels_file = _path_string(data_paths.FSD_LABELS_FILE)
    data.update({
        "fsd_base": fsd_base,
        "train_curated_dir": os.path.relpath(_path_string(data_paths.FSD_TRAIN_DIR), fsd_base),
        "test_dir": os.path.relpath(_path_string(data_paths.FSD_TEST_DIR), fsd_base),
        "meta_dir": os.path.relpath(str(Path(labels_file).parent), fsd_base),
        "fsd_labels_file": labels_file,
        "pico_base": _path_string(data_paths.DATASET_DIRS["picoaudio"]),
        "pico_sliced_dir": _path_string(data_paths.PICO_SLICED_DIR),
        "pico_labels_file": _path_string(data_paths.PICO_LABELS_FILE),
    })
    if domain == "speech_audio":
        subsets = data.get("librispeech_subsets", list(data_paths.LIBRISPEECH_SUBSET_DIRS))
        unknown = [subset for subset in subsets if subset not in data_paths.LIBRISPEECH_SUBSET_DIRS]
        if unknown:
            raise ValueError(f"Unknown LibriSpeech subsets in config: {unknown}")
        data.update({
            "speech_base": _path_string(data_paths.DATASET_DIRS["librispeech"]),
            "speakers_file": _path_string(data_paths.LIBRISPEECH_SPEAKERS_FILE),
            "librispeech_subsets": list(subsets),
            "subset_dirs": {
                subset: _path_string(data_paths.LIBRISPEECH_SUBSET_DIRS[subset])
                for subset in subsets
            },
        })
    else:
        for key in ("speech_base", "speakers_file", "subset_dirs"):
            data.pop(key, None)
    cfg.setdefault("generation", {})["cache_path"] = _cache_path(cfg)
    cfg.setdefault("output", {})["dir"] = _path_string(data_paths.SINGLE_SCENE_DIRS[domain][split])
    return cfg


def stable_int_hash(text: str, modulo: int) -> int:
    """Integer hash of `text` that is stable across Python processes (unlike hash())."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


@dataclass
class Assets:
    """Indexed source clips shared by all workers."""
    classes: List[str]
    class_index: Dict[str, List[str]]   # class -> [relative clip path]
    dur_cache: Dict[str, float]         # relative clip path -> duration (s)
    speech_meta: Dict[str, dict]        # LibriSpeech path -> {transcript, speaker_id, class}
    speech_classes: List[str]

    def scene_generator(self, cfg: dict) -> SceneGenerator:
        return SceneGenerator(cfg, self.classes, self.class_index, self.dur_cache,
                              self.speech_meta, self.speech_classes)


def _check_data_dirs(data_cfg: dict):
    required = [("fsd_base", data_cfg.get("fsd_base"))]
    if data_cfg.get("speech_base"):
        required.append(("speech_base", data_cfg["speech_base"]))
    missing = [f"  data.{key} = {path}" for key, path in required if not Path(path).is_dir()]
    if missing:
        raise SystemExit("Source data not found:\n" + "\n".join(missing)
                         + "\nDownload the datasets and set their locations in data_paths.py.")


def _log(progress, message):
    if progress is None:
        print(message)
    else:
        progress.write(message)


def load_assets(cfg: dict, progress=None) -> Assets:
    """Index FSDKaggle2019 + PicoAudio (+ LibriSpeech if configured) and load clip durations."""
    _log(progress, "\n[1/4] Loading metadata ...")
    data_cfg = cfg["data"]
    _check_data_dirs(data_cfg)
    loader = FSDLoader(cfg)
    exclude_audio = data_cfg.get("exclude_audio_classes", [])
    class_index, classes = loader.build_class_index(exclude_classes=exclude_audio, progress=progress)
    if exclude_audio:
        _log(progress, f"  Excluded {len(exclude_audio)} audio classes: {exclude_audio}")
    _log(progress, f"  {len(classes)} audio classes, {sum(len(v) for v in class_index.values())} audio clips indexed")

    speech_meta, speech_classes = {}, []
    speech_base = data_cfg.get("speech_base", "")
    if speech_base:
        _log(progress, "  Loading LibriSpeech ...")
        speech_index, speech_meta = LibriSpeechLoader(cfg).build_speech_index(progress=progress)
        speech_classes = sorted(speech_index.keys())
        if progress is not None:
            progress.reset(total=len(speech_classes), desc="Merge speech index", unit="classes")
        for cls in speech_classes:
            class_index[cls].extend(speech_index[cls])
            if progress is not None:
                progress.update()
        _log(progress, f"  {len(speech_classes)} speech classes, "
              f"{sum(len(v) for v in speech_index.values())} speech clips indexed")

    _log(progress, "\n[2/4] Pre-computing audio durations ...")
    gen_cfg = cfg["generation"]
    dur_cache = precompute_durations(
        class_index,
        loader,
        rms_min=gen_cfg.get("rms_min", 0.02),
        cache_path=gen_cfg.get("cache_path") or _cache_path(cfg),
        speech_base=speech_base,
        subset_dirs=data_cfg.get("subset_dirs"),
        progress=progress,
    )
    classes = sorted(class_index.keys())
    _log(progress, f"  {len(dur_cache)} clips kept, {len(classes)} classes remaining")
    return Assets(classes, class_index, dur_cache, speech_meta, speech_classes)
