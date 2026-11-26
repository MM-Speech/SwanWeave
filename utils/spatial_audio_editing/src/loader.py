"""FSDKaggle2019 + PicoAudio + LibriSpeech data loaders and duration cache.

Clips are referenced by paths relative to their dataset root:
  FSDKaggle2019.audio_train_curated/<fname>.wav   (relative to data.fsd_base)
  pico_sliced/<fname>.wav                         (data.pico_base/<pico_sliced_dir>)
  LibriSpeech/<subset>/<spk>/<chapter>/<utt>.flac (relative to data.speech_base)
"""

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


def _log(progress, message):
    if progress is None:
        print(message)
    else:
        progress.write(message)


def _track(iterable, progress=None, *, desc="", total=None, unit="records", reset=True):
    """Advance an existing parent-owned bar; never create a progress instance."""
    if progress is not None and reset:
        progress.reset(total=total, desc=desc, unit=unit)
    for item in iterable:
        yield item
        if progress is not None:
            progress.update()


class FSDLoader:
    """Loads FSDKaggle2019 + PicoAudio sliced labels and builds a class->clip index."""

    def __init__(self, cfg: dict):
        d = cfg["data"]
        self.base = Path(d["fsd_base"])
        self.train_dir = d["train_curated_dir"]
        self.test_dir = d["test_dir"]
        self.meta_dir = self.base / d["meta_dir"]
        self.labels_file = Path(d.get("fsd_labels_file") or
                                self.meta_dir / "train_curated_post_competition.csv")

        pico_base = d.get("pico_base", "")
        if pico_base:
            self.pico_base = Path(pico_base)
            self.pico_sliced_dir = self.pico_base / d.get("pico_sliced_dir", "sliced")
            self.pico_labels_file = self.pico_base / d.get("pico_labels_file", "sliced_labels.csv")
        else:
            self.pico_base = None
            self.pico_sliced_dir = None
            self.pico_labels_file = None

    @staticmethod
    def _read_labels_csv(path: Path, progress=None) -> List[Tuple[str, str]]:
        with open(path) as f:
            rows = _track(csv.DictReader(f), progress, desc=f"Read {path.name}", unit="records")
            return [(row["fname"], row["labels"]) for row in rows]

    def load_file_labels(self, progress=None) -> List[Tuple[str, str]]:
        return self._read_labels_csv(self.labels_file, progress=progress)

    def load_pico_labels(self, progress=None) -> List[Tuple[str, str]]:
        if not self.pico_labels_file or not self.pico_labels_file.exists():
            return []
        return self._read_labels_csv(self.pico_labels_file, progress=progress)

    def build_class_index(self, exclude_classes: List[str] = None, progress=None) -> Tuple[defaultdict, List[str]]:
        index = defaultdict(list)
        exclude = set(exclude_classes or [])

        file_labels = self.load_file_labels(progress=progress)
        for fname, label_str in _track(file_labels, progress, desc="Index FSD clips",
                                      total=len(file_labels), unit="clips"):
            for lbl in (s.strip() for s in label_str.split(",")):
                if lbl in exclude:
                    continue
                rel = self._resolve_relative_path(fname)
                if rel:
                    index[lbl].append(rel)

        pico_labels = self.load_pico_labels(progress=progress)
        for fname, label_str in _track(pico_labels, progress, desc="Index PicoAudio clips",
                                      total=len(pico_labels), unit="clips"):
            p = self.pico_sliced_dir / fname
            if p.exists():
                rel = f"pico_sliced/{fname}"
                for lbl in (s.strip() for s in label_str.split(",")):
                    if lbl in exclude:
                        continue
                    index[lbl].append(rel)

        classes = sorted(index.keys())
        return index, classes

    def _resolve_relative_path(self, fname: str) -> Optional[str]:
        for d in (self.train_dir, self.test_dir):
            p = self.base / d / fname
            if p.exists():
                return f"{d}/{fname}"
        return None

    def _resolve_abs_path(self, rel_path: str) -> Optional[Path]:
        p = self.base / rel_path
        if p.exists():
            return p
        if self.pico_base and rel_path.startswith("pico_sliced/"):
            p = self.pico_sliced_dir / rel_path[len("pico_sliced/"):]
            if p.exists():
                return p
        return None


class LibriSpeechLoader:
    """Loads LibriSpeech utterances with transcripts and speaker metadata."""

    def __init__(self, cfg: dict):
        d = cfg["data"]
        self.speech_base = Path(d["speech_base"])
        self.speakers_file = Path(d.get("speakers_file") or self.speech_base / "SPEAKERS.TXT")
        self.subsets = d.get("librispeech_subsets", ["train-clean-100", "train-clean-360"])
        configured_dirs = d.get("subset_dirs")
        if configured_dirs is not None:
            missing = [subset for subset in self.subsets if subset not in configured_dirs]
            if missing:
                raise ValueError(f"Missing LibriSpeech subset directories: {missing}")
        self.subset_dirs = {
            subset: Path(directory) for subset, directory in (configured_dirs or {}).items()
        }
        self.dur_min = d.get("speech_duration_min", 1.0)
        self.dur_max = d.get("speech_duration_max", 30.0)

    def load_speaker_genders(self, progress=None) -> Dict[str, str]:
        """Parse SPEAKERS.TXT, return {speaker_id: 'F'|'M'} for configured subsets."""
        speakers_file = self.speakers_file
        genders = {}
        valid_subsets = set(self.subsets)
        if not speakers_file.exists():
            return genders
        with open(speakers_file) as f:
            for line in _track(f, progress, desc="Read speaker metadata", unit="records"):
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 4:
                    continue
                speaker_id, gender, subset = parts[0], parts[1], parts[2]
                if gender in ("F", "M") and subset in valid_subsets:
                    genders[speaker_id] = gender
        return genders

    def build_speech_index(self, progress=None) -> Tuple[defaultdict, Dict[str, dict]]:
        """Walk LibriSpeech dirs, collect FLAC paths and transcript metadata.

        Returns:
            class_index: class_name -> [relative_path, ...]  (for FSDLoader-style compat)
            speech_meta:  relative_path -> {transcript, speaker_id, class}
        """
        speaker_genders = self.load_speaker_genders(progress=progress)
        class_index = defaultdict(list)
        speech_meta = {}

        if progress is not None:
            progress.reset(total=None, desc="Index LibriSpeech clips", unit="clips")
        for subset in self.subsets:
            subset_path = self.subset_dirs.get(subset, self.speech_base / subset)
            if not subset_path.exists():
                continue
            for speaker_dir in sorted(subset_path.iterdir()):
                if not speaker_dir.is_dir():
                    continue
                speaker_id = speaker_dir.name
                gender = speaker_genders.get(speaker_id)
                if gender is None:
                    continue
                speech_class = "Female_speech" if gender == "F" else "Male_speech"

                for chapter_dir in sorted(speaker_dir.iterdir()):
                    if not chapter_dir.is_dir():
                        continue
                    chapter_id = chapter_dir.name

                    # Load transcripts for this chapter
                    trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
                    transcripts = {}
                    if trans_file.exists():
                        with open(trans_file) as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                parts = line.split(" ", 1)
                                if len(parts) == 2:
                                    transcripts[parts[0]] = parts[1]

                    for flac_file in _track(sorted(chapter_dir.glob("*.flac")),
                                            progress, reset=False):
                        utt_id = flac_file.stem  # e.g. "19-198-0000"
                        transcript = transcripts.get(utt_id, "")

                        # Quick text-based duration estimate (will be refined by RMS/dur scan)
                        word_count = len(transcript.split())
                        # Rough: avg 150 words/min = 2.5 words/sec
                        est_dur = max(word_count / 2.5, 0.5) if transcript else 2.0
                        if est_dur < self.dur_min or est_dur > self.dur_max:
                            continue

                        rel = f"LibriSpeech/{subset}/{speaker_id}/{chapter_id}/{flac_file.name}"
                        class_index[speech_class].append(rel)
                        speech_meta[rel] = {
                            "transcript": transcript,
                            "speaker_id": speaker_id,
                            "class": speech_class,
                        }

        return class_index, speech_meta


def _resolve_abs_path(rel_path: str, loader: FSDLoader, speech_base: str = None,
                      subset_dirs: Optional[Dict[str, str]] = None) -> Optional[Path]:
    """Resolve a relative path from FSD, PicoAudio, or LibriSpeech to an absolute path."""
    # LibriSpeech paths
    if speech_base and rel_path.startswith("LibriSpeech/"):
        relative = Path(rel_path[len("LibriSpeech/"):])
        if subset_dirs is not None:
            subset = relative.parts[0] if relative.parts else ""
            if subset not in subset_dirs:
                raise ValueError(f"Missing LibriSpeech subset directory: {subset!r}")
            p = Path(subset_dirs[subset]).joinpath(*relative.parts[1:])
        else:
            p = Path(speech_base) / relative
        if p.exists():
            return p
        return None
    # FSD / PicoAudio paths
    return loader._resolve_abs_path(rel_path)


def _open_cache(path, mode: str):
    """Open a JSON cache file; paths ending in .gz are gzip-compressed."""
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def precompute_durations(class_index: defaultdict, loader: FSDLoader,
                         rms_min: float = 0.02, cache_path: str = None,
                         speech_base: str = None,
                         subset_dirs: Optional[Dict[str, str]] = None,
                         progress=None) -> dict:
    """Return {rel_path: duration} and prune class_index in place.

    Drops clips that are missing/unreadable, too quiet (RMS < rms_min) or, for
    LibriSpeech, shorter than 0.5 s. If cache_path exists it is loaded instead
    (including the pruned class index), so results stay reproducible.
    """
    if cache_path and Path(cache_path).exists():
        if progress is not None:
            progress.reset(total=1, desc="Load duration cache", unit="file")
        with _open_cache(cache_path, "rt") as f:
            saved = json.load(f)
        dur_cache = saved["durations"]
        filtered_index = saved["class_index"]
        class_index.clear()
        class_index.update({k: v for k, v in filtered_index.items()})
        if progress is not None:
            progress.update()
        _log(progress, f"    Loaded cache: {len(dur_cache)} clips, {len(class_index)} classes")
        return dur_cache

    cache = {}
    all_paths = [p for paths in class_index.values() for p in paths]
    skipped_low_rms = 0
    skipped_missing = 0

    for path in _track(all_paths, progress, desc="Check clip durations",
                       total=len(all_paths), unit="clips"):
        abs_path = _resolve_abs_path(path, loader, speech_base, subset_dirs)
        try:
            if abs_path is None:
                skipped_missing += 1
            elif path.startswith("LibriSpeech/"):
                # Clean read speech: skip the full RMS read
                dur = sf.info(str(abs_path)).duration
                if dur >= 0.5:
                    cache[path] = dur
            else:
                audio, _ = sf.read(str(abs_path))
                rms = float(np.sqrt(np.mean(np.asarray(audio) ** 2)))
                if rms < rms_min:
                    skipped_low_rms += 1
                else:
                    cache[path] = sf.info(str(abs_path)).duration
        except Exception as e:
            skipped_missing += 1
            _log(progress, f"    [WARN] cannot read {abs_path}: {e}")

    # Remove filtered clips from class_index
    removed_classes = []
    for cls in _track(list(class_index.keys()), progress, desc="Filter clip index",
                      total=len(class_index), unit="classes"):
        class_index[cls] = [p for p in class_index[cls] if p in cache]
        if not class_index[cls]:
            removed_classes.append(cls)
            del class_index[cls]

    if skipped_low_rms:
        _log(progress, f"    Skipped {skipped_low_rms} clips with RMS < {rms_min}")
    if skipped_missing:
        _log(progress, f"    Skipped {skipped_missing} missing/unreadable clips")
    if removed_classes:
        _log(progress, f"    Removed {len(removed_classes)} empty classes: {removed_classes}")

    # Save cache
    if cache_path:
        if progress is not None:
            progress.reset(total=1, desc="Save duration cache", unit="file")
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with _open_cache(cache_path, "wt") as f:
            json.dump({"durations": cache, "class_index": dict(class_index)}, f)
        if progress is not None:
            progress.update()
        _log(progress, f"    Saved cache to {cache_path}")

    return cache
