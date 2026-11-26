#!/usr/bin/env python3
"""
Slice PicoAudio multi-event WAVs into individual segments based on on/off annotations.

Input:  PicoAudio/meta_data/train.json + PicoAudio/data/multi_event_train/*.wav
Output: PicoAudio/sliced/*.flac + PicoAudio/sliced_labels.csv

Usage:
  python scripts/slice_picoaudio.py                         # paths from data_paths.py
  python scripts/slice_picoaudio.py --pico-base data/PicoAudio --min-duration 0.5 --target-sr 44100
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import soundfile as sf

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.spatial_audio_editing import data_paths
from utils.spatial_audio_editing.src.progress import Progress


def parse_onoff_caption(onoff: str) -> list:
    """
    Parse an onoffCaption string into a list of (class_name, start, end) tuples.

    Example input:
      "duck quacking at 0.309-2.309, 3.42-5.42, 6.714-8.714 and cow mooing at 2.038-5.007, 6.542-8.58"

    Returns:
      [("duck quacking", 0.309, 2.309), ("duck quacking", 3.42, 5.42), ...]
    """
    segments = []
    class_groups = [g.strip() for g in onoff.split(" and ")]
    for group in class_groups:
        match = re.match(r"(.+?) at (.+)", group)
        if not match:
            continue
        cls = match.group(1).strip()
        times_str = match.group(2)
        times = re.findall(r"([\d.]+)-([\d.]+)", times_str)
        for start, end in times:
            segments.append((cls, float(start), float(end)))
    return segments


def main():
    parser = argparse.ArgumentParser(description="Slice PicoAudio multi-event WAVs")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Minimum segment duration in seconds (default: 0.5)")
    parser.add_argument("--pico-base", type=str, default=None,
                        help="Override PicoAudio root (default: paths from data_paths.py)")
    parser.add_argument("--target-sr", type=int, default=None,
                        help="Resample to target sample rate (default: keep original)")
    args = parser.parse_args()

    if args.pico_base is None:
        pico_base = Path(data_paths.DATASET_DIRS["picoaudio"])
        meta_file = Path(data_paths.PICO_METADATA_FILE)
        data_dir = Path(data_paths.PICO_AUDIO_DIR)
        sliced_dir = Path(data_paths.PICO_SLICED_DIR)
        labels_csv = Path(data_paths.PICO_LABELS_FILE)
    else:
        pico_base = Path(args.pico_base)
        meta_file = pico_base / "meta_data" / "train.json"
        data_dir = pico_base / "data" / "multi_event_train"
        sliced_dir = pico_base / "sliced"
        labels_csv = pico_base / "sliced_labels.csv"

    if not meta_file.exists():
        print(f"ERROR: metadata not found at {meta_file}")
        return 1
    if not data_dir.exists():
        print(f"ERROR: data directory not found at {data_dir}")
        return 1

    sliced_dir.mkdir(parents=True, exist_ok=True)

    with Progress(desc="Loading PicoAudio metadata", unit="entries") as progress:
        # Load metadata
        entries = []
        with open(meta_file) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
                    progress.update()
        progress.write(f"Loaded {len(entries)} metadata entries")
        progress.reset(total=len(entries), desc="Slicing PicoAudio", unit="entries")

        total_segments = 0
        written = 0
        skipped_short = 0
        skipped_missing = 0
        label_rows = []

        for entry in entries:
            try:
                rel_path = Path(entry["filepath"])
                basename = rel_path.stem
                try:
                    audio_rel_path = rel_path.relative_to("data/multi_event_train")
                except ValueError:
                    wav_path = pico_base / rel_path
                else:
                    wav_path = data_dir / audio_rel_path

                if not wav_path.exists():
                    skipped_missing += 1
                    continue

                segments = parse_onoff_caption(entry["onoffCaption"])
                audio, sr = sf.read(str(wav_path))

                for seg_idx, (cls, start, end) in enumerate(segments):
                    total_segments += 1
                    duration = end - start

                    if duration < args.min_duration:
                        skipped_short += 1
                        continue

                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    segment_audio = audio[start_sample:end_sample]

                    if args.target_sr and args.target_sr != sr:
                        import samplerate  # optional dependency, only needed for resampling
                        ratio = args.target_sr / sr
                        segment_audio = samplerate.resample(segment_audio, ratio, "sinc_best")
                        out_sr = args.target_sr
                    else:
                        out_sr = sr

                    out_name = f"{basename}_seg{seg_idx}{data_paths.AUDIO_SUFFIX}"
                    out_path = sliced_dir / out_name
                    sf.write(str(out_path), segment_audio, out_sr, format=data_paths.AUDIO_FORMAT)

                    label_rows.append({"fname": out_name, "labels": cls})
                    written += 1
            finally:
                progress.update()

        # Write labels CSV (same format as FSDKaggle2019's train_curated_post_competition.csv)
        progress.reset(total=1, desc="Writing PicoAudio labels", unit="file")
        labels_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(labels_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["fname", "labels"])
            writer.writeheader()
            writer.writerows(label_rows)
        progress.update()

        progress.write("Results:")
        progress.write(f"  Total segments found:  {total_segments}")
        progress.write(f"  Written:               {written}")
        progress.write(f"  Skipped (too short):   {skipped_short}")
        progress.write(f"  Skipped (file missing):{skipped_missing}")
        progress.write(f"  Output directory:      {sliced_dir}")
        progress.write(f"  Labels CSV:            {labels_csv}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
