#!/usr/bin/env python3
"""渲染静态单声源 FOA。

接受的 JSONL metadata（每行一个 JSON 对象）：
{"sample_id":"00000000","duration":4.25,
 "room_dims":{"x":21,"y":10,"z":21},
 "source_path":"/path/to/input.wav",
 "wav_path":"/path/to/output.flac",
 "position":{"x":4.29,"y":3.78,"z":4.81}}

坐标约定：+X 向右、+Y 向上、-Z 向前；position 是相对房间中心的
静态声源位置，整段音频渲染期间保持不变。输出为 ACN/SN3D W,Y,Z,X 四声道 FLAC。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

SIM_MODULE_DIR = Path("users/spat_sim_pyacoustic").resolve()
if str(SIM_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_MODULE_DIR))

from sim_audio_add_event import (  # noqa: E402
    DEFAULT_ABSORPTION,
    DEFAULT_IR_SECONDS,
    DEFAULT_IR_TRIM_MS,
    DEFAULT_LATE_REVERB_GAIN_DB,
    DEFAULT_LATE_REVERB_START_MS,
    DEFAULT_MAX_ORDER,
    DEFAULT_ROOM_DIMS_XYZ,
    DEFAULT_SOUND_SPEED,
    compute_shared_save_scale,
    get_effective_room,
    load_metadata_rows,
    render_scene_foa_mix,
)

DEFAULT_METADATA = Path(
    "/mnt/bn/sa-ag-data/leike/spatial_edit/dataset/metadata/fsd50k_solo_foa_random.jsonl"
)
REQUIRED_KEYS = {
    "sample_id", "duration", "room_dims", "source_path", "wav_path", "position"
}
MIN_DURATION_SECONDS = 0.5


def is_short_audio(row: dict) -> bool:
    try:
        return float(row["duration"]) < MIN_DURATION_SECONDS
    except (KeyError, TypeError, ValueError):
        return False


def render_one(row: dict, sample_rate: int, fallback_room: np.ndarray,
               skip_existing: bool) -> bool:
    missing = REQUIRED_KEYS.difference(row)
    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")

    output_path = Path(row["wav_path"])
    if skip_existing and output_path.is_file():
        return True

    source_path = Path(row["source_path"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Source audio not found: {source_path}")

    duration = float(row["duration"])
    if duration <= 0:
        raise ValueError(f"Invalid duration: {duration}")

    room_dim, listener_origin = get_effective_room(row, fallback_room, None)
    event = {
        "event_id": 0,
        "audio_path": str(source_path),
        "audio_duration": duration,
        "position": row["position"],
        "start_time": 0.0,
        "end_time": duration,
        "gain": 1.0,
    }
    foa, _ = render_scene_foa_mix(
        [event], duration, sample_rate, room_dim, listener_origin,
        DEFAULT_ABSORPTION, DEFAULT_MAX_ORDER, DEFAULT_SOUND_SPEED,
        DEFAULT_IR_SECONDS, DEFAULT_IR_TRIM_MS,
        DEFAULT_LATE_REVERB_START_MS, DEFAULT_LATE_REVERB_GAIN_DB,
    )
    scale, _ = compute_shared_save_scale([foa], peak_dbfs=-1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, np.clip(foa * scale, -1.0, 1.0), sample_rate,
             format="FLAC", subtype="PCM_16")
    return True


def process_task(task: tuple) -> tuple[str, bool, str | None]:
    row, sample_rate, fallback_room, skip_existing = task
    sample_id = str(row.get("sample_id", "unknown"))
    try:
        return sample_id, render_one(
            row, sample_rate, fallback_room, skip_existing
        ), None
    except Exception as error:
        return sample_id, False, str(error)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render static FSD50K FOA FLAC files.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    rows = load_metadata_rows(args.metadata, args.limit)
    filtered_source_paths = {
        str(row.get("source_path")) for row in rows if is_short_audio(row)
    }
    rows = [
        row for row in rows
        if str(row.get("source_path")) not in filtered_source_paths
    ]
    fallback_room = np.array(
        [DEFAULT_ROOM_DIMS_XYZ[0], DEFAULT_ROOM_DIMS_XYZ[2],
         DEFAULT_ROOM_DIMS_XYZ[1]], dtype=np.float32
    )
    tasks = [
        (row, args.sample_rate, fallback_room, args.skip_existing) for row in rows
    ]
    if args.num_workers <= 1:
        iterator = map(process_task, tasks)
        pool = None
    else:
        pool = mp.get_context("fork").Pool(args.num_workers)
        iterator = pool.imap(process_task, tasks)

    succeeded = failed = 0
    try:
        for sample_id, ok, error in tqdm(
            iterator, total=len(tasks), desc="Rendering static FOA", unit="sample"
        ):
            succeeded += int(ok)
            failed += int(not ok)
            if error:
                tqdm.write(f"[ERROR] sample_id={sample_id}: {error}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print("\n--- Summary ---")
    print(f"Succeeded metadata rows: {succeeded}")
    print(f"Failed metadata rows: {failed}")
    print(f"Filtered source WAVs (< {MIN_DURATION_SECONDS}s): "
          f"{len(filtered_source_paths)}")


if __name__ == "__main__":
    main()