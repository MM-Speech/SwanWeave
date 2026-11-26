"""Output writer — append results to per-edit-type JSONL files with resume support."""

import json
from pathlib import Path


def _output_dir(cfg: dict) -> Path:
    return Path(cfg["output"]["dir"])


def _jsonl_path(cfg: dict, edit_type: str) -> Path:
    return _output_dir(cfg) / f"{edit_type}.jsonl"


def load_completed_tasks(cfg: dict, progress=None) -> set:
    completed = set()
    if progress is not None:
        progress.reset(desc="Checking completed tasks", unit="rows")
    out_dir = _output_dir(cfg)
    if not out_dir.exists():
        return completed
    for jsonl_file in out_dir.glob("*.jsonl"):
        edit_type = jsonl_file.stem
        if edit_type.endswith("_text") or edit_type.endswith("_negative"):
            continue
        try:
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if progress is not None:
                        progress.update()
                    r = json.loads(line)
                    sample_idx = int(r["sample_id"])
                    completed.add((edit_type, sample_idx))
        except Exception as e:
            message = f"  [WARN] Failed to read {jsonl_file.name}: {e}"
            if progress is not None:
                progress.write(message)
            else:
                print(message)
            continue
    return completed


def append_result(result: dict, cfg: dict):
    out_dir = _output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = _jsonl_path(cfg, result["edit_type"])
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def append_text_result(result: dict, cfg: dict):
    """Write a lightweight text-only record for text-to-audio training."""
    out_dir = _output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{result['edit_type']}_text.jsonl"
    record = {
        "sample_id": result["sample_id"],
        "edit_type": result["edit_type"],
        "original_text": result["original_text"],
        "target_text": result["target_text"],
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
