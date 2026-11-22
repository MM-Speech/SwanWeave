"""Run a SwanWeave training task from the repository root."""

import importlib
import os
from pathlib import Path
import sys

# Also support ``python tasks/run.py`` without an external PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils.commons.single_thread_env  # noqa: E402,F401
from utils.commons.hparams import hparams, set_hparams  # noqa: E402


def run_task():
    task_path = hparams.get("task_cls", "")
    if not task_path:
        raise ValueError("The training configuration must define task_cls")
    module_name, class_name = task_path.rsplit(".", 1)
    task_cls = getattr(importlib.import_module(module_name), class_name)
    task_cls.start()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    set_hparams()
    run_task()
