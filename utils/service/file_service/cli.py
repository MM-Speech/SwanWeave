import argparse
import importlib
import inspect
import json
import os
import queue
import random
import shutil
import signal
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _now_s() -> float:
    return time.time()


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _atomic_write_bytes(dst_path: str, data: bytes) -> None:
    dst_dir = os.path.dirname(dst_path)
    _safe_makedirs(dst_dir)

    do_fsync = _env_flag("FILE_SERVICE_FSYNC", True)

    tmp_path = f"{dst_path}.tmp.{os.getpid()}.{random.randint(0, 1_000_000)}"
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        if do_fsync:
            os.fsync(f.fileno())
    os.replace(tmp_path, dst_path)


def _atomic_write_json(dst_path: str, obj: Any) -> None:
    _atomic_write_bytes(dst_path, (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def _atomic_write_npy(dst_path: str, obj: Any) -> None:
    import numpy as np

    dst_dir = os.path.dirname(dst_path)
    _safe_makedirs(dst_dir)

    do_fsync = _env_flag("FILE_SERVICE_FSYNC", True)

    tmp_path = f"{dst_path}.tmp.{os.getpid()}.{random.randint(0, 1_000_000)}"
    with open(tmp_path, "wb") as f:
        np.save(f, obj, allow_pickle=True)
        f.flush()
        if do_fsync:
            os.fsync(f.fileno())
    os.replace(tmp_path, dst_path)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_npy(path: str) -> Any:
    import numpy as np

    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object and data.size == 1:
        return data.item()
    return data


class JobCodec:
    def load_job(self, job_path: str) -> Dict[str, Any]:
        raise NotImplementedError


class JsonJobCodec(JobCodec):
    def load_job(self, job_path: str) -> Dict[str, Any]:
        job = _load_json(job_path)
        if not isinstance(job, dict):
            raise ValueError(f"Job must be a dict, got {type(job)}")
        return job


class NpyJobCodec(JobCodec):
    def load_job(self, job_path: str) -> Dict[str, Any]:
        job = _load_npy(job_path)
        if not isinstance(job, dict):
            raise ValueError(f"Job must be a dict, got {type(job)}")
        return job


class ResultCodec:
    def save_result(self, result_path: str, result: Dict[str, Any]) -> None:
        raise NotImplementedError


class JsonResultCodec(ResultCodec):
    def save_result(self, result_path: str, result: Dict[str, Any]) -> None:
        _atomic_write_json(result_path, result)


class NpyResultCodec(ResultCodec):
    def save_result(self, result_path: str, result: Dict[str, Any]) -> None:
        _atomic_write_npy(result_path, result)


@dataclass(frozen=True)
class QueuePaths:
    root_dir: str
    inputs_dir: str
    processing_dir: str
    outputs_dir: str
    failed_dir: str
    done_dir: str


@dataclass
class FileQueueConfig:
    poll_interval_s: float = 0.2
    max_in_flight: int = 512
    input_ext: str = ".json"
    output_ext: str = ".json"
    job_codec: str = "json"  # json|npy
    result_codec: str = "json"  # json|npy
    keep_processed_inputs: bool = False
    lease_timeout_s: float = 3600.0
    recovery_interval_s: float = 5.0

    cleanup_enabled: bool = False
    cleanup_interval_s: float = 600.0
    cleanup_failed_ttl_s: float = 7.0 * 24.0 * 3600.0
    cleanup_outputs_ttl_s: float = 0.0
    cleanup_done_ttl_s: float = 0.0
    cleanup_max_delete_per_dir: int = 10000


def build_queue_paths(work_dir: str) -> QueuePaths:
    root_dir = os.path.abspath(work_dir)
    return QueuePaths(
        root_dir=root_dir,
        inputs_dir=os.path.join(root_dir, "inputs"),
        processing_dir=os.path.join(root_dir, "processing"),
        outputs_dir=os.path.join(root_dir, "outputs"),
        failed_dir=os.path.join(root_dir, "failed"),
        done_dir=os.path.join(root_dir, "done"),
    )


def _build_codecs(job_codec: str, result_codec: str) -> Tuple[JobCodec, ResultCodec]:
    job_codec = (job_codec or "json").lower()
    result_codec = (result_codec or "json").lower()

    if job_codec == "json":
        jc = JsonJobCodec()
    elif job_codec == "npy":
        jc = NpyJobCodec()
    else:
        raise ValueError(f"Unsupported job_codec={job_codec}, expected json|npy")

    if result_codec == "json":
        rc = JsonResultCodec()
    elif result_codec == "npy":
        rc = NpyResultCodec()
    else:
        raise ValueError(f"Unsupported result_codec={result_codec}, expected json|npy")

    return jc, rc


class FileSystemQueue:
    def __init__(self, paths: QueuePaths, cfg: FileQueueConfig):
        self.paths = paths
        self.cfg = cfg

        self.job_codec, self.result_codec = _build_codecs(cfg.job_codec, cfg.result_codec)

        for d in [
            paths.inputs_dir,
            paths.processing_dir,
            paths.outputs_dir,
            paths.failed_dir,
            paths.done_dir,
        ]:
            _safe_makedirs(d)

    def _iter_input_candidates(self):
        try:
            with os.scandir(self.paths.inputs_dir) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    name = entry.name
                    if not name.endswith(self.cfg.input_ext):
                        continue
                    if ".tmp." in name:
                        continue
                    yield entry.path
        except FileNotFoundError:
            return

    def claim_one(self) -> Optional[str]:
        for in_path in self._iter_input_candidates():
            base = os.path.basename(in_path)
            processing_path = os.path.join(self.paths.processing_dir, base)
            try:
                os.rename(in_path, processing_path)  # fail if dst exists (no clobber)
            except FileNotFoundError:
                continue
            except OSError:
                continue

            try:
                os.utime(processing_path, None)
            except Exception:
                pass

            return processing_path
        return None

    def load_job(self, processing_path: str) -> Dict[str, Any]:
        return self.job_codec.load_job(processing_path)

    def output_path_for(self, processing_path: str) -> str:
        base = os.path.splitext(os.path.basename(processing_path))[0]
        return os.path.join(self.paths.outputs_dir, base + self.cfg.output_ext)

    def _failed_paths_for(self, processing_path: str) -> Tuple[str, str]:
        base = os.path.splitext(os.path.basename(processing_path))[0]
        fail_job_path = os.path.join(self.paths.failed_dir, base + ".job" + self.cfg.input_ext)
        fail_err_path = os.path.join(self.paths.failed_dir, base + ".error.json")
        return fail_job_path, fail_err_path

    def mark_done(self, processing_path: str) -> None:
        if self.cfg.keep_processed_inputs:
            dst = os.path.join(self.paths.done_dir, os.path.basename(processing_path))
            try:
                os.rename(processing_path, dst)
            except FileNotFoundError:
                return
            except OSError:
                return
        else:
            try:
                os.remove(processing_path)
            except FileNotFoundError:
                return

    def mark_failed(self, processing_path: str, error_obj: Dict[str, Any]) -> None:
        fail_job_path, fail_err_path = self._failed_paths_for(processing_path)
        try:
            os.rename(processing_path, fail_job_path)
        except Exception:
            pass
        try:
            _atomic_write_json(fail_err_path, error_obj)
        except Exception:
            pass

    def save_result(self, output_path: str, result: Dict[str, Any]) -> None:
        self.result_codec.save_result(output_path, result)

    def requeue_stale_processing(self) -> int:
        if self.cfg.lease_timeout_s <= 0:
            return 0

        moved = 0
        now = _now_s()
        try:
            with os.scandir(self.paths.processing_dir) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    if not entry.name.endswith(self.cfg.input_ext):
                        continue

                    try:
                        st = entry.stat()
                    except FileNotFoundError:
                        continue

                    age = now - st.st_mtime
                    if age < self.cfg.lease_timeout_s:
                        continue

                    src = entry.path
                    dst = os.path.join(self.paths.inputs_dir, entry.name)
                    try:
                        os.rename(src, dst)
                        moved += 1
                    except Exception:
                        continue
        except FileNotFoundError:
            return moved

        return moved

    def _cleanup_dir(self, dir_path: str, ttl_s: float, now: float, max_delete: int) -> int:
        if ttl_s <= 0:
            return 0
        if max_delete <= 0:
            return 0

        deleted = 0
        try:
            with os.scandir(dir_path) as it:
                for entry in it:
                    if deleted >= max_delete:
                        break

                    try:
                        if not entry.is_file():
                            continue
                    except OSError:
                        continue

                    name = entry.name
                    if ".tmp." in name:
                        continue

                    try:
                        st = entry.stat()
                    except FileNotFoundError:
                        continue
                    except OSError:
                        continue

                    if (now - st.st_mtime) < ttl_s:
                        continue

                    try:
                        os.remove(entry.path)
                        deleted += 1
                    except FileNotFoundError:
                        continue
                    except IsADirectoryError:
                        continue
                    except PermissionError:
                        continue
                    except OSError:
                        continue
        except FileNotFoundError:
            return 0

        return deleted

    def cleanup_old_files(self, now: Optional[float] = None) -> Dict[str, int]:
        if not bool(getattr(self.cfg, "cleanup_enabled", False)):
            return {}

        if now is None:
            now_eff = _now_s()
        else:
            now_eff = float(now)

        try:
            max_delete = int(getattr(self.cfg, "cleanup_max_delete_per_dir", 10000))
        except Exception:
            max_delete = 10000
        if max_delete <= 0:
            max_delete = 0

        try:
            failed_ttl_s = float(getattr(self.cfg, "cleanup_failed_ttl_s", 0.0))
        except Exception:
            failed_ttl_s = 0.0
        try:
            outputs_ttl_s = float(getattr(self.cfg, "cleanup_outputs_ttl_s", 0.0))
        except Exception:
            outputs_ttl_s = 0.0
        try:
            done_ttl_s = float(getattr(self.cfg, "cleanup_done_ttl_s", 0.0))
        except Exception:
            done_ttl_s = 0.0

        deleted_failed = self._cleanup_dir(self.paths.failed_dir, failed_ttl_s, now_eff, max_delete)
        deleted_outputs = self._cleanup_dir(self.paths.outputs_dir, outputs_ttl_s, now_eff, max_delete)
        deleted_done = self._cleanup_dir(self.paths.done_dir, done_ttl_s, now_eff, max_delete)

        return {"failed": deleted_failed, "outputs": deleted_outputs, "done": deleted_done}


class BaseProcessor:
    """
    你只需要实现这个接口：
    - setup(): (可选) 加载模型、初始化资源
    - process(job): 输入 job(dict)，返回 result(dict)

    device:
    - CPU worker: "cpu"
    - GPU worker: "cuda:N"（N 是当前进程可见 GPU 的 index）

    说明：
    - 如果 server 进程启动时设置了 CUDA_VISIBLE_DEVICES，那么这里的 N 是子集后的 index（0..k-1）。
    - 如果开启了 isolate_cuda_visible_devices，那么每个 worker 会把 CUDA_VISIBLE_DEVICES 隔离成单卡，此时 device 通常是 "cuda:0"。
    
    扩展接口：
    - process_batch(jobs): (可选) 批量处理多个 job。
      默认实现是逐个调用 process()。
      如果你的模型天然支持 batch（例如一次 forward 多条音频），建议覆写这个方法。
    """

    def __init__(
        self,
        device: str = "cpu",
        num_workers: Optional[int] = None,
        worker_id: Optional[int] = None,
        worker_i: Optional[int] = None,
        **kwargs,
    ):
        self.device = device
        self.num_workers = num_workers
        self.worker_id = worker_id if worker_id is not None else worker_i
        self.worker_i = worker_i if worker_i is not None else worker_id

    def setup(self) -> None:
        return

    def process(self, job: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
    
    def process_batch(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(jobs, list):
            raise ValueError(f"process_batch() expects list[dict], got {type(jobs)}")
        return [self.process(j) for j in jobs]


def _load_class(module_path: str, class_name: str):
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def _build_processor(processor_cfg: Dict[str, Any], device: str, worker_id: int, num_workers: int) -> Any:
    module_path = processor_cfg["module"]
    class_name = processor_cfg["class"]
    kwargs = processor_cfg.get("kwargs", {})

    processor_cls = _load_class(module_path, class_name)

    injected_kwargs = {
        "num_workers": num_workers,
        "worker_id": worker_id,
        "worker_i": worker_id,
    }

    try:
        sig = inspect.signature(processor_cls.__init__)
        params = sig.parameters
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not accepts_var_kwargs:
            injected_kwargs = {k: v for k, v in injected_kwargs.items() if k in params}
    except Exception:
        injected_kwargs = {}

    merged_kwargs = dict(kwargs or {})
    merged_kwargs.update(injected_kwargs)

    processor = processor_cls(device=device, **merged_kwargs)
    if hasattr(processor, "setup"):
        processor.setup()
    return processor


def _format_exc(e: BaseException) -> Dict[str, Any]:
    tb = traceback.format_exc()
    # ... existing code ...
    if tb.strip() == "NoneType: None":
        tb = ""
    return {"type": type(e).__name__, "message": str(e), "traceback": tb}


def worker_main(
    worker_id: int,
    num_workers: int,
    device_env: Optional[str],
    device_str: str,
    task_queue,
    ack_queue,
    queue_paths: QueuePaths,
    fq_cfg: FileQueueConfig,
    processor_cfg: Dict[str, Any],
    isolate_cuda_visible_devices: bool,
    max_jobs_per_worker: int,
):
    # - 默认不依赖 per-worker 修改 CUDA_VISIBLE_DEVICES，而是把 device_str 传给 processor。
    # - 如需强隔离（防止误用其它卡），可配置 isolate_cuda_visible_devices=True。
    if isolate_cuda_visible_devices and device_env is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_env)

    # Best-effort: set torch current device early (before user processor imports torch models).
    if isinstance(device_str, str) and device_str.startswith("cuda:"):
        try:
            import torch

            cuda_idx = int(device_str.split(":", 1)[1])
            if torch.cuda.is_available():
                torch.cuda.set_device(cuda_idx)
        except Exception:
            # Don't hard fail here; user processor may not use torch.
            pass

    try:
        max_jobs_per_worker_eff = int(max_jobs_per_worker)
    except Exception:
        max_jobs_per_worker_eff = 1
    if max_jobs_per_worker_eff <= 0:
        max_jobs_per_worker_eff = 1

    fsq = FileSystemQueue(queue_paths, fq_cfg)
    processor = _build_processor(processor_cfg, device=device_str, worker_id=worker_id, num_workers=num_workers)

    while True:
        try:
            item = task_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if item is None:
            break

        processing_paths = [item]
        should_stop_after_batch = False

        # Grab up to N-1 more tasks in this worker "tick".
        for _ in range(max_jobs_per_worker_eff - 1):
            try:
                extra = task_queue.get_nowait()
            except queue.Empty:
                break

            if extra is None:
                should_stop_after_batch = True
                try:
                    task_queue.put_nowait(None)
                except Exception:
                    pass
                break

            processing_paths.append(extra)

        load_ok_jobs: List[Dict[str, Any]] = []
        load_ok_paths: List[str] = []
        load_ok_t0s: List[float] = []

        # Per-path: touch mtime and load job; loading failures should not poison the whole batch.
        for processing_path in processing_paths:
            t0 = _now_s()
            try:
                os.utime(processing_path, None)
            except Exception:
                pass

            try:
                job = fsq.load_job(processing_path)
            except Exception as e:
                error_obj = _format_exc(e)
                fsq.mark_failed(
                    processing_path,
                    {
                        "ok": False,
                        "worker_id": worker_id,
                        "device": device_str,
                        "t_process_s": _now_s() - t0,
                        "error": error_obj,
                    },
                )
                try:
                    ack_queue.put(
                        {
                            "ok": False,
                            "worker_id": worker_id,
                            "device": device_str,
                            "processing_path": processing_path,
                            "t_process_s": _now_s() - t0,
                            "error": error_obj,
                        }
                    )
                except Exception:
                    pass
                continue

            load_ok_jobs.append(job)
            load_ok_paths.append(processing_path)
            load_ok_t0s.append(t0)

        if load_ok_jobs:
            try:
                result_objs = processor.process_batch(load_ok_jobs)
            except Exception as e:
                error_obj = _format_exc(e)
                for processing_path, t0 in zip(load_ok_paths, load_ok_t0s):
                    fsq.mark_failed(
                        processing_path,
                        {
                            "ok": False,
                            "worker_id": worker_id,
                            "device": device_str,
                            "t_process_s": _now_s() - t0,
                            "error": error_obj,
                        },
                    )
                    try:
                        ack_queue.put(
                            {
                                "ok": False,
                                "worker_id": worker_id,
                                "device": device_str,
                                "processing_path": processing_path,
                                "t_process_s": _now_s() - t0,
                                "error": error_obj,
                            }
                        )
                    except Exception:
                        pass
            else:
                if not isinstance(result_objs, list) or len(result_objs) != len(load_ok_jobs):
                    error_obj = _format_exc(
                        ValueError(
                            f"process_batch() must return list[dict] with same length as jobs, got {type(result_objs)}"
                        )
                    )
                    for processing_path, t0 in zip(load_ok_paths, load_ok_t0s):
                        fsq.mark_failed(
                            processing_path,
                            {
                                "ok": False,
                                "worker_id": worker_id,
                                "device": device_str,
                                "t_process_s": _now_s() - t0,
                                "error": error_obj,
                            },
                        )
                        try:
                            ack_queue.put(
                                {
                                    "ok": False,
                                    "worker_id": worker_id,
                                    "device": device_str,
                                    "processing_path": processing_path,
                                    "t_process_s": _now_s() - t0,
                                    "error": error_obj,
                                }
                            )
                        except Exception:
                            pass
                else:
                    for processing_path, t0, result_obj in zip(load_ok_paths, load_ok_t0s, result_objs):
                        ok = False
                        error_obj = None

                        try:
                            if not isinstance(result_obj, dict):
                                raise ValueError(f"process() must return dict, got {type(result_obj)}")

                            out_path = fsq.output_path_for(processing_path)
                            fsq.save_result(
                                out_path,
                                {
                                    "ok": True,
                                    "worker_id": worker_id,
                                    "device": device_str,
                                    "t_process_s": _now_s() - t0,
                                    "result": result_obj,
                                },
                            )
                            fsq.mark_done(processing_path)
                            ok = True

                        except Exception as e:
                            error_obj = _format_exc(e)
                            fsq.mark_failed(
                                processing_path,
                                {
                                    "ok": False,
                                    "worker_id": worker_id,
                                    "device": device_str,
                                    "t_process_s": _now_s() - t0,
                                    "error": error_obj,
                                },
                            )

                        try:
                            ack_queue.put(
                                {
                                    "ok": ok,
                                    "worker_id": worker_id,
                                    "device": device_str,
                                    "processing_path": processing_path,
                                    "t_process_s": _now_s() - t0,
                                    "error": error_obj,
                                }
                            )
                        except Exception:
                            pass

        if should_stop_after_batch:
            break


class AsyncFileService:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        work_dir = cfg["work_dir"]
        self.queue_paths = build_queue_paths(work_dir)

        q_cfg_raw = cfg.get("queue", {})
        cleanup_cfg = q_cfg_raw.get("cleanup", {}) or {}

        self.fq_cfg = FileQueueConfig(
            poll_interval_s=float(q_cfg_raw.get("poll_interval_s", 0.2)),
            max_in_flight=int(q_cfg_raw.get("max_in_flight", 512)),
            input_ext=str(q_cfg_raw.get("input_ext", ".json")),
            output_ext=str(q_cfg_raw.get("output_ext", ".json")),
            job_codec=str(q_cfg_raw.get("job_codec", "json")),
            result_codec=str(q_cfg_raw.get("result_codec", "json")),
            keep_processed_inputs=bool(q_cfg_raw.get("keep_processed_inputs", False)),
            lease_timeout_s=float(q_cfg_raw.get("lease_timeout_s", 3600.0)),
            recovery_interval_s=float(q_cfg_raw.get("recovery_interval_s", 5.0)),
            cleanup_enabled=bool(cleanup_cfg.get("enabled", False)),
            cleanup_interval_s=float(cleanup_cfg.get("interval_s", 600.0)),
            cleanup_failed_ttl_s=float(cleanup_cfg.get("failed_ttl_s", 7.0 * 24.0 * 3600.0)),
            cleanup_outputs_ttl_s=float(cleanup_cfg.get("outputs_ttl_s", 0.0)),
            cleanup_done_ttl_s=float(cleanup_cfg.get("done_ttl_s", 0.0)),
            cleanup_max_delete_per_dir=int(cleanup_cfg.get("max_delete_per_dir", 10000)),
        )

        self.fsq = FileSystemQueue(self.queue_paths, self.fq_cfg)

        w_cfg = cfg.get("workers", {})
        self.start_method = str(w_cfg.get("start_method", "spawn"))
        self.daemon = bool(w_cfg.get("daemon", True))

        # 每个 worker 每次最多拉取/处理多少个任务（一个 "tick" 内）。
        # 语义上相当于：每个 worker 的 in-flight 上限。
        self.max_jobs_per_worker = int(w_cfg.get("max_jobs_per_worker", 1))
        if self.max_jobs_per_worker <= 0:
            self.max_jobs_per_worker = 1

        # 是否把每个 worker 的 CUDA_VISIBLE_DEVICES 隔离成单卡。
        # - True: 更强隔离（每个 worker 只看到一张卡），device 通常为 cuda:0
        # - False: 更通用（worker 看到所有可见卡），device 为 cuda:N
        self.isolate_cuda_visible_devices = bool(w_cfg.get("isolate_cuda_visible_devices", False))

        self.workers_per_device = int(w_cfg.get("workers_per_device", 1))
        self.cpu_workers = int(w_cfg.get("cpu_workers", 0))
        self.devices = self._resolve_devices(w_cfg.get("devices", "auto"), isolate=self.isolate_cuda_visible_devices)

        # 把全局 max_in_flight 软限制到：num_workers * max_jobs_per_worker。
        # 这样服务不会一次性 claim 太多文件到 processing/，而是按 worker 并发能力逐步领取。
        try:
            self.max_in_flight_effective = min(
                int(self.fq_cfg.max_in_flight),
                int(len(self.devices)) * int(self.max_jobs_per_worker),
            )
        except Exception:
            self.max_in_flight_effective = int(self.fq_cfg.max_in_flight)
        if self.max_in_flight_effective <= 0:
            self.max_in_flight_effective = int(self.fq_cfg.max_in_flight)

        self.processor_cfg = cfg["processor"]

        import multiprocessing as mp

        self.ctx = mp.get_context(self.start_method)
        self.task_queue = self.ctx.Queue(maxsize=int(w_cfg.get("task_queue_size", 1024)))
        self.ack_queue = self.ctx.Queue()  # unbounded
        self.procs = []

        self._stop = False
        self._in_flight = 0

        self._stats_last_t = _now_s()
        self._stats_interval_s = float(cfg.get("stats_interval_s", 30.0))
        self._total_claimed = 0
        self._total_done = 0
        self._total_failed = 0

        self._last_recovery_t = 0.0
        self._last_cleanup_t = 0.0

    def _detect_cuda_device_count(self) -> int:
        """Best-effort GPU count detection without assuming torch is always installed."""
        # If CUDA_VISIBLE_DEVICES is set, that defines the visible device count.
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cuda_visible:
            return len([d for d in cuda_visible.split(",") if d.strip()])

        # Fallback: try torch.
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.device_count())
        except Exception:
            pass

        return 0

    def _resolve_devices(self, devices_cfg, isolate: bool) -> Tuple[Tuple[Optional[str], str], ...]:
        """Return per-worker assignment: (device_env, device_str).

        - isolate=True:
          - device_env is a single GPU id string (set as CUDA_VISIBLE_DEVICES inside worker)
          - device_str is always "cuda:0" for GPU workers

        - isolate=False:
          - device_env is None
          - device_str is "cuda:N" where N is the visible index in this process
        """
        assignments = []

        if self.cpu_workers > 0:
            for _ in range(self.cpu_workers):
                assignments.append((None, "cpu"))

        device_list: list = []

        if devices_cfg == "auto":
            # Prefer env-defined visibility, otherwise detect count.
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if cuda_visible:
                raw = [d.strip() for d in cuda_visible.split(",") if d.strip()]
                if isolate:
                    # Treat raw values as physical IDs (since we'll re-assign CUDA_VISIBLE_DEVICES per worker).
                    device_list = raw
                else:
                    # Treat as visible indices 0..k-1 in current process.
                    device_list = list(range(len(raw)))
            else:
                n = self._detect_cuda_device_count()
                device_list = list(range(n))

        elif isinstance(devices_cfg, int):
            # Interpreted as number of GPUs.
            device_list = list(range(int(devices_cfg)))

        elif isinstance(devices_cfg, (list, tuple)):
            device_list = list(devices_cfg)

        else:
            device_list = []

        # GPU workers
        for d in device_list:
            for _ in range(self.workers_per_device):
                if isolate:
                    # Each worker only sees one GPU.
                    assignments.append((str(d), "cuda:0"))
                else:
                    # Worker sees all GPUs; pick explicit visible index.
                    assignments.append((None, f"cuda:{int(d)}"))

        if not assignments:
            assignments.append((None, "cpu"))

        return tuple(assignments)

    def _install_signal_handlers(self):
        def _handler(signum, frame):
            self._stop = True

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def start_workers(self):
        num_workers = len(self.devices)
        for worker_id, (device_env, device_str) in enumerate(self.devices):
            p = self.ctx.Process(
                target=worker_main,
                args=(
                    worker_id,
                    num_workers,
                    device_env,
                    device_str,
                    self.task_queue,
                    self.ack_queue,
                    self.queue_paths,
                    self.fq_cfg,
                    self.processor_cfg,
                    self.isolate_cuda_visible_devices,
                    self.max_jobs_per_worker,
                ),
                daemon=self.daemon,
            )
            p.start()
            self.procs.append(p)

    def _stop_workers(self):
        for _ in self.procs:
            try:
                self.task_queue.put_nowait(None)
            except Exception:
                pass

        deadline = _now_s() + 10.0
        for p in self.procs:
            remaining = max(0.0, deadline - _now_s())
            try:
                p.join(timeout=remaining)
            except Exception:
                pass

        for p in self.procs:
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass

    def _drain_acks(self, max_items: int = 5000) -> None:
        drained = 0
        while drained < max_items:
            try:
                ack = self.ack_queue.get_nowait()
            except queue.Empty:
                break

            drained += 1
            self._in_flight = max(0, self._in_flight - 1)
            if ack.get("ok"):
                self._total_done += 1
            else:
                self._total_failed += 1

    def _print_stats_if_needed(self):
        now = _now_s()
        if now - self._stats_last_t < self._stats_interval_s:
            return
        self._stats_last_t = now

        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        alive = sum(1 for p in self.procs if p.is_alive())
        print(
            f"[{ts}] [STATS] workers_alive={alive}/{len(self.procs)} "
            f"in_flight={self._in_flight}/{self.max_in_flight_effective} claimed={self._total_claimed} "
            f"done={self._total_done} failed={self._total_failed}",
            flush=True,
        )

    def _recovery_if_needed(self):
        if self.fq_cfg.recovery_interval_s <= 0:
            return
        now = _now_s()
        if now - self._last_recovery_t < self.fq_cfg.recovery_interval_s:
            return
        self._last_recovery_t = now

        moved = self.fsq.requeue_stale_processing()
        if moved > 0:
            print(f"[WARN] requeued stale processing jobs: {moved}", flush=True)

    def _cleanup_if_needed(self):
        if not self.fq_cfg.cleanup_enabled:
            return

        try:
            interval_s = float(self.fq_cfg.cleanup_interval_s)
        except Exception:
            interval_s = 0.0
        if interval_s <= 0:
            return

        now = _now_s()
        if now - self._last_cleanup_t < interval_s:
            return
        self._last_cleanup_t = now

        try:
            deleted = self.fsq.cleanup_old_files(now=now)
        except Exception as e:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
            print(f"[{ts}] [WARN] cleanup failed: {type(e).__name__}: {e}", flush=True)
            return

        if not isinstance(deleted, dict):
            return

        total = 0
        for v in deleted.values():
            try:
                total += int(v)
            except Exception:
                pass

        if total > 0:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
            detail = " ".join([f"{k}={int(v)}" for k, v in deleted.items() if int(v) > 0])
            if not detail:
                detail = "n/a"
            print(f"[{ts}] [CLEANUP] deleted={total} {detail}", flush=True)

    def serve_forever(self):
        self._install_signal_handlers()
        self.start_workers()

        print(
            f"AsyncFileService listening on {self.queue_paths.root_dir} | "
            f"inputs={self.queue_paths.inputs_dir} outputs={self.queue_paths.outputs_dir} "
            f"job_codec={self.fq_cfg.job_codec} result_codec={self.fq_cfg.result_codec}",
            flush=True,
        )

        try:
            while not self._stop:
                self._drain_acks()
                self._recovery_if_needed()
                self._cleanup_if_needed()

                while not self._stop and self._in_flight < self.max_in_flight_effective:
                    processing_path = self.fsq.claim_one()
                    if processing_path is None:
                        break

                    try:
                        self.task_queue.put_nowait(processing_path)
                    except queue.Full:
                        try:
                            os.rename(
                                processing_path,
                                os.path.join(self.queue_paths.inputs_dir, os.path.basename(processing_path)),
                            )
                        except Exception:
                            pass
                        break

                    self._in_flight += 1
                    self._total_claimed += 1

                self._print_stats_if_needed()
                time.sleep(self.fq_cfg.poll_interval_s)

        finally:
            self._stop_workers()


class FileQueueClient:
    def __init__(self, work_dir: str, input_ext: str = ".npy", output_ext: str = ".npy", codec: str = "npy"):
        self.paths = build_queue_paths(work_dir)
        _safe_makedirs(self.paths.inputs_dir)
        _safe_makedirs(self.paths.outputs_dir)

        self.input_ext = input_ext
        self.output_ext = output_ext
        self.codec = (codec or "json").lower()

        if self.codec not in ("json", "npy"):
            raise ValueError(f"Unsupported codec={self.codec}, expected json|npy")

    def _job_path(self, job_id: str) -> str:
        return os.path.join(self.paths.inputs_dir, job_id + self.input_ext)

    def _result_path(self, job_id: str) -> str:
        return os.path.join(self.paths.outputs_dir, job_id + self.output_ext)

    def submit_job(self, job_obj: Dict[str, Any], job_id: Optional[str] = None) -> str:
        if not isinstance(job_obj, dict):
            raise ValueError(f"job_obj must be a dict, got {type(job_obj)}")

        if job_id is None:
            job_id = str(job_obj.get("job_id") or "").strip() or None

        if job_id is None:
            job_id = f"{int(_now_s() * 1e6)}_{os.getpid()}_{random.randint(0, 1_000_000)}"

        job_obj = dict(job_obj)
        job_obj["job_id"] = job_id

        dst_path = self._job_path(job_id)
        if self.codec == "json":
            _atomic_write_json(dst_path, job_obj)
        else:
            _atomic_write_npy(dst_path, job_obj)
        return job_id

    def submit_payload(self, payload: Dict[str, Any], job_id: Optional[str] = None) -> str:
        if not isinstance(payload, dict):
            raise ValueError(f"payload must be a dict, got {type(payload)}")

        job_obj = {
            "job_id": job_id,
            "payload": payload,
            "meta": {"submit_time_s": _now_s(), "pid": os.getpid()},
        }
        return self.submit_job(job_obj, job_id=job_id)

    def submit(self, payload: Dict[str, Any], job_id: Optional[str] = None) -> str:
        return self.submit_payload(payload, job_id=job_id)

    def wait_result(
        self,
        job_id: str,
        timeout_s: Optional[float] = None,
        poll_s: float = 0.2,
        delete_after_read: bool = False,
    ) -> Dict[str, Any]:
        t0 = _now_s()
        result_path = self._result_path(job_id)

        while True:
            if os.path.isfile(result_path):
                if result_path.endswith(".json"):
                    result = _load_json(result_path)
                else:
                    result = _load_npy(result_path)

                if delete_after_read:
                    try:
                        os.remove(result_path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass

                return result

            if timeout_s is not None and (_now_s() - t0) > timeout_s:
                raise TimeoutError(f"Result not found for job_id={job_id} after {timeout_s}s")
            time.sleep(poll_s)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_pid_file(work_dir: str) -> str:
    return os.path.join(os.path.abspath(work_dir), "service.pid")


def _write_pid_file(pid_file: str) -> None:
    _atomic_write_bytes(pid_file, (str(os.getpid()) + "\n").encode("utf-8"))


def _read_pid_file(pid_file: str) -> int:
    with open(pid_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return int(content)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _resolve_work_dir_and_queue_defaults(cfg: Optional[Dict[str, Any]], work_dir_arg: Optional[str]):
    work_dir = work_dir_arg or (cfg.get("work_dir") if cfg else None)
    if not work_dir:
        raise ValueError("work_dir is required (pass --work_dir or provide it in --config)")

    queue_cfg = (cfg.get("queue", {}) if cfg else {})
    input_ext = str(queue_cfg.get("input_ext", ".json"))
    output_ext = str(queue_cfg.get("output_ext", ".json"))
    job_codec = str(queue_cfg.get("job_codec", "json"))

    return work_dir, input_ext, output_ext, job_codec


def _resolve_pid_file(cfg: Optional[Dict[str, Any]], work_dir: str, pid_file_arg: Optional[str]) -> str:
    if pid_file_arg:
        return pid_file_arg
    service_cfg = (cfg.get("service", {}) if cfg else {})
    pid_file_cfg = service_cfg.get("pid_file")
    if pid_file_cfg:
        return str(pid_file_cfg)
    return _default_pid_file(work_dir)


def _reset_work_dir(work_dir: str) -> None:
    work_dir_abs = os.path.abspath(str(work_dir))

    if work_dir_abs in (os.path.abspath(os.sep), ""):
        raise ValueError(f"Refusing to reset dangerous work_dir={work_dir_abs!r}")

    parts = [p for p in work_dir_abs.split(os.sep) if p]
    if len(parts) < 2:
        raise ValueError(f"Refusing to reset suspiciously-broad work_dir={work_dir_abs!r}")

    if os.path.isfile(work_dir_abs):
        raise ValueError(f"work_dir must be a directory, got file: {work_dir_abs}")

    if os.path.isdir(work_dir_abs):
        shutil.rmtree(work_dir_abs)

    os.makedirs(work_dir_abs, exist_ok=True)
    print(f"[WARN] reset work_dir: {work_dir_abs}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_srv = sub.add_parser("serve")
    p_srv.add_argument("--config", required=True)
    p_srv.add_argument(
        "--work_dir",
        default=None,
        help="Override config.work_dir. Priority: CLI > config.",
    )
    p_srv.add_argument(
        "--reset",
        action="store_true",
        help="If set, delete and recreate work_dir before serving (DANGEROUS).",
    )
    p_srv.add_argument("--pid_file", default=None)
    p_srv.add_argument("--no_pid_file", action="store_true")

    p_cli = sub.add_parser("submit")
    p_cli.add_argument("--config", default=None)
    p_cli.add_argument("--work_dir", default=None)
    p_cli.add_argument("--job_id", default=None)

    job_group = p_cli.add_mutually_exclusive_group(required=True)
    job_group.add_argument("--payload_json", default=None)
    job_group.add_argument("--payload_json_file", default=None)
    job_group.add_argument("--job_json", default=None)
    job_group.add_argument("--job_json_file", default=None)

    p_cli.add_argument("--codec", default=None)
    p_cli.add_argument("--input_ext", default=None)
    p_cli.add_argument("--output_ext", default=None)

    p_cli.add_argument("--wait", action="store_true")
    p_cli.add_argument("--timeout_s", type=float, default=None)
    p_cli.add_argument("--poll_s", type=float, default=0.2)

    p_stop = sub.add_parser("stop")
    p_stop.add_argument("--config", default=None)
    p_stop.add_argument("--work_dir", default=None)
    p_stop.add_argument("--pid_file", default=None)
    p_stop.add_argument("--timeout_s", type=float, default=10.0)

    args = parser.parse_args()

    if args.cmd == "serve":
        cfg = _load_yaml(args.config)

        if args.work_dir:
            cfg["work_dir"] = args.work_dir

        work_dir = cfg.get("work_dir")
        if not work_dir:
            raise ValueError("work_dir is required (pass --work_dir or provide it in --config)")

        if args.reset:
            _reset_work_dir(work_dir)

        svc = AsyncFileService(cfg)

        pid_file = _resolve_pid_file(cfg, cfg["work_dir"], args.pid_file)
        if not args.no_pid_file:
            _write_pid_file(pid_file)
            print(f"[INFO] pid_file={pid_file}", flush=True)

        try:
            svc.serve_forever()
        finally:
            if not args.no_pid_file:
                try:
                    os.remove(pid_file)
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
        return

    if args.cmd == "submit":
        cfg = _load_yaml(args.config) if args.config else None
        work_dir, input_ext, output_ext, job_codec = _resolve_work_dir_and_queue_defaults(cfg, args.work_dir)

        codec = (args.codec or job_codec)
        input_ext = (args.input_ext or input_ext)
        output_ext = (args.output_ext or output_ext)

        cli = FileQueueClient(work_dir, input_ext=input_ext, output_ext=output_ext, codec=codec)

        if args.payload_json_file:
            with open(args.payload_json_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            job_id = cli.submit_payload(payload, job_id=args.job_id)
        elif args.payload_json:
            payload = json.loads(args.payload_json)
            job_id = cli.submit_payload(payload, job_id=args.job_id)
        elif args.job_json_file:
            with open(args.job_json_file, "r", encoding="utf-8") as f:
                job_obj = json.load(f)
            job_id = cli.submit_job(job_obj, job_id=args.job_id)
        else:
            job_obj = json.loads(args.job_json)
            job_id = cli.submit_job(job_obj, job_id=args.job_id)

        print(job_id)

        if args.wait:
            res = cli.wait_result(job_id, timeout_s=args.timeout_s, poll_s=args.poll_s)
            if isinstance(res, dict):
                print(json.dumps(res, ensure_ascii=False))
            else:
                print(res)
        return

    if args.cmd == "stop":
        cfg = _load_yaml(args.config) if args.config else None
        work_dir, _, _, _ = _resolve_work_dir_and_queue_defaults(cfg, args.work_dir)
        pid_file = _resolve_pid_file(cfg, work_dir, args.pid_file)

        if not os.path.isfile(pid_file):
            raise FileNotFoundError(f"pid_file not found: {pid_file}")

        pid = _read_pid_file(pid_file)
        if not _pid_alive(pid):
            print(f"[WARN] pid {pid} not alive; removing stale pid_file {pid_file}", flush=True)
            try:
                os.remove(pid_file)
            except Exception:
                pass
            return

        print(f"[INFO] sending SIGTERM to pid={pid}", flush=True)
        os.kill(pid, signal.SIGTERM)

        deadline = _now_s() + float(args.timeout_s)
        while _now_s() < deadline:
            if not _pid_alive(pid):
                print("[INFO] stopped", flush=True)
                try:
                    os.remove(pid_file)
                except Exception:
                    pass
                return
            time.sleep(0.1)

        raise TimeoutError(f"Service pid={pid} did not exit within {args.timeout_s}s")


if __name__ == "__main__":
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')
    main()
