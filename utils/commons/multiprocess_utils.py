import os
from typing import Optional
import re
import traceback
import multiprocessing
import multiprocessing as mp
from multiprocessing import Process, Queue
from queue import Empty
import random
import time

setproctitle_installed = False
try:
    import setproctitle
    setproctitle_installed = True
except:
    setproctitle_installed = False

DEFAULT_FATAL_ERROR_PATTERNS = [
    r"device-side assert",
    r"illegal memory access",
    r"unspecified launch failure",
    r"CUBLAS_STATUS_EXECUTION_FAILED",
    r"CUBLAS_STATUS_INTERNAL_ERROR",
    r"CUDNN_STATUS_EXECUTION_FAILED",
    r"CUDNN_STATUS_INTERNAL_ERROR",
    r"CUDA error: an illegal memory access",
    r"CUDA error: device-side assert triggered",
]


def _normalize_fatal_error_patterns(fatal_error_patterns=None, use_default_fatal_error_patterns=True):
    patterns = []
    if use_default_fatal_error_patterns:
        patterns.extend(DEFAULT_FATAL_ERROR_PATTERNS)
    if fatal_error_patterns:
        patterns.extend(list(fatal_error_patterns))
    return [str(pattern) for pattern in patterns if str(pattern).strip()]


def is_fatal_worker_error(
        exc,
        traceback_text,
        data_pipe=None,
        fatal_error_patterns=None,
        use_default_fatal_error_patterns=True,
        fatal_error_checker=None):
    if data_pipe is not None and hasattr(data_pipe, "is_fatal_error"):
        try:
            if bool(data_pipe.is_fatal_error(exc, traceback_text)):
                return True
        except Exception:
            traceback.print_exc()

    if fatal_error_checker is not None:
        try:
            if bool(fatal_error_checker(exc, traceback_text)):
                return True
        except Exception:
            traceback.print_exc()

    error_text = f"{type(exc).__name__}: {exc}\n{traceback_text}"
    for pattern in _normalize_fatal_error_patterns(fatal_error_patterns, use_default_fatal_error_patterns):
        if re.search(pattern, error_text, flags=re.IGNORECASE):
            return True
    return False


def _build_fatal_worker_event(worker_id, job_id, exc, traceback_text):
    return {
        "type": "fatal_worker_error",
        "worker_id": int(worker_id),
        "job_id": None if job_id is None else int(job_id),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback_text,
        "timestamp_sec": round(time.time(), 6),
    }


def _stop_process(process, timeout=5.0):
    if process is None:
        return
    try:
        process.join(timeout=max(0.0, float(timeout)))
    except Exception:
        pass
    if process.is_alive():
        try:
            process.terminate()
        except Exception:
            pass
        try:
            process.join(timeout=max(0.0, float(timeout)))
        except Exception:
            pass
    if process.is_alive() and hasattr(process, "kill"):
        try:
            process.kill()
        except Exception:
            pass
        try:
            process.join(timeout=max(0.0, float(timeout)))
        except Exception:
            pass


def chunked_worker(worker_id, map_func, args, results_queue=None, init_ctx_func=None):
    ctx = init_ctx_func(worker_id) if init_ctx_func is not None else None
    for job_idx, arg in args:
        try:
            if not isinstance(arg, tuple) and not isinstance(arg, list):
                arg = [arg]
            if ctx is not None:
                res = map_func(*arg, ctx=ctx)
            else:
                res = map_func(*arg)
            results_queue.put((job_idx, res))
        except:
            traceback.print_exc()
            results_queue.put((job_idx, None))

def chunked_multiprocess_run(
        map_func, args, num_workers=None, ordered=True,
        init_ctx_func=None, q_max_size=1000, multithread=False):
    if multithread:
        from multiprocessing.dummy import Queue, Process
    else:
        from multiprocessing import Queue, Process
    args = zip(range(len(args)), args)
    args = list(args)
    n_jobs = len(args)
    if num_workers is None:
        num_workers = int(os.getenv('N_PROC', os.cpu_count()))
    results_queues = []
    if ordered:
        for i in range(num_workers):
            results_queues.append(Queue(maxsize=q_max_size // num_workers))
    else:
        results_queue = Queue(maxsize=q_max_size)
        for i in range(num_workers):
            results_queues.append(results_queue)
    workers = []
    for i in range(num_workers):
        args_worker = args[i::num_workers]
        p = Process(target=chunked_worker, args=(
            i, map_func, args_worker, results_queues[i], init_ctx_func), daemon=True)
        workers.append(p)
        p.start()
    for n_finished in range(n_jobs):
        results_queue = results_queues[n_finished % num_workers]
        job_idx, res = results_queue.get()
        assert job_idx == n_finished or not ordered, (job_idx, n_finished)
        yield res
    for w in workers:
        w.join()


def multiprocess_glob(pattern, num_workers=None):
    split_pattern = pattern.split("/")
    recursive_depth = 0  # number of recursive depth
    for split in split_pattern:
        if '*' in split:
            recursive_depth += 1

    if recursive_depth <= 1:
        return glob.glob(pattern)

    dirs = multiprocess_glob('/'.join(split_pattern[:-1]), num_workers=num_workers)
    args = [f'{d}/{split_pattern[-1]}' for d in dirs]

    if '*' not in split_pattern[-1]:
        return args

    if len(args) == 0:
        return []

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    num_workers = max(1, min(int(num_workers), len(args)))

    ret = []
    with dummy.Pool(num_workers) as p:
        for res in tqdm(p.imap_unordered(glob.glob, args), total=len(args), desc=f"globing {pattern}"):
            ret.extend(res)
    return ret


class MultiprocessDataPipe:
    def __init__(self, device='cuda'):
        raise NotImplementedError

    def process(self, *args, **kwargs):
        raise NotImplementedError
    
def data_pipe_worker(
        data_pipe_cls, init_kwargs, task_queue: Queue, result_queue: Queue, pbar_queue: Queue,
        worker_proc_name='data_pipe_worker',
        restart_worker_on_fatal=False,
        fatal_error_patterns=None,
        use_default_fatal_error_patterns=True,
        fatal_error_checker=None,
        control_queue=None
    ):
    if setproctitle_installed:
        setproctitle.setproctitle(f'{worker_proc_name}:({init_kwargs["worker_id"]}/{init_kwargs["num_workers"]})')
    data_pipe = data_pipe_cls(**init_kwargs)
    while True:
        job_id = None
        try:
            job_args = task_queue.get()
            if job_args is None:
                break
            job_id, (args, kwargs) = job_args
            result = data_pipe.process(*args, **kwargs)
            result_queue.put((job_id, result))
            pbar_queue.put(1)
        except KeyboardInterrupt:
            break
        except Exception as exc:
            traceback_text = traceback.format_exc()
            print(traceback_text, end="")
            is_fatal = restart_worker_on_fatal and is_fatal_worker_error(
                exc,
                traceback_text,
                data_pipe=data_pipe,
                fatal_error_patterns=fatal_error_patterns,
                use_default_fatal_error_patterns=use_default_fatal_error_patterns,
                fatal_error_checker=fatal_error_checker,
            )
            if job_id is not None:
                result_queue.put((job_id, None))
                pbar_queue.put(1)
            if is_fatal:
                if control_queue is not None:
                    control_queue.put(_build_fatal_worker_event(init_kwargs["worker_id"], job_id, exc, traceback_text))
                break

def pbar_worker(pbar_queue: Queue, total=None, desc=None, timeout=None):
    from tqdm import tqdm
    pbar = tqdm(total=total, desc=desc)
    retry = 0
    cnt = 0
    while True:
        try:
            item = pbar_queue.get(timeout=timeout)
            if item is None:
                break
            pbar.update(item)
            cnt += item
            retry = 0
            if total is not None and total > 0 and cnt >= total:
                break
        except Empty:
            retry += 1
            print(f"pbar_queue is empty after {timeout * retry} seconds")

def multiprocess_data_pipe_run(
        data_pipe_cls=MultiprocessDataPipe,
        job_args=[],
        job_kwargs=[],
        total=None,
        cls_init_kwargs={},
        init_shared_data={},
        n_devices=None,
        workers_per_device=1,
        ordered=True,
        desc=None,
        q_max_size=10000,
        use_tqdm=True,
        time_window=None,   # limited rate
        max_jobs_per_time_window=None,
        start_method='spawn',
        daemon=True,
        worker_proc_name='data_pipe_worker',
        restart_worker_on_fatal=False,
        fatal_error_patterns=None,
        use_default_fatal_error_patterns=True,
        fatal_error_checker=None,
        max_worker_restarts: Optional[int] = None,
        worker_restart_timeout=5.0,
        supervisor_poll_interval=0.2,
        on_fatal_worker_event=None
    ):
    try:
        ctx = multiprocessing.get_context(start_method) if start_method else multiprocessing.get_context()
    except ValueError:
        ctx = multiprocessing.get_context()

    manager = ctx.Manager()
    shared_dict = manager.dict()
    shared_lock = manager.Lock()
    
    if init_shared_data is not None and len(init_shared_data) > 0:
        shared_dict.update(init_shared_data)

    assert len(job_args) > 0 or len(job_kwargs) > 0 or total
    if len(job_args) > 0 and not (isinstance(job_args[0], list) or isinstance(job_args[0], tuple)):
        job_args = [(job_arg,) for job_arg in job_args]
    if len(job_args) > 0 and len(job_kwargs) > 0:
        assert len(job_args) == len(job_kwargs)
        jobs = [(a, kw) for a, kw in zip(job_args, job_kwargs)]
    elif len(job_args) > 0:
        jobs = [(a, {}) for a in job_args]
    elif len(job_kwargs) > 0:
        jobs = [((), kw) for kw in job_kwargs]
    else:
        jobs = [((), {}) for _ in range(total)]
    jobs = list(enumerate(jobs))

    if n_devices is None or n_devices < 0:
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
    elif n_devices == 0:    # cpu
        devices = []
    elif n_devices > 0:
        devices = list(range(n_devices))
    use_cuda = len(devices) > 0

    num_workers = len(devices) * workers_per_device if use_cuda else workers_per_device
    num_workers = max(1, num_workers)

    pbar_queue = ctx.Queue()
    result_queue = ctx.Queue(maxsize=q_max_size)
    task_queue = ctx.Queue(maxsize=q_max_size)
    control_queue = ctx.Queue() if restart_worker_on_fatal else None

    workers = [None for _ in range(num_workers)]
    worker_init_kwargs = []
    worker_restart_counts = [0 for _ in range(num_workers)]

    for i in range(num_workers):
        init_kwargs_ = {
            **cls_init_kwargs, 
            "worker_id": i, 
            "num_workers": num_workers,
            "shared_dict": shared_dict,
            "shared_lock": shared_lock
        }
        if use_cuda:
            init_kwargs_["device"] = f"cuda:{i % len(devices)}"
        worker_init_kwargs.append(init_kwargs_)

    def start_worker(worker_id):
        p = ctx.Process(
            target=data_pipe_worker,
            args=(
                data_pipe_cls,
                worker_init_kwargs[worker_id],
                task_queue,
                result_queue,
                pbar_queue,
                worker_proc_name,
                restart_worker_on_fatal,
                fatal_error_patterns,
                use_default_fatal_error_patterns,
                fatal_error_checker,
                control_queue,
            ),
            daemon=daemon,
        )
        p.start()
        return p

    def handle_fatal_worker_event(event):
        worker_id = int(event["worker_id"])
        worker_restart_counts[worker_id] += 1
        event = dict(event)
        event["restart_count"] = worker_restart_counts[worker_id]
        print(
            f"[WARN] fatal worker error: worker_id={worker_id} "
            f"job_id={event.get('job_id')} restart_count={event['restart_count']} "
            f"error={event.get('error')}"
        )
        if on_fatal_worker_event is not None:
            on_fatal_worker_event(event)
        _stop_process(workers[worker_id], timeout=worker_restart_timeout)
        if max_worker_restarts is not None and worker_restart_counts[worker_id] > int(max_worker_restarts):
            raise RuntimeError(
                f"worker_id={worker_id} fatal restart 次数超过 max_worker_restarts={max_worker_restarts}; "
                f"last_error={event.get('error')}"
            )
        workers[worker_id] = start_worker(worker_id)

    def drain_control_events(wait_timeout=0.0):
        if control_queue is None:
            return
        if wait_timeout is not None and wait_timeout > 0:
            try:
                event = control_queue.get(timeout=wait_timeout)
                handle_fatal_worker_event(event)
            except Empty:
                pass
        while True:
            try:
                event = control_queue.get_nowait()
            except Empty:
                break
            handle_fatal_worker_event(event)

    for i in range(num_workers):
        workers[i] = start_worker(i)
    if use_tqdm:
        pbar = ctx.Process(target=pbar_worker, args=(pbar_queue, len(jobs), desc), daemon=daemon)
        pbar.start()
        
    apply_rate_limit = (time_window is not None and max_jobs_per_time_window is not None)
    start_time = time.time()
    job_cnt_in_window = 0
    
    in_flight_window = max(1, q_max_size)
    sent = 0
    got = 0
    yielded = 0
    next_id = 0
    buffer = {}
    
    try:
        while got < len(jobs):
            drain_control_events()
            while sent - got < in_flight_window and sent < len(jobs):
                if apply_rate_limit:
                    now = time.time()
                    elapsed = now - start_time
                    if job_cnt_in_window >= max_jobs_per_time_window and elapsed < time_window:
                        time.sleep(time_window - elapsed)
                        start_time = time.time()
                        job_cnt_in_window = 0
                        
                task_queue.put(jobs[sent])
                sent += 1
                job_cnt_in_window += 1

            if restart_worker_on_fatal:
                while True:
                    drain_control_events()
                    try:
                        job_id, result = result_queue.get(timeout=supervisor_poll_interval)
                        break
                    except Empty:
                        continue
            else:
                job_id, result = result_queue.get()
            got += 1
            drain_control_events(supervisor_poll_interval if result is None else 0.0)
            if ordered:
                buffer[job_id] = result
                while next_id in buffer:
                    res = buffer.pop(next_id)
                    yielded += 1
                    next_id += 1
                    yield res
            else:
                yield result
            
        for i in range(num_workers):
            task_queue.put(None)

    except KeyboardInterrupt:
        for i in range(num_workers):
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass
        raise
    
    finally:
        try:
            pbar_queue.put(None)
        except Exception:
            pass
        
        for p in workers:
            if p.is_alive():
                p.join()
                
        if use_tqdm and pbar.is_alive():
            pbar.terminate()
            
        if ordered:
            while not result_queue.empty():
                job_id, result = result_queue.get()
                buffer[job_id] = result
            while next_id in buffer:
                yield buffer.pop(next_id)
                next_id += 1
        else:
            while not result_queue.empty():
                job_id, result = result_queue.get()
                yield result
                
        if use_tqdm and pbar.is_alive():
            pbar.join()
        os.system('stty echo')
    

def data_pipe_worker_server(
        data_pipe_cls, init_kwargs, task_queue, result_queue, worker_proc_name='data_pipe_worker',
        restart_worker_on_fatal=False,
        fatal_error_patterns=None,
        use_default_fatal_error_patterns=True,
        fatal_error_checker=None,
        control_queue=None):
    if setproctitle_installed:
        setproctitle.setproctitle(
            f'{worker_proc_name}:({init_kwargs["worker_id"]}/{init_kwargs["num_workers"]})'
        )

    data_pipe = data_pipe_cls(**init_kwargs)

    while True:
        job_id = None
        try:
            job = task_queue.get()
            if job is None:
                break
            job_id, (args, kwargs) = job
            result = data_pipe.process(*args, **kwargs)
            result_queue.put((job_id, result))
        except KeyboardInterrupt:
            break
        except Exception as exc:
            traceback_text = traceback.format_exc()
            print(traceback_text, end="")
            is_fatal = restart_worker_on_fatal and is_fatal_worker_error(
                exc,
                traceback_text,
                data_pipe=data_pipe,
                fatal_error_patterns=fatal_error_patterns,
                use_default_fatal_error_patterns=use_default_fatal_error_patterns,
                fatal_error_checker=fatal_error_checker,
            )
            if job_id is not None:
                result_queue.put((job_id, None))
            if is_fatal:
                if control_queue is not None:
                    control_queue.put(_build_fatal_worker_event(init_kwargs["worker_id"], job_id, exc, traceback_text))
                break

class LocalDataPipeQueueService:
    
    def __init__(
        self,
        data_pipe_cls,
        cls_init_kwargs=None,
        init_shared_data=None,
        n_devices=None,
        workers_per_device=1,
        ordered=True,
        q_max_size=10000,
        start_method='spawn',
        daemon=True,
        worker_proc_name='data_pipe_worker',
        restart_worker_on_fatal=False,
        fatal_error_patterns=None,
        use_default_fatal_error_patterns=True,
        fatal_error_checker=None,
        max_worker_restarts: Optional[int] = None,
        worker_restart_timeout=5.0,
        supervisor_poll_interval=0.2,
        on_fatal_worker_event=None
    ):
        if cls_init_kwargs is None:
            cls_init_kwargs = {}
        if init_shared_data is None:
            init_shared_data = {}
            
        self.ordered = ordered
        self.daemon = daemon
        self.worker_proc_name = worker_proc_name
        self.restart_worker_on_fatal = bool(restart_worker_on_fatal)
        self.fatal_error_patterns = fatal_error_patterns
        self.use_default_fatal_error_patterns = use_default_fatal_error_patterns
        self.fatal_error_checker = fatal_error_checker
        self.max_worker_restarts = max_worker_restarts
        self.worker_restart_timeout = worker_restart_timeout
        self.supervisor_poll_interval = supervisor_poll_interval
        self.on_fatal_worker_event = on_fatal_worker_event
        self.data_pipe_cls = data_pipe_cls

        try:
            self.ctx = mp.get_context(start_method) if start_method else mp.get_context()
        except ValueError:
            self.ctx = mp.get_context()

        manager = self.ctx.Manager()
        self.shared_dict = manager.dict()
        self.shared_lock = manager.Lock()
        if init_shared_data:
            self.shared_dict.update(init_shared_data)

        if n_devices is None or n_devices < 0:
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_visible.strip():
                devices = cuda_visible.split(',')
            else:
                devices = []
        elif n_devices == 0:
            devices = []
        else:
            devices = list(range(n_devices))
        use_cuda = len(devices) > 0

        self.num_workers = len(devices) * workers_per_device if use_cuda else workers_per_device
        self.num_workers = max(1, self.num_workers)

        self.input_queue = self.ctx.Queue(maxsize=q_max_size)
        self.output_queue = self.ctx.Queue(maxsize=q_max_size)
        self.control_queue = self.ctx.Queue() if self.restart_worker_on_fatal else None

        self.workers = []
        self.worker_init_kwargs = []
        self.worker_restart_counts = []
        for i in range(self.num_workers):
            init_kwargs_ = {
                **cls_init_kwargs,
                "worker_id": i,
                "num_workers": self.num_workers,
                "shared_dict": self.shared_dict,
                "shared_lock": self.shared_lock,
            }
            if use_cuda:
                init_kwargs_["device"] = f"cuda:{i % len(devices)}"

            self.worker_init_kwargs.append(init_kwargs_)
            self.worker_restart_counts.append(0)
            self.workers.append(self._start_worker(i))
            
        # ---------- submit / get_result 所需的状态 ----------
        self._next_job_id = 0        # 提交时自增
        self._next_expect_id = 0     # ordered=True 时，下一个应该返回的 job_id
        self._buffer = {}            # ordered=True 时暂存乱序结果：job_id -> result
        self._ignored_job_ids = set()
        self._ignored_job_meta = {}
        self._ignored_result_events = []
        self._fatal_worker_events = []

    def _start_worker(self, worker_id):
        p = self.ctx.Process(
            target=data_pipe_worker_server,
            args=(
                self.data_pipe_cls,
                self.worker_init_kwargs[worker_id],
                self.input_queue,
                self.output_queue,
                self.worker_proc_name,
                self.restart_worker_on_fatal,
                self.fatal_error_patterns,
                self.use_default_fatal_error_patterns,
                self.fatal_error_checker,
                self.control_queue,
            ),
            daemon=self.daemon,
        )
        p.start()
        return p

    def _handle_fatal_worker_event(self, event):
        worker_id = int(event["worker_id"])
        self.worker_restart_counts[worker_id] += 1
        event = dict(event)
        event["restart_count"] = self.worker_restart_counts[worker_id]
        self._fatal_worker_events.append(event)
        print(
            f"[WARN] fatal worker error: worker_id={worker_id} "
            f"job_id={event.get('job_id')} restart_count={event['restart_count']} "
            f"error={event.get('error')}"
        )
        if self.on_fatal_worker_event is not None:
            self.on_fatal_worker_event(event)
        _stop_process(self.workers[worker_id], timeout=self.worker_restart_timeout)
        if self.max_worker_restarts is not None and self.worker_restart_counts[worker_id] > int(self.max_worker_restarts):
            raise RuntimeError(
                f"worker_id={worker_id} fatal restart 次数超过 max_worker_restarts={self.max_worker_restarts}; "
                f"last_error={event.get('error')}"
            )
        if not getattr(self, "_closed", False):
            self.workers[worker_id] = self._start_worker(worker_id)

    def _drain_control_events(self, wait_timeout=0.0):
        if self.control_queue is None:
            return
        if wait_timeout is not None and wait_timeout > 0:
            try:
                event = self.control_queue.get(timeout=wait_timeout)
                self._handle_fatal_worker_event(event)
            except Empty:
                pass
        while True:
            try:
                event = self.control_queue.get_nowait()
            except Empty:
                break
            self._handle_fatal_worker_event(event)

    def pop_fatal_events(self):
        self._drain_control_events()
        events = list(self._fatal_worker_events)
        self._fatal_worker_events.clear()
        return events
        
    def submit(self, *args, **kwargs) -> int:
        """
        提交一个任务，返回分配的 job_id。
        """
        self._drain_control_events()
        job_id = self._next_job_id
        self._next_job_id += 1
        self.input_queue.put((job_id, (args, kwargs)))
        return job_id

    def ignore_job(self, job_id, metadata=None) -> None:
        if job_id is None:
            return
        job_id_int = int(job_id)
        self._ignored_job_ids.add(job_id_int)
        if metadata is not None:
            self._ignored_job_meta[job_id_int] = dict(metadata)

    def ignore_jobs(self, job_ids, metadata_by_job_id=None) -> None:
        for job_id in job_ids:
            metadata = None
            if isinstance(metadata_by_job_id, dict):
                metadata = metadata_by_job_id.get(job_id)
            self.ignore_job(job_id, metadata=metadata)

    def pop_ignored_results(self):
        events = list(self._ignored_result_events)
        self._ignored_result_events.clear()
        return events

    def _handle_ignored_result(self, job_id) -> bool:
        try:
            job_id_int = int(job_id)
        except Exception:
            return False
        if job_id_int not in self._ignored_job_ids:
            return False
        self._ignored_job_ids.discard(job_id_int)
        metadata = self._ignored_job_meta.pop(job_id_int, {})
        self._ignored_result_events.append(
            {
                "job_id": job_id_int,
                "metadata": metadata,
                "timestamp_sec": round(time.time(), 6),
            }
        )
        return True
    
    def get_result(self, timeout=None, return_job_id=False):
        """
        从 output_queue 取一个结果：
        
        - ordered=False:  直接返回任意一个已完成任务的 result（乱序）。
        - ordered=True:   按 submit 顺序返回第 next_expect_id 个任务的 result。
        
        注意：如果使用 ordered=True，请不要在其它地方直接从
        service.output_queue 自己消费结果，否则会打乱顺序逻辑。
        """
        if not self.ordered:
            # 无序：直接取一条结果；如遇到被主进程标记为忽略的迟到结果，则就地吞掉并继续取下一条。
            deadline = None if timeout is None else (time.time() + max(0.0, float(timeout)))
            while True:
                self._drain_control_events()
                wait_timeout = None
                if deadline is not None:
                    wait_timeout = max(0.0, deadline - time.time())
                    if wait_timeout <= 0:
                        raise Empty
                job_id, result = (
                    self.output_queue.get(timeout=wait_timeout)
                    if wait_timeout is not None
                    else self.output_queue.get()
                )
                self._drain_control_events(self.supervisor_poll_interval if result is None else 0.0)
                if self._handle_ignored_result(job_id):
                    continue
                if return_job_id:
                    return job_id, result
                return result
        
        # 有序：维护一个缓冲区
        deadline = None if timeout is None else (time.time() + max(0.0, float(timeout)))
        while True:
            self._drain_control_events()
            wait_timeout = None
            if deadline is not None:
                wait_timeout = max(0.0, deadline - time.time())
                if wait_timeout <= 0:
                    raise Empty
            job_id, result = self.output_queue.get(timeout=wait_timeout) if wait_timeout is not None else self.output_queue.get()
            self._drain_control_events(self.supervisor_poll_interval if result is None else 0.0)
            if self._handle_ignored_result(job_id):
                continue
            self._buffer[job_id] = result
            
            if self._next_expect_id in self._buffer:
                next_job_id = self._next_expect_id
                res = self._buffer.pop(self._next_expect_id)
                self._next_expect_id += 1
                if return_job_id:
                    return next_job_id, res
                return res

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True

        for _ in range(self.num_workers):
            try:
                self.input_queue.put(None, timeout=0.2)
            except Exception:
                pass

        for p in self.workers:
            if p.is_alive():
                p.join(timeout=1.0)
        for p in self.workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class DummyDataPipe(MultiprocessDataPipe):
    def __init__(self, device='cuda', fuck=True, **kwargs):
        pass
    
    def process(self, a, b):
        import time
        time.sleep(random.random())
        return a + b

if __name__ == '__main__':
        
    test = 2
    
    if test == 1:
            
        jobs = [(i, i) for i in range(1000)]

        results = []
        for result in multiprocess_data_pipe_run(
            DummyDataPipe, 
            job_args=jobs, 
            cls_init_kwargs={'fuck': False}, 
            desc='fuck', 
            ordered=False,
            workers_per_device=50,
            q_max_size=100
        ):
            results.append(result)
        
        print(results[:100])
        
    elif test == 2:
        
        # 1) 低级用法：直接用 input_queue/output_queue，自行管理 job_id 和顺序
        service = LocalDataPipeQueueService(
            data_pipe_cls=DummyDataPipe,
            cls_init_kwargs={'fuck': False},
            n_devices=0,              # CPU
            workers_per_device=4,
            q_max_size=100,
            ordered=False
        )
        
        # 往 input_queue 丢 10 个任务
        # for i in range(10):
        #     # job 结构： (job_id, (args, kwargs))
        #     service.input_queue.put((i, ((i, i), {})))
            
        # 从 output_queue 拿结果（无序）
        # results = {}
        # for _ in range(10):
        #     job_id, result = service.output_queue.get()
        #     results[job_id] = result
        # print("queue 模式结果（乱序）:", results)
        
        # # 2) 高级用法：直接 submit 同步调用（内部还是用 queue）
        print("submit 模式:", [service.submit(i, i) for i in range(5)])
        results = [service.get_result() for _ in range(5)]
        print(results)
        service.close()
        
