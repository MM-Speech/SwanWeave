import os
import time
import socket
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _now_ts() -> float:
    return time.time()


def _fmt_time(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return str(ts)


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def human_bytes(num_bytes: Optional[int], *, precision: int = 2, signed: bool = False) -> str:
    if num_bytes is None:
        return "-"
    try:
        n = float(num_bytes)
    except Exception:
        return "-"

    sign = ""
    if n < 0:
        sign = "-"
        n = -n
    elif signed:
        sign = "+"

    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    u = 0
    while n >= 1024.0 and u < len(units) - 1:
        n /= 1024.0
        u += 1

    if u == 0:
        return f"{sign}{int(n)}{units[u]}"
    return f"{sign}{n:.{precision}f}{units[u]}"


def _pct(numer: Optional[int], denom: Optional[int], *, precision: int = 1) -> str:
    if numer is None or denom is None or denom <= 0:
        return "-"
    return f"{(100.0 * float(numer) / float(denom)):.{precision}f}%"


def _get_dist_info() -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if torch is None:
        return None, None, None
    try:
        import torch.distributed as dist  # type: ignore
    except Exception:
        return None, None, None

    if not dist.is_available() or not dist.is_initialized():
        return None, None, None

    try:
        rank = int(dist.get_rank())
    except Exception:
        rank = None
    try:
        world_size = int(dist.get_world_size())
    except Exception:
        world_size = None
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank < 0:
            local_rank = None
    except Exception:
        local_rank = None
    return rank, world_size, local_rank


@dataclass(frozen=True)
class GPUMemoryDeviceSnapshot:
    device_index: int
    device_name: str

    allocated_bytes: Optional[int]
    reserved_bytes: Optional[int]
    max_allocated_bytes: Optional[int]
    max_reserved_bytes: Optional[int]

    active_bytes: Optional[int]
    inactive_split_bytes: Optional[int]
    num_alloc_retries: Optional[int]
    num_ooms: Optional[int]

    cuda_free_bytes: Optional[int]
    cuda_total_bytes: Optional[int]
    device_total_bytes: Optional[int]

    error: Optional[str] = None


@dataclass(frozen=True)
class GPUMemorySnapshot:
    tag: str
    time_s: float
    host: str
    pid: int
    rank: Optional[int]
    world_size: Optional[int]
    local_rank: Optional[int]
    cuda_visible_devices: str
    torch_cuda_version: str
    torch_version: str
    devices: Tuple[GPUMemoryDeviceSnapshot, ...]

    def format(self) -> str:
        return format_gpu_memory_snapshot(self)

    def log(self, printer: Callable[[str], None] = print) -> None:
        printer(self.format())


@dataclass(frozen=True)
class GPUMemoryDeviceDiff:
    device_index: int
    device_name: str
    before: Optional[GPUMemoryDeviceSnapshot]
    after: Optional[GPUMemoryDeviceSnapshot]

    delta_allocated_bytes: Optional[int]
    delta_reserved_bytes: Optional[int]
    delta_max_allocated_bytes: Optional[int]
    delta_max_reserved_bytes: Optional[int]
    delta_active_bytes: Optional[int]
    delta_inactive_split_bytes: Optional[int]
    delta_num_alloc_retries: Optional[int]
    delta_num_ooms: Optional[int]


@dataclass(frozen=True)
class GPUMemoryDiff:
    before: GPUMemorySnapshot
    after: GPUMemorySnapshot
    dt_s: float
    devices: Tuple[GPUMemoryDeviceDiff, ...]

    def format(self) -> str:
        return format_gpu_memory_diff(self)

    def log(self, printer: Callable[[str], None] = print) -> None:
        printer(self.format())


def _default_devices() -> List[int]:
    if torch is None:
        return []
    if not torch.cuda.is_available():
        return []
    try:
        return list(range(int(torch.cuda.device_count())))
    except Exception:
        return []


def _maybe_synchronize(devices: Sequence[int], synchronize: bool) -> None:
    if not synchronize:
        return
    if torch is None:
        return
    if not torch.cuda.is_available():
        return
    for d in devices:
        try:
            torch.cuda.synchronize(d)
        except Exception:
            pass


def take_gpu_memory_snapshot(
    tag: str = "",
    *,
    devices: Optional[Sequence[int]] = None,
    synchronize: bool = True,
    reset_peak: bool = True,
    include_allocator_stats: bool = True,
    include_cuda_mem_info: bool = True,
) -> GPUMemorySnapshot:
    """
    - devices: 默认所有可见 GPU（0..device_count-1）
    - synchronize: 采样前是否对每张卡 synchronize（更准但更慢）
    - reset_peak: 采样前是否 reset peak stats（用于区间 peak）
    - include_allocator_stats: 是否读取 torch.cuda.memory_stats（包含 active/inactive/oom/retry 等）
    - include_cuda_mem_info: 是否读取 torch.cuda.mem_get_info（driver 侧 free/total）
    """
    host = socket.gethostname()
    pid = os.getpid()
    rank, world_size, local_rank = _get_dist_info()

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    torch_cuda_version = ""
    torch_version = ""
    if torch is not None:
        try:
            torch_cuda_version = str(getattr(torch.version, "cuda", "") or "")
        except Exception:
            torch_cuda_version = ""
        try:
            torch_version = str(getattr(torch, "__version__", "") or "")
        except Exception:
            torch_version = ""

    if not tag:
        tag = "snapshot"

    if torch is None or not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        return GPUMemorySnapshot(
            tag=tag,
            time_s=_now_ts(),
            host=host,
            pid=pid,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            cuda_visible_devices=cuda_visible_devices,
            torch_cuda_version=torch_cuda_version,
            torch_version=torch_version,
            devices=tuple(),
        )

    if devices is None:
        devices = _default_devices()
    devices = list(devices)

    _maybe_synchronize(devices, synchronize=synchronize)

    if reset_peak:
        for d in devices:
            try:
                torch.cuda.reset_peak_memory_stats(d)
            except Exception:
                pass

    snapshots: List[GPUMemoryDeviceSnapshot] = []
    for d in devices:
        device_name = f"cuda:{d}"
        try:
            try:
                device_name = str(torch.cuda.get_device_name(d))
            except Exception:
                device_name = f"cuda:{d}"

            allocated_bytes = _safe_int(torch.cuda.memory_allocated(d))
            reserved_bytes = _safe_int(torch.cuda.memory_reserved(d))
            max_allocated_bytes = _safe_int(torch.cuda.max_memory_allocated(d))
            max_reserved_bytes = _safe_int(torch.cuda.max_memory_reserved(d))

            active_bytes = None
            inactive_split_bytes = None
            num_alloc_retries = None
            num_ooms = None
            if include_allocator_stats:
                try:
                    stats = torch.cuda.memory_stats(d)
                    active_bytes = _safe_int(stats.get("active_bytes.all.current"))
                    inactive_split_bytes = _safe_int(stats.get("inactive_split_bytes.all.current"))
                    num_alloc_retries = _safe_int(stats.get("num_alloc_retries"))
                    num_ooms = _safe_int(stats.get("num_ooms"))
                except Exception:
                    pass

            cuda_free_bytes = None
            cuda_total_bytes = None
            if include_cuda_mem_info:
                try:
                    free_b, total_b = torch.cuda.mem_get_info(d)
                    cuda_free_bytes = _safe_int(free_b)
                    cuda_total_bytes = _safe_int(total_b)
                except Exception:
                    pass

            device_total_bytes = None
            try:
                props = torch.cuda.get_device_properties(d)
                device_total_bytes = _safe_int(getattr(props, "total_memory", None))
            except Exception:
                pass

            snapshots.append(
                GPUMemoryDeviceSnapshot(
                    device_index=int(d),
                    device_name=device_name,
                    allocated_bytes=allocated_bytes,
                    reserved_bytes=reserved_bytes,
                    max_allocated_bytes=max_allocated_bytes,
                    max_reserved_bytes=max_reserved_bytes,
                    active_bytes=active_bytes,
                    inactive_split_bytes=inactive_split_bytes,
                    num_alloc_retries=num_alloc_retries,
                    num_ooms=num_ooms,
                    cuda_free_bytes=cuda_free_bytes,
                    cuda_total_bytes=cuda_total_bytes,
                    device_total_bytes=device_total_bytes,
                    error=None,
                )
            )
        except Exception as e:
            snapshots.append(
                GPUMemoryDeviceSnapshot(
                    device_index=int(d),
                    device_name=device_name,
                    allocated_bytes=None,
                    reserved_bytes=None,
                    max_allocated_bytes=None,
                    max_reserved_bytes=None,
                    active_bytes=None,
                    inactive_split_bytes=None,
                    num_alloc_retries=None,
                    num_ooms=None,
                    cuda_free_bytes=None,
                    cuda_total_bytes=None,
                    device_total_bytes=None,
                    error=str(e),
                )
            )

    return GPUMemorySnapshot(
        tag=tag,
        time_s=_now_ts(),
        host=host,
        pid=pid,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        cuda_visible_devices=cuda_visible_devices,
        torch_cuda_version=torch_cuda_version,
        torch_version=torch_version,
        devices=tuple(sorted(snapshots, key=lambda x: x.device_index)),
    )


def diff_gpu_memory_snapshots(before: GPUMemorySnapshot, after: GPUMemorySnapshot) -> GPUMemoryDiff:
    before_map: Dict[int, GPUMemoryDeviceSnapshot] = {d.device_index: d for d in before.devices}
    after_map: Dict[int, GPUMemoryDeviceSnapshot] = {d.device_index: d for d in after.devices}

    all_dev_ids = sorted(set(before_map.keys()) | set(after_map.keys()))

    diffs: List[GPUMemoryDeviceDiff] = []
    for dev_id in all_dev_ids:
        b = before_map.get(dev_id)
        a = after_map.get(dev_id)

        def delta(getter: str) -> Optional[int]:
            vb = getattr(b, getter) if b is not None else None
            va = getattr(a, getter) if a is not None else None
            if vb is None or va is None:
                return None
            try:
                return int(va) - int(vb)
            except Exception:
                return None

        name = (a.device_name if a is not None else None) or (b.device_name if b is not None else None) or f"cuda:{dev_id}"

        diffs.append(
            GPUMemoryDeviceDiff(
                device_index=dev_id,
                device_name=name,
                before=b,
                after=a,
                delta_allocated_bytes=delta("allocated_bytes"),
                delta_reserved_bytes=delta("reserved_bytes"),
                delta_max_allocated_bytes=delta("max_allocated_bytes"),
                delta_max_reserved_bytes=delta("max_reserved_bytes"),
                delta_active_bytes=delta("active_bytes"),
                delta_inactive_split_bytes=delta("inactive_split_bytes"),
                delta_num_alloc_retries=delta("num_alloc_retries"),
                delta_num_ooms=delta("num_ooms"),
            )
        )

    dt_s = float(after.time_s) - float(before.time_s)
    return GPUMemoryDiff(before=before, after=after, dt_s=dt_s, devices=tuple(diffs))


def _make_table(rows: List[List[str]], headers: List[str], aligns: Optional[List[str]] = None) -> List[str]:
    if aligns is None:
        aligns = ["l"] * len(headers)
    assert len(aligns) == len(headers)

    all_rows = [headers] + rows
    col_widths = [0] * len(headers)
    for r in all_rows:
        for i, cell in enumerate(r):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(r: List[str]) -> str:
        out: List[str] = []
        for i, cell in enumerate(r):
            if aligns[i] == "r":
                out.append(cell.rjust(col_widths[i]))
            else:
                out.append(cell.ljust(col_widths[i]))
        return "  ".join(out)

    lines = [fmt_row(headers), fmt_row(["-" * w for w in col_widths])]
    lines.extend(fmt_row(r) for r in rows)
    return lines


def format_gpu_memory_snapshot(s: GPUMemorySnapshot) -> str:
    prefix = "[GPUMem]"
    meta = [
        f"tag={s.tag}",
        f"time={_fmt_time(s.time_s)}",
        f"pid={s.pid}",
    ]
    if s.rank is not None:
        meta.append(f"rank={s.rank}/{s.world_size if s.world_size is not None else '?'}")
    if s.local_rank is not None:
        meta.append(f"local_rank={s.local_rank}")
    if s.cuda_visible_devices != "":
        meta.append(f"CUDA_VISIBLE_DEVICES={s.cuda_visible_devices}")
    if s.torch_version:
        meta.append(f"torch={s.torch_version}")
    if s.torch_cuda_version:
        meta.append(f"cuda={s.torch_cuda_version}")

    lines: List[str] = [f"{prefix} snapshot " + " ".join(meta)]

    if len(s.devices) == 0:
        lines.append(f"{prefix} (no cuda devices or torch.cuda unavailable)")
        return "\n".join(lines)

    has_active = any(d.active_bytes is not None for d in s.devices)
    has_inactive = any(d.inactive_split_bytes is not None for d in s.devices)
    has_retries = any(d.num_alloc_retries is not None for d in s.devices)
    has_ooms = any(d.num_ooms is not None for d in s.devices)
    has_cuda_mem = any(d.cuda_total_bytes is not None for d in s.devices)

    headers = ["dev", "name", "alloc", "reserv", "peak_alloc", "peak_reserv"]
    aligns = ["r", "l", "r", "r", "r", "r"]
    if has_active:
        headers.append("active")
        aligns.append("r")
    if has_inactive:
        headers.append("inact_split")
        aligns.append("r")
    if has_retries:
        headers.append("retries")
        aligns.append("r")
    if has_ooms:
        headers.append("ooms")
        aligns.append("r")
    if has_cuda_mem:
        headers.extend(["free", "total", "alloc%", "reserv%"])
        aligns.extend(["r", "r", "r", "r"])

    rows: List[List[str]] = []
    sum_alloc = 0
    sum_reserv = 0
    sum_peak_alloc = 0
    sum_peak_reserv = 0
    sum_active = 0
    sum_inactive = 0
    sum_free = 0
    sum_total = 0

    for d in s.devices:
        if d.error:
            rows.append([str(d.device_index), d.device_name, "ERR", "ERR", "ERR", "ERR"])
            continue

        alloc_b = d.allocated_bytes
        reserv_b = d.reserved_bytes
        peak_alloc_b = d.max_allocated_bytes
        peak_reserv_b = d.max_reserved_bytes

        sum_alloc += int(alloc_b or 0)
        sum_reserv += int(reserv_b or 0)
        sum_peak_alloc += int(peak_alloc_b or 0)
        sum_peak_reserv += int(peak_reserv_b or 0)

        row = [
            str(d.device_index),
            d.device_name,
            human_bytes(alloc_b),
            human_bytes(reserv_b),
            human_bytes(peak_alloc_b),
            human_bytes(peak_reserv_b),
        ]

        if has_active:
            row.append(human_bytes(d.active_bytes))
            sum_active += int(d.active_bytes or 0)
        if has_inactive:
            row.append(human_bytes(d.inactive_split_bytes))
            sum_inactive += int(d.inactive_split_bytes or 0)
        if has_retries:
            row.append(str(d.num_alloc_retries if d.num_alloc_retries is not None else "-"))
        if has_ooms:
            row.append(str(d.num_ooms if d.num_ooms is not None else "-"))

        if has_cuda_mem:
            free_b = d.cuda_free_bytes
            total_b = d.cuda_total_bytes or d.device_total_bytes
            row.extend(
                [
                    human_bytes(free_b),
                    human_bytes(total_b),
                    _pct(alloc_b, total_b),
                    _pct(reserv_b, total_b),
                ]
            )
            sum_free += int(free_b or 0)
            sum_total += int(total_b or 0)

        rows.append(row)

    total_row = [
        "ALL",
        f"{len(s.devices)} gpus",
        human_bytes(sum_alloc),
        human_bytes(sum_reserv),
        human_bytes(sum_peak_alloc),
        human_bytes(sum_peak_reserv),
    ]
    if has_active:
        total_row.append(human_bytes(sum_active))
    if has_inactive:
        total_row.append(human_bytes(sum_inactive))
    if has_retries:
        total_row.append("-")
    if has_ooms:
        total_row.append("-")
    if has_cuda_mem:
        total_row.extend([human_bytes(sum_free), human_bytes(sum_total), _pct(sum_alloc, sum_total), _pct(sum_reserv, sum_total)])
    rows.append(total_row)

    table_lines = _make_table(rows, headers, aligns=aligns)
    lines.extend(f"{prefix} {ln}" for ln in table_lines)
    return "\n".join(lines)


def format_gpu_memory_diff(d: GPUMemoryDiff) -> str:
    prefix = "[GPUMem]"
    lines: List[str] = [
        f"{prefix} diff tag={d.before.tag}->{d.after.tag} dt={d.dt_s:.3f}s pid={d.after.pid}"
    ]

    if len(d.devices) == 0:
        lines.append(f"{prefix} (no devices)")
        return "\n".join(lines)

    has_active = any(x.delta_active_bytes is not None for x in d.devices)
    has_inactive = any(x.delta_inactive_split_bytes is not None for x in d.devices)
    has_retries = any(x.delta_num_alloc_retries is not None for x in d.devices)
    has_ooms = any(x.delta_num_ooms is not None for x in d.devices)

    headers = ["dev", "name", "alloc", "Δalloc", "reserv", "Δreserv", "peak_alloc", "Δpeak_alloc", "peak_reserv", "Δpeak_reserv"]
    aligns = ["r", "l", "r", "r", "r", "r", "r", "r", "r", "r"]

    if has_active:
        headers.extend(["active", "Δactive"])
        aligns.extend(["r", "r"])
    if has_inactive:
        headers.extend(["inact_split", "Δinact_split"])
        aligns.extend(["r", "r"])
    if has_retries:
        headers.extend(["retries", "Δretries"])
        aligns.extend(["r", "r"])
    if has_ooms:
        headers.extend(["ooms", "Δooms"])
        aligns.extend(["r", "r"])

    rows: List[List[str]] = []
    sum_da = 0
    sum_dr = 0
    sum_dpa = 0
    sum_dpr = 0
    sum_dact = 0
    sum_dinact = 0

    for x in d.devices:
        a = x.after
        b = x.before
        if (a is not None and a.error) or (b is not None and b.error):
            rows.append([str(x.device_index), x.device_name, "ERR", "-", "ERR", "-", "ERR", "-", "ERR", "-"])
            continue

        alloc = a.allocated_bytes if a is not None else None
        reserv = a.reserved_bytes if a is not None else None
        peak_alloc = a.max_allocated_bytes if a is not None else None
        peak_reserv = a.max_reserved_bytes if a is not None else None

        row = [
            str(x.device_index),
            x.device_name,
            human_bytes(alloc),
            human_bytes(x.delta_allocated_bytes, signed=True),
            human_bytes(reserv),
            human_bytes(x.delta_reserved_bytes, signed=True),
            human_bytes(peak_alloc),
            human_bytes(x.delta_max_allocated_bytes, signed=True),
            human_bytes(peak_reserv),
            human_bytes(x.delta_max_reserved_bytes, signed=True),
        ]

        if has_active:
            active = a.active_bytes if a is not None else None
            row.extend([human_bytes(active), human_bytes(x.delta_active_bytes, signed=True)])
        if has_inactive:
            inact = a.inactive_split_bytes if a is not None else None
            row.extend([human_bytes(inact), human_bytes(x.delta_inactive_split_bytes, signed=True)])
        if has_retries:
            retries = a.num_alloc_retries if a is not None else None
            row.extend(
                [
                    str(retries if retries is not None else "-"),
                    str(x.delta_num_alloc_retries if x.delta_num_alloc_retries is not None else "-"),
                ]
            )
        if has_ooms:
            ooms = a.num_ooms if a is not None else None
            row.extend([str(ooms if ooms is not None else "-"), str(x.delta_num_ooms if x.delta_num_ooms is not None else "-")])

        rows.append(row)

        sum_da += int(x.delta_allocated_bytes or 0)
        sum_dr += int(x.delta_reserved_bytes or 0)
        sum_dpa += int(x.delta_max_allocated_bytes or 0)
        sum_dpr += int(x.delta_max_reserved_bytes or 0)
        sum_dact += int(x.delta_active_bytes or 0)
        sum_dinact += int(x.delta_inactive_split_bytes or 0)

    total_row = [
        "ALL",
        f"{len(d.devices)} gpus",
        "-",
        human_bytes(sum_da, signed=True),
        "-",
        human_bytes(sum_dr, signed=True),
        "-",
        human_bytes(sum_dpa, signed=True),
        "-",
        human_bytes(sum_dpr, signed=True),
    ]
    if has_active:
        total_row.extend(["-", human_bytes(sum_dact, signed=True)])
    if has_inactive:
        total_row.extend(["-", human_bytes(sum_dinact, signed=True)])
    if has_retries:
        total_row.extend(["-", "-"])
    if has_ooms:
        total_row.extend(["-", "-"])
    rows.append(total_row)

    table_lines = _make_table(rows, headers, aligns=aligns)
    lines.extend(f"{prefix} {ln}" for ln in table_lines)
    return "\n".join(lines)


class GPUMemoryMonitor:
    """
    用法（最常见）：
        with GPUMemoryMonitor("forward", reset_peak_on_enter=True).log_on_exit():
            ...
    或者手动：
        mon = GPUMemoryMonitor("step")
        mon.start()
        ...
        mon.stop_and_log()
    """

    def __init__(
        self,
        tag: str,
        *,
        devices: Optional[Sequence[int]] = None,
        enabled: bool = True,
        synchronize: bool = False,
        reset_peak_on_enter: bool = False,
        include_allocator_stats: bool = False,
        include_cuda_mem_info: bool = True,
        printer: Callable[[str], None] = print,
    ):
        self.tag = tag
        self.devices = devices
        self.enabled = enabled
        self.synchronize = synchronize
        self.reset_peak_on_enter = reset_peak_on_enter
        self.include_allocator_stats = include_allocator_stats
        self.include_cuda_mem_info = include_cuda_mem_info
        self.printer = printer

        self.before: Optional[GPUMemorySnapshot] = None
        self.after: Optional[GPUMemorySnapshot] = None

    def start(self) -> GPUMemorySnapshot:
        self.before = take_gpu_memory_snapshot(
            tag=f"{self.tag}:before",
            devices=self.devices,
            synchronize=self.synchronize,
            reset_peak=self.reset_peak_on_enter,
            include_allocator_stats=self.include_allocator_stats,
            include_cuda_mem_info=self.include_cuda_mem_info,
        )
        return self.before

    def stop(self) -> GPUMemorySnapshot:
        self.after = take_gpu_memory_snapshot(
            tag=f"{self.tag}:after",
            devices=self.devices,
            synchronize=self.synchronize,
            reset_peak=False,
            include_allocator_stats=self.include_allocator_stats,
            include_cuda_mem_info=self.include_cuda_mem_info,
        )
        return self.after

    def diff(self) -> Optional[GPUMemoryDiff]:
        if self.before is None or self.after is None:
            return None
        return diff_gpu_memory_snapshots(self.before, self.after)

    def stop_and_log(self) -> Optional[GPUMemoryDiff]:
        if not self.enabled:
            return None
        self.stop()
        d = self.diff()
        if d is not None:
            d.log(self.printer)
        return d

    def __enter__(self) -> "GPUMemoryMonitor":
        if self.enabled:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.enabled:
            self.stop_and_log()

    def log_on_exit(self) -> "GPUMemoryMonitor":
        return self