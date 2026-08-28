import time
from contextlib import contextmanager
from collections import defaultdict
from typing import Optional

import torch

class CudaWallProfiler:
    """
    轻量级分阶段 profiler：
    - wall_ms: Python 端墙钟时间（包含同步等待）
    - cuda_ms: CUDA event 时间（仅 GPU 且当前 stream 上的 kernel 时间）
    注意：为保证分段准确，profile 启用时每段会做 event synchronize，会比正常推理更慢。
    """
    def __init__(self, enabled: bool, device: Optional[torch.device] = None):
        self.enabled = bool(enabled)
        self.device = device
        self.is_cuda = bool(
            self.enabled and isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available()
        )
        self.records = []  # raw records: [{"phase":..., "wall_ms":..., "cuda_ms":...}, ...]

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return

        t0 = time.perf_counter()
        start_evt = end_evt = None
        cuda_ms = None

        if self.is_cuda:
            # 用当前 device/current stream 的 event 计时
            with torch.cuda.device(self.device):
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()

        try:
            yield
        finally:
            if self.is_cuda:
                with torch.cuda.device(self.device):
                    end_evt.record()
                    end_evt.synchronize()  # 保证该 section 的 CUDA 时间可读
                    cuda_ms = float(start_evt.elapsed_time(end_evt))

            wall_ms = float((time.perf_counter() - t0) * 1000.0)
            self.records.append({
                "phase": name,
                "wall_ms": wall_ms,
                "cuda_ms": cuda_ms,
            })

    def report(self, extra: Optional[dict] = None) -> dict:
        if not self.enabled:
            return {"enabled": False}

        total_wall = sum(r["wall_ms"] for r in self.records)
        total_cuda = sum((r["cuda_ms"] or 0.0) for r in self.records) if self.is_cuda else None

        # 聚合同名 phase（比如 pack_debug_cpu_copy 会在 batch 循环里出现多次）
        agg = defaultdict(lambda: {"count": 0, "wall_ms": 0.0, "cuda_ms": 0.0, "has_cuda": False})
        for r in self.records:
            a = agg[r["phase"]]
            a["count"] += 1
            a["wall_ms"] += float(r["wall_ms"])
            if r["cuda_ms"] is not None:
                a["cuda_ms"] += float(r["cuda_ms"])
                a["has_cuda"] = True

        phases_agg = []
        for k, v in agg.items():
            item = {
                "phase": k,
                "count": int(v["count"]),
                "wall_ms": float(v["wall_ms"]),
                "wall_pct": float(v["wall_ms"] / max(total_wall, 1e-12) * 100.0),
            }
            if v["has_cuda"]:
                item["cuda_ms"] = float(v["cuda_ms"])
                if total_cuda is not None and total_cuda > 0:
                    item["cuda_pct"] = float(v["cuda_ms"] / total_cuda * 100.0)
            phases_agg.append(item)

        phases_agg = sorted(phases_agg, key=lambda x: x["wall_ms"], reverse=True)

        out = {
            "enabled": True,
            "device": str(self.device) if self.device is not None else "unknown",
            "num_records": len(self.records),
            "total_wall_ms": float(total_wall),
            "total_cuda_ms": (float(total_cuda) if total_cuda is not None else None),
            "phases_agg_sorted_by_wall": phases_agg,
            "phases_raw": self.records,  # 如嫌太长可删掉
        }
        if extra:
            out.update(extra)
        return out

