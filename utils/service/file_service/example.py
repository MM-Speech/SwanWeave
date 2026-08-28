from typing import Any, Dict

from utils.service.file_service import BaseProcessor


class ExampleProcessor(BaseProcessor):
    def setup(self) -> None:
        # 这里放模型加载/初始化（例如 import torch 后加载 ckpt）
        return

    def process(self, job: Dict[str, Any]) -> Dict[str, Any]:
        payload = job.get("payload", {})
        job_id = job.get("job_id")

        # 这里写你的实际逻辑：reward/对齐/ASR/特征抽取/质检/缓存构建……
        return {
            "echo_job_id": job_id,
            "echo_payload": payload,
            "device": self.device,
        }
