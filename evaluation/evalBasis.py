from typing import Any, Dict, List, Optional

import numpy as np


class EvalBasis:
    """
    评价指标基类（整段评估，不分窗）。

    约定 data 格式：
      data = {
        "audio": [audio0, audio1, ...],  # list[np.ndarray]
        "rate":  24000                   # int
      }

    约定音频 shape：
      - (T,) 或 (T, C)，并且时间维在 axis=0
    """

    def __init__(self, name: Optional[str] = None):
        # 指标计算时使用的采样率；None 表示跟随输入 data['rate']
        self.score_rate: Optional[int] = None

        # intrusive=True 表示需要 reference（通常至少两路音频）
        self.intrusive: bool = True

        self.name: str = name or self.__class__.__name__
        self.model = None
        self.device: str = "cpu"

    # -------------------------
    # 内部函数（子类必须实现）
    # -------------------------
    def _scoring(self, audios: List[np.ndarray], score_rate: int) -> Any:
        """
        子类实现：输入为（必要时已重采样后的）整段 audios，返回分数/字典等结果。
        """
        raise NotImplementedError(f"In {self.name}, _scoring is not yet implemented")

    # -------------------------
    # 外部接口（统一入口）
    # -------------------------
    def scoring(self, data: Dict[str, Any], score_rate: Optional[int] = None) -> Any:
        """
        外部调用接口：不分窗；必要时重采样；然后调用 _scoring()。

        采样率优先级：
          self.score_rate  >  scoring(..., score_rate=...)  >  data['rate']
        """
        # 基本校验
        if "audio" not in data or "rate" not in data:
            raise ValueError('data 必须包含 key: "audio" 和 "rate"')

        audios_in = data["audio"]
        in_rate = int(data["rate"])

        if not isinstance(audios_in, (list, tuple)):
            raise ValueError('data["audio"] 必须是 list/tuple，形如 [ref, test]')

        # 决定目标采样率
        if self.score_rate is not None:
            target_rate = int(self.score_rate)
        elif score_rate is not None:
            target_rate = int(score_rate)
        else:
            target_rate = in_rate

        # intrusive 指标检查
        audios = list(audios_in)  # 浅拷贝，避免改原 list
        if self.intrusive and len(audios) < 2:
            raise ValueError(f"{self.name} 是 intrusive 指标，至少需要 2 路音频（ref/test）")

        # 必要时重采样
        if target_rate != in_rate:
            import resampy  # 延迟导入，避免不需要时引入依赖
            for i, a in enumerate(audios):
                a = np.asarray(a)
                audios[i] = resampy.resample(a, in_rate, target_rate, axis=0)

        return self._scoring(audios, target_rate)
