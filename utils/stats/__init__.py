import numpy as np
import matplotlib.pyplot as plt

def plot_hist_and_stats(
    data,
    bins=30,
    title=None,
    fig_path=None,
    xscale: str = "linear",
    yscale: str = "linear",
    log_bins: bool = False,
):
    """
    data: 可迭代的数值数据（list, tuple, numpy array 等）
    bins: 直方图的柱子数量（int）或 bin edges（array-like）
    title: 图标题
    fig_path: 保存路径（可选）
    xscale: "linear" 或 "log"
    yscale: "linear" 或 "log"
    log_bins: 是否使用对数间隔的 bins（仅当数据全为正时可用）
    """

    # 转成 numpy array，过滤掉不能转成数字的、NaN、Inf
    arr = np.array(data, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        print("数据为空或无法转换为数值。")
        return {}

    # 计算常用统计量
    stats = {
        "count": int(arr.size),                    # 样本数
        "min": float(np.min(arr)),                 # 最小值
        "max": float(np.max(arr)),                 # 最大值
        "mean": float(np.mean(arr)),               # 平均值
        "median": float(np.median(arr)),           # 中位数
        "std": float(np.std(arr, ddof=1)),         # 样本标准差
        "q25": float(np.percentile(arr, 25)),      # 25% 分位数
        "q75": float(np.percentile(arr, 75)),      # 75% 分位数
    }

    # 画直方图
    plt.figure(figsize=(6, 4))

    # log x / log bins 要求数据为正
    need_positive = (xscale == "log") or log_bins
    if need_positive:
        arr = arr[arr > 0]
        if arr.size == 0:
            print("log xscale/log bins 需要数据全为正，但过滤后为空。")
            return {}

    hist_bins = bins
    if log_bins:
        if isinstance(bins, int):
            lo = float(np.min(arr))
            hi = float(np.max(arr))
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= 0 or lo == hi:
                print("无法为 log_bins 构造有效区间。")
                return {}
            hist_bins = np.logspace(np.log10(lo), np.log10(hi), bins + 1)

    plt.hist(arr, bins=hist_bins, edgecolor='black', alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title or "Histogram")
    plt.grid(alpha=0.3)

    if xscale in ("linear", "log"):
        plt.xscale(xscale)
    else:
        raise ValueError(f"Unsupported xscale: {xscale}")

    if yscale in ("linear", "log"):
        plt.yscale(yscale)
    else:
        raise ValueError(f"Unsupported yscale: {yscale}")

    if fig_path is not None:
        import os
        from pathlib import Path
        os.makedirs(Path(fig_path).parent, exist_ok=True)
        plt.savefig(fig_path)
        plt.close()

    # 打印统计结果
    print(f"{title or ''} 统计结果：")
    for k, v in stats.items():
        print(f"{k:>6} : {v:.4f}")

    return stats
