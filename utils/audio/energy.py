import torch
import math

def short_time_energy(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    use_log_db: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    可微分的短时能量计算（Short-Time Energy, STE）

    参数
    ----
    waveform : torch.Tensor
        - 形状可以是 [T] 或 [B, T]，值可为 float32/float64
        - 允许 requires_grad=True，用于端到端训练
    sample_rate : int
        采样率（Hz），用于把毫秒转换为采样点数
    frame_ms : float
        帧长（毫秒），典型值 20~25 ms（语音）
    hop_ms : float
        帧移（毫秒），典型值 5~10 ms
    use_log_db : bool
        是否输出对数能量（dB）。True 时：返回 10 * log10(E + eps)
    eps : float
        防止 log(0) 的小常数

    返回
    ----
    energies : torch.Tensor
        - 若输入为 [T]，输出形状为 [num_frames]
        - 若输入为 [B, T]，输出形状为 [B, num_frames]
        - 每个元素是对应帧的能量（或 dB 能量）
    """

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [T] -> [1, T]
    elif waveform.dim() != 2:
        raise ValueError("`waveform` 必须是 [T] 或 [B, T] 形状的张量")

    B, T = waveform.shape
    device = waveform.device
    dtype = waveform.dtype

    # 1. 长度（采样点）
    frame_len = int(round(frame_ms * sample_rate / 1000.0))
    hop_len = int(round(hop_ms * sample_rate / 1000.0))
    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame_ms / hop_ms 太小导致帧长或帧移为 0")

    # 不做 padding：不足一帧的尾部直接丢弃
    if T < frame_len:
        # 没有完整的帧
        return waveform.new_zeros((B, 0))

    # 2. 使用 `unfold` 做分帧（是 view 操作，完全可微）
    #   waveform: [B, T] -> frames: [B, num_frames, frame_len]
    num_frames = 1 + (T - frame_len) // hop_len
    frames = waveform.unfold(dimension=1, size=frame_len, step=hop_len)  # [B, num_frames, frame_len]

    # 3. 加窗（Hamming 窗）（窗本身不需要梯度）
    window = torch.hamming_window(frame_len, periodic=False, dtype=dtype, device=device)
    frames = frames * window  # 广播到 [B, num_frames, frame_len]

    # 4. 短时能量：sum(x^2)
    energies = (frames ** 2).sum(dim=-1)  # [B, num_frames]

    # 5. 可选：转 dB
    if use_log_db:
        energies = 10.0 * torch.log10(energies + eps)

    # 如果原来是 [T]，就 squeeze 掉 batch 维
    if energies.shape[0] == 1:
        energies = energies.squeeze(0)

    return energies
