from typing import List, Optional, Sequence
import math
import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.core.convert import hz_to_mel
from scipy.io.wavfile import read
import torch.nn as nn

MAX_WAV_VALUE = 32768.0

def get_mel_len(wav_len, hop_size):
    return (wav_len + hop_size - 1) // hop_size


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log10(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.power(10.0, x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log10(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.pow(10.0, x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


class MelNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.n_fft = hparams['fft_size']
        self.num_mels = hparams['audio_num_mel_bins']
        self.sampling_rate = hparams['audio_sample_rate']
        self.hop_size = hparams['hop_size']
        self.win_size = hparams['win_size']
        self.fmin = hparams['fmin']
        self.fmax = hparams['fmax']

        mel = librosa_mel_fn(
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.num_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        mel = torch.from_numpy(mel).float()  # [num_mels, n_fft//2 + 1]

        self.register_buffer("mel_basis", mel)
        self.register_buffer("hann_window", torch.hann_window(self.win_size))

    def forward(self, y, center=False, return_complex=False):
        """
        y: Tensor 或 np.ndarray，形状 [B, T] 或 [T]
        返回:
          - return_complex=False: log-mel, 形状 [B, T_mel, num_mels]
          - return_complex=True : STFT 复数谱，形状 [B, T_spec, F] (complex tensor)
        """
        if isinstance(y, np.ndarray):
            y = torch.as_tensor(y, dtype=torch.float32)
        if y.dim() == 1:
            y = y.unsqueeze(0)  # [1, T]
        y = y.clamp(min=-1.0, max=1.0)
        device = self.mel_basis.device
        if y.device != device:
            y = y.to(device)

        pad_length = math.ceil(y.shape[1] / self.hop_size) * self.hop_size - y.shape[1]
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            [int((self.n_fft - self.hop_size) / 2),
             int((self.n_fft - self.hop_size) / 2 + pad_length)],
            mode='reflect'
        ).squeeze(1)    # [B, T']

        spec = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=center,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )

        if return_complex:
            spec = spec.transpose(1, 2)  # [B, T_spec, F]
            return spec

        mag = torch.abs(spec) + 1e-9       # [B, F, T_spec]
        mel = torch.matmul(self.mel_basis, mag) # [num_mels, F] @ [B, F, T] -> [B, num_mels, T]
        mel = spectral_normalize_torch(mel)   # log10(mel)
        mel = mel.transpose(1, 2)

        return mel


class MultiResolutionMelLoss(nn.Module):
    def __init__(self, hparams, loss_fn: nn.Module = nn.L1Loss()):
        super().__init__()
        self.mel_nets = nn.ModuleList()
        for hparams_ in hparams:
            self.mel_nets.append(MelNet(hparams_))
        self.loss_fn = loss_fn

    def forward(self, y_pred, y_ref):
        loss = 0.0
        for mel_net in self.mel_nets:
            mel_pred = mel_net(y_pred)
            mel_ref = mel_net(y_ref)
            loss += self.loss_fn(mel_pred, mel_ref)
        loss /= len(self.mel_nets)
        return loss


class MultiResolutionMultiBandMelLoss(nn.Module):
    """
    Multi-Resolution + Hz 边界 Multi-Band Mel Loss

    - 与原来的 MultiResolutionMelLoss 类似，也是传入一组 hparams_list，每个 hparams 对应一个 MelNet
    - 额外传入 band_edges_hz：按 Hz 边界切分频带，例如 [0, 3000, 8000, 12000]
    - 内部会对每个 MelNet，根据其 fmin/fmax/num_mels，把 Hz 边界映射为 mel bin index 进行切片
    """

    def __init__(
        self,
        hparams_list: List[dict],
        band_edges_hz: Sequence[float],
        band_weights: Optional[Sequence[float]] = None,
        loss_fn: nn.Module = nn.L1Loss()
    ):
        """
        Args:
            hparams_list:  每个元素是一个 dict，传给 MelNet 的参数（和你现有的 MultiResolutionMelLoss 一样）
            band_edges_hz: Hz 边界列表，长度为 N+1，对应 N 个 band。
                           例如 [0, 3000, 8000, 12000] → 3 个 band：
                           [0,3000], [3000,8000], [8000,12000]
            band_weights : 每个 band 的权重，长度为 N；如果为 None，则默认为全 1。
            loss_fn      : 基础回归损失（L1/L2），默认 L1。
        """
        super().__init__()

        assert len(band_edges_hz) >= 2, "band_edges_hz 至少需要两个元素（一个 band 的左右边界）"
        self.mel_nets = nn.ModuleList([MelNet(hp) for hp in hparams_list])
        self.loss_fn = loss_fn

        # 频带数量 = len(edges) - 1
        self.num_bands = len(band_edges_hz) - 1

        # band 权重
        if band_weights is None:
            band_weights = [1.0] * self.num_bands
        assert len(band_weights) == self.num_bands, "band_weights 长度必须等于 len(band_edges_hz) - 1"
        self.band_weights = [float(w) for w in band_weights]

        # 保留原始 Hz 边界（排序 & 去重防御）
        band_edges_hz = list(band_edges_hz)
        band_edges_hz = sorted(band_edges_hz)
        self.band_edges_hz = band_edges_hz

        # 为每个 MelNet 预计算：对应的 mel-bin 切片 (start, end)
        # 注意：不同 MelNet 的 num_mels / fmin / fmax 不同，因此要分别计算
        self.band_slices_per_melnet = []
        for mel_net in self.mel_nets:
            band_slices = self._compute_band_slices_for_melnet(mel_net, band_edges_hz)
            self.band_slices_per_melnet.append(band_slices)

    def _compute_band_slices_for_melnet(
        self,
        mel_net: nn.Module,
        band_edges_hz: Sequence[float],
    ):
        """
        对单个 MelNet，根据 fmin/fmax/num_mels，将 Hz 边界映射为 mel bin slice (start, end)，
        返回 [(start, end), ...]，长度 = num_bands

        注意：
        - 使用 [start, end) 作为切片区间
        - start 使用 floor，end 使用 ceil，允许 end == num_mels，这样最后一个 band 可以覆盖到最后一个 bin
        """
        fmin = float(mel_net.fmin)
        fmax = float(mel_net.fmax)
        num_mels = int(mel_net.num_mels)

        # 限制 Hz 边界在 [fmin, fmax] 内
        def clamp_hz(f):
            return max(fmin, min(fmax, f))

        # fmin / fmax 在 mel 标度上的位置
        mel_fmin = hz_to_mel(fmin)
        mel_fmax = hz_to_mel(fmax)
        mel_range = max(mel_fmax - mel_fmin, 1e-9)  # 防止除零

        # Hz -> 归一化位置 t in [0, 1]
        def hz_to_t(f_hz: float) -> float:
            f_clamped = clamp_hz(f_hz)
            m = hz_to_mel(f_clamped)
            t = (m - mel_fmin) / mel_range
            # 数值防御
            return max(0.0, min(1.0, t))

        band_slices = []
        for i in range(len(band_edges_hz) - 1):
            f_low = band_edges_hz[i]
            f_high = band_edges_hz[i + 1]

            t_low = hz_to_t(f_low)
            t_high = hz_to_t(f_high)

            # start: floor 到 [0, num_mels-1]
            start = int(math.floor(t_low * num_mels))
            start = max(0, min(start, num_mels - 1))

            # end: ceil 到 [1, num_mels]
            end = int(math.ceil(t_high * num_mels))
            end = max(1, min(end, num_mels))

            # 确保至少包含一个 bin
            if end <= start:
                end = min(start + 1, num_mels)

            band_slices.append((start, end))

        return band_slices

    def forward(self, y_pred: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: 生成波形 [B, T]
            y_ref : 参考/GT 波形 [B, T]

        Returns:
            标量 loss（Tensor）
        """
        total_loss = 0.0
        num_melnets = len(self.mel_nets)
        band_weight_sum = sum(self.band_weights)

        for mel_net, band_slices in zip(self.mel_nets, self.band_slices_per_melnet):
            # mel: [B, T_mel, num_mels]
            mel_pred = mel_net(y_pred)
            mel_ref = mel_net(y_ref)

            # 对每个频带单独算 loss
            for (start, end), w in zip(band_slices, self.band_weights):
                if end <= start:
                    continue  #极端防御，正常不会走到这里
                band_mel_pred = mel_pred[..., start:end]
                band_mel_ref = mel_ref[..., start:end]

                band_loss = self.loss_fn(band_mel_pred, band_mel_ref)
                total_loss = total_loss + w * band_loss

        total_loss = total_loss / (num_melnets * band_weight_sum)
        return total_loss



## below can be used in one gpu, but not ddp
mel_basis = {}
hann_window = {}


def mel_spectrogram(y, hparams, center=False, complex=False):  # y should be a tensor with shape (b,wav_len)
    # hop_size: 512  # For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
    # win_size: 2048  # For 22050Hz, 1100 ~= 50 ms (If None, win_size: fft_size) (0.05 * sample_rate)
    # fmin: 55  # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    # fmax: 10000  # To be increased/reduced depending on data.
    # fft_size: 2048  # Extra window size is filled with 0 paddings to match this parameter
    # n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax,
    n_fft = hparams['fft_size']
    num_mels = hparams['audio_num_mel_bins']
    sampling_rate = hparams['audio_sample_rate']
    hop_size = hparams['hop_size']
    win_size = hparams['win_size']
    fmin = hparams['fmin']
    fmax = hparams['fmax']
    if isinstance(y, np.ndarray):
        y = torch.FloatTensor(y)
    if len(y.shape) == 1:
        y = y.unsqueeze(0)
    y = y.clamp(min=-1., max=1.)
    global mel_basis, hann_window
    key = f"{fmax}_{y.device}"
    if key not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
    if str(y.device) not in hann_window:
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), [int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)],
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=complex)

    if not complex:
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
        spec = spectral_normalize_torch(spec)
    else:
        B, C, T, _ = spec.shape
        spec = spec.transpose(1, 2)  # [B, T, n_fft, 2]
    return spec