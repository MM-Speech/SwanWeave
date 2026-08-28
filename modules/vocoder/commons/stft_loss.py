# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


def _flatten_audio_batch(x, y=None):
    if len(x.shape) == 3:
        x = x.view(-1, x.size(2))  # (B, C, T) -> (B x C, T)
        if y is not None:
            y = y.view(-1, y.size(2))
    return (x, y) if y is not None else x


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro").clamp_min(1e-6)


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        win = getattr(torch, window)(win_length)
        self.register_buffer("window", win)
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        window = self.window
        if window.device != x.device:
            window = window.to(x.device)

        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
            self,
            fft_sizes=None,
            hop_sizes=None,
            win_lengths=None,
            resolutions=None,
            window="hann_window",
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            resolutions (list): List of (fft_size, hop_size, win_length).
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()

        if resolutions is None:
            if fft_sizes is None:
                fft_sizes = [1024, 2048, 512]
            if hop_sizes is None:
                hop_sizes = [120, 240, 50]
            if win_lengths is None:
                win_lengths = [600, 1200, 240]
            assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
            resolutions = list(zip(fft_sizes, hop_sizes, win_lengths))

        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in resolutions:
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        x, y = _flatten_audio_batch(x, y)
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss


class TransientSTFTLoss(torch.nn.Module):
    """STFT temporal-difference loss for transient reconstruction."""

    def __init__(
            self,
            fft_size=256,
            shift_size=32,
            win_length=128,
            window="hann_window",
            use_log_mag=True,
            positive_only=True,
            onset_weight=0.0,
            loss_type="l1",
            eps=1e-7,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.use_log_mag = use_log_mag
        self.positive_only = positive_only
        self.onset_weight = float(onset_weight)
        self.loss_type = loss_type
        self.eps = float(eps)
        if self.loss_type not in {"l1", "l2"}:
            raise ValueError(f"Unsupported transient STFT loss_type: {loss_type}")
        win = getattr(torch, window)(win_length)
        self.register_buffer("window", win)

    def _delta_mag(self, x):
        window = self.window
        if window.device != x.device:
            window = window.to(x.device)

        mag = stft(x, self.fft_size, self.shift_size, self.win_length, window)
        if self.use_log_mag:
            mag = torch.log(torch.clamp(mag, min=self.eps))
        delta = mag[:, 1:, :] - mag[:, :-1, :]
        if self.positive_only:
            delta = F.relu(delta)
        return delta

    def forward(self, x, y):
        x_delta = self._delta_mag(x)
        y_delta = self._delta_mag(y)
        if x_delta.shape[1] == 0 or y_delta.shape[1] == 0:
            return x.new_zeros(())

        diff = x_delta - y_delta
        if self.loss_type == "l2":
            diff = diff.pow(2)
        else:
            diff = diff.abs()

        if self.onset_weight > 0.0:
            onset = F.relu(y_delta)
            denom = torch.clamp(onset.mean(dim=(1, 2), keepdim=True), min=self.eps)
            weights = 1.0 + self.onset_weight * (onset / denom)
            diff = diff * weights

        return diff.mean()


class MultiResolutionTransientSTFTLoss(torch.nn.Module):
    """Multi-resolution transient STFT loss."""

    def __init__(
            self,
            fft_sizes=None,
            hop_sizes=None,
            win_lengths=None,
            resolutions=None,
            window="hann_window",
            use_log_mag=True,
            positive_only=True,
            onset_weight=0.0,
            loss_type="l1",
            eps=1e-7,
    ):
        super().__init__()

        if resolutions is None:
            if fft_sizes is None:
                fft_sizes = [256, 512, 1024]
            if hop_sizes is None:
                hop_sizes = [32, 64, 120]
            if win_lengths is None:
                win_lengths = [128, 256, 600]
            assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
            resolutions = list(zip(fft_sizes, hop_sizes, win_lengths))

        self.transient_losses = torch.nn.ModuleList()
        for fs, ss, wl in resolutions:
            self.transient_losses.append(
                TransientSTFTLoss(
                    fft_size=fs,
                    shift_size=ss,
                    win_length=wl,
                    window=window,
                    use_log_mag=use_log_mag,
                    positive_only=positive_only,
                    onset_weight=onset_weight,
                    loss_type=loss_type,
                    eps=eps,
                )
            )

    def forward(self, x, y):
        x, y = _flatten_audio_batch(x, y)
        loss = 0.0
        for f in self.transient_losses:
            loss += f(x, y)
        loss /= len(self.transient_losses)
        return loss
