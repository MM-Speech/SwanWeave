import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any

from utils.audio.transform import batch_resample
from utils.audio.pitch_utils import resample_align_curve
from data_gen.pitch.rmvpe.constants import *
from data_gen.pitch.rmvpe.model import E2E0
from data_gen.pitch.rmvpe.spec import MelSpectrogram
from data_gen.pitch.rmvpe.rmvpe_utils import to_local_average_f0, to_viterbi_f0


class RMVPE:
    def __init__(self, model_path, device='auto', hop_length=160, precision='fp32'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.precision = precision.strip().lower()
        if self.precision not in ('fp32', 'fp16', 'bf16'):
            raise ValueError(f"precision must be 'fp32', 'fp16', or 'bf16', got '{precision}'")
        if self.precision in ('fp16', 'bf16') and self.device.type != 'cuda':
            self.precision = 'fp32'
        self.model = E2E0(4, 1, (2, 2)).eval().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        ).to(self.device)
        self.hop_length = hop_length
        self.resamplers = {}
        self.model_sr = SAMPLE_RATE

    def _autocast_context(self):
        if self.precision == 'fp16':
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        elif self.precision == 'bf16':
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return torch.autocast(device_type='cuda', enabled=False)

    @torch.no_grad()
    def mel2hidden(self, mel):
        n_frames = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode='constant')
        with self._autocast_context():
            hidden = self.model(mel)
        return hidden[:, :n_frames].float()

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            f0 = to_viterbi_f0(hidden, thred=thred)
        else:
            f0 = to_local_average_f0(hidden, thred=thred)
        return f0

    def postprocess(self, f0, fmin=50, fmax=1000, min_gap=2):
        f0[f0 < fmin] = 0
        f0[f0 > fmax] = 0
        for idx in range(f0.shape[0] - min_gap - 1):
            if f0[idx] == 0 and f0[idx + min_gap + 1] == 0 and np.sum(f0[idx: idx + min_gap + 2]) > 0:
                f0[idx: idx + min_gap + 2] = 0
        return f0

    @torch.no_grad()
    def infer(
        self,
        audio: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
        sr: Optional[int] = None,
        sr_list: Optional[List[int]] = None,
        hop_size: int = 100,
        fmin: int = 50,
        fmax: int = 900,
        batch_size: int = 16,
        thred: float = 0.03,
        use_viterbi: bool = False,
        interp_uv: bool = False,
        min_gap: int = 2,
    ) -> Dict[str, Any]:
        is_single = not isinstance(audio, list)
        if is_single:
            audio = [audio]

        wavs = []
        for a in audio:
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
            a = np.asarray(a, dtype=np.float32)
            if a.ndim > 1:
                a = a.squeeze()
            wavs.append(a)

        if sr is not None and sr_list is not None:
            raise ValueError("Provide either sr or sr_list, not both.")
        if sr is not None:
            sample_rates = [sr] * len(wavs)
        elif sr_list is not None:
            if len(sr_list) != len(wavs):
                raise ValueError("sr_list length must match audio list length.")
            sample_rates = list(sr_list)
        else:
            raise ValueError("Either sr or sr_list must be provided.")

        lengths = [(len(w) + hop_size - 1) // hop_size for w in wavs]

        wavs = batch_resample(
            wavs, sample_rates, self.model_sr,
            resamplers=self.resamplers,
            batch_size=batch_size,
            device=self.device,
        )

        results = []
        for start in range(0, len(wavs), batch_size):
            batch_wavs = wavs[start:start + batch_size]
            batch_lengths = lengths[start:start + batch_size]
            batch_srs = sample_rates[start:start + batch_size]

            wav_tensors = [torch.from_numpy(w).float() for w in batch_wavs]
            max_wav_len = max(len(w) for w in wav_tensors)
            padded = torch.stack([
                F.pad(w, (0, max_wav_len - len(w))) for w in wav_tensors
            ]).to(self.device)

            with self._autocast_context():
                mels = self.mel_extractor(padded, center=True)
                hiddens = self.mel2hidden(mels)
            hiddens = hiddens.float()
            f0_batch = self.decode(hiddens, thred=thred, use_viterbi=use_viterbi)

            for i in range(len(batch_wavs)):
                mel_len = math.ceil((len(batch_wavs[i]) + 1) / self.hop_length)
                f0 = f0_batch[i, :mel_len]
                f0 = self.postprocess(f0, fmin, fmax, min_gap)
                uv = f0 == 0

                time_step = hop_size / batch_srs[i]
                f0_res = resample_align_curve(f0, 0.01, time_step, batch_lengths[i])
                uv_res = resample_align_curve(uv.astype(np.float32), 0.01, time_step, batch_lengths[i]) > 0.5
                if not interp_uv:
                    f0_res[uv_res] = 0

                voiced_f0 = f0_res[~uv_res]
                avg_f0 = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
                avg_f0_log = float(np.exp(np.mean(np.log(voiced_f0)))) if len(voiced_f0) > 0 else 0.0

                results.append({"f0": f0_res, "uv": uv_res, "avg_f0": avg_f0, "avg_f0_log": avg_f0_log})

        if is_single:
            return results[0]
        return results

    def visualize(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sr: int,
        save_path: str,
        hop_size: int = 100,
        fmin: int = 50,
        fmax: int = 900,
        thred: float = 0.03,
        use_viterbi: bool = False,
        interp_uv: bool = False,
        min_gap: int = 2,
        fig_width: int = 14,
        fig_height: int = 10,
        dpi: int = 150,
    ):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        result = self.infer(
            audio, sr=sr, hop_size=hop_size,
            fmin=fmin, fmax=fmax, thred=thred,
            use_viterbi=use_viterbi, interp_uv=interp_uv, min_gap=min_gap,
        )
        f0 = result["f0"]
        uv = result["uv"]

        wav_16k = batch_resample([audio_np], sr, self.model_sr, resamplers=self.resamplers, device=self.device)[0]
        wav_tensor = torch.from_numpy(wav_16k).float().unsqueeze(0).to(self.device)
        with self._autocast_context():
            mel = self.mel_extractor(wav_tensor, center=True)
        mel_np = mel.float().squeeze(0).cpu().numpy()

        duration = len(audio_np) / sr
        time_wav = np.linspace(0, duration, len(audio_np))
        time_f0 = np.arange(len(f0)) * (hop_size / sr)

        fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1, 1]})

        ax_mel = axes[0]
        mel_extent = [0, duration, 0, self.mel_extractor.n_mel_channels]
        ax_mel.imshow(mel_np, aspect='auto', origin='lower', extent=mel_extent, cmap='magma')
        ax_mel.set_ylabel('Mel bin')
        ax_mel.set_title('Mel Spectrogram')

        ax_f0 = axes[1]
        voiced = f0.copy()
        voiced[uv] = np.nan
        unvoiced = f0.copy()
        unvoiced[~uv] = np.nan
        ax_f0.plot(time_f0, voiced, 'b.', markersize=2, label='Voiced')
        ax_f0.plot(time_f0, unvoiced, 'r.', markersize=2, label='Unvoiced')
        ax_f0.set_ylabel('F0 (Hz)')
        ax_f0.set_ylim(fmin - 10, fmax + 100)
        ax_f0.legend(loc='upper right', fontsize=8)
        ax_f0.set_title('F0 Curve')

        ax_wav = axes[2]
        ax_wav.plot(time_wav, audio_np, linewidth=0.3, color='steelblue')
        ax_wav.set_ylabel('Amplitude')
        ax_wav.set_xlabel('Time (s)')
        ax_wav.set_title('Waveform')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    def release_cuda(self):
        self.model = self.model.cpu()
        self.mel_extractor = self.mel_extractor.cpu()


if __name__ == '__main__':

    import librosa

    model = RMVPE(
        model_path='pretrained_models/rmvpe/model.pt',
        device='cuda',
        precision='fp16'
    )

    wav, sr = librosa.load('user/temp/audio.wav', sr=None)
    wav = wav[:int(20 * sr)]

    model.visualize(
        audio=wav,
        sr=sr,
        save_path='user/temp/audio_pitch.png',
        use_viterbi=True
    )



