from evaluation.evalBasis import EvalBasis
import math
import numpy as np
import pyworld
import pysptk
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class MCD(EvalBasis):
    def __init__(self, mcd_mode="plain"):
        super().__init__(name="MCD")
        self.intrusive = True
        self.mcd_mode = mcd_mode  # "plain" / "dtw" / "dtw_sl"
        self.score_rate = 24000  # 常用 16k/22.05k/24k
        self.mcd_toolbox = Calculate_MCD(MCD_mode=mcd_mode)

    def _scoring(self, audios, score_rate: int):
        if len(audios) != 2:
            raise ValueError("MCD 需要 2 路音频：[reference, test]")
        # 这里沿用你之前的顺序：audios[0]=ref, audios[1]=test（如与你实现相反可交换）
        return self.mcd_toolbox.calculate_mcd(audios[1], audios[0], score_rate)
    
class Calculate_MCD:
    def __init__(self, MCD_mode="dtw", frame_period=5.0, mcep_order=24, alpha=0.68, fft_size=1024, drop_c0=True):
        """
        MCD_mode: "plain" / "dtw" / "dtw_sl"
        drop_c0: 计算距离与 DTW 时是否丢弃 c0（常见做法是 True）
        """
        self.MCD_mode = MCD_mode
        self.FRAME_PERIOD = float(frame_period)
        self.mcep_order = int(mcep_order)
        self.alpha = float(alpha)
        self.fft_size = int(fft_size)
        self.drop_c0 = bool(drop_c0)
        # 10 / ln(10) * sqrt(2) ~= 6.14185
        self.log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    
    def wav2mcep_numpy(self, loaded_wav, score_rate):
        x = loaded_wav.astype(np.double)
        # 1) F0 + time axis
        f0, t = pyworld.harvest(x, fs=score_rate, frame_period=self.FRAME_PERIOD)
        # 2) spectral envelope
        sp = pyworld.cheaptrick(x, f0, t, fs=score_rate, fft_size=self.fft_size)
        # 3) MCEP
        mcep = pysptk.sptk.mcep(
            sp,
            order=self.mcep_order,
            alpha=self.alpha,
            maxiter=0,
            etype=1,
            eps=1.0E-8,
            min_det=0.0,
            itype=3,
        )
        return mcep  # [num_frames, mcep_dim]
    
    @staticmethod
    def _path_to_indices(path):
        pathx = [p[0] for p in path]
        pathy = [p[1] for p in path]
        return pathx, pathy
    
    def calculate_mcd_distance(self, x, y, path):
        """
        x, y: [Tx, D], [Ty, D]
        path: list[(i, j)]
        return: (frames_tot, min_cost_tot) 其中 min_cost_tot 是沿 path 的欧氏距离之和（未乘常数）
        """
        pathx, pathy = self._path_to_indices(path)
        x2, y2 = x[pathx], y[pathy]
        frames_tot = x2.shape[0]
        z = x2 - y2
        min_cost_tot = np.sqrt((z * z).sum(-1)).sum()
        return frames_tot, float(min_cost_tot)
    
    def average_mcd(self, loaded_ref_wav, loaded_syn_wav, score_rate):
        # plain 模式才需要 pad 到同长度
        if self.MCD_mode == "plain":
            if len(loaded_ref_wav) < len(loaded_syn_wav):
                loaded_ref_wav = np.pad(loaded_ref_wav, (0, len(loaded_syn_wav) - len(loaded_ref_wav)))
            else:
                loaded_syn_wav = np.pad(loaded_syn_wav, (0, len(loaded_ref_wav) - len(loaded_syn_wav)))
        ref_mcep = self.wav2mcep_numpy(loaded_ref_wav, score_rate)
        syn_mcep = self.wav2mcep_numpy(loaded_syn_wav, score_rate)
        if ref_mcep.size == 0 or syn_mcep.size == 0:
            raise ValueError("音频过短或特征提取失败，导致 mcep 为空。")
        # 是否丢弃 c0（推荐距离与 DTW 都一致）
        if self.drop_c0:
            ref_feat = ref_mcep[:, 1:]
            syn_feat = syn_mcep[:, 1:]
        else:
            ref_feat = ref_mcep
            syn_feat = syn_mcep
        if self.MCD_mode == "plain":
            L = min(len(ref_feat), len(syn_feat))
            path = [(i, i) for i in range(L)]
            cof = 1.0
        elif self.MCD_mode == "dtw":
            _, path = fastdtw(ref_feat, syn_feat, dist=euclidean)
            cof = 1.0
        elif self.MCD_mode == "dtw_sl":
            _, path = fastdtw(ref_feat, syn_feat, dist=euclidean)
            cof = (len(ref_feat) / len(syn_feat)) if len(ref_feat) > len(syn_feat) else (len(syn_feat) / len(ref_feat))
        else:
            raise ValueError(f"Unknown MCD_mode: {self.MCD_mode}")
        frames_tot, min_cost_tot = self.calculate_mcd_distance(ref_feat, syn_feat, path)
        mean_mcd = cof * self.log_spec_dB_const * min_cost_tot / frames_tot
        return float(mean_mcd)
    
    def calculate_mcd(self, reference_audio, synthesized_audio, score_rate):
        return self.average_mcd(reference_audio, synthesized_audio, score_rate)
    
    
if __name__ == "__main__":
    metric = MCD(mcd_mode="dtw")
    ref_wav_path = "/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/user/dac_44khz/test/44100_8kbps/speech_clean/AISHELL-3_0000000109[G].wav"
    test_wav_path = "/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/user/dac_44khz/test/44100_8kbps/speech_clean/AISHELL-3_0000000109[P].wav"
    import librosa
    ref_wav, _ = librosa.load(ref_wav_path, sr=24000, mono=True)
    test_wav, _ = librosa.load(test_wav_path, sr=24000, mono=True)
    data = {"audio": [test_wav, ref_wav], "rate": 24000}
    measure = metric.scoring(data)
    print("MCD:", measure)