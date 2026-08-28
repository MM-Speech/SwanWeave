import numpy as np
from evaluation.evalBasis import EvalBasis


class SNR(EvalBasis):
    """
    SNR (Signal-to-Noise Ratio), intrusive 指标，需要 reference + test。
    约定 audios = [test, ref]（与 SISDR 示例一致）。
    """

    def __init__(self):
        super().__init__(name="SNR")
        self.intrusive = True
        self.score_rate = 24000  # 强制用 24kHz 进行评分（EvalBasis 会在需要时重采样）

    def _scoring(self, audios, score_rate: int):
        # audios = [test, ref]
        test = np.asarray(audios[0], dtype=np.float64).reshape(-1)
        ref  = np.asarray(audios[1], dtype=np.float64).reshape(-1)

        # 对齐长度（codec 重建经常长度不同/有 padding）
        T = min(ref.shape[0], test.shape[0])
        if T == 0:
            raise ValueError("输入音频为空，无法计算 SNR")
        ref = ref[:T]
        test = test[:T]

        eps = 1e-10

        # Remove DC
        ref = ref - ref.mean()
        test = test - test.mean()

        # 缩放 test 使其与 ref 动态范围一致（参考你提供的 cal_SNR）
        ref_peak = np.max(np.abs(ref))
        test_peak = np.max(np.abs(test))
        if test_peak > eps:
            test = test * (ref_peak / (test_peak + eps))
        # else: test 全零就不缩放，后面公式会自然得到很差的 SNR

        noise = ref - test
        snr = 10.0 * np.log10((np.sum(ref ** 2) + eps) / (np.sum(noise ** 2) + eps))
        return float(snr)

if __name__ == "__main__":
    import librosa

    metric = SNR()

    ref_wav_path = "/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/user/dac_44khz/test/44100_8kbps/speech_clean/AISHELL-3_0000000109[G].wav"
    test_wav_path = "/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/user/dac_44khz/test/44100_8kbps/speech_clean/AISHELL-3_0000000109[P].wav"

    # 你可以直接按原采样率读入，让 EvalBasis 去重采样到 24k；
    # 也可以像下面这样直接读成 24k。
    ref_wav, _ = librosa.load(ref_wav_path, sr=24000, mono=True)
    test_wav, _ = librosa.load(test_wav_path, sr=24000, mono=True)

    data = {"audio": [test_wav, ref_wav], "rate": 24000}
    measure = metric.scoring(data)
    print("SNR:", measure)
