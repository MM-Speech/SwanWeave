from evaluation.evalBasis import EvalBasis
import numpy as np
import librosa

EPS = 1e-12

def wav_to_spectrogram(wav, rate):
    hop_length = int(rate / 100)
    n_fft = int(2048 / (48000 / rate)) 
    spec = np.abs(librosa.stft(wav, hop_length=hop_length, n_fft=n_fft))
    spec = np.transpose(spec, (1, 0))
    return spec

def cal_LSD(est, target):
    log_ratio = np.log10(target**2 / ((est + EPS) ** 2) + EPS) ** 2
    lsd_ = np.mean(np.mean(log_ratio, axis=1) ** 0.5, axis=0)
    return lsd_

class LSD(EvalBasis):
    def __init__(self):
        super(LSD, self).__init__(name='LSD')
        self.intrusive = True
        self.mono = True
        self.score_rate = 24000  # 常用 16k/22.05k/24k

    def _scoring(self, audios, score_rate: int):
        if len(audios) != 2:
            raise ValueError('LSD needs a reference and a test signals.')
        ref_wav = np.asarray(audios[1], dtype=np.float64).reshape(-1)
        test_wav = np.asarray(audios[0], dtype=np.float64).reshape(-1)
        est = wav_to_spectrogram(test_wav, score_rate)
        target = wav_to_spectrogram(ref_wav, score_rate)
        return cal_LSD(est, target)
    
if __name__ == "__main__":
    metric = LSD()
    ref_wav_path = "/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/user/dac_44khz/test/44100_8kbps/speech_clean/AISHELL-3_0000000109[G].wav"
    test_wav_path = "/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/user/dac_44khz/test/44100_8kbps/speech_clean/AISHELL-3_0000000109[P].wav"
    ref_wav, _ = librosa.load(ref_wav_path, sr=24000, mono=True)
    test_wav, _ = librosa.load(test_wav_path, sr=24000, mono=True)
    data = {"audio": [test_wav, ref_wav], "rate": 24000}
    measure = metric.scoring(data)
    print("LSD:", measure)
