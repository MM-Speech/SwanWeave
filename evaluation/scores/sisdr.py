import numpy as np
from numpy.linalg import norm
from evaluation.evalBasis import EvalBasis

class SISDR(EvalBasis):
    def __init__(self):
        super(SISDR, self).__init__(name='SISDR')
        self.intrusive = True
        self.score_rate = 24000 # 常用 16k/22.05k/24k
        
    def _scoring(self, audios, score_rate: int):
        eps = np.finfo(audios[0].dtype).eps
        reference = audios[1].reshape(audios[1].size, 1)
        estimate = audios[0].reshape(audios[0].size, 1)
        # compute Rss
        Rss = np.dot(reference.T, reference)
        
        # get the scaling factor for clean sources
        a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

        e_true = a * reference
        e_res = estimate - e_true

        Sss = (e_true**2).sum()
        Snn = (e_res**2).sum()

        return 10 * np.log10((eps+ Sss)/(eps + Snn))
    
if __name__ == "__main__":
    metric = SISDR()
    ref_wav_path = "/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/user/dac_44khz/test/44100_8kbps/speech_clean/AISHELL-3_0000000109[G].wav"
    test_wav_path = "/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/user/dac_44khz/test/44100_8kbps/speech_clean/AISHELL-3_0000000109[P].wav"
    import librosa
    ref_wav, _ = librosa.load(ref_wav_path, sr=24000, mono=True)
    test_wav, _ = librosa.load(test_wav_path, sr=24000, mono=True)
    data = {"audio": [test_wav, ref_wav], "rate": 24000}
    measure = metric.scoring(data)
    print("SISDR:", measure)