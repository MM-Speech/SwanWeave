from evaluation.evalBasis import EvalBasis

class STOI(EvalBasis):
    def __init__(self):
        super().__init__(name="STOI")
        self.intrusive = True
        self.score_rate = 24000  # 强制用 24kHz 进行评分（EvalBasis 会在需要时重采样）
        
    def _scoring(self, audios, score_rate: int):
        from pystoi.stoi import stoi
        if len(audios) != 2:
            raise ValueError('STOI needs a reference and a test signals.')
        return stoi(audios[1], audios[0], score_rate, extended=False)
    
if __name__ == "__main__":
    import librosa
    metric = STOI()
    
    ref_wav_path = "/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/user/dac_44khz/test/44100_8kbps/speech_clean/AISHELL-3_0000000109[G].wav"
    test_wav_path = "/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/user/dac_44khz/test/44100_8kbps/speech_clean/AISHELL-3_0000000109[P].wav"

    # 你可以直接按原采样率读入，让 EvalBasis 去重采样到 24k；
    # 也可以像下面这样直接读成 24k。
    ref_wav, _ = librosa.load(ref_wav_path, sr=24000, mono=True)
    test_wav, _ = librosa.load(test_wav_path, sr=24000, mono=True)

    data = {"audio": [test_wav, ref_wav], "rate": 24000}
    measure = metric.scoring(data)
    print("STOI:", measure)
