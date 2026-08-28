from evaluation.evalBasis import EvalBasis

class PESQ(EvalBasis):
    def __init__(self, pesq_type: str = 'wb'):
        super(PESQ, self).__init__(name='PESQ')
        self.intrusive = False
        self.mono = True
        self.score_rate = 16000
        self.type = pesq_type  # wideband

    def _scoring(self, audios, rate):
        from pesq import pesq
        if len(audios) != 2:
            raise ValueError('PESQ needs a reference and a test signals.')
            return None
        return pesq(rate, audios[1], audios[0], self.type)
    
if __name__ == "__main__":
    metric = PESQ()
    ref_wav_path = "/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/user/dac_44khz/test/44100_8kbps/speech_clean/AISHELL-3_0000000109[G].wav"
    test_wav_path = "/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/user/dac_44khz/test/44100_8kbps/speech_clean/AISHELL-3_0000000109[P].wav"
    import librosa
    ref_wav, _ = librosa.load(ref_wav_path, sr=24000, mono=True)
    test_wav, _ = librosa.load(test_wav_path, sr=24000, mono=True)
    data = {"audio": [test_wav, ref_wav], "rate": 24000}
    measure = metric.scoring(data)
    print("PESQ:", measure)