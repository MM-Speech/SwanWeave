from evaluation.evalBasis import EvalBasis
# non-intrusive
from evaluation.scores.dnsmos import DNSMOS
# intrusive
from evaluation.scores.PESQ import PESQ
from evaluation.scores.stoi import STOI
from evaluation.scores.snr import SNR
from evaluation.scores.mcd import MCD
from evaluation.scores.sisdr import SISDR

class SpeechEval:
    def __init__(self):
        # 初始化所有支持的非侵入式和侵入式评价指标
        
        self.non_intrusive_metrics = {
            "DNSMOS": DNSMOS(),
        }
        self.intrusive_metrics = {
            "PESQ": PESQ(pesq_type='wb'),
            "STOI": STOI(),
            "SNR": SNR(),
            "MCD": MCD(),
            "SISDR": SISDR()
        }
    
    def evaluate(self, metric_name: str, audios: list, rate: int):
        if metric_name not in self.non_intrusive_metrics and metric_name not in self.intrusive_metrics:
            raise ValueError(f"Metric {metric_name} not supported.")
        if metric_name in self.non_intrusive_metrics:
            metric = self.non_intrusive_metrics[metric_name]
        else:
            metric = self.intrusive_metrics[metric_name]
        return metric.scoring({"audio": audios, "rate": rate})

if __name__ == "__main__":
    import librosa

    ref_wav_path = "/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/user/dac_44khz/eval/44100_8kbps/speech_clean/AISHELL-3_0000000109[G].wav"
    test_wav_path = "/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/user/dac_44khz/eval/44100_8kbps/speech_clean/AISHELL-3_0000000109[P].wav"

    ref_wav, _ = librosa.load(ref_wav_path, sr=24000, mono=True)
    test_wav, _ = librosa.load(test_wav_path, sr=24000, mono=True)

    evaluator = SpeechEval()
    
    # 首先处理非侵入式指标
    for metric_name in evaluator.non_intrusive_metrics.keys():
        score = evaluator.evaluate(metric_name, [test_wav], 24000)
        print(f"{metric_name}: {score}")
    
    # 然后处理侵入式指标
    for metric_name in evaluator.intrusive_metrics.keys():
        score = evaluator.evaluate(metric_name, [test_wav, ref_wav], 24000)
        print(f"{metric_name}: {score}")