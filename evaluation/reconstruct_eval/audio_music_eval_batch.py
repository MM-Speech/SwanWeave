import os
import glob
from tqdm import tqdm

from utils.commons.io import json_dumps
from utils.commons.multiprocess_utils import chunked_multiprocess_run

from evaluation.music_eval import MusicEval

def init_worker_evaluator(worker_id: int):
    # 每个 worker 启动时只初始化一次，后续所有 job 复用
    return MusicEval()


import os
import librosa

def eval_one_pair(gt_path: str, pred_path: str, ctx=None):
    """
    ctx: 由 init_worker_evaluator 返回，这里就是 SpeechEval() 实例
    """
    evaluator = ctx
    file_id = os.path.basename(gt_path).replace("[G].wav", "")

    # 文件检查（避免 pred 缺失导致异常）
    if not os.path.exists(gt_path):
        return {"file": file_id, "_error": f"gt_not_found: {gt_path}"}
    if not os.path.exists(pred_path):
        return {"file": file_id, "_error": f"pred_not_found: {pred_path}"}

    # 加载音频（在 worker 内做，减少主进程内存压力）
    gt_wav, sr = librosa.load(gt_path, sr=24000, mono=True)
    pred_wav, sr2 = librosa.load(pred_path, sr=24000, mono=True)
    rate = sr  # 理论上就是 24000

    out = {"file": file_id}

    # 非侵入式：只用 pred
    for metric_name in evaluator.non_intrusive_metrics.keys():
        out[metric_name] = evaluator.evaluate(metric_name, [pred_wav], rate)

    # 侵入式：用 (pred, gt)
    for metric_name in evaluator.intrusive_metrics.keys():
        out[metric_name] = evaluator.evaluate(metric_name, [pred_wav, gt_wav], rate)

    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, required=True)
    ap.add_argument("--output-path", type=str, required=True)
    ap.add_argument("--num-workers", type=int, default=32)
    args = ap.parse_args()

    input_dir = args.input_dir
    output_path = args.output_path
    num_workers = args.num_workers
    os.environ.setdefault("DNSMOS_NUM_WORKERS", str(num_workers))

    gt_pat = os.path.join(input_dir, "*" + glob.escape("[G]") + ".wav")
    gt_wav_paths = sorted(glob.glob(gt_pat))
    if not gt_wav_paths:
        raise RuntimeError(f"未匹配到任何 [G].wav：pattern={gt_pat}")

    pred_wav_paths = [p.replace("[G].wav", "[P].wav") for p in gt_wav_paths]

    # 关键：args 的每个元素是 tuple => worker 会 *arg 展开成 (gt_path, pred_path)
    jobs = list(zip(gt_wav_paths, pred_wav_paths))

    results = []
    for r in tqdm(
        chunked_multiprocess_run(
            eval_one_pair,
            jobs,
            num_workers=num_workers,
            ordered=True,
            init_ctx_func=init_worker_evaluator,  # 注意：会传 worker_id
        ),
        total=len(jobs),
        desc="evaluating",
        dynamic_ncols=True,
    ):
        # 你的 worker 异常时会 put None
        if r is not None:
            results.append(r)
        else:
            results.append({"file": None, "_error": "worker_exception_returned_none"})

    # 统计平均：跳过错误项
    ok_results = [x for x in results if x.get("_error") is None]
    if not ok_results:
        final_output = {"average": {}, "details": results}
    else:
        metric_names = [k for k in ok_results[0].keys() if k != "file" and not k.startswith("_")]

        avg_results = {}
        for m in metric_names:
            avg_results[m] = round(sum(r[m] for r in ok_results) / len(ok_results), 4)

        final_output = {"average": avg_results, "details": results}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_dumps(final_output))
