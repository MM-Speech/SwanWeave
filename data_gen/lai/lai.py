import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HOME"] = "/mnt/bn/sa-ag-data/liruiqi/code/huggingface"

import sys
import contextlib
import lattifai
import textgrid
from typing import Any, Dict, List, Optional, Tuple
import traceback
import httpx


class LaiAligner:
    def __init__(self, model_name="LattifAI/Lattice-1", device='cuda', api_key="lf_15e20d1f307eff7ec8b876e456d9356d"):
        self.client = lattifai.LattifAI(
            client_config=lattifai.ClientConfig(
                api_key=api_key,
                timeout=120.0,
                max_retries=10,
                profile=False
            ),
            alignment_config=lattifai.AlignmentConfig(
                model_name=model_name,
                model_hub="huggingface",
                device=device
            ),
            caption_config=lattifai.CaptionConfig(
                word_level=True
            )
        )

        # print(f"{self.client.config = }")
        # print(f"{self.client.aligner.config = }")
        # print(f"{self.client.caption_config = }")

    # ===== 原类外函数，改为类内方法 =====
    def find_tier(
        self,
        tg: textgrid.TextGrid,
        name: str,
        index_fallback: Optional[int] = None,
    ):
        """
        优先通过 tier.name 查找，找不到再用 index_fallback.
        """
        for tier in tg.tiers:
            if tier.name == name:
                return tier
        if index_fallback is not None and index_fallback < len(tg.tiers):
            return tg[index_fallback]
        raise ValueError(f"找不到名为 {name!r} 的 tier，且 index_fallback 也不可用")

    def key_interval(self, interval) -> Tuple[float, float]:
        """
        用 (minTime, maxTime) 构造 key，做一点小数点的四舍五入，避免浮点误差。
        """
        return (round(float(interval.minTime), 6), round(float(interval.maxTime), 6))

    def textgrid_to_json_obj(self, tg: textgrid.TextGrid) -> Dict[str, Any]:
        """
        按需求，把 TextGrid 对象转成：
        {
          "text": str,
          "sentence_conf": float | None,
          "word_info": [ {start, end, text, conf}, ... ]
        }
        """

        # 1. 找到四个 tier（兼容名字和固定顺序）
        utter_tier = self.find_tier(tg, "utterances", index_fallback=0)          # 第一行
        words_tier = self.find_tier(tg, "words", index_fallback=1)               # 第二行
        utt_score_tier = self.find_tier(tg, "utterance_scores", index_fallback=2)  # 第三行
        word_score_tier = self.find_tier(tg, "word_scores", index_fallback=3)      # 第四行

        # 2. text：来自 utterances 的所有非空 mark 拼接
        text_pieces: List[str] = [
            itv.mark for itv in utter_tier.intervals
            if itv.mark is not None and itv.mark.strip() != ""
        ]
        full_text = "".join(text_pieces)

        # 3. sentence_conf：utterance_scores 中第一个非空 mark
        sentence_conf = None
        for itv in utt_score_tier.intervals:
            mark = (itv.mark or "").strip()
            if mark != "":
                try:
                    sentence_conf = float(mark)
                except ValueError:
                    sentence_conf = None
                break

        # 4. word_info：words & word_scores 对齐
        #    先构建一个 (start,end) -> conf 的字典
        score_map: Dict[Tuple[float, float], Optional[float]] = {}
        for itv in word_score_tier.intervals:
            k = self.key_interval(itv)
            mark = (itv.mark or "").strip()
            if mark == "":
                score_map[k] = None
            else:
                try:
                    score_map[k] = float(mark)
                except ValueError:
                    score_map[k] = None

        word_info: List[Dict[str, Any]] = []
        for itv in words_tier.intervals:
            word = (itv.mark or "").strip()
            # if word == "":
            #     # 通常空的间隔不写入 word_info
            #     continue
            k = self.key_interval(itv)
            conf = score_map.get(k, None)
            word_info.append(
                {
                    "start": float(itv.minTime),
                    "end": float(itv.maxTime),
                    "text": word,
                    "conf": conf,
                }
            )

        return {
            "text": full_text,
            "sentence_conf": sentence_conf,
            "word_info": word_info,
        }

    # ===== 原有类方法 =====
    def align(self, input_media, input_caption, output_caption_path):
        try:
            caption = self.client.alignment(
                input_media=input_media,
                input_caption=input_caption,
                output_caption_path=output_caption_path,
            )
            return caption
        except Exception as err:
            print(f"{err.error_code = }", file=sys.__stdout__)
            traceback.print_exc()


    def process_alignment(self, wav_path, text, verbose=False):
        import tempfile
        with tempfile.TemporaryDirectory(dir='/dev/shm') as tmpdir:
            temp_txt_path = os.path.join(tmpdir, "temp.txt")
            temp_tg_path = os.path.join(tmpdir, "temp.TextGrid")
            with open(temp_txt_path, 'w', encoding='utf-8') as f:
                f.write(text)

            if verbose:
                ret = self.align(
                    input_media=wav_path,
                    input_caption=temp_txt_path,
                    output_caption_path=temp_tg_path,
                )
            else:
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                    ret = self.align(
                        input_media=wav_path,
                        input_caption=temp_txt_path,
                        output_caption_path=temp_tg_path,
                    )
            if ret is None:
                return None

            tg = textgrid.TextGrid.fromFile(temp_tg_path)
            json_obj = self.textgrid_to_json_obj(tg)
            return json_obj


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', type=str, required=True, help='Input wav file path')
    parser.add_argument('--text', type=str, required=True, help='Input text to align')
    args = parser.parse_args()

    lai_aligner = LaiAligner(model_name='/mnt/bn/sa-ag-data/panchanghao/code/ScriptSpeech/pretrained_models/LattifAI/Lattice-1')
    result = lai_aligner.process_alignment(args.wav, args.text)
    # print(json.dumps(result, ensure_ascii=False, indent=2))
    result['word_info'] = [(w['start'], w['end'], w['text'], w['conf']) for w in result['word_info']]
    from utils.commons.io import json_dumps
    print(json_dumps(result))

    # source /mnt/bn/sa-ag-data/zhangyu.34/.bashrc; conda activate lai