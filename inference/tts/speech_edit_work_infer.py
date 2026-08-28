import os
import random
import re
import tempfile
from datetime import datetime, timedelta
import collections
import collections.abc
import math
import csv
import json
from typing import Optional, Dict, List, Tuple
import numpy as np
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))
from attrdictionary import AttrDict
import socket
from contextlib import closing
import yaml
from argparse import ArgumentParser
import torch.distributed as dist
import torch
import soundfile as sf
import librosa
import shutil
import tqdm

os.environ.setdefault("MODELSCOPE_CACHE", "/mnt/bn/sa-ag-data/liruiqi/code/modelscope")

from multiprocessing import Process, set_start_method
from utils.commons.os_utils import kill_void



from inference.tts.speech_edit_infer import SpeechEditInfer, SpeechEditInferWrapper

from utils.text.zh_text_norm import num2chn

from inference.tts.speech_edit_infer import set_seed


def _extract_media_id_from_path(p: str) -> str:
    base = os.path.basename(p)
    for ext in ('.wav', '.mp4'):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def _iter_local_vid_basenames(local_vids_value: str) -> List[str]:
    s = (local_vids_value or '').strip()
    if not s:
        return []

    items: List[str] = []
    if s.startswith('[') and s.endswith(']'):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                items = [str(x) for x in parsed]
            else:
                items = [str(parsed)]
        except Exception:
            items = [s]
    else:
        items = re.split(r"[\s,;|]+", s)

    out: List[str] = []
    for it in items:
        it = (it or '').strip().strip('"').strip("'")
        if not it:
            continue
        out.append(_extract_media_id_from_path(it))
    return out


def _num_string_to_price_zh(num_str: str) -> str:
    ns = (num_str or '').strip()
    if not ns:
        return ''

    if '.' in ns:
        int_part, frac_part = ns.split('.', 1)
        int_part = int_part or '0'
        frac_part = re.sub(r"\D+", "", frac_part)
        frac_part = frac_part.rstrip('0')

        zh_int = num2chn(int_part, alt_two=False)
        if not frac_part:
            return zh_int

        zh_frac = ''.join(num2chn(d, use_units=False, alt_two=False) for d in frac_part)
        return f"{zh_int}块{zh_frac}"

    return num2chn(ns, alt_two=False)


def normalize_price_text(text: str) -> str:
    s = (text or '').strip()
    if not s or not re.search(r"\d", s):
        return s

    s = s.replace('．', '.').replace('。', '.')

    def _repl(m: re.Match) -> str:
        num = m.group('num')
        suffix = m.group('suffix') or ''
        return _num_string_to_price_zh(num) + suffix

    return re.sub(r"(?P<num>\d+(?:\.\d+)?)(?P<suffix>多)?", _repl, s)


def load_prices_from_edit_0326(
    csv_path: str = "/mnt/bn/sa-ag-data/leike/work/edit_0326_2/data.csv",
    media_dir: str = "/mnt/bn/sa-ag-data/leike/work/edit_0326_2",
) -> Tuple[List[str], List[str], List[str]]:
    wav_paths = sorted(
        [
            os.path.join(media_dir, f)
            for f in os.listdir(media_dir)
            if f.endswith('.wav')
        ]
    )

    mp4_paths = sorted(
        [
            os.path.join(media_dir, f)
            for f in os.listdir(media_dir)
            if f.endswith('.mp4')
        ]
    )

    _ = mp4_paths

    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)

    vid2row: Dict[str, Dict[str, str]] = {}
    # rows = rows[:30]

    for row in rows:
        local_vids = row.get('local_vids', '')
        for vid_base in _iter_local_vid_basenames(local_vids):
            if vid_base and vid_base not in vid2row:
                vid2row[vid_base] = row

    price_before_list: List[str] = []
    price_after_list: List[str] = []
    wav_path_list: List[str] = []

    for wav_path in wav_paths:
        wav_id = _extract_media_id_from_path(wav_path)
        row = vid2row.get(wav_id)
        if row is None:
            continue

        price_before = normalize_price_text(str(row.get('price_before', '') or ''))
        price_after = normalize_price_text(str(row.get('price_after', '') or ''))
        if not price_before or not price_after:
            continue

        price_before_list.append(price_before)
        price_after_list.append(price_after)
        wav_path_list.append(wav_path)

    return price_before_list, price_after_list, wav_path_list


def ads_infer(args, cfg, out_path):
    device = torch.device('cuda:0')

    dit_ckpt = args.dit_ckpt
    infer_ins = SpeechEditInfer(
        device,
        dit_ckpt=dit_ckpt,
        vae_ckpt=args.vae_ckpt,
        merge_ckpt=args.merge_ckpt,
        merge_weight=args.merge_weight,
        use_sa_front=cfg.get('use_sa_front', False),
    )
    infer_ins = SpeechEditInferWrapper(infer_ins)

    price_before_list, price_after_list, wav_path_list = load_prices_from_edit_0326()
    # for price_before, price_after, wav_path in zip(price_before_list, price_after_list, wav_path_list):
    timings = []
    for idx in tqdm.tqdm(range(len(price_before_list))):
        price_before = price_before_list[idx]
        price_after = price_after_list[idx]
        wav_path = wav_path_list[idx]

        wav_name = os.path.basename(wav_path).replace('.wav', '')
        # infer_ins.model_infer3(cfg, '八十九', '一百三十九', '/mnt/bn/sa-ag-data/leike/work/edit_0326_2/v0345dg10003d71kq2iljht8i87ofhcg.wav', out_path+'_aligner_extend', 'v0345dg10003d71kq2iljht8i87ofhcg_89_139_016-latest.wav', overlap_time=(0.16, 0.16), debug=True)

        try:
            set_seed(42)
            # infer_ins.model_infer3(cfg, price_before, price_after, wav_path, out_path+'_aligner_extend', f"{wav_name}_{price_before}_{price_after}.wav", overlap_time=(0.16, 0.16) )
            # infer_ins.model_infer4(cfg, price_before, price_after, wav_path, out_path+'_aligner', f"{wav_name}_{price_before}_{price_after}.wav", overlap_time=(0.16, 0.16) )
            _, timing = infer_ins.inference(
                cfg=cfg,
                wav_path=wav_path,
                out_path=out_path + '_aligner',
                out_file=f"{wav_name}_{price_before}_{price_after}.wav",
                text_src=price_before,
                text_tgt=price_after,
                overlap_time=(0.16, 0.16),
                debug=True,
            )
            timings.append(timing)

            shutil.copy(wav_path, os.path.join(out_path+'_aligner', f"{wav_name}.wav"))
        except Exception as e:
            print(f'| [INFO] 运行 {wav_name} 出错 {e}')
    asr = sum([timing['after_asr'] - timing['before_asr'] for timing in timings]) / len(timings)
    uvr = sum([timing['after_uvr'] - timing['before_uvr'] for timing in timings]) / len(timings)
    vad = sum([timing['after_vad'] - timing['before_vad'] for timing in timings]) / len(timings)
    aligner = sum([timing['after_aligner'] - timing['before_aligner'] for timing in timings]) / len(timings)
    dit = sum([timing['after_dit'] - timing['before_dit'] for timing in timings]) / len(timings)
    start_to_before_asr = sum([timing['before_asr'] - timing['start'] for timing in timings]) / len(timings)
    after_uvr_to_before_vad = sum([- timing['after_uvr'] + timing['before_vad'] for timing in timings]) / len(timings)
    after_dit_to_end = sum([timing['end'] - timing['after_dit'] for timing in timings]) / len(timings)
    print(f"asr: {asr:.4f}, uvr: {uvr:.4f}, vad: {vad:.4f}, aligner: {aligner:.4f}, dit: {dit:.4f}, start_to_before_asr: {start_to_before_asr:.4f}, after_uvr_to_before_vad: {after_uvr_to_before_vad:.4f}, after_dit_to_end: {after_dit_to_end:.4f}")

if __name__ == '__main__':
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv

        load_dotenv('.env.local')

    kill_void()

    try:
        set_start_method('spawn')  # 多进程启动方式，Linux/Windows 通用
    except RuntimeError:
        pass

    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to YAML config", type=str,
                        default='egs/tts/inference/swan_bench_caption_1spk.yaml')
    parser.add_argument("--dit_ckpt", help="Path to model", type=str,
                        default='checkpoints/260416_speechedit_alldata')
    parser.add_argument("--merge_ckpt", help="Path to merge model", type=str)
    parser.add_argument("--merge_weight", help="Weight to merge model", type=float)
    parser.add_argument("--vae_ckpt", help="Path to VAE ckpt", type=str,
                        default='checkpoints/251120_wavvae_v4_unfreeze')
    args = parser.parse_args()
    # 读取 config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_path = f'infer_out/speech_edit/260420/{os.path.basename(args.dit_ckpt)}_cfg[1.5,3.0]_step31000'

    ads_infer(args, cfg, out_path)
