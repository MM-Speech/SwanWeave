import argparse
from pathlib import Path
from typing import Iterable, Tuple, Optional

import torch
import torchaudio
import torchaudio.functional as AF

from utils.audiotools import AudioSignal
from modules.codec.dac.model.dac import DAC


def load_audio_any(path: Path, mono: bool = True) -> Tuple[torch.Tensor, int]:
    """
    返回: (waveform[C, T], sample_rate)
    依赖 torchaudio 对该格式的支持（m4a 往往需要 ffmpeg backend）。
    """
    wav, sr = torchaudio.load(str(path))  # [C, T]
    wav = wav.to(torch.float32)
    if mono and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def resample(wav_ct: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if orig_sr == new_sr:
        return wav_ct
    return AF.resample(wav_ct, orig_freq=orig_sr, new_freq=new_sr)


def save_wav(path: Path, wav_ct: torch.Tensor, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav_ct.detach().cpu(), sr)


def get_recon_wav_from_audiosignal(sig: AudioSignal) -> torch.Tensor:
    """
    兼容不同版本 audiotools：尽可能从常见属性取出音频张量。
    期望得到 [C, T]（若是 [B, C, T] 则取 batch=0）。
    """
    for attr in ("audio_data", "audio", "samples"):
        if hasattr(sig, attr):
            x = getattr(sig, attr)
            if torch.is_tensor(x):
                if x.dim() == 3:   # [B, C, T]
                    return x[0]
                if x.dim() == 2:   # [C, T]
                    return x
    raise RuntimeError("无法从 AudioSignal 中提取音频张量（请检查 audiotools 版本/属性名）。")


def iter_manifest_lines(txt_path: Path) -> Iterable[str]:
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            yield s


@torch.no_grad()
def process_one_audio(
    model: DAC,
    audio_path: Path,
    out_dir: Path,
    target_sr_model: int,
    save_sr: int,
    device: str,
    mono: bool = True,
):
    wav, sr = load_audio_any(audio_path, mono=mono)                    # [C, T] @ sr
    wav_target = resample(wav, sr, target_sr_model)                    # [C, T] @ target_sr_model

    # [G]：原音频（统一到 target_sr_model 后）再下采样到 save_sr 落盘
    wav_g = resample(wav_target, target_sr_model, save_sr)
    save_wav(out_dir / f"{audio_path.stem}[G].wav", wav_g, save_sr)

    # [P]：模型重建（输入 target_sr_model）-> 输出 target_sr_model -> 再下采样到 save_sr 落盘
    x = AudioSignal(wav_target.unsqueeze(0).to(device), target_sr_model)  # [1, C, T]
    compressed = model.compress(x, verbose=False)
    recon_sig = model.decompress(compressed, verbose=False)
    if recon_sig is None:
        raise RuntimeError("model.decompress 返回 None（请检查 DAC 版本/接口）。")

    recon_target = get_recon_wav_from_audiosignal(recon_sig)           # [C, T] @ target_sr_model（期望）
    recon_p = resample(recon_target, target_sr_model, save_sr)
    save_wav(out_dir / f"{audio_path.stem}[P].wav", recon_p, save_sr)


def is_manifest_file(p: Path) -> bool:
    # 你也可以改成更严格的判断方式（例如要求 .txt 且文件内容像路径列表）
    return p.is_file() and p.suffix.lower() in {".txt", ".manifest"}


def run(
    input_path: Path,
    output_dir: Path,
    weights: str,
    target_sr_model: int,
    save_sr: int,
    device: str,
    mono: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型（只加载一次）
    model = DAC.load(weights).to(device).eval()

    if is_manifest_file(input_path):
        out_dir = output_dir / input_path.stem  # output_dir/{manifest_basename}
        out_dir.mkdir(parents=True, exist_ok=True)

        log_path = out_dir / "_process.log"
        lines = list(iter_manifest_lines(input_path))

        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"manifest={input_path}\n")
            log.write(f"out_dir={out_dir}\n")
            log.write(f"device={device}\n")
            log.write(f"weights={weights}\n")
            log.write(f"target_sr_model={target_sr_model}\n")
            log.write(f"save_sr={save_sr}\n")
            log.write(f"mono={mono}\n")
            log.write(f"total_lines={len(lines)}\n")

            for i, p in enumerate(lines, start=1):
                audio_path = Path(p)
                try:
                    if not audio_path.exists():
                        raise FileNotFoundError(str(audio_path))
                    process_one_audio(
                        model=model,
                        audio_path=audio_path,
                        out_dir=out_dir,
                        target_sr_model=target_sr_model,
                        save_sr=save_sr,
                        device=device,
                        mono=mono,
                    )
                except Exception as e:
                    log.write(f"[ERROR] line={i} path={p} err={repr(e)}\n")
    else:
        # 单文件：输出直接在 output_dir 下
        audio_path = input_path
        if not audio_path.exists():
            raise FileNotFoundError(str(audio_path))
        process_one_audio(
            model=model,
            audio_path=audio_path,
            out_dir=output_dir,
            target_sr_model=target_sr_model,
            save_sr=save_sr,
            device=device,
            mono=mono,
        )


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("DAC batch/single inference: save [G] and [P]")
    ap.add_argument("--input", type=str, required=True, help="单个音频路径 或 manifest txt 路径")
    ap.add_argument("--output_dir", type=str, required=True, help="输出根目录")
    ap.add_argument("--weights", type=str, required=True, help="DAC 权重路径")
    ap.add_argument("--target_sr_model", type=int, default=44100, help="模型期望采样率（如 44100）")
    ap.add_argument("--save_sr", type=int, default=24000, help="落盘采样率（如 24000）")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--mono", action="store_true", help="强制转单声道（默认 False）")
    return ap


def main():
    args = build_argparser().parse_args()
    run(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        weights=args.weights,
        target_sr_model=args.target_sr_model,
        save_sr=args.save_sr,
        device=args.device,
        mono=args.mono,
    )
    print("Done.")


if __name__ == "__main__":
    # 可选：避免某些环境下 CPU 线程过多
    # torch.set_num_threads(4)
    main()
