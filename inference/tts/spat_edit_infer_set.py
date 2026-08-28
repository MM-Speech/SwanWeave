"""
统一的空间音频编辑推理入口。

本模块支持一次读取一个或多个 JSONL 配置，也支持直接编辑单个 FOA 音频。每条记录中的
多条编辑指令都从同一个原始音频独立推理，输出文件使用跨配置统一序号和指令序号命名。
模型配置会从 checkpoint 所在目录自动发现，DiT 权重始终使用 strict=True 加载。

单条音频编辑示例：
python inference/tts/spat_edit_infer_set.py \
  --src_wav /path/to/origin/foa.wav \
  --instruction "Move the sound to the left." \
  --instruction "Move the sound farther away." \
  --ckpt checkpoints/260523_edit_latent_from_base_3 \
  --out_dir users/infer_out/single_edit

一个或多个 JSONL 配置编辑示例：
python inference/tts/spat_edit_infer_set.py \
  --config /path/to/set_a.jsonl /path/to/set_b.jsonl \
  --ckpt checkpoints/260523_edit_latent_from_base_3 \
  --out_dir users/infer_out/config_edit
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams, set_hparams


@dataclass(frozen=True)
class InputRecord:
    global_index: int
    config_name: str
    config_path: Path | None
    line_number: int
    src_wav: Path
    instructions: tuple[str, ...]


@dataclass(frozen=True)
class InferenceJob:
    global_index: int
    instruction_index: int
    config_name: str
    config_path: Path | None
    line_number: int
    src_wav: Path
    instruction: str
    out_path: Path
    seed: int


def safe_slug(text: str, max_length: int = 48) -> str:
    slug = re.sub(r"[^\w]+", "_", text.strip().lower(), flags=re.UNICODE).strip("_")
    return slug[:max_length].rstrip("_") or "instruction"


def build_output_path(record: InputRecord, instruction_index: int, out_dir: Path) -> Path:
    instruction = record.instructions[instruction_index]
    filename = "_".join(
        [
            f"{record.global_index:06d}",
            safe_slug(record.src_wav.stem),
            safe_slug(record.config_name),
            safe_slug(instruction),
            str(instruction_index),
        ]
    )
    return Path(out_dir) / f"{filename}.wav"


def build_jobs(records: Sequence[InputRecord], out_dir: Path, base_seed: int) -> list[InferenceJob]:
    jobs: list[InferenceJob] = []
    for record in records:
        for instruction_index, instruction in enumerate(record.instructions):
            jobs.append(
                InferenceJob(
                    global_index=record.global_index,
                    instruction_index=instruction_index,
                    config_name=record.config_name,
                    config_path=record.config_path,
                    line_number=record.line_number,
                    src_wav=record.src_wav,
                    instruction=instruction,
                    out_path=build_output_path(record, instruction_index, out_dir),
                    seed=int(base_seed) + record.global_index * 1000 + instruction_index,
                )
            )
    return jobs


def resolve_model_config(ckpt: Path) -> Path:
    ckpt = Path(ckpt).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    config_path = (ckpt.parent if ckpt.is_file() else ckpt) / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found beside checkpoint: {config_path}")
    return config_path.resolve()


def load_dit_checkpoint(model, ckpt: Path, use_ema: bool) -> None:
    load_ckpt(
        model,
        str(ckpt),
        "ema_model" if use_ema else "dit",
        force=True,
        strict=True,
        delete_unmatch=False,
    )


def load_records(config_paths: Sequence[Path]) -> list[InputRecord]:
    records: list[InputRecord] = []
    for raw_config_path in config_paths:
        config_path = Path(raw_config_path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"JSONL config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {config_path}:{line_number}: {exc}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"Expected an object at {config_path}:{line_number}")

                src_value = item.get("src_wav")
                instructions_value = item.get("instructions")
                if not isinstance(src_value, str) or not src_value.strip():
                    raise ValueError(f"src_wav must be a non-empty string at {config_path}:{line_number}")
                if (
                    not isinstance(instructions_value, list)
                    or not instructions_value
                    or any(not isinstance(value, str) or not value.strip() for value in instructions_value)
                ):
                    raise ValueError(
                        f"instructions must be a list of non-empty strings at {config_path}:{line_number}"
                    )
                instructions = tuple(value.strip() for value in instructions_value)

                src_wav = Path(src_value).expanduser()
                if not src_wav.is_absolute():
                    src_wav = config_path.parent / src_wav
                src_wav = src_wav.resolve()
                if not src_wav.is_file():
                    raise FileNotFoundError(f"Source wav not found at {config_path}:{line_number}: {src_wav}")
                records.append(
                    InputRecord(
                        global_index=len(records),
                        config_name=config_path.stem,
                        config_path=config_path,
                        line_number=line_number,
                        src_wav=src_wav,
                        instructions=instructions,
                    )
                )
    if not records:
        raise ValueError("No inference records found in the JSONL configs")
    return records


def make_single_record(src_wav: Path, instructions: Sequence[str]) -> list[InputRecord]:
    src_wav = Path(src_wav).expanduser().resolve()
    if not src_wav.is_file():
        raise FileNotFoundError(f"Source wav not found: {src_wav}")
    cleaned = tuple(value.strip() for value in instructions)
    if not cleaned or any(not value for value in cleaned):
        raise ValueError("Single-audio mode requires at least one non-empty --instruction")
    return [InputRecord(0, "single", None, 0, src_wav, cleaned)]


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def pad_last_dim(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    if tensor.shape[-1] >= target_len:
        return tensor[..., :target_len]
    return F.pad(tensor, (0, target_len - tensor.shape[-1]))


def pad_seq_dim(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    if tensor.shape[1] >= target_len:
        return tensor[:, :target_len]
    return F.pad(tensor, (0, 0, 0, target_len - tensor.shape[1]))


def encode_audio(vae, audio: torch.Tensor, chunked: bool, chunk_size: int, overlap: int) -> torch.Tensor:
    if hasattr(vae, "encode_audio"):
        if not chunked or audio.shape[-1] <= chunk_size * int(vae.downsampling_ratio):
            return vae.encode_audio(audio, chunked=False)
        return vae.encode_audio(audio, chunked=True, chunk_size=chunk_size, overlap=overlap)
    encoded = vae.encode(audio)
    latent_dist = getattr(encoded, "latent_dist", None)
    if latent_dist is None:
        return getattr(encoded, "latents", encoded[0] if isinstance(encoded, tuple) else encoded)
    return latent_dist.mode() if hparams.get("vae_deterministic", True) else latent_dist.sample()


def decode_audio(vae, latent: torch.Tensor, chunked: bool, chunk_size: int, overlap: int) -> torch.Tensor:
    if hasattr(vae, "decode_audio"):
        if not chunked or latent.shape[-1] <= chunk_size:
            return vae.decode_audio(latent, chunked=False)
        return vae.decode_audio(latent, chunked=True, chunk_size=chunk_size, overlap=overlap)
    decoded = vae.decode(latent)
    return getattr(decoded, "sample", decoded[0] if isinstance(decoded, tuple) else decoded)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_foa_wav(path: Path, target_sr: int) -> torch.Tensor:
    import torchaudio

    wav, sample_rate = torchaudio.load(str(path))
    if sample_rate != target_sr:
        wav = torchaudio.functional.resample(wav.float(), sample_rate, target_sr)
    wav = wav.float().transpose(0, 1).contiguous()
    if wav.ndim != 2 or wav.shape[1] < 4:
        raise ValueError(f"Expected at least 4 FOA channels, got {tuple(wav.shape)} from {path}")
    wav = wav[:, :4]
    peak = float(wav.abs().max()) if wav.numel() else 0.0
    return wav / peak if peak > 1.0 else wav


def save_foa_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    if not np.isfinite(wav).all():
        raise ValueError("Inference produced NaN or Inf waveform")
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak * 0.99
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sample_rate, subtype="PCM_16")


@dataclass(frozen=True)
class SourceFeatures:
    latent: torch.Tensor
    duration_sec: float


class SpatEditSetInfer:
    def __init__(self, ckpt: Path, device: str, precision: str, use_ema: bool):
        from modules.tts.spat_edit.build_model_utils import DiTBuildModelMixin

        self.device = torch.device(device)
        self.precision = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        }[precision]
        config_path = resolve_model_config(ckpt)
        set_hparams(config=str(config_path), print_hparams=False, global_hparams=True)
        if bool(hparams.get("train_base", True)):
            raise ValueError(f"Expected an edit config with train_base=false: {config_path}")
        hparams["exp_name"] = "spat_edit_infer_set"
        hparams["use_fsdp"] = False

        class ModelBuilder(DiTBuildModelMixin):
            pass

        self.model = ModelBuilder()
        self.model.config = AttrDict(hparams)
        self.model.trainer = SimpleNamespace(device=self.device)
        self.model._build_model(attn_implementation=hparams.get("attn_implementation", "sdpa"))
        load_dit_checkpoint(self.model.dit, ckpt, use_ema)

        self.model.vae.eval().to(self.device)
        self.vae_dtype = next(self.model.vae.parameters()).dtype
        self.model.dit.eval().to(self.device, dtype=self.precision)
        if getattr(self.model, "goku_text_encoder", None) is None:
            raise RuntimeError("Caption encoder is not built; check use_caption in config.yaml")
        self.model.goku_text_encoder.eval().to(self.device, dtype=self.precision)
        self.input_sample_rate = int(hparams["audio_sample_rate"])
        self.output_sample_rate = int(self.model.vae_sample_rate)

    def _preprocess_pair(self, audio: torch.Tensor) -> torch.Tensor:
        vae = self.model.vae
        if hasattr(vae, "preprocess_audio_list_for_encoder"):
            return vae.preprocess_audio_list_for_encoder(
                [audio[index].float() for index in range(audio.shape[0])],
                self.input_sample_rate,
            )
        if self.input_sample_rate != self.output_sample_rate:
            import torchaudio

            audio = torchaudio.functional.resample(audio.float(), self.input_sample_rate, self.output_sample_rate)
        pad = (-audio.shape[-1]) % int(self.model.vae_downsampling_ratio)
        return F.pad(audio.float(), (0, pad)) if pad else audio.float()

    @torch.no_grad()
    def prepare_source(self, src_wav: Path) -> SourceFeatures:
        wav = load_foa_wav(src_wav, self.input_sample_rate)
        duration_sec = wav.shape[0] / self.input_sample_rate
        batch = wav.unsqueeze(0).to(self.device)
        pairs = hparams.get("stable_audio_foa_pair_a", [0, 1]), hparams.get(
            "stable_audio_foa_pair_b", [2, 3]
        )
        chunked = bool(hparams.get("stable_audio_vae_chunked", True))
        chunk_size = int(hparams.get("stable_audio_vae_chunk_size", 128))
        overlap = int(hparams.get("stable_audio_vae_overlap", 32))
        latents = []
        for channel_pair in pairs:
            stereo = batch[:, :, list(channel_pair)].transpose(1, 2).contiguous()
            stereo = self._preprocess_pair(stereo).to(self.device, dtype=self.vae_dtype)
            latents.append(encode_audio(self.model.vae, stereo, chunked, chunk_size, overlap))
        latent_len = min(latent.shape[-1] for latent in latents)
        latent = torch.cat([pad_last_dim(value, latent_len) for value in latents], dim=1)
        return SourceFeatures(latent.transpose(1, 2).contiguous().float(), duration_sec)

    @torch.no_grad()
    def _text_embedding(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.model.goku_tokenizer(
            [instruction],
            padding=False,
            truncation=True,
            max_length=hparams.get("text_max_token_length", 256),
            return_tensors="pt",
        )
        attention = tokens.attention_mask.to(self.device, dtype=torch.long)
        embedding = self.model.goku_text_encoder(
            input_ids=tokens.input_ids.to(self.device, dtype=torch.long),
            attention_mask=attention,
            return_dict=False,
        )[0]
        return embedding * attention[..., None], attention.sum(-1).long()

    @torch.no_grad()
    def infer_instruction(
        self,
        source: SourceFeatures,
        instruction: str,
        out_path: Path,
        target_latent_len: int,
        num_steps: int,
        caption_cfg: float,
        source_cfg: float,
        use_amo_sampler: bool,
        use_sway: bool,
        match_src_duration: bool,
    ) -> dict[str, Any]:
        latent_len = target_latent_len if target_latent_len > 0 else source.latent.shape[1]
        src_latent = pad_seq_dim(source.latent, int(latent_len)).to(self.device).float()
        positive, positive_len = self._text_embedding(instruction)
        zero_source = torch.zeros_like(src_latent)
        inputs = {
            "caption_emb": torch.cat([positive, positive, torch.zeros_like(positive)], dim=0),
            "caption_lens": torch.cat([positive_len, positive_len, positive_len], dim=0),
            "tgt_len": torch.full((3,), int(latent_len), dtype=torch.long, device=self.device),
            "src_lat": torch.cat([src_latent, zero_source, zero_source], dim=0),
        }
        autocast = self.device.type == "cuda" and self.precision in (torch.float16, torch.bfloat16)
        with torch.autocast(self.device.type, dtype=self.precision, enabled=autocast):
            edited = self.model.dit.inference(
                inputs,
                timesteps=int(num_steps),
                seq_cfg_w=(float(caption_cfg), float(source_cfg)),
                timestep_annealing_w=(0.6, 0.6, 1.0),
                use_amo_sampler=bool(use_amo_sampler),
                use_sway=bool(use_sway),
            ).float()
        wav = self._decode_foa(edited)[0].cpu().numpy()
        if match_src_duration:
            wav = wav[: int(round(source.duration_sec * self.output_sample_rate))]
        save_foa_wav(out_path, wav, self.output_sample_rate)
        return {
            "sample_rate": self.output_sample_rate,
            "num_samples": int(wav.shape[0]),
            "num_channels": int(wav.shape[1]),
            "duration_sec": wav.shape[0] / self.output_sample_rate,
            "target_latent_len": int(latent_len),
        }

    @torch.no_grad()
    def _decode_foa(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[-1] != 128:
            raise ValueError(f"Expected latent [B, L, 128], got {tuple(latent.shape)}")
        chunked = bool(hparams.get("stable_audio_vae_chunked", True))
        chunk_size = int(hparams.get("stable_audio_vae_chunk_size", 128))
        overlap = int(hparams.get("stable_audio_vae_overlap", 32))
        decoded = []
        for value in (latent[:, :, :64], latent[:, :, 64:]):
            value = value.transpose(1, 2).contiguous().to(self.device, dtype=self.vae_dtype)
            decoded.append(decode_audio(self.model.vae, value, chunked, chunk_size, overlap).float())
        wav_len = min(value.shape[-1] for value in decoded)
        return torch.cat([value[..., :wav_len] for value in decoded], dim=1).transpose(1, 2).contiguous()


def job_payload(job: InferenceJob) -> dict[str, Any]:
    return {
        "global_index": job.global_index,
        "instruction_index": job.instruction_index,
        "config_name": job.config_name,
        "config_path": str(job.config_path) if job.config_path else None,
        "line_number": job.line_number,
        "src_wav": str(job.src_wav),
        "instruction": job.instruction,
        "out_path": str(job.out_path),
        "seed": job.seed,
    }


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace, records: Sequence[InputRecord]) -> None:
    out_dir = args.out_dir.expanduser().resolve()
    jobs = build_jobs(records, out_dir, args.seed)
    existing = [job.out_path for job in jobs if job.out_path.exists()]
    if existing and not (args.skip_existing or args.overwrite):
        raise FileExistsError(f"Output exists (use --skip_existing or --overwrite): {existing[0]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "plan.jsonl", [job_payload(job) for job in jobs])
    if args.dry_run:
        print(f"planned jobs: {len(jobs)}")
        print(f"plan: {out_dir / 'plan.jsonl'}")
        return

    infer = SpatEditSetInfer(args.ckpt, args.device, args.precision, args.use_ema)
    jobs_by_record = {
        record.global_index: [job for job in jobs if job.global_index == record.global_index] for record in records
    }
    summary_path = out_dir / "summary.jsonl"
    results: list[dict[str, Any]] = []
    with summary_path.open("w", encoding="utf-8") as summary:
        def record_result(result: dict[str, Any]) -> None:
            results.append(result)
            summary.write(json.dumps(result, ensure_ascii=False) + "\n")
            summary.flush()

        for record in records:
            record_jobs = jobs_by_record[record.global_index]
            pending = [job for job in record_jobs if not (args.skip_existing and job.out_path.exists())]
            for job in record_jobs:
                if job not in pending:
                    record_result({"status": "skipped", **job_payload(job)})
            if not pending:
                continue
            try:
                set_seed(args.seed + record.global_index * 1000)
                source = infer.prepare_source(record.src_wav)
            except Exception as exc:
                for job in pending:
                    record_result({"status": "error", **job_payload(job), "error": repr(exc)})
                if not args.continue_on_error:
                    raise
                continue

            for job in pending:
                print(f"[{len(results) + 1}/{len(jobs)}] {job.out_path.name}", flush=True)
                try:
                    set_seed(job.seed)
                    infer_result = infer.infer_instruction(
                        source,
                        job.instruction,
                        job.out_path,
                        args.target_latent_len,
                        args.num_steps,
                        args.caption_cfg,
                        args.source_cfg,
                        args.use_amo_sampler,
                        not args.no_sway,
                        not args.no_match_src_duration,
                    )
                    record_result({"status": "ok", **job_payload(job), "infer_result": infer_result})
                except Exception as exc:
                    record_result({"status": "error", **job_payload(job), "error": repr(exc)})
                    if not args.continue_on_error:
                        raise
    print(f"summary: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统一的空间音频编辑推理入口")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", nargs="+", type=Path, help="One or more JSONL configs")
    source.add_argument("--src_wav", type=Path, help="Single source FOA wav")
    parser.add_argument("--instruction", action="append", default=[], help="Repeat for single-audio mode")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--target_latent_len", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--caption_cfg", type=float, default=3.0)
    parser.add_argument("--source_cfg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_amo_sampler", action="store_true")
    parser.add_argument("--no_sway", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--no_match_src_duration", action="store_true")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--skip_existing", action="store_true")
    output.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.config:
        if args.instruction:
            parser.error("--instruction is only valid with --src_wav")
        records = load_records(args.config)
    else:
        records = make_single_record(args.src_wav, args.instruction)
    run(args, records)


if __name__ == "__main__":
    main()
