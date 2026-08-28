import os
import re
import gc
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import tempfile

from langdetect import detect as classify_language, LangDetectException

from utils.commons.io import get_wav_duration
from utils.text import isPUNC
from utils.text.split_text import get_word_list
from utils.service.file_service import BaseProcessor

from data_gen.asr.pipeline import ASRPipeline


class ASRDataServerProcessor(BaseProcessor):
    def __init__(
        self,
        device: str = "cpu",
        num_workers: Optional[int] = None,
        worker_id: Optional[int] = None,
        worker_i: Optional[int] = None,
        max_batch_size: int = 0,
        max_total_duration_s: float = 0.0,
        long_single_duration_s: float = 0.0,
        long_single_max_total_duration_scale: float = 0.0,
        aligner_backend='qwen3aligner',
        attn_implementation='flash_attention_2',
        **kwargs,
    ):
        super().__init__(
            device=device,
            num_workers=num_workers,
            worker_id=worker_id,
            worker_i=worker_i,
            **kwargs,
        )

        try:
            self.max_batch_size = int(max_batch_size)
        except Exception:
            self.max_batch_size = 0
        if self.max_batch_size <= 0:
            self.max_batch_size = 0

        try:
            self.max_total_duration_s = float(max_total_duration_s)
        except Exception:
            self.max_total_duration_s = 0.0
        if self.max_total_duration_s <= 0:
            self.max_total_duration_s = 0.0

        self.long_single_duration_s = long_single_duration_s
        self.long_single_max_total_duration_scale = long_single_max_total_duration_scale

        self.aligner_backend = aligner_backend
        self.kwargs = kwargs
        self.attn_implementation = attn_implementation

    def setup(self) -> None:
        cls_init_kwargs = dict(
            device=self.device,
            attn_implementation=self.attn_implementation,
            special_token_ignore_literals=["<S1>", "</S1>", "<S2>", "</S2>", "<S3>", "</S3>", "<S4>", "</S4>"],
            restore_special_tokens=True
        )
        if self.aligner_backend == 'qwen3aligner':
            pass
        elif self.aligner_backend == 'aligner_2tower_v5':
            cls_init_kwargs.update(dict(
                punc_backend='aligner_2tower_v5',
                aligner_ckpt=self.kwargs.get('aligner_ckpt', 'checkpoints/260225_aligner_2tower_v5'),
            ))
        self.pipe = ASRPipeline(**cls_init_kwargs)
        self._job_i = 0

        if os.environ.get("ASR_SERVICE_LOG_CUDA_MEM", "").strip():
            try:
                import torch

                if isinstance(self.device, str) and self.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        print(f"ASRDataServerProcessor {self.worker_id}/{self.num_workers} setup on device {self.device}")

    def _maybe_log_cuda_mem(self, *, when: str) -> None:
        if not os.environ.get("ASR_SERVICE_LOG_CUDA_MEM", "").strip():
            return

        if not (isinstance(self.device, str) and self.device.startswith("cuda")):
            return

        try:
            import torch

            if not torch.cuda.is_available():
                return

            try:
                dev = torch.device(self.device)
                dev_index = 0 if dev.index is None else int(dev.index)
            except Exception:
                dev_index = 0

            allocated_mb = torch.cuda.memory_allocated(dev_index) / 1024 / 1024
            reserved_mb = torch.cuda.memory_reserved(dev_index) / 1024 / 1024
            max_allocated_mb = torch.cuda.max_memory_allocated(dev_index) / 1024 / 1024
            max_reserved_mb = torch.cuda.max_memory_reserved(dev_index) / 1024 / 1024

            print(
                f"[ASRDataServerProcessor cuda_mem] worker={self.worker_id} job_i={self._job_i} when={when} "
                f"allocated_mb={allocated_mb:.1f} reserved_mb={reserved_mb:.1f} "
                f"max_allocated_mb={max_allocated_mb:.1f} max_reserved_mb={max_reserved_mb:.1f} device={self.device}"
            )
        except Exception:
            return

    def _maybe_cuda_gc(self) -> None:
        every_s = os.environ.get("ASR_SERVICE_EMPTY_CACHE_EVERY", "").strip()
        if not every_s:
            return

        try:
            every = int(every_s)
        except Exception:
            return

        if every <= 0:
            return

        if (self._job_i % every) != 0:
            return

        if not (isinstance(self.device, str) and self.device.startswith("cuda")):
            return

        try:
            import torch

            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass

    def process(self, job: Dict[str, Any]) -> Dict[str, Any]:
        self._job_i += 1

        payload = job.get("payload", {})
        job_id = job.get("job_id")

        wav_bytes = payload.get("wav_bytes")
        texts = payload.get("texts")

        result = {
            "job_id": job_id,
        }

        if wav_bytes is None:
            return result
        if not isinstance(wav_bytes, list):
            wav_bytes = [wav_bytes]
        if texts is not None and not isinstance(texts, list):
            texts = [texts]

        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp_dir:
            wav_paths = []
            for wav_idx, wav_bytes_ in enumerate(wav_bytes):
                wav_path = os.path.join(temp_dir, f"{wav_idx}.wav")
                with open(wav_path, "wb") as f:
                    f.write(wav_bytes_)
                wav_paths.append(wav_path)

            self._maybe_log_cuda_mem(when="before")

            try:
                import torch

                if isinstance(self.device, str) and self.device.startswith("cuda"):
                    with torch.inference_mode():
                        asr_results = self.pipe.process(
                            audio=wav_paths,
                            text=texts,
                        )
                else:
                    asr_results = self.pipe.process(
                        audio=wav_paths,
                        text=texts,
                    )
            except Exception:
                asr_results = self.pipe.process(
                    audio=wav_paths,
                    text=texts,
                )

            self._maybe_log_cuda_mem(when="after")
            self._maybe_cuda_gc()

        result["asr_results"] = asr_results

        return result

    def _iter_minibatches(self, entries: List[Dict[str, Any]]):
        if not entries:
            return

        max_bs = int(self.max_batch_size) if getattr(self, "max_batch_size", 0) else 0
        max_dur = float(self.max_total_duration_s) if getattr(self, "max_total_duration_s", 0.0) else 0.0

        if max_bs <= 0 and max_dur <= 0:
            yield entries
            return

        # 对于“单条特别长”的样本：显存占用通常不是严格线性线性随总时长增长。
        # 因此当某条样本超过阈值时，把该 batch 的 max_total_duration_s 动态缩小。
        #
        # 可选配置（不要求一定存在；不存在时使用默认行为）：
        # - self.long_single_duration_s: 单条时长阈值（秒）。默认：max_dur * 0.5（若 max_dur>0），否则禁用。
        # - self.long_single_max_total_duration_scale: 缩放比例 (0, 1]。默认：0.5（即缩小一半）。
        long_single_duration_s = float(getattr(self, "long_single_duration_s", 0.0) or 0.0)
        if long_single_duration_s <= 0.0 and max_dur > 0.0:
            long_single_duration_s = 0.333 * max_dur

        long_scale = float(getattr(self, "long_single_max_total_duration_scale", 0.5) or 0.5)
        if not (0.0 < long_scale <= 1.0):
            long_scale = 0.5

        base_max_dur = max_dur

        cur: List[Dict[str, Any]] = []
        cur_dur = 0.0
        cur_max_dur = base_max_dur

        def reset_batch_state():
            nonlocal cur, cur_dur, cur_max_dur
            cur = []
            cur_dur = 0.0
            cur_max_dur = base_max_dur

        for ent in entries:
            ent_dur = float(ent.get("duration_s") or 0.0)

            ent_is_long = bool(long_single_duration_s > 0.0 and ent_dur > long_single_duration_s)
            next_batch_max_dur = cur_max_dur
            if ent_is_long and base_max_dur > 0.0:
                next_batch_max_dur = min(next_batch_max_dur, base_max_dur * long_scale)

            if cur:
                if max_bs > 0 and len(cur) >= max_bs:
                    yield cur
                    reset_batch_state()
                elif next_batch_max_dur > 0.0 and (cur_dur + ent_dur) > next_batch_max_dur:
                    yield cur
                    reset_batch_state()

            if not cur:
                # 重新开始一个 batch 时，基于当前样本决定该 batch 的有效 max_dur。
                cur_max_dur = base_max_dur
                if ent_is_long and base_max_dur > 0.0:
                    cur_max_dur = min(cur_max_dur, base_max_dur * long_scale)
            else:
                # 当前 batch 中加入 long sample 后，收紧本 batch 的 max_dur。
                cur_max_dur = next_batch_max_dur

            cur.append(ent)
            cur_dur += ent_dur

        if cur:
            yield cur

    def process_batch(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(jobs, list):
            raise ValueError(f"process_batch() expects list[dict], got {type(jobs)}")
        
        if self.worker_i == 0:
            print(f"{len(jobs) = }, n_wavs = {sum(len(p.get('payload', {}).get('wav_bytes', [])) for p in jobs)}")

        results: List[Optional[Dict[str, Any]]] = [None for _ in range(len(jobs))]

        prepared: List[Optional[Dict[str, Any]]] = []
        for job in jobs:
            self._job_i += 1

            payload = (job or {}).get("payload", {})
            job_id = (job or {}).get("job_id")

            wav_bytes = payload.get("wav_bytes")
            texts = payload.get("texts")
            durations = payload.get("durations")

            if wav_bytes is None:
                prepared.append(None)
                continue

            if not isinstance(wav_bytes, list):
                wav_bytes = [wav_bytes]

            if texts is not None and not isinstance(texts, list):
                texts = [texts]

            # Treat empty list as "not provided".
            if texts is not None and len(texts) == 0:
                texts = None

            # If text is provided, ensure per-audio alignment.
            if texts is not None:
                if len(texts) == 1 and len(wav_bytes) > 1:
                    texts = [texts[0] for _ in range(len(wav_bytes))]
                if len(texts) != len(wav_bytes):
                    raise ValueError(
                        f"texts length mismatch: job_id={job_id} len(texts)={len(texts)} len(wav_bytes)={len(wav_bytes)}"
                    )
                texts = ["" if t is None else str(t) for t in texts]

            prepared.append({"job_id": job_id, "wav_bytes": wav_bytes, "texts": texts, "durations": durations})

        if all(p is None for p in prepared):
            for i, job in enumerate(jobs):
                results[i] = {"job_id": (job or {}).get("job_id")}
            return results  # type: ignore

        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp_dir:
            groups: Dict[bool, List[int]] = {False: [], True: []}
            for i, p in enumerate(prepared):
                if p is None:
                    continue
                groups[bool(p.get("texts") is not None)].append(i)

            self._maybe_log_cuda_mem(when="before")

            for has_text, job_indices in groups.items():
                if not job_indices:
                    continue

                entries: List[Dict[str, Any]] = []
                for job_idx in job_indices:
                    p = prepared[job_idx]
                    assert p is not None

                    wav_list = p["wav_bytes"]
                    texts_list = p.get("texts") if has_text else None
                    durations_list = p.get("durations")

                    for wav_i, wav_bytes_ in enumerate(wav_list):
                        wav_path = os.path.join(temp_dir, f"job{job_idx}_wav{wav_i}.wav")
                        with open(wav_path, "wb") as f:
                            f.write(wav_bytes_)

                        if durations_list is not None and len(durations_list) > 0 and wav_i < len(durations_list) and durations_list[wav_i] is not None:
                            duration_s = float(durations_list[wav_i])
                        else:
                            duration_s = get_wav_duration(wav_path)
                        ent: Dict[str, Any] = {
                            "job_idx": job_idx,
                            "audio_path": wav_path,
                            "duration_s": duration_s,
                        }
                        if has_text:
                            assert texts_list is not None
                            ent["text"] = texts_list[wav_i]
                        entries.append(ent)

                per_job_lists: Dict[int, Dict[str, List[Any]]] = {idx: {} for idx in job_indices}
                group_nonlist: Dict[str, Any] = {}

                for batch_entries in self._iter_minibatches(entries):
                    audio_paths = [e["audio_path"] for e in batch_entries]
                    batch_texts: Optional[List[str]] = None
                    if has_text:
                        batch_texts = [str(e.get("text") or "") for e in batch_entries]

                    try:
                        import torch

                        if isinstance(self.device, str) and self.device.startswith("cuda"):
                            with torch.inference_mode():
                                pipe_out = self.pipe.process(audio=audio_paths, text=batch_texts)
                        else:
                            pipe_out = self.pipe.process(audio=audio_paths, text=batch_texts)
                    except Exception:
                        pipe_out = self.pipe.process(audio=audio_paths, text=batch_texts)

                    if not isinstance(pipe_out, dict):
                        raise ValueError(f"ASRPipeline.process must return dict, got {type(pipe_out)}")

                    for k, v in pipe_out.items():
                        if isinstance(v, list):
                            if len(v) != len(batch_entries):
                                raise ValueError(
                                    f"ASRPipeline output list length mismatch: key={k} len(v)={len(v)} batch={len(batch_entries)}"
                                )
                            for i_item, item in enumerate(v):
                                j = int(batch_entries[i_item]["job_idx"])
                                per_job_lists[j].setdefault(k, []).append(item)
                        else:
                            group_nonlist[k] = v

                for job_idx in job_indices:
                    p = prepared[job_idx]
                    assert p is not None

                    out: Dict[str, Any] = dict(group_nonlist)
                    for k, lst in per_job_lists[job_idx].items():
                        out[k] = lst

                    results[job_idx] = {
                        "job_id": p.get("job_id"),
                        "asr_results": out,
                    }

            self._maybe_log_cuda_mem(when="after")
            self._maybe_cuda_gc()

        for i, r in enumerate(results):
            if r is None:
                results[i] = {"job_id": (jobs[i] or {}).get("job_id")}

        return results  # type: ignore


if __name__ == "__main__":
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')

    pipe = ASRPipeline(device='cuda')

    out = pipe.process(
        audio="infer_out/tts_dialogue/infer_once/260108_prompt_base_dialogue/step84000/0-我家这餐馆，菜味儿正宗分量也足，怎么饭点.wav",
        text="<S1>我家这餐馆，菜味儿正宗分量也足，怎么饭点总坐不满啊？ </S1><S2>你试过做抖音团购套餐吗？ </S2><S1>抖音团购？哎呦，那是不是得花钱推广，还怕没人买亏了本啊？ </S1><S2>不用花推广费，套餐定价灵活还能吸引新客。 </S2><S1>真不用花钱？这靠谱吗？ </S1><S2>现在点击视频下方链接，就能零元上架抖音团购套餐，官方还会给本地流量推荐，帮你把附近想吃的人都引过来。像双人餐、家庭餐都能做，还能设置到店核销，不怕跑单。 </S2><S1>这听着行啊，那团购套餐咋设计啊？ </S1><S2>平台有现成的套餐模板，你按自家招牌菜搭配就行，还能自动生成图文海报，就算不会做宣传，用户刷到直接就能下单，用完都说好还会带朋友来。 </S2><S1>那我现在就去弄这个团购套餐！ </S1><S2>你们做餐饮的也赶紧点下方链接，上架抖音团购，让店里天天满座！</S2>",
        special_token_ignore_literals=["<S1>", "</S1>", "<S2>", "</S2>", "<S3>", "</S3>", "<S4>", "</S4>"],
        restore_special_tokens=True
    )

    print(out)


# ASR_SERVICE_LOG_CUDA_MEM=1 ASR_SERVICE_EMPTY_CACHE_EVERY=1000 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python utils/service/file_service/cli.py serve --config data_gen/services/asr/base_asr.yaml
