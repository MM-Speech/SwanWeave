"""Local, bounded-memory JSONL streaming for spatial-audio training."""
import bisect
import json
import random
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from utils.commons.dataset_utils import collate_xd
from utils.commons.import_utils import import_module_bystr


def valid_item_kv(item, key):
    return key in item and item[key] is not None


def get_from_global_stores(key, stores, factory):
    if key not in stores:
        stores[key] = factory()
    return stores[key]


class BaseDataset(IterableDataset):
    """Yield prebatched samples without Falcon, shared memory, or TTS services.

    Dataset-group weights select reader chunks, not individual samples. Chunk
    ownership stays fixed across passes so ranks/workers do not share records.
    Each consumer scans JSONL sequentially; only its own chunks are decoded.
    """
    BUCKETS = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,
               600, 650, 700, 750, 800, 850, 900, 950, 1000, 1200, 1400,
               1600, 1800, 2000, 2400, 2800, 3000, 4000, 5000, 6000,
               8000, 10000)

    def __init__(self, prefix, hparams, use_fast_dataloader=True, rank_id=0,
                 world_size=1, batch_size=1, node_id=None, node_size=None,
                 random_frame=True, **kwargs):
        super().__init__()
        self.prefix = prefix
        self.hparams = dict(hparams)
        self.use_fast_dataloader = bool(use_fast_dataloader)
        self.batch_size = max(1, int(batch_size))
        self.rank = int(rank_id or 0) + int(node_id or 0) * int(world_size)
        self.world_size = int(world_size) * int(node_size or 1)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid dataset rank/world_size")
        self.seed = int(hparams.get("dataloader_seed", 1231))
        self.iteration = 0
        self.endless = prefix == "train" and bool(hparams.get("endless_ds", True))
        self.sample_limit = int(hparams.get("stream_max_samples", 0))
        if prefix != "train":
            test_num = int(hparams.get("test_num", 100))
            if test_num > 0:
                self.sample_limit = min(self.sample_limit, test_num) if self.sample_limit > 0 else test_num
        if self.sample_limit > 0:
            if self.sample_limit < self.world_size:
                raise ValueError(
                    "Finite sample limit %d is smaller than world_size=%d; "
                    "increase test_num/stream_max_samples to give every rank a sample." %
                    (self.sample_limit, self.world_size)
                )
            # Equal rank budgets keep distributed validation collectives aligned.
            self.sample_limit = self.sample_limit // self.world_size * self.world_size
        self.groups = []
        for name, spec in hparams.get("datasets", {}).items():
            if name.startswith("__") or not isinstance(spec, dict):
                continue
            weight = float(spec.get("weight", 1.0))
            if weight <= 0:
                continue
            split_key = "train_sets" if prefix == "train" else prefix + "_sets"
            paths = spec.get(split_key)
            if paths is None and prefix == "val":
                paths = spec.get("valid_sets")
            if paths is None:
                paths = spec.get("train_sets", [])
                if prefix != "train" and paths:
                    warnings.warn("No validation manifests configured; using bounded training samples for smoke validation.")
            if isinstance(paths, str):
                paths = [paths]
            if not paths:
                continue
            paths = [Path(path).expanduser() for path in paths]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError("Dataset manifest does not exist: " + str(path))
                if path.suffix.lower() != ".jsonl":
                    raise ValueError("Streaming datasets require JSONL manifests: " + str(path))
            chunk_size = int(spec.get("reader_chunk_size", 32))
            if chunk_size <= 0:
                raise ValueError("reader_chunk_size must be positive")
            self.groups.append(dict(name=name, paths=paths, weight=weight,
                                    chunk_size=chunk_size,
                                    processor=import_module_bystr(spec["processer_fn"])))
        if not self.groups:
            raise ValueError("No active local JSONL dataset groups configured")

    def get_dataloader(self, seed=1231, num_workers=None, **kwargs):
        self.seed = int(seed)
        workers = int(num_workers if num_workers is not None else 0)
        options = dict(dataset=self, batch_size=None, collate_fn=self.collater,
                       num_workers=workers, pin_memory=bool(self.hparams.get("pin_memory", False)),
                       generator=torch.Generator().manual_seed(self.seed))
        if workers > 0:
            options["prefetch_factor"] = max(1, int(self.hparams.get("fast_ds_prefetch_steps", 2)))
        return DataLoader(**options)

    def _owned_chunks(self, group, shard, shards):
        chunk_index = 0
        for path in group["paths"]:
            chunk = []
            with path.open("r", encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    chunk.append((line_no, line))
                    if len(chunk) == group["chunk_size"]:
                        if chunk_index % shards == shard:
                            yield self._decode_chunk(chunk, path)
                        chunk_index += 1
                        chunk = []
            if chunk:
                if chunk_index % shards == shard:
                    yield self._decode_chunk(chunk, path)
                chunk_index += 1

    @staticmethod
    def _decode_chunk(chunk, path):
        rows = []
        for line_no, line in chunk:
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("Invalid JSON in %s:%d" % (path, line_no)) from error
            if not isinstance(row, dict):
                raise ValueError("Expected JSON object in %s:%d" % (path, line_no))
            rows.append(row)
        return rows

    def _group_chunks(self, group, shard, shards, rng):
        # A bounded buffer changes chunk order without indexing or loading the manifest.
        buffer_size = max(1, int(self.hparams.get("stream_shuffle_chunks", 8))) if self.prefix == "train" else 1
        buffer = []
        for chunk in self._owned_chunks(group, shard, shards):
            buffer.append(chunk)
            if len(buffer) >= buffer_size:
                yield buffer.pop(rng.randrange(len(buffer)))
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))

    def __iter__(self):
        info = get_worker_info()
        worker_id, workers = (info.id, info.num_workers) if info else (0, 1)
        shard, shards = self.rank * workers + worker_id, self.world_size * workers
        rng = random.Random(self.seed + 1009 * shard + 100003 * self.iteration)
        self.iteration += 1
        quota = None
        if self.sample_limit > 0:
            rank_limit = self.sample_limit // self.world_size
            quota = max(0, (rank_limit - worker_id + workers - 1) // workers)
            if quota == 0:
                return
        states = [dict(group=group, iterator=self._group_chunks(group, shard, shards, rng),
                       chunks=0) for group in self.groups]
        stores, buckets = {}, {}
        accepted = empty_chunks = 0
        saw_chunk = False
        empty_limit = max(1, int(self.hparams.get("stream_max_empty_chunks", 32)))
        max_sentences = max(1, int(self.hparams.get("max_sentences", 1))) if self.use_fast_dataloader else self.batch_size
        max_tokens = max(1, int(self.hparams.get("max_tokens", 3000)))
        dynamic = self.use_fast_dataloader and bool(self.hparams.get("dynamic_batch", True))
        while states and (quota is None or accepted < quota):
            state = rng.choices(states, weights=[state["group"]["weight"] for state in states])[0]
            try:
                chunk = next(state["iterator"])
            except StopIteration:
                if self.endless and state["chunks"]:
                    state["iterator"] = self._group_chunks(state["group"], shard, shards, rng)
                    state["chunks"] = 0
                else:
                    states.remove(state)
                continue
            state["chunks"] += 1
            saw_chunk = True
            if self.prefix == "train":
                rng.shuffle(chunk)
            tgt_size = rng.randint(int(self.hparams.get("tgt_size_min", 100)),
                                   int(self.hparams.get("tgt_size_max", 6000)))
            chunk_valid = 0
            for item in self._process_item(state["group"]["processor"], chunk, tgt_size,
                                           self.hparams, stores, shard, shards):
                if item is None:
                    continue
                chunk_valid += 1
                accepted += 1
                length = max(1, int(item["len"]))
                bucket_id = bisect.bisect_left(self.BUCKETS, length) if self.use_fast_dataloader else 0
                bucket = buckets.setdefault(bucket_id, [])
                if bucket and (len(bucket) >= max_sentences or
                               (dynamic and (len(bucket) + 1) * max(length, max(int(x["len"]) for x in bucket)) > max_tokens)):
                    yield bucket
                    bucket = buckets[bucket_id] = []
                bucket.append(item)
                if len(bucket) >= max_sentences or (dynamic and length >= max_tokens):
                    yield bucket
                    buckets[bucket_id] = []
                if quota is not None and accepted >= quota:
                    break
            empty_chunks = empty_chunks + 1 if not chunk_valid else 0
            if empty_chunks >= empty_limit:
                raise RuntimeError("No valid spatial samples in %d consecutive chunks; check audio/latent paths, captions and frame limits." % empty_chunks)
        if not saw_chunk:
            raise RuntimeError(
                "No JSONL chunks assigned to shard %d/%d (rank=%d, worker=%d/%d); "
                "reduce workers/ranks or add more data." %
                (shard, shards, self.rank, worker_id, workers)
            )
        if saw_chunk and accepted == 0:
            raise RuntimeError("No valid spatial samples found in configured manifests")
        for bucket in buckets.values():
            if bucket:
                yield bucket

    def collater(self, samples):
        if samples and isinstance(samples[0], list):
            samples = samples[0]
        if not samples:
            return {}
        batch = {"item_name": [sample["item_name"] for sample in samples],
                 "nsamples": len(samples)}
        if valid_item_kv(samples[0], "wav"):
            batch["wavs"] = collate_xd([sample["wav"] for sample in samples], 0.0)
            batch["wav_lengths"] = torch.LongTensor([sample["wav"].shape[0] for sample in samples])
        for key in ("caption", "global", "local", "text", "orig_text"):
            if valid_item_kv(samples[0], key):
                batch[key] = [sample[key] for sample in samples]
        return batch
