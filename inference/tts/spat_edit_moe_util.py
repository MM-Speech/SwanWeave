"""
SE-MoE 路由与专家利用率分析。

本脚本针对测试集统计两类信息：全部三条改写指令的 task-expert 路由，以及第一条
指令在 20 步编辑推理中 source+instruction CFG 分支的 routed/null 路由。脚本不保存
生成音频，只保存逐样本 JSONL、汇总 JSON/CSV 和三联图，支持多 GPU 分片与断点续跑。

单卡收集：
  python inference/tts/spat_edit_moe_util.py --phase collect --device cuda

八卡分片（每张卡分别设置 CUDA_VISIBLE_DEVICES 和 shard_index）：
  CUDA_VISIBLE_DEVICES=0 python inference/tts/spat_edit_moe_util.py \
    --phase collect --num_shards 8 --shard_index 0 --device cuda

所有分片完成后汇总：
  python inference/tts/spat_edit_moe_util.py --phase merge --num_shards 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.tts.spat_base_infer import AttrDict, resolve_model_config_path
from inference.tts.spat_edit_infer import SpatEditInfer, load_foa_wav, pad_latent_seq_time
from users.test.edit_infer_testset import (
    captions_from_metadata,
    load_metadata,
    parse_render_script,
    sample_dirs,
)
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams, set_hparams


DEFAULT_CKPT = PROJECT_ROOT / "checkpoints/260522_edit_latent_task_moe"
DEFAULT_RENDER_SCRIPT = PROJECT_ROOT / "users/spat_sim_pyacoustic/render_test_0520.sh"
DEFAULT_OUT_DIR = PROJECT_ROOT / "users/moe_util"

EDIT_TYPE_ORDER = [
    "add_event",
    "extract_event",
    "remove_event",
    "replace_event",
    "angle_change_plane",
    "distance_change",
    "relocation_plane",
    "angle_motion",
    "distance_motion",
    "room_change",
    "volume_change",
]
EDIT_TYPE_LABELS = {
    "add_event": "Add event",
    "extract_event": "Extract event",
    "remove_event": "Remove event",
    "replace_event": "Replace event",
    "angle_change_plane": "Azimuth change",
    "distance_change": "Distance change",
    "relocation_plane": "Relocation",
    "angle_motion": "Angular motion",
    "distance_motion": "Distance motion",
    "room_change": "Room change",
    "volume_change": "Volume change",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return records


def resolve_checkpoint_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")

    model_only_last = path / "model_only_last.ckpt"
    if model_only_last.is_file():
        return model_only_last

    candidates = []
    pattern = re.compile(r"model_ckpt_steps_(\d+)\.ckpt$")
    for candidate in path.glob("model_ckpt_steps_*.ckpt"):
        match = pattern.fullmatch(candidate.name)
        if match:
            candidates.append((int(match.group(1)), candidate))
    if not candidates:
        raise FileNotFoundError(f"no model checkpoint found under {path}")
    return max(candidates, key=lambda item: item[0])[1]


class StrictMoeEditInfer(SpatEditInfer):
    """SpatEditInfer with strict checkpoint loading and MoE topology checks."""

    def __init__(self, *args, expect_task_router: bool = True, **kwargs):
        self.expect_task_router = bool(expect_task_router)
        super().__init__(*args, **kwargs)

    def build_model(self, dit_ckpt: str, config_path: str = "") -> None:
        checkpoint_file = resolve_checkpoint_file(Path(dit_ckpt))
        config_path = resolve_model_config_path(str(checkpoint_file), config_path)
        set_hparams(config=config_path, print_hparams=False, global_hparams=True)
        hparams["exp_name"] = "infer"
        hparams["use_fsdp"] = False
        self.config = AttrDict(hparams)
        self.trainer = SimpleNamespace(device=self.device)

        self._build_model(attn_implementation=hparams.get("attn_implementation", "sdpa"))
        model_name = "ema_model" if self.use_ema and hparams.get("use_ema", False) else "dit"
        load_ckpt(
            self.dit,
            str(checkpoint_file),
            model_name,
            strict=True,
            delete_unmatch=False,
        )

        self.vae.eval().to(self.device)
        self.vae_dtype = next(self.vae.parameters()).dtype
        self.dit.eval().to(self.device, dtype=self.precision)
        if getattr(self, "goku_text_encoder", None) is not None:
            self.goku_text_encoder.eval().to(self.device, dtype=self.precision)

        if bool(hparams.get("train_base", True)):
            raise ValueError("MoE utilization requires an edit checkpoint with train_base=false")
        if not bool(hparams.get("use_moe_ffn", False)):
            raise ValueError("checkpoint config has use_moe_ffn=false")
        self.task_expert_router = getattr(self.dit.encoder, "task_expert_router", None)
        has_task_router = self.task_expert_router is not None
        if has_task_router != self.expect_task_router:
            expected = "with" if self.expect_task_router else "without"
            actual = "with" if has_task_router else "without"
            raise ValueError(f"expected a model {expected} task router, loaded one {actual} task router")

        moe_layers = [layer for layer in self.dit.encoder.layers if hasattr(layer.mlp, "num_routed")]
        if not moe_layers:
            raise ValueError("checkpoint model has no routed MoE layers")
        expected_layers = int(self.dit.encoder.config.num_hidden_layers)
        if len(moe_layers) != expected_layers:
            raise ValueError(f"expected {expected_layers} MoE layers, found {len(moe_layers)}")
        self.resolved_checkpoint_file = checkpoint_file


def load_task_items(render_script: Path) -> List[Dict[str, Any]]:
    items = []
    seen = set()
    for render_set in parse_render_script(render_script):
        with render_set.metadata_jsonl.open("r", encoding="utf-8") as f:
            for row_index, line in enumerate(f):
                metadata = json.loads(line)
                edit_type = str(metadata.get("edit_type", render_set.edit_type))
                sample_id = str(metadata.get("sample_id", f"{row_index:08d}"))
                for caption_index, caption in enumerate(captions_from_metadata(metadata)):
                    key = (edit_type, sample_id, caption_index)
                    if key in seen:
                        raise ValueError(f"duplicate task item: {key}")
                    seen.add(key)
                    items.append(
                        {
                            "edit_type": edit_type,
                            "sample_id": sample_id,
                            "caption_index": caption_index,
                            "caption": caption,
                            "metadata_jsonl": str(render_set.metadata_jsonl),
                        }
                    )
    return items


def load_frame_jobs(render_script: Path) -> List[Dict[str, Any]]:
    jobs = []
    for render_set in parse_render_script(render_script):
        for sample_dir in sample_dirs(render_set.output_dir):
            metadata_path = sample_dir / "metadata.json"
            src_wav = sample_dir / "origin/foa.wav"
            if not metadata_path.is_file() or not src_wav.is_file():
                continue
            metadata = load_metadata(metadata_path)
            captions = captions_from_metadata(metadata)
            if not captions:
                continue
            jobs.append(
                {
                    "edit_type": render_set.edit_type,
                    "sample_id": sample_dir.name,
                    "caption_index": 0,
                    "caption": captions[0],
                    "src_wav": str(src_wav),
                    "metadata_path": str(metadata_path),
                }
            )
    return jobs


def shard_items(items: Sequence[Dict[str, Any]], num_shards: int, shard_index: int) -> List[Dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    result = []
    for global_index, item in enumerate(items):
        if global_index % num_shards == shard_index:
            result.append({"global_index": global_index, **item})
    return result


def normalized_mutual_information(labels: Sequence[Any], assignments: Sequence[Any]) -> float:
    """Arithmetic-mean normalized mutual information, matching sklearn's default."""
    if len(labels) != len(assignments):
        raise ValueError("labels and assignments must have equal length")
    if not labels:
        return 0.0

    joint = defaultdict(int)
    label_counts = defaultdict(int)
    assignment_counts = defaultdict(int)
    for label, assignment in zip(labels, assignments):
        joint[(label, assignment)] += 1
        label_counts[label] += 1
        assignment_counts[assignment] += 1

    total = float(len(labels))
    mutual_information = 0.0
    for (label, assignment), count in joint.items():
        p_joint = count / total
        p_label = label_counts[label] / total
        p_assignment = assignment_counts[assignment] / total
        mutual_information += p_joint * math.log(p_joint / (p_label * p_assignment))

    def entropy(counts: Iterable[int]) -> float:
        value = 0.0
        for count in counts:
            probability = count / total
            if probability > 0:
                value -= probability * math.log(probability)
        return value

    label_entropy = entropy(label_counts.values())
    assignment_entropy = entropy(assignment_counts.values())
    denominator = label_entropy + assignment_entropy
    if denominator <= 0:
        return 1.0 if list(labels) == list(assignments) else 0.0
    return float(2.0 * mutual_information / denominator)


def normalized_entropy(counts: Sequence[float]) -> float:
    values = np.asarray(counts, dtype=np.float64)
    total = float(values.sum())
    if values.size <= 1 or total <= 0:
        return 0.0
    probabilities = values / total
    nonzero = probabilities > 0
    return float(-(probabilities[nonzero] * np.log(probabilities[nonzero])).sum() / np.log(values.size))


def summarize_task_routes(records: Sequence[Dict[str, Any]], num_task_experts: int) -> Dict[str, Any]:
    valid = [record for record in records if "top1_expert" in record]
    labels = [record["edit_type"] for record in valid]
    assignments = [int(record["top1_expert"]) for record in valid]

    groups = defaultdict(list)
    for record in valid:
        groups[(record["edit_type"], record["sample_id"])].append(record)

    agreeing_pairs = 0
    total_pairs = 0
    all_agree = 0
    paraphrase_groups = 0
    for group in groups.values():
        group = sorted(group, key=lambda item: int(item["caption_index"]))
        experts = [int(item["top1_expert"]) for item in group]
        if len(experts) < 2:
            continue
        paraphrase_groups += 1
        all_agree += int(len(set(experts)) == 1)
        for left in range(len(experts)):
            for right in range(left + 1, len(experts)):
                total_pairs += 1
                agreeing_pairs += int(experts[left] == experts[right])

    top1_counts = np.bincount(assignments, minlength=num_task_experts).astype(np.int64)
    weight_rows = [record.get("weights") for record in valid if record.get("weights") is not None]
    mean_weights = (
        np.asarray(weight_rows, dtype=np.float64).mean(axis=0).tolist()
        if weight_rows
        else [0.0] * num_task_experts
    )
    return {
        "num_task_routes": len(valid),
        "num_edit_types": len(set(labels)),
        "task_label_top1_nmi": normalized_mutual_information(labels, assignments),
        "num_paraphrase_groups": paraphrase_groups,
        "paraphrase_pairwise_agreement": agreeing_pairs / total_pairs if total_pairs else 0.0,
        "paraphrase_all_agree_rate": all_agree / paraphrase_groups if paraphrase_groups else 0.0,
        "task_top1_counts": top1_counts.tolist(),
        "task_top1_shares": (top1_counts / top1_counts.sum()).tolist() if top1_counts.sum() else [],
        "task_mean_routing_weights": mean_weights,
        "task_weight_normalized_entropy": normalized_entropy(mean_weights),
    }


@torch.no_grad()
def compute_route_snapshot(
    module: torch.nn.Module,
    *,
    x: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    t: Optional[torch.Tensor],
    cfg_branches: int = 3,
    branch_index: int = 0,
) -> Dict[str, Any]:
    """Reproduce eval-time Top-P selection/capacity and retain one CFG branch."""
    batch, seq_len, channels = x.shape
    if batch % cfg_branches != 0:
        raise ValueError(f"batch {batch} is not divisible by cfg_branches={cfg_branches}")
    if branch_index < 0 or branch_index >= cfg_branches:
        raise ValueError(f"branch_index must be in [0, {cfg_branches})")

    if padding_mask is None:
        padding_mask = torch.ones(batch, seq_len, device=x.device, dtype=torch.bool)
    else:
        padding_mask = padding_mask.to(device=x.device, dtype=torch.bool)
    valid_flat_ids = torch.nonzero(padding_mask.reshape(-1), as_tuple=False).squeeze(1)
    if valid_flat_ids.numel() == 0:
        return {
            "token_count": 0,
            "routed_counts": [0] * int(module.num_routed),
            "selected_routed_counts": [0] * int(module.num_routed),
            "null_count": 0,
            "active_routed_per_token": 0.0,
            "null_selection_rate": 0.0,
        }

    x_valid = x.reshape(batch * seq_len, channels).index_select(0, valid_flat_ids)
    batch_ids = torch.div(valid_flat_ids, seq_len, rounding_mode="floor")
    gate_input = x_valid
    budget = None

    if t is not None:
        if t.dim() == 2:
            t_expanded = t.unsqueeze(1).expand(batch, seq_len, t.shape[-1])
        elif t.dim() == 3 and t.shape[1] == 1:
            t_expanded = t.expand(batch, seq_len, t.shape[-1])
        elif t.dim() == 3 and t.shape[1] == seq_len:
            t_expanded = t
        else:
            raise ValueError(f"unsupported t shape: {tuple(t.shape)}")
        t_valid = t_expanded.reshape(batch * seq_len, -1).index_select(0, valid_flat_ids)
        gate_input = gate_input + module.t_proj(t_valid)
        if bool(module.use_t_budget):
            budget = torch.sigmoid(module.t_budget_proj(t_valid))
    elif bool(module.use_t_budget):
        budget = x_valid.new_full((x_valid.shape[0], 1), 0.5)

    gate_logits = module.gate(gate_input)
    num_routed = int(module.num_routed)
    num_null = int(module.num_null)
    if num_null > 0:
        if bool(module.use_t_budget) and budget is not None:
            null_bias = module.null_logit_bias_max + (
                module.null_logit_bias_min - module.null_logit_bias_max
            ) * budget
            gate_logits[:, num_routed:] += null_bias.to(gate_logits.dtype)
        else:
            gate_logits[:, num_routed:] += float(module.null_logit_bias_min)

    clean_probs = F.softmax(gate_logits.float(), dim=-1)
    selection_probs = clean_probs
    if bool(module.use_bias_balance) and num_routed > 0:
        selection_logits = gate_logits.float().clone()
        selection_logits[:, :num_routed] += module.routing_bias.to(selection_logits)
        selection_probs = F.softmax(selection_logits, dim=-1)

    if bool(module.use_t_budget) and budget is not None:
        p_tok = (
            module.p_min + (module.p_max - module.p_min) * budget.squeeze(-1)
        ).clamp(0.01, 0.999).float()
    else:
        p_tok = torch.full(
            (x_valid.shape[0],),
            float(module.p),
            device=x.device,
            dtype=torch.float32,
        )
    selected = module._top_p_select(selection_probs, p_tok)
    if num_routed > 0:
        has_routed = selected[:, :num_routed].any(dim=-1)
        if not bool(has_routed.all()):
            best_routed = selection_probs[:, :num_routed].argmax(dim=-1)
            missing = ~has_routed
            selected[missing, best_routed[missing]] = True

    combine_probs = clean_probs.to(x.dtype)
    weights = combine_probs * selected.to(x.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    routed_selected = selected[:, :num_routed]
    dispatched = torch.zeros_like(routed_selected)

    if num_routed > 0:
        token_ids, expert_ids = torch.nonzero(routed_selected, as_tuple=True)
        assignment_count = int(token_ids.numel())
        if assignment_count > 0:
            if bool(module.use_t_budget) and budget is not None:
                mean_budget = float(budget.mean().detach().cpu())
                capacity_factor = module.capacity_factor_min + (
                    module.capacity_factor_max - module.capacity_factor_min
                ) * mean_budget
            else:
                capacity_factor = module.capacity_factor_max
            capacity = max(1, int(math.ceil(capacity_factor * assignment_count / num_routed)))
            for expert_id in range(num_routed):
                expert_token_ids = torch.nonzero(
                    routed_selected[:, expert_id], as_tuple=False
                ).squeeze(1)
                if expert_token_ids.numel() == 0:
                    continue
                if bool(module.overflow_drop) and expert_token_ids.numel() > capacity:
                    expert_weights = weights[expert_token_ids, expert_id]
                    keep = torch.topk(expert_weights, k=capacity, largest=True).indices
                    expert_token_ids = expert_token_ids[keep]
                dispatched[expert_token_ids, expert_id] = True

    base_batch = batch // cfg_branches
    branch_start = branch_index * base_batch
    branch_mask = (batch_ids >= branch_start) & (batch_ids < branch_start + base_batch)
    token_count = int(branch_mask.sum().item())
    routed_counts = dispatched[branch_mask].sum(dim=0).to(torch.int64).cpu().tolist()
    selected_routed_counts = routed_selected[branch_mask].sum(dim=0).to(torch.int64).cpu().tolist()
    null_count = (
        int(selected[branch_mask, num_routed:].any(dim=-1).sum().item()) if num_null > 0 else 0
    )
    routed_total = int(sum(routed_counts))
    return {
        "token_count": token_count,
        "routed_counts": routed_counts,
        "selected_routed_counts": selected_routed_counts,
        "null_count": null_count,
        "active_routed_per_token": routed_total / token_count if token_count else 0.0,
        "null_selection_rate": null_count / token_count if token_count else 0.0,
    }


class MoERouteRecorder:
    def __init__(self, encoder: torch.nn.Module, num_steps: int, cfg_branches: int = 3):
        self.encoder = encoder
        self.num_steps = int(num_steps)
        self.cfg_branches = int(cfg_branches)
        self.moe_modules = [layer.mlp for layer in encoder.layers if hasattr(layer.mlp, "num_routed")]
        if not self.moe_modules:
            raise ValueError("encoder has no MoE modules")
        self.num_routed = int(self.moe_modules[0].num_routed)
        self.num_shared = int(getattr(self.moe_modules[0], "num_shared", 0))
        if any(int(getattr(module, "num_shared", 0)) != self.num_shared for module in self.moe_modules):
            raise ValueError("inconsistent shared expert counts across MoE layers")
        self.handles = [encoder.register_forward_pre_hook(self._on_encoder, with_kwargs=True)]
        for layer_index, module in enumerate(self.moe_modules):
            self.handles.append(
                module.register_forward_pre_hook(
                    self._make_moe_hook(layer_index),
                    with_kwargs=True,
                )
            )
        self.active = False
        self.reset()

    def reset(self) -> None:
        num_layers = len(self.moe_modules)
        self.current_step = -1
        self.layer_calls = [0] * num_layers
        self.layer_routed_counts = np.zeros((num_layers, self.num_routed), dtype=np.int64)
        self.layer_selected_routed_counts = np.zeros((num_layers, self.num_routed), dtype=np.int64)
        self.layer_null_counts = np.zeros(num_layers, dtype=np.int64)
        self.layer_token_counts = np.zeros(num_layers, dtype=np.int64)
        self.step_routed_assignments = np.zeros(self.num_steps, dtype=np.int64)
        self.step_selected_routed_assignments = np.zeros(self.num_steps, dtype=np.int64)
        self.step_null_counts = np.zeros(self.num_steps, dtype=np.int64)
        self.step_token_layer_counts = np.zeros(self.num_steps, dtype=np.int64)

    def begin_sample(self) -> None:
        self.reset()
        self.active = True

    def _on_encoder(self, module, args, kwargs) -> None:
        if self.active:
            self.current_step += 1

    def _make_moe_hook(self, layer_index: int):
        def hook(module, args, kwargs) -> None:
            if not self.active:
                return
            if self.current_step < 0 or self.current_step >= self.num_steps:
                raise RuntimeError(f"unexpected inference step {self.current_step}")
            x = args[0] if args else kwargs["x"]
            snapshot = compute_route_snapshot(
                module,
                x=x,
                padding_mask=kwargs.get("padding_mask"),
                t=kwargs.get("t"),
                cfg_branches=self.cfg_branches,
                branch_index=0,
            )
            routed = np.asarray(snapshot["routed_counts"], dtype=np.int64)
            selected = np.asarray(snapshot["selected_routed_counts"], dtype=np.int64)
            tokens = int(snapshot["token_count"])
            nulls = int(snapshot["null_count"])
            self.layer_calls[layer_index] += 1
            self.layer_routed_counts[layer_index] += routed
            self.layer_selected_routed_counts[layer_index] += selected
            self.layer_null_counts[layer_index] += nulls
            self.layer_token_counts[layer_index] += tokens
            self.step_routed_assignments[self.current_step] += int(routed.sum())
            self.step_selected_routed_assignments[self.current_step] += int(selected.sum())
            self.step_null_counts[self.current_step] += nulls
            self.step_token_layer_counts[self.current_step] += tokens

        return hook

    def end_sample(self) -> Dict[str, Any]:
        self.active = False
        actual_steps = self.current_step + 1
        if actual_steps != self.num_steps:
            raise RuntimeError(f"expected {self.num_steps} denoise steps, recorded {actual_steps}")
        if any(count != self.num_steps for count in self.layer_calls):
            raise RuntimeError(f"unexpected MoE layer call counts: {self.layer_calls}")
        return {
            "num_layers": len(self.moe_modules),
            "num_routed_experts": self.num_routed,
            "num_shared_experts": self.num_shared,
            "num_steps": self.num_steps,
            "layer_routed_counts": self.layer_routed_counts.tolist(),
            "layer_selected_routed_counts": self.layer_selected_routed_counts.tolist(),
            "layer_null_counts": self.layer_null_counts.tolist(),
            "layer_token_counts": self.layer_token_counts.tolist(),
            "step_routed_assignments": self.step_routed_assignments.tolist(),
            "step_selected_routed_assignments": self.step_selected_routed_assignments.tolist(),
            "step_null_counts": self.step_null_counts.tolist(),
            "step_token_layer_counts": self.step_token_layer_counts.tolist(),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shard_file(out_dir: Path, kind: str, num_shards: int, shard_index: int) -> Path:
    return out_dir / "raw" / kind / f"shard_{shard_index:02d}_of_{num_shards:02d}.jsonl"


def existing_keys(path: Path, fields: Sequence[str]) -> set:
    keys = set()
    for record in read_jsonl(path):
        if record.get("status", "ok") == "ok":
            keys.add(tuple(record[field] for field in fields))
    return keys


@torch.no_grad()
def collect_task_routes(args, infer: StrictMoeEditInfer) -> Dict[str, Any]:
    router = getattr(infer.dit.encoder, "task_expert_router", None)
    if router is None:
        print("task routes: skipped (model has no task router)", flush=True)
        return {"skipped": True, "reason": "model has no task router"}

    all_items = load_task_items(args.render_script)
    items = shard_items(all_items, args.num_shards, args.shard_index)
    if args.max_task_records:
        items = items[: args.max_task_records]
    output_path = shard_file(args.out_dir, "task_routes", args.num_shards, args.shard_index)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    completed = existing_keys(output_path, ("edit_type", "sample_id", "caption_index"))
    pending = [
        item
        for item in items
        if (item["edit_type"], item["sample_id"], item["caption_index"]) not in completed
    ]

    tokenizer = infer.goku_tokenizer
    text_encoder = infer.goku_text_encoder
    caption_proj = infer.dit.caption_proj
    max_length = int(hparams.get("text_max_token_length", 256))
    total_batches = math.ceil(len(pending) / args.task_batch_size) if pending else 0
    print(f"task routes: total={len(items)}, pending={len(pending)}, batches={total_batches}", flush=True)

    for batch_start in range(0, len(pending), args.task_batch_size):
        batch_items = pending[batch_start : batch_start + args.task_batch_size]
        tokenized = tokenizer(
            [item["caption"] for item in batch_items],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(infer.device, dtype=torch.long)
        attention_mask = tokenized.attention_mask.to(infer.device, dtype=torch.long)
        hidden = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]
        hidden = hidden * attention_mask[..., None]
        caption_ctx = caption_proj(hidden.to(dtype=caption_proj.weight.dtype))
        selected, weights, logits = router(
            caption_ctx,
            attention_mask,
            batch_size=len(batch_items),
            device=infer.device,
            dtype=caption_ctx.dtype,
        )
        probabilities = F.softmax(logits.float(), dim=-1)

        selected_cpu = selected.cpu()
        weights_cpu = weights.float().cpu()
        probabilities_cpu = probabilities.cpu()
        for row, item in enumerate(batch_items):
            selected_experts = torch.nonzero(selected_cpu[row], as_tuple=False).squeeze(1).tolist()
            record = {
                "status": "ok",
                **item,
                "selected_experts": selected_experts,
                "top1_expert": int(probabilities_cpu[row].argmax().item()),
                "probabilities": probabilities_cpu[row].tolist(),
                "weights": weights_cpu[row].tolist(),
            }
            append_jsonl(output_path, record)
        batch_index = batch_start // args.task_batch_size + 1
        print(f"task batch {batch_index}/{total_batches}", flush=True)

    return {"expected_all": len(all_items), "assigned": len(items), "output": str(output_path)}


def flow_schedule(num_steps: int, use_sway: bool) -> List[float]:
    schedule = torch.linspace(0, 1, num_steps + 1)
    if use_sway:
        schedule = schedule - (torch.cos(torch.pi / 2 * schedule) - 1 + schedule)
    return schedule[:-1].tolist()


@torch.no_grad()
def collect_frame_routes(args, infer: StrictMoeEditInfer) -> Dict[str, Any]:
    all_jobs = load_frame_jobs(args.render_script)
    jobs = shard_items(all_jobs, args.num_shards, args.shard_index)
    if args.max_frame_jobs:
        jobs = jobs[: args.max_frame_jobs]
    output_path = shard_file(args.out_dir, "frame_routes", args.num_shards, args.shard_index)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    completed = existing_keys(output_path, ("edit_type", "sample_id"))
    pending = [job for job in jobs if (job["edit_type"], job["sample_id"]) not in completed]
    recorder = MoERouteRecorder(infer.dit.encoder, num_steps=args.num_steps, cfg_branches=3)
    print(f"frame routes: total={len(jobs)}, pending={len(pending)}", flush=True)

    try:
        for local_index, job in enumerate(pending, start=1):
            started = time.time()
            seed = int(args.seed) + int(job["global_index"])
            try:
                set_seed(seed)
                src_wav = load_foa_wav(job["src_wav"], int(hparams["audio_sample_rate"]))
                src_lat = infer.encode_foa_latent(src_wav.unsqueeze(0).to(infer.device), int(hparams["audio_sample_rate"]))
                target_latent_len = int(src_lat.shape[1])
                src_lat = pad_latent_seq_time(src_lat, target_latent_len)

                recorder.begin_sample()
                edited_lat = infer.sample_edit_latent(
                    caption=job["caption"],
                    src_lat=src_lat,
                    target_latent_len=target_latent_len,
                    num_steps=args.num_steps,
                    source_cfg_w=args.source_cfg,
                    caption_cfg_w=args.caption_cfg,
                    use_amo_sampler=False,
                    use_sway=not args.no_sway,
                )
                route_stats = recorder.end_sample()
                if not bool(torch.isfinite(edited_lat).all()):
                    raise ValueError("inference latent contains NaN or Inf")
                record = {
                    "status": "ok",
                    **job,
                    "seed": seed,
                    "target_latent_len": target_latent_len,
                    "source_cfg": float(args.source_cfg),
                    "caption_cfg": float(args.caption_cfg),
                    "flow_t": flow_schedule(args.num_steps, not args.no_sway),
                    "elapsed_sec": time.time() - started,
                    **route_stats,
                }
            except Exception as exc:
                recorder.active = False
                record = {
                    "status": "error",
                    **job,
                    "seed": seed,
                    "elapsed_sec": time.time() - started,
                    "error": repr(exc),
                }
                append_jsonl(output_path, record)
                if not args.continue_on_error:
                    raise
                print(f"frame error {job['edit_type']}/{job['sample_id']}: {exc!r}", flush=True)
                continue

            append_jsonl(output_path, record)
            print(
                f"frame {local_index}/{len(pending)}: {job['edit_type']}/{job['sample_id']} "
                f"({record['elapsed_sec']:.1f}s)",
                flush=True,
            )
            del src_wav, src_lat, edited_lat
    finally:
        recorder.close()

    return {"expected_all": len(all_jobs), "assigned": len(jobs), "output": str(output_path)}


def deduplicate_records(paths: Sequence[Path], key_fields: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records_by_key = {}
    errors = []
    for path in paths:
        for record in read_jsonl(path):
            if record.get("status", "ok") != "ok":
                errors.append(record)
                continue
            key = tuple(record[field] for field in key_fields)
            records_by_key[key] = record
    return list(records_by_key.values()), errors


def summarize_frame_routes(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("no frame route records to summarize")
    num_layers = int(records[0]["num_layers"])
    num_routed = int(records[0]["num_routed_experts"])
    num_shared = int(records[0].get("num_shared_experts", 0))
    num_steps = int(records[0]["num_steps"])
    routed = np.zeros((num_layers, num_routed), dtype=np.int64)
    selected = np.zeros_like(routed)
    nulls = np.zeros(num_layers, dtype=np.int64)
    tokens = np.zeros(num_layers, dtype=np.int64)
    step_routed = np.zeros(num_steps, dtype=np.int64)
    step_selected = np.zeros(num_steps, dtype=np.int64)
    step_nulls = np.zeros(num_steps, dtype=np.int64)
    step_tokens = np.zeros(num_steps, dtype=np.int64)

    for record in records:
        if (
            int(record["num_layers"]) != num_layers
            or int(record["num_routed_experts"]) != num_routed
            or int(record.get("num_shared_experts", 0)) != num_shared
            or int(record["num_steps"]) != num_steps
        ):
            raise ValueError("inconsistent frame route shapes across records")
        routed += np.asarray(record["layer_routed_counts"], dtype=np.int64)
        selected += np.asarray(record["layer_selected_routed_counts"], dtype=np.int64)
        nulls += np.asarray(record["layer_null_counts"], dtype=np.int64)
        tokens += np.asarray(record["layer_token_counts"], dtype=np.int64)
        step_routed += np.asarray(record["step_routed_assignments"], dtype=np.int64)
        step_selected += np.asarray(record["step_selected_routed_assignments"], dtype=np.int64)
        step_nulls += np.asarray(record["step_null_counts"], dtype=np.int64)
        step_tokens += np.asarray(record["step_token_layer_counts"], dtype=np.int64)

    routed_shares = routed / np.maximum(routed.sum(axis=1, keepdims=True), 1)
    routed_activation = routed / np.maximum(tokens[:, None], 1)
    shared_activation = np.ones((num_layers, num_shared), dtype=np.float64)
    null_activation = (nulls / np.maximum(tokens, 1))[:, None]
    activation_rates = np.concatenate([routed_activation, shared_activation, null_activation], axis=1)
    activation_labels = (
        [f"R{index}" for index in range(num_routed)]
        + [f"S{index}" for index in range(num_shared)]
        + ["Null"]
    )
    entropies = [normalized_entropy(row) for row in routed]
    dead_mask = routed_shares < 0.01
    flow_t = records[0]["flow_t"]
    return {
        "num_frame_samples": len(records),
        "num_layers": num_layers,
        "num_routed_experts": num_routed,
        "num_shared_experts": num_shared,
        "num_steps": num_steps,
        "layer_routed_counts": routed.tolist(),
        "layer_selected_routed_counts": selected.tolist(),
        "layer_null_counts": nulls.tolist(),
        "layer_token_counts": tokens.tolist(),
        "layer_routed_shares": routed_shares.tolist(),
        "layer_expert_activation_rates": activation_rates.tolist(),
        "expert_activation_labels": activation_labels,
        "layer_normalized_routed_entropy": entropies,
        "mean_layer_normalized_routed_entropy": float(np.mean(entropies)),
        "min_layer_normalized_routed_entropy": float(np.min(entropies)),
        "dead_layer_expert_count_below_1pct": int(dead_mask.sum()),
        "capacity_drop_rate": float(1.0 - routed.sum() / max(int(selected.sum()), 1)),
        "flow_t": flow_t,
        "step_active_routed_experts_per_token": (
            step_routed / np.maximum(step_tokens, 1)
        ).tolist(),
        "step_null_selection_rate": (step_nulls / np.maximum(step_tokens, 1)).tolist(),
        "step_routed_assignments": step_routed.tolist(),
        "step_selected_routed_assignments": step_selected.tolist(),
        "step_null_counts": step_nulls.tolist(),
        "step_token_layer_counts": step_tokens.tolist(),
    }


def task_heatmaps(records: Sequence[Dict[str, Any]], num_task_experts: int):
    edit_types = [edit_type for edit_type in EDIT_TYPE_ORDER if any(r["edit_type"] == edit_type for r in records)]
    extras = sorted({r["edit_type"] for r in records} - set(edit_types))
    edit_types.extend(extras)
    weight_rows = []
    selection_rows = []
    for edit_type in edit_types:
        group = [record for record in records if record["edit_type"] == edit_type]
        weights = np.asarray([record["weights"] for record in group], dtype=np.float64)
        selections = np.zeros((len(group), num_task_experts), dtype=np.float64)
        for row, record in enumerate(group):
            selections[row, record["selected_experts"]] = 1.0
        weight_rows.append(weights.mean(axis=0))
        selection_rows.append(selections.mean(axis=0))
    return edit_types, np.asarray(weight_rows), np.asarray(selection_rows)


def write_csv(path: Path, header: Sequence[Any], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_aggregate_tables(
    out_dir: Path,
    edit_types: Optional[Sequence[str]],
    task_weights: Optional[np.ndarray],
    task_selection: Optional[np.ndarray],
    frame_summary: Dict[str, Any],
) -> None:
    if task_weights is not None and task_selection is not None and edit_types is not None:
        task_header = ["edit_type"] + [f"task_{index}" for index in range(task_weights.shape[1])]
        write_csv(
            out_dir / "task_weight_by_edit_type.csv",
            task_header,
            ([edit_type] + task_weights[row].tolist() for row, edit_type in enumerate(edit_types)),
        )
        write_csv(
            out_dir / "task_selection_rate_by_edit_type.csv",
            task_header,
            ([edit_type] + task_selection[row].tolist() for row, edit_type in enumerate(edit_types)),
        )

    activation = np.asarray(frame_summary["layer_expert_activation_rates"])
    layer_header = ["layer"] + [label.lower() for label in frame_summary["expert_activation_labels"]]
    write_csv(
        out_dir / "layer_expert_activation_rate.csv",
        layer_header,
        ([layer + 1] + activation[layer].tolist() for layer in range(activation.shape[0])),
    )
    write_csv(
        out_dir / "flow_step_stats.csv",
        ["step", "flow_t", "active_routed_experts_per_token", "null_selection_rate"],
        (
            [step + 1, t, active, null_rate]
            for step, (t, active, null_rate) in enumerate(
                zip(
                    frame_summary["flow_t"],
                    frame_summary["step_active_routed_experts_per_token"],
                    frame_summary["step_null_selection_rate"],
                )
            )
        ),
    )


def plot_results(
    out_dir: Path,
    edit_types: Optional[Sequence[str]],
    task_weights: Optional[np.ndarray],
    frame_summary: Dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_task_router = task_weights is not None and edit_types is not None
    panel_count = 3 if has_task_router else 2
    fig_width = 16.5 if has_task_router else 11.5
    fig, axes = plt.subplots(1, panel_count, figsize=(fig_width, 6.0), constrained_layout=True)

    if has_task_router:
        ax = axes[0]
        image = ax.imshow(task_weights, aspect="auto", vmin=0.0, vmax=max(0.5, float(task_weights.max())), cmap="viridis")
        ax.set_xticks(range(task_weights.shape[1]), [f"T{index}" for index in range(task_weights.shape[1])])
        ax.set_yticks(range(len(edit_types)), [EDIT_TYPE_LABELS.get(name, name) for name in edit_types])
        ax.set_title("(a) Task-expert routing weight")
        ax.set_xlabel("Task expert")
        for row in range(task_weights.shape[0]):
            for column in range(task_weights.shape[1]):
                value = task_weights[row, column]
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value > task_weights.max() * 0.55 else "black")
        for boundary in (3.5, 6.5, 8.5):
            if boundary < len(edit_types) - 0.5:
                ax.axhline(boundary, color="white", linewidth=1.5)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    activation = np.asarray(frame_summary["layer_expert_activation_rates"])
    routed_panel = 1 if has_task_router else 0
    ax = axes[routed_panel]
    image = ax.imshow(activation, aspect="auto", vmin=0.0, cmap="magma")
    labels = frame_summary["expert_activation_labels"]
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(0, activation.shape[0], 2), [str(index + 1) for index in range(0, activation.shape[0], 2)])
    ax.set_title(f"({'b' if has_task_router else 'a'}) Layer-wise expert activation")
    ax.set_xlabel("Layer-local expert")
    ax.set_ylabel("DiT layer")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="Activations / valid tokens")

    flow_panel = 2 if has_task_router else 1
    ax = axes[flow_panel]
    flow_t = np.asarray(frame_summary["flow_t"])
    active = np.asarray(frame_summary["step_active_routed_experts_per_token"])
    null_rate = np.asarray(frame_summary["step_null_selection_rate"])
    line_active = ax.plot(flow_t, active, marker="o", markersize=3, color="#0072B2", label="Active routed experts")[0]
    ax.set_xlabel("Flow time t")
    ax.set_ylabel("Active routed experts / token", color="#0072B2")
    ax.tick_params(axis="y", labelcolor="#0072B2")
    ax.grid(alpha=0.25)
    twin = ax.twinx()
    line_null = twin.plot(flow_t, null_rate, marker="s", markersize=3, color="#D55E00", label="Null selection rate")[0]
    twin.set_ylabel("Null selection rate", color="#D55E00")
    twin.tick_params(axis="y", labelcolor="#D55E00")
    twin.set_ylim(0.0, max(1.0, float(null_rate.max()) * 1.05))
    ax.set_title(f"({'c' if has_task_router else 'b'}) Routing over flow steps")
    ax.legend([line_active, line_null], [line_active.get_label(), line_null.get_label()], loc="best", frameon=False)

    fig.savefig(out_dir / "moe_routing_analysis.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "moe_routing_analysis.pdf", bbox_inches="tight")
    plt.close(fig)


def merge_results(args) -> Dict[str, Any]:
    suffix = f"_of_{args.num_shards:02d}.jsonl"
    task_paths = sorted((args.out_dir / "raw/task_routes").glob(f"*{suffix}"))
    frame_paths = sorted((args.out_dir / "raw/frame_routes").glob(f"*{suffix}"))
    expected_task_shards = 0 if args.no_task_router else args.num_shards
    if len(task_paths) != expected_task_shards or len(frame_paths) != args.num_shards:
        raise ValueError(
            f"expected {expected_task_shards} task and {args.num_shards} frame shard files, found "
            f"{len(task_paths)} and {len(frame_paths)}"
        )

    task_records, task_errors = (
        deduplicate_records(task_paths, ("edit_type", "sample_id", "caption_index"))
        if task_paths
        else ([], [])
    )
    frame_records, frame_errors = deduplicate_records(frame_paths, ("edit_type", "sample_id"))
    expected_task = 0 if args.no_task_router else len(load_task_items(args.render_script))
    expected_frame = len(load_frame_jobs(args.render_script))
    if not args.allow_partial:
        if len(task_records) != expected_task or len(frame_records) != expected_frame:
            raise ValueError(
                f"incomplete results: task={len(task_records)}/{expected_task}, "
                f"frame={len(frame_records)}/{expected_frame}, "
                f"errors={len(task_errors) + len(frame_errors)}"
            )

    frame_summary = summarize_frame_routes(frame_records)
    if task_records:
        num_task_experts = len(task_records[0]["weights"])
        task_summary = summarize_task_routes(task_records, num_task_experts)
        edit_types, task_weights, task_selection = task_heatmaps(task_records, num_task_experts)
    else:
        task_summary = None
        edit_types, task_weights, task_selection = None, None, None
    metrics = {
        "checkpoint": str(args.dit_ckpt),
        "model_variant": "shared_moe" if args.no_task_router else "task_moe",
        "expected_task_routes": expected_task,
        "expected_frame_samples": expected_frame,
        "task_errors": len(task_errors),
        "frame_errors": len(frame_errors),
        "definitions": {
            "task_label_top1_nmi": "Arithmetic-mean NMI between 11 edit labels and top-1 task expert.",
            "paraphrase_pairwise_agreement": "Top-1 agreement over all caption pairs within each sample.",
            "layer_expert_activation_rates": (
                "Post-capacity routed dispatches or null selections divided by valid "
                "source+instruction tokens; shared experts are always active (1.0)."
            ),
            "dead_layer_expert": "A layer-local routed expert with less than 1% of that layer's routed dispatches.",
        },
        "task": task_summary,
        "frame": frame_summary,
    }
    write_json(args.out_dir / "metrics.json", metrics)
    write_json(args.out_dir / "errors.json", {"task": task_errors, "frame": frame_errors})
    write_aggregate_tables(args.out_dir, edit_types, task_weights, task_selection, frame_summary)
    plot_results(args.out_dir, edit_types, task_weights, frame_summary)
    print(json.dumps({"task": task_summary, "frame": {
        "num_frame_samples": frame_summary["num_frame_samples"],
        "mean_layer_normalized_routed_entropy": frame_summary["mean_layer_normalized_routed_entropy"],
        "min_layer_normalized_routed_entropy": frame_summary["min_layer_normalized_routed_entropy"],
        "dead_layer_expert_count_below_1pct": frame_summary["dead_layer_expert_count_below_1pct"],
        "capacity_drop_rate": frame_summary["capacity_drop_rate"],
    }}, indent=2), flush=True)
    return metrics


def serializable_args(args) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze SE-MoE task routing and expert utilization.")
    parser.add_argument("--phase", choices=["task", "frame", "collect", "merge"], default="collect")
    parser.add_argument("--dit_ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--render_script", type=Path, default=DEFAULT_RENDER_SCRIPT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--source_cfg", type=float, default=3.0)
    parser.add_argument("--caption_cfg", type=float, default=3.0)
    parser.add_argument("--no_sway", action="store_true")
    parser.add_argument("--task_batch_size", type=int, default=16)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max_task_records", type=int, default=0)
    parser.add_argument("--max_frame_jobs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument(
        "--no_task_router",
        action="store_true",
        help="Analyze a routed/shared MoE checkpoint that has no task expert router.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.render_script = args.render_script.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "merge":
        merge_results(args)
        return

    checkpoint_file = resolve_checkpoint_file(args.dit_ckpt)
    print(f"strict checkpoint: {checkpoint_file}", flush=True)
    infer = StrictMoeEditInfer(
        device=args.device,
        dit_ckpt=str(args.dit_ckpt),
        config_path=args.config,
        precision=args.precision,
        use_ema=args.use_ema,
        expect_task_router=not args.no_task_router,
    )
    task_router = getattr(infer.dit.encoder, "task_expert_router", None)
    first_moe = infer.dit.encoder.layers[0].mlp
    model_info = {
        "args": serializable_args(args),
        "resolved_checkpoint": str(infer.resolved_checkpoint_file),
        "num_layers": len(infer.dit.encoder.layers),
        "num_task_experts": int(task_router.num_task_experts) if task_router is not None else 0,
        "num_routed_experts": int(first_moe.num_routed),
        "num_shared_experts": int(getattr(first_moe, "num_shared", 0)),
        "num_null_experts": int(first_moe.num_null),
    }
    results = {}
    if args.phase in ("task", "collect"):
        results["task"] = collect_task_routes(args, infer)
    if args.phase in ("frame", "collect"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results["frame"] = collect_frame_routes(args, infer)
    model_info["results"] = results
    write_json(
        args.out_dir / "manifests" / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json",
        model_info,
    )


if __name__ == "__main__":
    main()
