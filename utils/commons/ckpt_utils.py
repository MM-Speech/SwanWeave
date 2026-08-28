import contextlib
import glob
import os
import re
import subprocess
import traceback

import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist


@contextlib.contextmanager
def dist_load(path):
    if not dist.is_initialized() or dist.get_world_size() == 1 or os.path.realpath(path).startswith('/dev/shm'):
        yield path
    else:
        from utils.commons.hparams import hparams
        from utils.commons.trainer import LOCAL_RANK
        tmpdir = '/dev/shm'
        assert len(os.path.basename(path)) > 0
        shm_ckpt_path = f'{tmpdir}/{hparams["exp_name"]}/{os.path.basename(path)}'
        if LOCAL_RANK == 0:
            subprocess.check_call(
                f'mkdir -p {os.path.dirname(shm_ckpt_path)}; '
                f'cp -Lr {path} {shm_ckpt_path}', shell=True)
        dist.barrier()
        yield shm_ckpt_path
        dist.barrier()
        if LOCAL_RANK == 0:
            subprocess.check_call(f'rm -rf {shm_ckpt_path}', shell=True)


def torch_load_dist(path, map_location='cpu', mmap=None):
    with dist_load(path) as tmp_path:
        checkpoint = torch.load(tmp_path, map_location=map_location, mmap=mmap)
    return checkpoint


def get_last_checkpoint(work_dir, steps=None, map_location='cpu', mmap=None, return_step=False):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch_load_dist(last_ckpt_path, map_location=map_location, mmap=mmap)
    if not return_step:
        return checkpoint, last_ckpt_path
    else:
        if last_ckpt_path is not None:
            pattern = r'.*steps_(\d+)(?:\.ckpt|_backbone\.ckpt)'
            global_steps = int(re.findall(pattern, last_ckpt_path)[0])
        else:
            global_steps = 0
        return checkpoint, last_ckpt_path, global_steps


def get_all_ckpts(work_dir, steps=None):
    if steps is None or steps == 0:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_*.ckpt'
    else:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_{steps}.ckpt'
    pattern = '.*steps_(\d+)(?:\.ckpt|_backbone\.ckpt)'
    all_ckpts = [x for x in glob.glob(ckpt_path_pattern) if len(re.findall(pattern, x)) > 0]
    return sorted(all_ckpts, key=lambda x: -int(re.findall(pattern, x)[0]))


def get_all_ckpt_steps(work_dir):
    ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_*.ckpt'
    pattern = '.*steps_(\d+)(?:\.ckpt|_backbone\.ckpt)'
    all_ckpts = [x for x in glob.glob(ckpt_path_pattern) if len(re.findall(pattern, x)) > 0]
    steps = [int(re.findall(pattern, c)[0]) for c in all_ckpts]
    steps = sorted(steps)
    return steps


def load_ckpt(cur_model, ckpt_base_dir, model_name='model', force=True, strict=True,
              silent=False, load_opt=False, opts=None, steps=None, checkpoint=None, ckpt_path='', delete_unmatch=True, map_location='cpu', mmap=None):
    if checkpoint is None:
        if os.path.isfile(ckpt_base_dir):
            base_dir = os.path.dirname(ckpt_base_dir)
            ckpt_path = ckpt_base_dir
            checkpoint = torch_load_dist(ckpt_base_dir, map_location=map_location, mmap=mmap)
        else:
            base_dir = ckpt_base_dir
            if load_opt:
                checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir, steps)
            else:
                ckpt_path = f'{ckpt_base_dir}/model_only_last.ckpt'
                if os.path.exists(ckpt_path):
                    checkpoint = torch_load_dist(ckpt_path, map_location=map_location, mmap=mmap)
                else:
                    checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir, steps)
    if checkpoint is not None:
        # ===== 新增：聚合 missing / unmatched key，避免在不同层重复打印 =====
        aggregated_missing = set()
        aggregated_unexpected = set()
        aggregated_unmatched = set()

        def _canonical_key(k: str) -> str:
            # 将形如 ".0." / ".1." 这类层索引归一化，避免“同一模块不同层”重复打印
            # 例如 encoder.layers.0.self_attn.q_proj.weight -> encoder.layers.<N>.self_attn.q_proj.weight
            return re.sub(r'\.\d+(\.|$)', '.<N>\\1', k)
        # ====================================================================

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        state_dict_all = {
            k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        if not isinstance(cur_model, list):
            cur_models = [cur_model]
            model_names = [model_name]
        else:
            cur_models = cur_model
            model_names = model_name
        for model_name, cur_model in zip(model_names, cur_models):
            if isinstance(cur_model, DistributedDataParallel):
                cur_model = cur_model.module
            device = next(cur_model.parameters()).device
            if '.' not in model_name:
                state_dict = state_dict_all[model_name]
            else:
                base_model_name = model_name.split('.')[0]
                rest_model_name = model_name[len(base_model_name) + 1:]
                state_dict = {
                    k[len(rest_model_name) + 1:]: v for k, v in state_dict_all[base_model_name].items()
                    if k.startswith(f'{rest_model_name}.')}
            state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            if not strict and delete_unmatch:
                try:
                    cur_model.load_state_dict(state_dict, strict=True)
                    if not silent:
                        print(f"| loaded '{model_name}' from '{ckpt_path}' with strict=True.")
                except:
                    cur_model_state_dict = cur_model.state_dict()
                    cur_model_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in
                                            cur_model_state_dict.items()}

                    state_dict, repaired, removed = repair_unmatched_state_dict(cur_model_state_dict, state_dict, silent)

            load_results = cur_model.load_state_dict(state_dict, strict=strict)
            cur_model.to(device)
            if not silent:
                print(f"| loaded '{model_name}' from '{ckpt_path}'.")
                missing_keys, unexpected_keys = load_results.missing_keys, load_results.unexpected_keys
                print(f"| Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                # 新增：记录 missing 的 key
                for k in missing_keys:
                    aggregated_missing.add(_canonical_key(k))
                for k in unexpected_keys:
                    aggregated_unexpected.add(_canonical_key(k))

        # 新增：在所有 model 都 load 完之后，统一按模块去重打印一次
        if not silent:
            if aggregated_missing:
                print("| ===== Missing key names (deduplicated by module) =====")
                for k in sorted(aggregated_missing):
                    print("|   ", k)
            if aggregated_unexpected:
                print("| ===== Unexpected key names (deduplicated by module) =====")
                for k in sorted(aggregated_unexpected):
                    print("|   ", k)
            if aggregated_unmatched:
                print("| ===== Unmatched (size-mismatch) key names (deduplicated by module) =====")
                for k in sorted(aggregated_unmatched):
                    print("|   ", k)

        if load_opt:
            if "optimizer_states" in checkpoint:
                optimizer_states = checkpoint['optimizer_states']
            else:
                optimizer_states = torch_load_dist(ckpt_path[:-5] + '_optm.ckpt', map_location=map_location, mmap=mmap)
            assert len(opts) == len(optimizer_states)
            for optimizer, opt_state in zip(opts, optimizer_states):
                opt_state = {k.replace('_orig_mod.', ''): v for k, v in opt_state.items()}
                if optimizer is None:
                    return
                try:
                    optimizer.load_state_dict(opt_state)
                    for i, state in enumerate(optimizer.state.values()):
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                except ValueError:
                    print(f"| WARMING: optimizer {optimizer} parameters not match !!!")
        return checkpoint.get('global_step', 0)
    else:
        e_msg = f"| ckpt not found in {base_dir}."
        if force:
            assert False, e_msg
        else:
            print(e_msg)

def load_ckpt2(cur_model, ckpt_base_dir, model_name='model', force=True, strict=True,
              silent=False, load_opt=False, opts=None, steps=None, checkpoint=None,
              ckpt_path='', delete_unmatch=True, map_location='cpu', mmap=None,
              ckpt_path2=None, checkpoint2=None,
              ckpt2_ratio=0.5, ckpt1_ratio=None, normalize_ratio=True):
    """
    支持：
    1) 仅加载一个 ckpt（旧行为，完全兼容）
    2) 同时加载两个 ckpt（通过 ckpt_path2 / checkpoint2 传第二个），并对参数做：
       - 两个 ckpt 中都存在且 shape 相同的参数：按比例混合 w1*v1 + w2*v2（默认 w1=w2=0.5）
       - 只在一个 ckpt 中存在的参数：直接使用该 ckpt 的参数
       - 两个 ckpt 中都存在但 shape 不同：使用第一个 ckpt 的参数，并将 key 记入 aggregated_unmatched 打印

    权重说明：
      - 只传 ckpt2_ratio：表示第二个 ckpt 的占比 alpha，w2=alpha，w1=1-alpha
      - 同时传 ckpt1_ratio/ckpt2_ratio：表示原始比例，若 normalize_ratio=True 会自动归一化到和为 1
    """

    # ====== 先照原逻辑，解析第一个 ckpt ======
    if checkpoint is None:
        if os.path.isfile(ckpt_base_dir):
            base_dir = os.path.dirname(ckpt_base_dir)
            ckpt_path = ckpt_base_dir
            checkpoint = torch_load_dist(ckpt_base_dir, map_location=map_location, mmap=mmap)
        else:
            base_dir = ckpt_base_dir
            if load_opt:
                checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir, steps)
            else:
                ckpt_path = f'{ckpt_base_dir}/model_only_last.ckpt'
                if os.path.exists(ckpt_path):
                    checkpoint = torch_load_dist(ckpt_path, map_location=map_location, mmap=mmap)
                else:
                    checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir, steps)

    # ====== 解析第二个 ckpt（可选）======
    if checkpoint2 is None and ckpt_path2 is not None:
        if os.path.isfile(ckpt_path2):
            checkpoint2 = torch_load_dist(ckpt_path2, map_location=map_location, mmap=mmap)
        else:
            if not silent:
                print(f"| WARNING: second ckpt file '{ckpt_path2}' not found, ignore second ckpt.")
            checkpoint2 = None

    # ====== 计算混合权重 ======
    if ckpt1_ratio is None:
        w2 = float(ckpt2_ratio)
        w1 = 1.0 - w2
    else:
        w1 = float(ckpt1_ratio)
        w2 = float(ckpt2_ratio)
        if normalize_ratio:
            s = w1 + w2
            if s == 0:
                raise ValueError("ckpt1_ratio + ckpt2_ratio == 0，无法归一化")
            w1 /= s
            w2 /= s

    if checkpoint is not None:
        aggregated_missing = set()
        aggregated_unexpected = set()
        aggregated_unmatched = set()

        def _canonical_key(k: str) -> str:
            return re.sub(r'\.\d+(\.|$)', '.<N>\\1', k)

        # ---------- 第一个 ckpt 的 state_dict ----------
        if "state_dict" in checkpoint:
            state_dict_1 = checkpoint["state_dict"]
        else:
            state_dict_1 = checkpoint
        state_dict_all_1 = {
            k.replace('module.', '').replace('_orig_mod.', ''): v
            for k, v in state_dict_1.items()
        }

        # ---------- 第二个 ckpt 的 state_dict（如果有） ----------
        state_dict_all_2 = None
        if checkpoint2 is not None:
            if "state_dict" in checkpoint2:
                state_dict_2 = checkpoint2["state_dict"]
            else:
                state_dict_2 = checkpoint2
            state_dict_all_2 = {
                k.replace('module.', '').replace('_orig_mod.', ''): v
                for k, v in state_dict_2.items()
            }

        # ---------- 统一把 cur_model / model_name 变成 list 处理 ----------
        if not isinstance(cur_model, list):
            cur_models = [cur_model]
            model_names = [model_name]
        else:
            cur_models = cur_model
            model_names = model_name

        for model_name, cur_model in zip(model_names, cur_models):
            if isinstance(cur_model, DistributedDataParallel):
                cur_model = cur_model.module
            device = next(cur_model.parameters()).device

            # ====== 按 model_name 取出对应子 state_dict（第一个 ckpt）======
            if '.' not in model_name:
                state_dict_1_sub = state_dict_all_1[model_name]
                state_dict_2_sub = (
                    state_dict_all_2[model_name] if state_dict_all_2 is not None and model_name in state_dict_all_2
                    else None
                )
            else:
                base_model_name = model_name.split('.')[0]
                rest_model_name = model_name[len(base_model_name) + 1:]

                state_dict_1_sub = {
                    k[len(rest_model_name) + 1:]: v
                    for k, v in state_dict_all_1[base_model_name].items()
                    if k.startswith(f'{rest_model_name}.')
                }

                if state_dict_all_2 is not None and base_model_name in state_dict_all_2:
                    state_dict_2_sub = {
                        k[len(rest_model_name) + 1:]: v
                        for k, v in state_dict_all_2[base_model_name].items()
                        if k.startswith(f'{rest_model_name}.')
                    }
                else:
                    state_dict_2_sub = None

            # 再做一遍 module/_orig_mod 清理（防御性，多做无害）
            state_dict_1_sub = {
                k.replace('module.', '').replace('_orig_mod.', ''): v
                for k, v in state_dict_1_sub.items()
            }
            if state_dict_2_sub is not None:
                state_dict_2_sub = {
                    k.replace('module.', '').replace('_orig_mod.', ''): v
                    for k, v in state_dict_2_sub.items()
                }

            # ====== 关键：合并两个 ckpt（按 key 做 union + 加权混合）======
            if state_dict_2_sub is not None:
                merged_state_dict = {}
                all_keys = set(state_dict_1_sub.keys()) | set(state_dict_2_sub.keys())
                for k in all_keys:
                    v1 = state_dict_1_sub.get(k, None)
                    v2 = state_dict_2_sub.get(k, None)

                    if v1 is not None and v2 is not None:
                        # shape 相同才尝试混合
                        if hasattr(v1, "shape") and hasattr(v2, "shape") and v1.shape == v2.shape:
                            # 只对浮点/复数张量做混合；否则保留 v1（比如某些 int buffer）
                            can_mix = (
                                isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor) and
                                ((v1.is_floating_point() or v1.is_complex()) and (v2.is_floating_point() or v2.is_complex()))
                            )
                            if can_mix:
                                # dtype 对齐到 v1，避免不必要的类型提升/报错
                                if v2.dtype != v1.dtype:
                                    v2 = v2.to(dtype=v1.dtype)

                                # 权重也用同 dtype/device 的标量，避免类型提升
                                w1_t = torch.as_tensor(w1, dtype=v1.dtype, device=v1.device)
                                w2_t = torch.as_tensor(w2, dtype=v1.dtype, device=v1.device)

                                merged_state_dict[k] = v1 * w1_t + v2 * w2_t
                            else:
                                merged_state_dict[k] = v1
                                aggregated_unmatched.add(_canonical_key(k))
                        else:
                            # shape 不一致：保留第一个 ckpt 的参数，同时记录 unmatched
                            merged_state_dict[k] = v1
                            aggregated_unmatched.add(_canonical_key(k))
                    elif v1 is not None:
                        merged_state_dict[k] = v1
                    else:
                        merged_state_dict[k] = v2

                state_dict_to_load = merged_state_dict
            else:
                state_dict_to_load = state_dict_1_sub

            # ====== 非 strict + delete_unmatch 时，仍然用原来的 repair_unmatched_state_dict ======
            if not strict and delete_unmatch:
                try:
                    cur_model.load_state_dict(state_dict_to_load, strict=True)
                    if not silent:
                        print(f"| loaded '{model_name}' from '{ckpt_path}' with strict=True.")
                except Exception:
                    cur_model_state_dict = cur_model.state_dict()
                    cur_model_state_dict = {
                        k.replace('module.', '').replace('_orig_mod.', ''): v
                        for k, v in cur_model_state_dict.items()
                    }

                    state_dict_to_load, repaired, removed = repair_unmatched_state_dict(
                        cur_model_state_dict, state_dict_to_load, silent
                    )

            # ====== 真正 load 到模型 ======
            load_results = cur_model.load_state_dict(state_dict_to_load, strict=strict)
            cur_model.to(device)

            if not silent:
                print(f"| loaded '{model_name}' from '{ckpt_path}'"
                      f"{'' if ckpt_path2 is None else f' and merged with {ckpt_path2}'} "
                      f"(w1={w1:.6f}, w2={w2:.6f}).")

                missing_keys, unexpected_keys = load_results.missing_keys, load_results.unexpected_keys
                print(f"| Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                for k in missing_keys:
                    aggregated_missing.add(_canonical_key(k))
                for k in unexpected_keys:
                    aggregated_unexpected.add(_canonical_key(k))

        if not silent:
            if aggregated_missing:
                print("| ===== Missing key names (deduplicated by module) =====")
                for k in sorted(aggregated_missing):
                    print("|   ", k)
            if aggregated_unexpected:
                print("| ===== Unexpected key names (deduplicated by module) =====")
                for k in sorted(aggregated_unexpected):
                    print("|   ", k)
            if aggregated_unmatched:
                print("| ===== Unmatched (size-mismatch / non-mixable) key names (deduplicated by module) =====")
                for k in sorted(aggregated_unmatched):
                    print("|   ", k)

        # ====== optimizer 仍然只从第一个 ckpt 恢复，不做 merge ======
        if load_opt:
            if "optimizer_states" in checkpoint:
                optimizer_states = checkpoint['optimizer_states']
            else:
                optimizer_states = torch_load_dist(
                    ckpt_path[:-5] + '_optm.ckpt', map_location=map_location, mmap=mmap
                )
            assert len(opts) == len(optimizer_states)
            for optimizer, opt_state in zip(opts, optimizer_states):
                opt_state = {k.replace('_orig_mod.', ''): v for k, v in opt_state.items()}
                if optimizer is None:
                    return
                try:
                    optimizer.load_state_dict(opt_state)
                    for i, state in enumerate(optimizer.state.values()):
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                except ValueError:
                    print(f"| WARMING: optimizer {optimizer} parameters not match !!!")

        return checkpoint.get('global_step', 0)
    else:
        e_msg = f"| ckpt not found in {base_dir}."
        if force:
            assert False, e_msg
        else:
            print(e_msg)


def repair_unmatched_state_dict(cur_model_state_dict, state_dict, silent=False):
    repaired, removed = 0, []
    for key, old_param in list(state_dict.items()):
        if key in cur_model_state_dict:
            new_param = cur_model_state_dict[key]
            if new_param.shape != old_param.shape:
                print("| Unmatched keys:", key, "cur model:", tuple(new_param.shape), "ckpt model:", tuple(old_param.shape))
                merged = merge_unmatched_params(new_param, old_param, key)
                if merged is not None:
                    state_dict[key] = merged
                    repaired += 1
                    print(f"| Unmatched key {key} is partially loaded")
                else:
                    removed.append(key)

    for key in removed:
        del state_dict[key]
    
    if repaired > 0 and not silent:
        print(f"| Partially loaded {repaired} tensor(s) with size mismatch by copying overlapping slices.")

    return state_dict, repaired, removed


def merge_unmatched_params(new_tensor: torch.Tensor, old_tensor: torch.Tensor, key: str = ""):
    """
    Return a tensor with the same shape as new_tensor, where the overlapping
    slice is copied from old_tensor. If not possible, return None.
    """
    try:
        if new_tensor.ndim != old_tensor.ndim:
            return None
        old_tensor = old_tensor.to(dtype=new_tensor.dtype)

        slices = tuple(slice(0, min(n, o)) for n, o in zip(new_tensor.shape, old_tensor.shape))

        out = new_tensor.clone()
        out[slices] = old_tensor[slices]
        return out
    except Exception as e:
        print(f"| merge failed on '{key}': {e}")
        return None


def load_with_size_mismatch(model, state_dict, prefix=""):
    current_model_dict = model.state_dict()
    cm_keys = current_model_dict.keys()
    mismatch_keys = {k.replace(prefix, "") for k, v in state_dict.items() if k.replace(prefix, "") in cm_keys and v.size() != current_model_dict[k.replace(prefix, "")].size()}
    new_state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.replace(prefix, "") in cm_keys and v.size() == current_model_dict[k.replace(prefix, "")].size()}
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"| mismatch keys: ", mismatch_keys)
    if len(missing_keys) > 0:
        print(f"| missing_keys in: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"| unexpected_keys in: {unexpected_keys}")

def load_ckpt_moe(
    cur_model,
    ckpt_base_dir,
    ckpt_audio_base_dir="",
    model_name="model",
    expert_idx=None,
    force=True,
    strict=False,
    silent=False,
    steps=None,
    checkpoint=None,
    ckpt_path="",
    delete_unmatch=True,
    map_location="cpu",
    mmap=None,
):
    """
    MoE-FFN 专用 load（对齐 load_ckpt 的核心语义：不match只load匹配部分）

    功能：
      - 主 ckpt：加载所有“非 dense-FFN”参数（key/shape 匹配才加入）
      - 主 ckpt：dense FFN 广播初始化到一部分 routed/task experts（由 split 决定）
        * 支持旧式 feed_forward.w1/w2/w3 和当前 DiT mlp.gate/up/down_proj 两套 key
      - audio ckpt：dense FFN 广播初始化到剩余 routed/task experts（由 split 决定）

    打印（不冗余）：
      - main/audio 的 expert split 分配
      - 主 dense->broadcast 是否覆盖到每一层（缺失层列表）
      - audio 覆盖到了哪些层（以及覆盖的 routed/task 专家集合）
      - Missing/Unexpected/Unmatched（canonical 去重）
    """
    import os
    import re
    import torch
    from torch.nn.parallel import DistributedDataParallel
    from utils.commons.hparams import hparams
    from utils.commons.io import print_once
    from utils.nn.model_utils import unwrap_model

    # ===== 去重打印聚合 =====
    aggregated_missing = set()
    aggregated_unexpected = set()
    aggregated_unmatched = set()

    def _canonical_key(k: str) -> str:
        return re.sub(r"\.\d+(\.|$)", ".<N>\\1", k)

    # ===== expert idx =====
    if expert_idx is None:
        expert_idx = hparams.get("moe_audio_expert_idx", 1)

    # ===== 解析并读取主 ckpt：复用 load_ckpt 的路径逻辑（但不 load opt）=====
    if checkpoint is None:
        if os.path.isfile(ckpt_base_dir):
            ckpt_path = ckpt_base_dir
            checkpoint = torch_load_dist(ckpt_base_dir, map_location=map_location, mmap=mmap)
        else:
            ckpt_path = f"{ckpt_base_dir}/model_only_last.ckpt"
            if os.path.exists(ckpt_path):
                checkpoint = torch_load_dist(ckpt_path, map_location=map_location, mmap=mmap)
            else:
                checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir, steps)

    # ===== 读取 audio ckpt =====
    audio_checkpoint = None
    audio_ckpt_path = ""
    if ckpt_audio_base_dir:
        if os.path.isfile(ckpt_audio_base_dir):
            audio_ckpt_path = ckpt_audio_base_dir
            audio_checkpoint = torch_load_dist(audio_ckpt_path, map_location=map_location, mmap=mmap)
        else:
            audio_ckpt_path = f"{ckpt_audio_base_dir}/model_only_last.ckpt"
            if os.path.exists(audio_ckpt_path):
                audio_checkpoint = torch_load_dist(audio_ckpt_path, map_location=map_location, mmap=mmap)
            else:
                audio_checkpoint, audio_ckpt_path = get_last_checkpoint(ckpt_audio_base_dir, steps)

    if checkpoint is None and audio_checkpoint is None:
        msg = f"| ckpt not found in main='{ckpt_base_dir}' audio='{ckpt_audio_base_dir}'"
        if force:
            raise FileNotFoundError(msg)
        if not silent:
            print_once(msg)
        return 0

    # ===== unwrap model =====
    if isinstance(cur_model, DistributedDataParallel):
        cur_model = cur_model.module
    cur_model = unwrap_model(cur_model)
    device = next(cur_model.parameters()).device

    # ===== ckpt state_dict_all 归一化 =====
    def _get_state_dict_all(ckpt_obj):
        if ckpt_obj is None:
            return None
        sd = ckpt_obj["state_dict"] if (isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj) else ckpt_obj
        sd_all = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
        return sd_all

    main_all = _get_state_dict_all(checkpoint)
    audio_all = _get_state_dict_all(audio_checkpoint)

    # ===== 按 load_ckpt 的 model_name 切片逻辑 =====
    def _slice_model_sd(state_dict_all, model_name_):
        if state_dict_all is None:
            return None
        if "." not in model_name_:
            return {
                k.replace("module.", "").replace("_orig_mod.", ""): v
                for k, v in state_dict_all[model_name_].items()
            }
        else:
            base_model_name = model_name_.split(".")[0]
            rest_model_name = model_name_[len(base_model_name) + 1 :]
            sub = {
                k[len(rest_model_name) + 1 :]: v
                for k, v in state_dict_all[base_model_name].items()
                if k.startswith(f"{rest_model_name}.")
            }
            return {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sub.items()}

    main_sd = _slice_model_sd(main_all, model_name) if main_all is not None else None
    audio_sd = _slice_model_sd(audio_all, model_name) if audio_all is not None else None

    # ===== 当前模型 state_dict（用于 shape 对齐 / repair）=====
    cur_sd = cur_model.state_dict()
    cur_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in cur_sd.items()}

    # ===== 推断层数 / expert 数（从当前模型 key 推断，写死 encoder.layers 前缀）=====
    def _infer_max_layer():
        pat = re.compile(r"^encoder\.layers\.(\d+)\.")
        mx = -1
        for k in cur_sd.keys():
            m = pat.match(k)
            if m:
                mx = max(mx, int(m.group(1)))
        return mx + 1

    def _infer_expert_count(prefix_pat: str) -> int:
        pat = re.compile(prefix_pat)
        mx = -1
        for k in cur_sd.keys():
            m = pat.search(k)
            if m:
                mx = max(mx, int(m.group(1)))
        return mx + 1

    n_layers = _infer_max_layer()
    num_routed = max(
        _infer_expert_count(r"encoder\.layers\.\d+\.feed_forward\.routed_experts\.(\d+)\."),
        _infer_expert_count(r"encoder\.layers\.\d+\.mlp\.routed_experts\.(\d+)\.ffn\."),
    )
    num_shared = max(
        _infer_expert_count(r"encoder\.layers\.\d+\.feed_forward\.shared_experts\.(\d+)\."),
        _infer_expert_count(r"encoder\.layers\.\d+\.mlp\.shared_experts\.(\d+)\.ffn\."),
    )
    num_task = max(
        _infer_expert_count(r"encoder\.layers\.\d+\.feed_forward\.task_experts\.(\d+)\."),
        _infer_expert_count(r"encoder\.layers\.\d+\.mlp\.task_experts\.(\d+)\.ffn\."),
    )

    # ===== dense ffn key patterns（ckpt）=====
    dense_ffn_pats = (
        ("feed_forward", re.compile(r"^encoder\.layers\.(\d+)\.feed_forward\.(w1|w2|w3)\.(weight|bias)$")),
        ("mlp", re.compile(r"^encoder\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$")),
    )

    def _match_dense_ffn_key(key: str):
        for layout, pat in dense_ffn_pats:
            m = pat.match(key)
            if m:
                return int(m.group(1)), layout, m.group(2), m.group(3)
        return None

    def _expert_ffn_key(layer_id: int, layout: str, expert_kind: str, expert_idx_: int, proj_name: str, tensor_name: str) -> str:
        if layout == "mlp":
            return (
                f"encoder.layers.{layer_id}.mlp.{expert_kind}_experts."
                f"{expert_idx_}.ffn.{proj_name}.{tensor_name}"
            )
        return (
            f"encoder.layers.{layer_id}.feed_forward.{expert_kind}_experts."
            f"{expert_idx_}.{proj_name}.{tensor_name}"
        )

    # ===== expert split: half(main) / half(audio) =====
    def _split_half_half(num: int, prefer_audio_idx: int = None, lock_odd_audio: bool = False):
        """
        返回 (main_set, audio_set)
        - 默认：odd -> audio, even -> main（约一半一半）
        - routed：尽量保证 prefer_audio_idx 在 audio_set（必要时翻转 parity）
        - task：默认同样按奇偶切分
        """
        if num <= 0:
            return set(), set()

        # 默认 odd->audio
        audio_set = {i for i in range(num) if (i % 2 == 1)}
        main_set = set(range(num)) - audio_set

        if prefer_audio_idx is not None and 0 <= prefer_audio_idx < num and (prefer_audio_idx not in audio_set):
            if not lock_odd_audio:
                # 翻转 parity 让 prefer_audio_idx 进 audio
                audio_set = {i for i in range(num) if (i % 2 == 0)}
                main_set = set(range(num)) - audio_set

        # 如果 audio_set 为空但确实有 audio ckpt，至少给一个（避免“全走 main”）
        if len(audio_set) == 0:
            if prefer_audio_idx is not None and 0 <= prefer_audio_idx < num:
                audio_set = {prefer_audio_idx}
            else:
                audio_set = {num - 1}
            main_set = set(range(num)) - audio_set

        return main_set, audio_set

    has_audio = (audio_sd is not None)

    if has_audio:
        # routed：尽量让 expert_idx 落到 audio
        routed_main_set, routed_audio_set = _split_half_half(
            num_routed, prefer_audio_idx=expert_idx, lock_odd_audio=False
        )
        task_main_set, task_audio_set = _split_half_half(
            num_task, prefer_audio_idx=expert_idx, lock_odd_audio=False
        )
    else:
        # 没有 audio ckpt：全部用 main 广播（保持原语义）
        routed_main_set, routed_audio_set = set(range(num_routed)), set()
        task_main_set, task_audio_set = set(range(num_task)), set()

    # ===== 构建“只 load 匹配部分”的 to_load =====
    to_load = {}
    copied_non_ffn = 0
    copied_main_ffn = 0
    copied_audio_ffn = 0
    main_ffn_layers = set()
    audio_ffn_layers = set()

    def _add_if_match(dst_key: str, src_tensor, where: str, layer_id: int = None):
        nonlocal copied_non_ffn, copied_main_ffn, copied_audio_ffn
        if dst_key not in cur_sd:
            aggregated_unexpected.add(_canonical_key(dst_key))
            return False
        if not isinstance(src_tensor, torch.Tensor):
            return False
        if cur_sd[dst_key].shape != src_tensor.shape:
            aggregated_unmatched.add(_canonical_key(dst_key))
            return False
        to_load[dst_key] = src_tensor
        if where == "non_ffn":
            copied_non_ffn += 1
        elif where == "main_ffn":
            copied_main_ffn += 1
            if layer_id is not None:
                main_ffn_layers.add(layer_id)
        elif where == "audio_ffn":
            copied_audio_ffn += 1
            if layer_id is not None:
                audio_ffn_layers.add(layer_id)
        return True

    # (A) 主 ckpt：加载非 dense-FFN（只 key/shape 匹配的）
    if main_sd is not None:
        for k, v in main_sd.items():
            if _match_dense_ffn_key(k) is not None and k not in cur_sd:
                continue
            if k in cur_sd:
                _add_if_match(k, v, where="non_ffn")
            else:
                aggregated_unexpected.add(_canonical_key(k))

    # (B) 主 ckpt：dense->broadcast 到 routed/task（只覆盖 main_set）
    if main_sd is not None and (num_routed > 0 or num_task > 0):
        for k, v in main_sd.items():
            dense_match = _match_dense_ffn_key(k)
            if dense_match is None:
                continue
            layer_id, layout, proj_name, tensor_name = dense_match

            # routed experts (main subset)
            for e in routed_main_set:
                dst = _expert_ffn_key(layer_id, layout, "routed", e, proj_name, tensor_name)
                if dst in cur_sd:
                    _add_if_match(dst, v, where="main_ffn", layer_id=layer_id)

            # task experts (main subset)
            for s in task_main_set:
                dst = _expert_ffn_key(layer_id, layout, "task", s, proj_name, tensor_name)
                if dst in cur_sd:
                    _add_if_match(dst, v, where="main_ffn", layer_id=layer_id)

    # (C) audio ckpt：dense->broadcast 覆盖 routed/task（只覆盖 audio_set）
    if audio_sd is not None:
        for k, v in audio_sd.items():
            dense_match = _match_dense_ffn_key(k)
            if dense_match is None:
                continue
            layer_id, layout, proj_name, tensor_name = dense_match

            # routed experts (audio subset)
            if num_routed > 0:
                for e in routed_audio_set:
                    dst = _expert_ffn_key(layer_id, layout, "routed", e, proj_name, tensor_name)
                    if dst in cur_sd:
                        _add_if_match(dst, v, where="audio_ffn", layer_id=layer_id)

            # task experts (audio subset)
            if num_task > 0:
                for s in task_audio_set:
                    dst = _expert_ffn_key(layer_id, layout, "task", s, proj_name, tensor_name)
                    if dst in cur_sd:
                        _add_if_match(dst, v, where="audio_ffn", layer_id=layer_id)

    # ===== 对齐 load_ckpt：strict=False && delete_unmatch=True 时调用 repair_unmatched_state_dict =====
    if (not strict) and delete_unmatch:
        to_load, repaired, removed = repair_unmatched_state_dict(cur_sd, to_load, silent)
        if removed:
            for k in removed:
                aggregated_unmatched.add(_canonical_key(k))

    # ===== 真正 load =====
    load_results = cur_model.load_state_dict(to_load, strict=strict)
    cur_model.to(device)

    # ===== 聚合 missing/unexpected =====
    for k in load_results.missing_keys:
        aggregated_missing.add(_canonical_key(k))
    for k in load_results.unexpected_keys:
        aggregated_unexpected.add(_canonical_key(k))

    # ===== 打印（不冗余）=====
    if not silent:
        print_once(
            f"| loaded '{model_name}' from main='{ckpt_path}' audio='{audio_ckpt_path}' "
            f"(strict={strict}, delete_unmatch={delete_unmatch})"
        )
        print_once(
            f"| Copied: non_ffn={copied_non_ffn}, main_ffn_broadcast={copied_main_ffn}, audio_ffn_broadcast={copied_audio_ffn} "
            f"| routed={num_routed}, task={num_task}, legacy_shared={num_shared}, layers={n_layers}"
        )

        if has_audio:
            print_once(
                f"| Expert split: routed main={sorted(routed_main_set)} audio={sorted(routed_audio_set)}; "
                f"task main={sorted(task_main_set)} audio={sorted(task_audio_set)}"
            )

        # 主 FFN broadcast 是否覆盖每层（以“该层至少写进过一个专家参数”为覆盖）
        if n_layers > 0 and copied_main_ffn > 0:
            if len(main_ffn_layers) == n_layers:
                print_once(f"| Main FFN broadcast: covered ALL layers (0..{n_layers-1})")
            else:
                miss = [i for i in range(n_layers) if i not in main_ffn_layers]
                print_once(f"| Main FFN broadcast: covered {len(main_ffn_layers)}/{n_layers}; missing_layers={miss}")

        # audio 覆盖层（注意：现在是覆盖一组 routed/task 专家）
        if has_audio and n_layers > 0:
            layers_sorted = sorted(list(audio_ffn_layers))
            if len(layers_sorted) == n_layers:
                print_once(
                    f"| Audio FFN broadcast: routed={sorted(routed_audio_set)} task={sorted(task_audio_set)} "
                    f"covered ALL layers (0..{n_layers-1})"
                )
            else:
                print_once(
                    f"| Audio FFN broadcast: routed={sorted(routed_audio_set)} task={sorted(task_audio_set)} "
                    f"applied_layers={layers_sorted}"
                )

        # canonical 去重打印
        if aggregated_missing:
            print_once("| ===== Missing key names (deduplicated by module) =====")
            for k in sorted(aggregated_missing):
                print_once(f"|   {k}")
        if aggregated_unexpected:
            print_once("| ===== Unexpected key names (deduplicated by module) =====")
            for k in sorted(aggregated_unexpected):
                print_once(f"|   {k}")
        if aggregated_unmatched:
            print_once("| ===== Unmatched (size-mismatch) key names (deduplicated by module) =====")
            for k in sorted(aggregated_unmatched):
                print_once(f"|   {k}")

    # ===== 返回 global_step（对齐 load_ckpt）=====
    if isinstance(checkpoint, dict):
        return checkpoint.get("global_step", 0)
    return 0
