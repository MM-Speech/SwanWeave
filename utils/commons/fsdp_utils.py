from typing import Dict
import torch
from torch._dynamo.eval_frame import OptimizedModule
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer


def get_model_states(model: Module, *, sharded: bool = False):
    """
    Get model state dict.
    Call by all ranks.
    If full state dict, only use the result on rank 0.
    If sharded state dict, only for fsdp model.
    """
    if isinstance(model, OptimizedModule):
        model = model._orig_mod
    if isinstance(model, DistributedDataParallel):
        model = model.module
    if isinstance(model, FullyShardedDataParallel):
        configure_fsdp_states(model, sharded=sharded)
    return model.state_dict()


def get_optimizer_states(optimizer: Optimizer):
    return optimizer.state_dict()


def get_fsdp_optimizer_states(
        optimizer: Optimizer,
        model: FullyShardedDataParallel,
        *,
        sharded: bool = False,
):
    """
    Get fsdp optimizer state dict.
    Call by all ranks.
    If full state dict, only use the result on rank 0.
    If sharded state dict, only for fsdp model.
    """
    configure_fsdp_states(model, sharded=sharded)
    states = optimizer.state_dict()
    states = FullyShardedDataParallel.optim_state_dict(
        model=model,
        optim=optimizer,
        optim_state_dict=states,
    )
    return states


def set_fsdp_optimizer_states(
        states: Dict,
        optimizer: Optimizer,
        model: FullyShardedDataParallel,
):
    """
    Set fsdp optimizer state dict.
    Call by all ranks.
    """
    configure_fsdp_states(model, rank0_only=False)

    try:
        states_to_load = FullyShardedDataParallel.optim_state_dict_to_load(
            model=model,
            optim=optimizer,
            optim_state_dict=states,
        )
    except KeyError:
        states_remapped = _remap_fsdp_optim_state_dict_param_names_for_compile(states, model)
        states_to_load = FullyShardedDataParallel.optim_state_dict_to_load(
            model=model,
            optim=optimizer,
            optim_state_dict=states_remapped,
        )

    optimizer.load_state_dict(states_to_load)


def configure_fsdp_states(
        model: FullyShardedDataParallel,
        *,
        rank0_only: bool = True,
        sharded: bool = False,
):
    """
    Configure fsdp state dict type.
    """
    if not sharded:
        FullyShardedDataParallel.set_state_dict_type(
            module=model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=rank0_only
            ),
        )
    else:
        FullyShardedDataParallel.set_state_dict_type(
            module=model,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
            optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
        )


def _remap_fsdp_optim_state_dict_param_names_for_compile(
        optim_state_dict: Dict,
        model: FullyShardedDataParallel,
) -> Dict:
    if not isinstance(optim_state_dict, dict):
        return optim_state_dict
    if "state" not in optim_state_dict or "param_groups" not in optim_state_dict:
        return optim_state_dict
    if not isinstance(optim_state_dict["state"], dict) or not isinstance(optim_state_dict["param_groups"], list):
        return optim_state_dict

    root = model
    if isinstance(root, OptimizedModule):
        root = root._orig_mod
    if isinstance(root, FullyShardedDataParallel):
        root = root.module

    compiled_prefixes = set()
    try:
        for name, mod in root.named_modules():
            if isinstance(mod, OptimizedModule) and name != "":
                compiled_prefixes.add(name)
    except Exception:
        compiled_prefixes = set()

    compiled_prefixes = sorted(compiled_prefixes, key=len, reverse=True)

    def add_orig_mod_if_needed(param_name: str) -> str:
        if not compiled_prefixes:
            return param_name
        out = param_name
        for p in compiled_prefixes:
            p_dot = p + "."
            p_orig = p + "._orig_mod."
            if out.startswith(p_dot) and not out.startswith(p_orig):
                out = p_orig + out[len(p_dot):]
                break
        return out

    def strip_orig_mod(param_name: str) -> str:
        return param_name.replace("._orig_mod.", ".")

    sample_keys = list(optim_state_dict["state"].keys())
    has_orig_mod = any(isinstance(k, str) and "._orig_mod." in k for k in sample_keys[:256])

    if compiled_prefixes:
        state_new = {}
        changed = False
        for k, v in optim_state_dict["state"].items():
            if isinstance(k, str):
                nk = add_orig_mod_if_needed(k)
                changed = changed or (nk != k)
                state_new[nk] = v
            else:
                state_new[k] = v

        param_groups_new = []
        for group in optim_state_dict["param_groups"]:
            if not isinstance(group, dict):
                param_groups_new.append(group)
                continue
            group_new = dict(group)
            params = group_new.get("params", None)
            if isinstance(params, list):
                new_params = []
                for p in params:
                    if isinstance(p, str):
                        new_params.append(add_orig_mod_if_needed(p))
                    else:
                        new_params.append(p)
                group_new["params"] = new_params
            param_groups_new.append(group_new)

        if not changed and param_groups_new == optim_state_dict["param_groups"]:
            return optim_state_dict

        out = dict(optim_state_dict)
        out["state"] = state_new
        out["param_groups"] = param_groups_new
        return out

    if (not compiled_prefixes) and has_orig_mod:
        state_new = {}
        for k, v in optim_state_dict["state"].items():
            if isinstance(k, str):
                state_new[strip_orig_mod(k)] = v
            else:
                state_new[k] = v

        param_groups_new = []
        for group in optim_state_dict["param_groups"]:
            if not isinstance(group, dict):
                param_groups_new.append(group)
                continue
            group_new = dict(group)
            params = group_new.get("params", None)
            if isinstance(params, list):
                new_params = []
                for p in params:
                    if isinstance(p, str):
                        new_params.append(strip_orig_mod(p))
                    else:
                        new_params.append(p)
                group_new["params"] = new_params
            param_groups_new.append(group_new)

        out = dict(optim_state_dict)
        out["state"] = state_new
        out["param_groups"] = param_groups_new
        return out

    return optim_state_dict