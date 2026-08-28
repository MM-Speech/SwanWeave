from typing import Tuple, List
import torch
import torch.distributed as dist
import numpy as np


def reduce_tensors(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            dist.all_reduce(v)
            v = v / dist.get_world_size()
        if type(v) is dict:
            v = reduce_tensors(v)
        new_metrics[k] = v
    return new_metrics


def tensors_to_scalars(tensors):
    if isinstance(tensors, torch.Tensor):
        tensors = tensors.item()
        return tensors
    elif isinstance(tensors, dict):
        new_tensors = {}
        for k, v in tensors.items():
            v = tensors_to_scalars(v)
            new_tensors[k] = v
        return new_tensors
    elif isinstance(tensors, list):
        return [tensors_to_scalars(v) for v in tensors]
    else:
        return tensors


def convert_to_np(tensors):
    if isinstance(tensors, np.ndarray):
        return tensors
    elif isinstance(tensors, dict):
        new_np = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = convert_to_np(v)
            new_np[k] = v
    elif isinstance(tensors, list):
        new_np = []
        for v in tensors:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = convert_to_np(v)
            new_np.append(v)
    elif isinstance(tensors, torch.Tensor):
        v = tensors
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if type(v) is dict:
            v = convert_to_np(v)
        new_np = v
    else:
        raise Exception(f'tensors_to_np does not support type {type(tensors)}.')
    return new_np


def convert_to_tensor(arrays):
    from scipy.sparse import csr_matrix
    if isinstance(arrays, np.ndarray):
        ret = torch.from_numpy(arrays)
        if arrays.dtype in [np.float64, np.float32]:
            ret = ret.float()
        if arrays.dtype in [np.int64, np.uint32, np.int32, np.uint16, np.int16, np.int8, np.uint8]:
            ret = ret.long()
    elif isinstance(arrays, csr_matrix):
        ret = torch.from_numpy(arrays.todense())
    elif isinstance(arrays, torch.Tensor):
        ret = arrays
    elif isinstance(arrays, list):
        ret = []
        for arr in arrays:
            ret.append(convert_to_tensor(arr))
    elif type(arrays) is dict:
        ret = {}
        for k, v in arrays.items():
            v = convert_to_tensor(v)
            ret[k] = v
    else:
        ret = arrays
    return ret

def convert_like(inp, target):
    if isinstance(target, np.ndarray):
        return convert_to_np(inp)
    elif isinstance(target, torch.Tensor):
        inp = convert_to_tensor(inp)
        inp = inp.to()
        if target.device == 'cpu':
            return move_to_cpu(inp)
        else:
            return move_to_cuda(inp)

def move_to_cpu(batch):
    if callable(getattr(batch, 'cpu', None)):
        return batch.cpu()
    elif callable(getattr(batch, 'to', None)):
        return batch.to('cpu')
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = move_to_cpu(x)
        return batch
    elif isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = move_to_cpu(x)
        return tuple(batch)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = move_to_cpu(v)
        return batch
    elif isinstance(batch, int) or isinstance(batch, float) or isinstance(batch, str):
        return batch
    elif batch is None:
        return None
    return batch


def move_to_cuda(batch, gpu_id=0, device=None):
    if device is None:
        device = torch.device('cuda', gpu_id)
    else:
        if isinstance(device, torch.device):
            gpu_id = device.index
        elif isinstance(device, str):
            device = torch.device(device)
            gpu_id = device.index
    if callable(getattr(batch, 'to', None)):
        return batch.to(device, non_blocking=True)
    elif callable(getattr(batch, 'cuda', None)):
        return batch.cuda(gpu_id, non_blocking=True)
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id, device)
        return batch
    elif isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id, device)
        return tuple(batch)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = move_to_cuda(v, gpu_id, device)
        return batch
    elif isinstance(batch, int) or isinstance(batch, float) or isinstance(batch, str):
        return batch
    elif batch is None:
        return None
    return batch


def convert_tensors_to_dtype(batch, dtype=torch.float32):
    if callable(getattr(batch, 'to', None)):
        return batch.to(dtype, non_blocking=True)
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = convert_tensors_to_dtype(x, dtype)
        return batch
    elif isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = convert_tensors_to_dtype(x, dtype)
        return tuple(batch)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = convert_tensors_to_dtype(v, dtype)
        return batch
    elif isinstance(batch, int) or isinstance(batch, float) or isinstance(batch, str):
        return batch
    elif batch is None:
        return None
    return batch


def all_gather_varlen_tensor(t, dim=0):
    ws = torch.distributed.get_world_size()
    local_len = torch.tensor([t.size(dim)], device=t.device, dtype=torch.long)
    lens = [torch.zeros_like(local_len) for _ in range(ws)]
    torch.distributed.all_gather(lens, local_len)
    lens = torch.cat(lens).cpu().tolist()
    max_len = max(lens)

    if t.size(dim) < max_len:
        pad_shape = list(t.shape)
        pad_shape[dim] = max_len - t.size(dim)
        pad = torch.zeros(pad_shape, device=t.device, dtype=t.dtype)
        t_pad = torch.cat([t, pad], dim=dim)
    else:
        t_pad = t

    gathered = [torch.zeros_like(t_pad) for _ in range(ws)]
    torch.distributed.all_gather(gathered, t_pad)
    # 截断
    chunks = []
    for r, g in enumerate(gathered):
        sl = [slice(None)] * g.ndim
        sl[dim] = slice(0, lens[r])
        chunks.append(g[tuple(sl)])
    return torch.cat(chunks, dim=dim)


def all_gather_varlen_tensor_stack(t, dim=0, pad_value=0.0, return_lens=False):
    import torch

    ws = torch.distributed.get_world_size()

    local_len = torch.tensor([t.size(dim)], device=t.device, dtype=torch.long)
    lens_t = [torch.zeros_like(local_len) for _ in range(ws)]
    torch.distributed.all_gather(lens_t, local_len)
    lens = torch.cat(lens_t)  # shape [ws]
    max_len = lens.max().item()

    if t.size(dim) < max_len:
        pad_shape = list(t.shape)
        pad_shape[dim] = max_len - t.size(dim)
        pad = torch.full(pad_shape, pad_value, device=t.device, dtype=t.dtype)
        t_pad = torch.cat([t, pad], dim=dim)
    else:
        t_pad = t

    gathered = [torch.zeros_like(t_pad) for _ in range(ws)]
    torch.distributed.all_gather(gathered, t_pad)
    stacked = torch.stack(gathered, dim=0)

    if return_lens:
        return stacked, lens
    else:
        return stacked


def tensor_mean_per_element(t, t_mask):
    if t_mask.ndim == t.ndim - 1:
        t_mask = t_mask.unsqueeze(-1)
    t_mask = t_mask.to(t.dtype)
    num = (t * t_mask).sum()
    denom = t_mask.sum() * t.shape[2]
    out = torch.where(denom > 0, num / denom, torch.zeros_like(num))
    return out


def tensor_mean_per_seq(t, t_mask):
    mask = t_mask
    if mask.ndim == t.ndim - 1:
        mask = mask.unsqueeze(-1)          # [B, T, 1]
    mask = mask.to(t.dtype)
    num = (t * mask).sum(dim=(1, 2))     # [B]，有效位置的总和
    steps = mask.sum(dim=1).squeeze(-1)    # [B]，每序列有效时间步数
    denom = steps * t.shape[2]             # 有效步数 × C
    out = torch.where(denom > 0, num / denom, torch.zeros_like(num))
    return out


def slice_batch_value(v, slc):
    if isinstance(v, torch.Tensor):
        return v[slc]
    if isinstance(v, list):
        return v[slc] if isinstance(slc, slice) else [v[i] for i in slc]
    if isinstance(v, dict):
        return {kk: slice_batch_value(vv, slc) for kk, vv in v.items()}
    return v


def repeat_batch_value(v, repeat_k: int):
    if repeat_k <= 1:
        return v
    if torch.is_tensor(v) and v.ndim >= 1:
        return v.repeat_interleave(repeat_k, dim=0)
    if isinstance(v, list):
        out = []
        for x in v:
            out.extend([x] * repeat_k)
        return out
    if isinstance(v, dict):
        return {kk: repeat_batch_value(vv, repeat_k) for kk, vv in v.items()}
    return v

