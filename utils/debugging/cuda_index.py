import torch
from torch.utils._python_dispatch import TorchDispatchMode

def _is_int_index(t):
    return torch.is_tensor(t) and t.dtype in (torch.int64, torch.int32)

def _safe_minmax(t):
    if not torch.is_tensor(t) or t.numel() == 0:
        return None, None
    return int(t.min().item()), int(t.max().item())

class IndexDebugMode(TorchDispatchMode):
    def __init__(self, stop_on_oob=True, verbose=True):
        self.stop_on_oob = stop_on_oob
        self.verbose = verbose

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        name = func.name()  # e.g. 'aten::gather'
        try:
            if name.startswith("aten::gather"):
                # signature: (self, dim, index, sparse_grad=False)
                inp, dim, index = args[0], int(args[1]), args[2]
                m, M = _safe_minmax(index)
                if self.verbose:
                    print(f"[gather] dim={dim} input.shape={tuple(inp.shape)} index.shape={tuple(index.shape)} "
                          f"index[min,max]={m,M} valid=[0,{inp.size(dim)-1}]")
                if self.stop_on_oob and m is not None and (m < 0 or M >= inp.size(dim)):
                    raise RuntimeError(f"OOB in gather: dim={dim}, input.size={inp.size(dim)}, min={m}, max={M}")

            elif name.startswith("aten::take_along_dim"):
                # signature: (self, indices, dim=None)
                inp, indices = args[0], args[1]
                dim = kwargs.get("dim", args[2] if len(args) > 2 else None)
                dim = 0 if dim is None else int(dim)
                m, M = _safe_minmax(indices)
                if self.verbose:
                    print(f"[take_along_dim] dim={dim} input.shape={tuple(inp.shape)} index.shape={tuple(indices.shape)} "
                          f"index[min,max]={m,M} valid=[0,{inp.size(dim)-1}]")
                if self.stop_on_oob and m is not None and (m < 0 or M >= inp.size(dim)):
                    raise RuntimeError(f"OOB in take_along_dim: dim={dim}, size={inp.size(dim)}, min={m}, max={M}")

            elif name.startswith("aten::index_select"):
                # signature: (self, dim, index)
                inp, dim, index = args[0], int(args[1]), args[2]
                m, M = _safe_minmax(index)
                if self.verbose:
                    print(f"[index_select] dim={dim} input.shape={tuple(inp.shape)} index.shape={tuple(index.shape)} "
                          f"index[min,max]={m,M} valid=[0,{inp.size(dim)-1}]")
                if self.stop_on_oob and m is not None and (m < 0 or M >= inp.size(dim)):
                    raise RuntimeError(f"OOB in index_select: dim={dim}, size={inp.size(dim)}, min={m}, max={M}")

            elif name.startswith("aten::scatter") or name.startswith("aten::scatter_add"):
                # signature: (self, dim, index, src, reduce=None)
                inp, dim, index = args[0], int(args[1]), args[2]
                if _is_int_index(index):
                    m, M = _safe_minmax(index)
                    if self.verbose:
                        print(f"[{name}] dim={dim} input.shape={tuple(inp.shape)} index.shape={tuple(index.shape)} "
                              f"index[min,max]={m,M} valid=[0,{inp.size(dim)-1}]")
                    if self.stop_on_oob and m is not None and (m < 0 or M >= inp.size(dim)):
                        raise RuntimeError(f"OOB in {name}: dim={dim}, size={inp.size(dim)}, min={m}, max={M}")

            elif name.startswith("aten::embedding"):
                # signature: (weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False)
                weight, indices = args[0], args[1]
                m, M = _safe_minmax(indices)
                if self.verbose:
                    print(f"[embedding] num_embeddings={weight.size(0)} index.shape={tuple(indices.shape)} index[min,max]={m,M}")
                if self.stop_on_oob and m is not None and (m < 0 or M >= weight.size(0)):
                    raise RuntimeError(f"OOB in embedding: num_embeddings={weight.size(0)}, min={m}, max={M}")

            elif name.startswith("aten::index"):
                # 高级索引 x[indices]，indices 是一个 list/tuple（可含 slice/None/bool/int/张量）
                inp, indices = args[0], args[1]
                if isinstance(indices, (list, tuple)):
                    msg_parts = []
                    dim_cursor = 0
                    for i, ind in enumerate(indices):
                        if torch.is_tensor(ind) and ind.dtype in (torch.int64, torch.int32):
                            m, M = _safe_minmax(ind)
                            msg_parts.append(f"dim? idx_tensor[{i}] min={m} max={M}")
                        elif torch.is_tensor(ind) and ind.dtype == torch.bool:
                            msg_parts.append(f"bool_mask[{i}] shape={tuple(ind.shape)}")
                        elif ind is Ellipsis:
                            msg_parts.append("...")
                        elif ind is None:
                            msg_parts.append("None(newaxis)")
                        else:
                            msg_parts.append(f"{ind}")
                    if self.verbose:
                        print(f"[advanced index] input.shape={tuple(inp.shape)} indices=({', '.join(msg_parts)})")
                # 高级索引的越界检查更复杂（广播/整型索引会触发新张量构造），此处先记录；如需可进一步严格检查。

        except Exception:
            # 保持原始栈，便于定位
            raise

        # 放行实际的算子执行
        return func(*args, **kwargs)
