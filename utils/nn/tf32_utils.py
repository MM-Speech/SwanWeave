import torch
from contextlib import contextmanager

@contextmanager
def tf32_enable(enable: bool):
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32

    old_prec = None
    has_prec = hasattr(torch, "get_float32_matmul_precision") and hasattr(torch, "set_float32_matmul_precision")
    if has_prec:
        old_prec = torch.get_float32_matmul_precision()

    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(enable)
        torch.backends.cudnn.allow_tf32 = bool(enable)

        # torch>=2.0：对 fp32 matmul 精度也有影响（可选）
        if has_prec and (not enable):
            torch.set_float32_matmul_precision("highest")

        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
        if has_prec and old_prec is not None:
            torch.set_float32_matmul_precision(old_prec)
