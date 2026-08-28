from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
from safetensors.torch import load_file


PathLike = Union[str, Path]


def _import_snapshot_download():
    from huggingface_hub import snapshot_download

    return snapshot_download


def _import_accelerate():
    import accelerate

    return accelerate


def download_tokenizer_snapshot(
    cache_dir: Optional[PathLike] = None,
    repo_id: str = "amphion/Vevo",
) -> Path:
    snapshot_download = _import_snapshot_download()
    kwargs = dict(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=["tokenizer/vq32/*", "tokenizer/vq8192/*"],
    )
    if cache_dir is None:
        snapshot_path = snapshot_download(**kwargs)
        return Path(snapshot_path)

    local_dir = _as_path(cache_dir)
    snapshot_path = snapshot_download(
        **kwargs,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return Path(snapshot_path)


def _as_path(path: PathLike) -> Path:
    return Path(path).expanduser()


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def default_content_checkpoint(root: PathLike) -> Path:
    root = _as_path(root)
    return root / "tokenizer" / "vq32" / "hubert_large_l18_c32.pkl"


def default_content_style_checkpoint(root: PathLike) -> Path:
    root = _as_path(root)
    return root / "tokenizer" / "vq8192"


def resolve_content_checkpoint_path(path: PathLike) -> Path:
    path = _as_path(path)
    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"content checkpoint not found: {path}")

    direct = _first_existing(
        [
            path / "hubert_large_l18_c32.pkl",
            path / "hubert_large_l18_c32.safetensors",
            path / "model.safetensors",
        ]
    )
    if direct is not None:
        return direct

    matches = sorted(path.rglob("hubert_large_l18_c32.pkl"))
    if matches:
        return matches[0]

    matches = sorted(path.rglob("*.pkl")) + sorted(path.rglob("*.safetensors"))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(f"unable to resolve content checkpoint under: {path}")


def resolve_content_style_checkpoint_path(path: PathLike) -> Path:
    path = _as_path(path)
    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"content-style checkpoint not found: {path}")

    direct = _first_existing(
        [
            path / "model.safetensors",
            path / "pytorch_model.bin",
        ]
    )
    if direct is not None:
        return direct

    safetensors_files = sorted(path.glob("*.safetensors"))
    if len(safetensors_files) == 1:
        return safetensors_files[0]

    bin_files = sorted(path.glob("*.bin"))
    if len(bin_files) == 1:
        return bin_files[0]

    index_files = list(path.glob("*.index.json"))
    if index_files or safetensors_files or bin_files:
        return path

    raise FileNotFoundError(f"unable to resolve content-style checkpoint under: {path}")


def _unwrap_state_dict(
    payload: object,
    candidate_paths: Sequence[Tuple[str, ...]],
) -> dict:
    for candidate_path in candidate_paths:
        current = payload
        valid = True
        for key in candidate_path:
            if not isinstance(current, dict) or key not in current:
                valid = False
                break
            current = current[key]
        if valid and isinstance(current, dict):
            return current

    if isinstance(payload, dict) and all(torch.is_tensor(value) for value in payload.values()):
        return payload

    raise ValueError("unable to unwrap checkpoint payload into a state_dict")


def load_state_dict_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: PathLike,
    candidate_paths: Sequence[Tuple[str, ...]],
    strict: bool = True,
) -> None:
    checkpoint_path = _as_path(checkpoint_path)

    if checkpoint_path.suffix == ".safetensors":
        state_dict = load_file(str(checkpoint_path))
        model.load_state_dict(state_dict, strict=strict)
        return

    payload = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = _unwrap_state_dict(payload, candidate_paths)
    model.load_state_dict(state_dict, strict=strict)


def load_dispatch_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: PathLike,
) -> torch.nn.Module:
    checkpoint_path = _as_path(checkpoint_path)
    accelerate = _import_accelerate()
    return accelerate.load_checkpoint_and_dispatch(model, str(checkpoint_path))
