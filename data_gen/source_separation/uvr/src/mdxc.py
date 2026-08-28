from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import torch

from data_gen.source_separation.common import normalize_device, resolve_uvr_weight_dir

from .models_dir.mdxc import mdxc_interface as mdxc_api
from .utils.fastio import read

MODELS_JSON_PATH = Path(__file__).resolve().parent / "models_dir" / "models.json"
MODELPARAMS_ROOT = Path(__file__).resolve().parent / "models_dir" / "mdxc" / "modelparams"

with MODELS_JSON_PATH.open("r") as f:
    MODELS_JSON = json.load(f)


class BaseModel:
    def __init__(self, name: str, architecture: str, other_metadata: dict, device=None, logger=None):
        self.name = name
        self.architecture = architecture
        self.other_metadata = other_metadata
        self.logger = logger
        self.device = normalize_device(device)

    def __call__(self, audio: Union[npt.NDArray, str], sampling_rate: int = None, **kwargs) -> dict:
        if isinstance(audio, str):
            return self.predict_path(audio, **kwargs)
        return self.predict(audio, sampling_rate, **kwargs)

    def predict(self, audio: npt.NDArray, sampling_rate: int, **kwargs) -> dict:
        raise NotImplementedError

    def predict_path(self, audio: str, **kwargs) -> dict:
        raise NotImplementedError

    def separate(self, audio: npt.NDArray, sampling_rate: int = None) -> dict:
        return self.__call__(audio, sampling_rate)

    def __repr__(self):
        return f"Architecture {self.architecture}, model {self.name}. With other_metadata {self.other_metadata}"

    def to(self, device):
        self.device = normalize_device(device)

    def update_metadata(self, metadata: dict):
        self.other_metadata.update(metadata)

    @staticmethod
    def list_models() -> list:
        models_list = []
        for arch, arch_models in MODELS_JSON.items():
            models_list.extend([f"{arch}: {model}" for model in arch_models.keys()])
        return models_list


class MDXC(BaseModel):
    def __init__(
        self,
        name: str,
        other_metadata: dict,
        device=None,
        logger=None,
        model_root=None,
        allow_legacy_fallback: bool = True,
        precision: str = "fp32",
    ):
        super().__init__(name, "mdxc", other_metadata, device, logger)

        weight_dir = resolve_uvr_weight_dir(
            "mdxc",
            name,
            model_root=model_root,
            allow_legacy_fallback=allow_legacy_fallback,
        )
        files = sorted(weight_dir.iterdir())
        model_files = [p for p in files if p.is_file()]
        if not model_files:
            raise FileNotFoundError(f"uvr 缺少 mdxc 权重文件: {weight_dir}")

        self.sample_rate = 44100
        self.precision = precision
        self.models_data = mdxc_api.load_mdxc_models_data(str(MODELPARAMS_ROOT / "model_data.json"))

        model_path = model_files[0]
        model_hash = mdxc_api.get_model_hash_from_path(model_path=str(model_path))
        model_data = mdxc_api.load_mdxc_model_data(
            self.models_data,
            model_hash,
            model_path=str(MODELPARAMS_ROOT),
        )
        model_run = mdxc_api.load_modle(str(model_path), model_data, self.device)

        self.model_dir = weight_dir
        self.model_path = model_path
        self.model_data = model_data
        self.model_run = model_run

        self.init_metadata()
        self.update_metadata(other_metadata)

    def init_metadata(self):
        self.other_metadata = {
            "is_mdx_c_seg_def": False,
            "segment_size": 256,
            "batch_size": 1,
            "overlap_mdx23": 8,
            "semitone_shift": 0,
        }

    def to(self, device):
        self.device = normalize_device(device)
        self.model_run.to(self.device).eval()

    def predict(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        *,
        batch_size: int = 1,
        chunk_size_sec: Optional[float] = None,
        **kwargs,
    ) -> dict:
        return self.predict_batch(
            [audio],
            sampling_rate=sampling_rate,
            batch_size=batch_size,
            chunk_size_sec=chunk_size_sec,
            **kwargs,
        )[0]

    def predict_batch(
        self,
        audio_list: list[np.ndarray],
        sampling_rate: int,
        *,
        batch_size: int = 1,
        chunk_size_sec: Optional[float] = None,
        **kwargs,
    ) -> list[dict]:
        mixes = [mdxc_api.prepare_mix(audio) for audio in audio_list]
        chunk_size_samples = None
        if chunk_size_sec is not None:
            chunk_size_samples = mdxc_api.seconds_to_chunk_size_samples(
                float(chunk_size_sec),
                sample_rate=self.sample_rate,
                hop_length=int(self.model_data.audio.hop_length),
            )

        stems_list = mdxc_api.demix_batch(
            mixes,
            self.other_metadata,
            self.model_run,
            self.model_data,
            self.device,
            batch_size=int(batch_size),
            precision=self.precision,
            chunk_size_samples=chunk_size_samples,
        )
        return [{"origin": audio, "separated": mdxc_api.rename_stems(stems)} for audio, stems in zip(audio_list, stems_list)]

    def predict_path(self, audio: str, **kwargs) -> dict:
        audio_arr, sampling_rate = read(audio, target_sampling_rate=self.sample_rate)
        return self.predict(audio_arr, sampling_rate, **kwargs)

    @staticmethod
    def list_models() -> dict:
        return list(MODELS_JSON["mdxc"].keys())
