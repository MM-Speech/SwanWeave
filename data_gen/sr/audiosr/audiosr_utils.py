import contextlib
import importlib
import shutil
from inspect import isfunction, signature
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

import os
import subprocess
import tempfile
import json
import soundfile as sf
import time
import wave
import torchaudio
import progressbar
from librosa.filters import mel as librosa_mel_fn
from data_gen.sr.audiosr.lowpass import lowpass

hann_window = {}
mel_basis = {}

AUDIO_SR_CHECKPOINTS = {
    "basic": ("haoheliu/audiosr_basic", "pytorch_model.bin"),
    "speech": ("haoheliu/audiosr_speech", "pytorch_model.bin"),
}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def _locate_cutoff_freq(stft, percentile=0.97):
    def _find_cutoff(x, percentile=0.95):
        percentile = x[-1] * percentile
        for i in range(1, x.shape[0]):
            if x[-i] < percentile:
                return x.shape[0] - i
        return 0

    magnitude = torch.abs(stft)
    energy = torch.cumsum(torch.sum(magnitude, dim=0), dim=0)
    return _find_cutoff(energy, percentile)


def pad_wav(waveform, target_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

    if waveform_length == target_length:
        return waveform

    # Pad
    temp_wav = np.zeros((1, target_length), dtype=np.float32)
    rand_start = 0

    temp_wav[:, rand_start : rand_start + waveform_length] = waveform
    return temp_wav


def lowpass_filtering_prepare_inference(dl_output):
    waveform = dl_output["waveform"]  # [1, samples]
    sampling_rate = dl_output["sampling_rate"]

    cutoff_freq = (
        _locate_cutoff_freq(dl_output["stft"], percentile=0.985) / 1024
    ) * 24000
    
    # If the audio is almost empty. Give up processing
    if(cutoff_freq < 1000):
        cutoff_freq = 24000

    order = 8
    ftype = dl_output.get("lowpass_filter_type", "butter")
    filtered_audio = lowpass(
        waveform.numpy().squeeze(),
        highcut=cutoff_freq,
        fs=sampling_rate,
        order=order,
        _type=ftype,
    )

    filtered_audio = torch.FloatTensor(filtered_audio.copy()).unsqueeze(0)

    if waveform.size(-1) <= filtered_audio.size(-1):
        filtered_audio = filtered_audio[..., : waveform.size(-1)]
    else:
        filtered_audio = F.pad(
            filtered_audio, (0, waveform.size(-1) - filtered_audio.size(-1))
        )

    return {"waveform_lowpass": filtered_audio}


def mel_spectrogram_train(y):
    global mel_basis, hann_window

    sampling_rate = 48000
    filter_length = 2048
    hop_length = 480
    win_length = 2048
    n_mel = 256
    mel_fmin = 20
    mel_fmax = 24000

    if 24000 not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=filter_length, n_mels=n_mel, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis[str(mel_fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_length).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((filter_length - hop_length) / 2), int((filter_length - hop_length) / 2)),
        mode="reflect",
    )

    y = y.squeeze(1)

    stft_spec = torch.stft(
        y,
        filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[str(y.device)],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    stft_spec = torch.abs(stft_spec)

    mel = spectral_normalize_torch(
        torch.matmul(mel_basis[str(mel_fmax) + "_" + str(y.device)], stft_spec)
    )

    return mel[0], stft_spec[0]


def pad_spec(log_mel_spec, target_frame):
    n_frames = log_mel_spec.shape[0]
    p = target_frame - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        log_mel_spec = m(log_mel_spec)
    elif p < 0:
        log_mel_spec = log_mel_spec[0:target_frame, :]

    if log_mel_spec.size(-1) % 2 != 0:
        log_mel_spec = log_mel_spec[..., :-1]

    return log_mel_spec


def wav_feature_extraction(waveform, target_frame):
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    log_mel_spec, stft = mel_spectrogram_train(waveform.unsqueeze(0))

    log_mel_spec = torch.FloatTensor(log_mel_spec.T)
    stft = torch.FloatTensor(stft.T)

    log_mel_spec, stft = pad_spec(log_mel_spec, target_frame), pad_spec(
        stft, target_frame
    )
    return log_mel_spec, stft


def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5

def read_wav_file(filename):
    waveform, sr = torchaudio.load(filename)
    duration = waveform.size(-1) / sr

    if(duration > 10.24):
        print("\033[93m {}\033[00m" .format("Warning: audio is longer than 10.24 seconds, may degrade the model performance. It's recommand to truncate your audio to 5.12 seconds before input to AudioSR to get the best performance."))

    if(duration % 5.12 != 0):
        pad_duration = duration + (5.12 - duration % 5.12)
    else:
        pad_duration = duration

    target_frame = int(pad_duration * 100)

    waveform = torchaudio.functional.resample(waveform, sr, 48000)

    waveform = waveform.numpy()[0, ...]

    waveform = normalize_wav(
        waveform
    )  # TODO rescaling the waveform will cause low LSD score

    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, target_length=int(48000 * pad_duration))
    return waveform, target_frame, pad_duration

def read_audio_file(filename):
    waveform, target_frame, duration = read_wav_file(filename)
    log_mel_spec, stft = wav_feature_extraction(waveform, target_frame)
    return log_mel_spec, stft, waveform, duration, target_frame


def read_list(fname):
    result = []
    with open(fname, "r", encoding="utf-8") as f:
        for each in f.readlines():
            each = each.strip("\n")
            result.append(each)
    return result


def get_duration(fname):
    with contextlib.closing(wave.open(fname, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


def get_bit_depth(fname):
    with contextlib.closing(wave.open(fname, "r")) as f:
        bit_depth = f.getsampwidth() * 8
        return bit_depth


def get_time():
    t = time.localtime()
    return time.strftime("%d_%m_%Y_%H_%M_%S", t)


def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def strip_silence(orignal_path, input_path, output_path):
    get_dur = subprocess.run([
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'format=duration',
        '-sexagesimal',
        '-of', 'json',
        orignal_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    duration = json.loads(get_dur.stdout)['format']['duration']
    
    subprocess.run([
        'ffmpeg',
        '-loglevel', 'error',
        '-y',
        '-ss', '00:00:00',
        '-i', input_path,
        '-t', duration,
        '-c', 'copy',
        output_path
    ])
    os.remove(input_path)



def save_wave(waveform, inputpath, savepath, name="outwav", samplerate=16000):
    if type(name) is not list:
        name = [name] * waveform.shape[0]

    for i in range(waveform.shape[0]):
        if waveform.shape[0] > 1:
            fname = "%s_%s.wav" % (
                os.path.basename(name[i])
                if (not ".wav" in name[i])
                else os.path.basename(name[i]).split(".")[0],
                i,
            )
        else:
            fname = (
                "%s.wav" % os.path.basename(name[i])
                if (not ".wav" in name[i])
                else os.path.basename(name[i]).split(".")[0]
            )
            # Avoid the file name too long to be saved
            if len(fname) > 255:
                fname = f"{hex(hash(fname))}.wav"

        save_path = os.path.join(savepath, fname)
        temp_path = os.path.join(tempfile.gettempdir(), fname)
        print("\033[98m {}\033[00m" .format("Don't forget to try different seeds by setting --seed <int> so that AudioSR can have optimal performance on your hardware."))
        print("Save audio to %s." % save_path)
        
        # Reshape waveform for soundfile
        data_to_save = waveform[i].T.cpu().numpy()
        if data_to_save.ndim == 1:
            data_to_save = data_to_save[:, np.newaxis]

        sf.write(temp_path, data_to_save, samplerate=samplerate)
        strip_silence(inputpath, temp_path, save_path)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    try:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))
    except:
        import ipdb

        ipdb.set_trace()


def default_audioldm_config(model_name="basic"):
    basic_config = get_basic_config()
    return basic_config


class MyProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def get_default_audiosr_model_root() -> Path:
    pretrained_root = os.environ.get("PRETRAINED_MODELS_ROOT")
    if pretrained_root:
        return Path(pretrained_root).expanduser() / "sr" / "audiosr"
    return Path(__file__).resolve().parents[3] / "pretrained_models" / "sr" / "audiosr"


def resolve_audiosr_model_root(model_root: Optional[Union[os.PathLike, str]] = None) -> Path:
    if model_root is not None:
        return Path(model_root).expanduser()
    env_model_root = os.environ.get("AUDIOSR_MODEL_ROOT")
    if env_model_root:
        return Path(env_model_root).expanduser()
    return get_default_audiosr_model_root()


def get_audiosr_checkpoint_path(
    checkpoint_name: str = "basic",
    model_root: Optional[Union[os.PathLike, str]] = None,
) -> Path:
    if checkpoint_name not in AUDIO_SR_CHECKPOINTS:
        raise ValueError("Invalid Model Name %s" % checkpoint_name)
    _, filename = AUDIO_SR_CHECKPOINTS[checkpoint_name]
    return resolve_audiosr_model_root(model_root) / checkpoint_name / filename


def get_audiosr_aux_model_path(
    model_name: str,
    model_root: Optional[Union[os.PathLike, str]] = None,
) -> Path:
    return resolve_audiosr_model_root(model_root) / "aux" / model_name


def _call_hf_hub_download(**kwargs) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required only when allow_download=True for AudioSR checkpoints."
        ) from exc

    download_kwargs = dict(kwargs)
    if "local_dir_use_symlinks" in signature(hf_hub_download).parameters:
        download_kwargs["local_dir_use_symlinks"] = False
    return hf_hub_download(**download_kwargs)


def _materialize_downloaded_file(downloaded_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if downloaded_path.resolve() != target_path.resolve():
        shutil.copy2(downloaded_path, target_path)
    if target_path.is_symlink():
        resolved_path = target_path.resolve()
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        shutil.copy2(resolved_path, tmp_path)
        target_path.unlink()
        tmp_path.replace(target_path)


def download_checkpoint(
    checkpoint_name="basic",
    model_root: Optional[Union[os.PathLike, str]] = None,
    allow_download: bool = False,
    local_files_only: bool = False,
    force_download: bool = False,
):
    if checkpoint_name not in AUDIO_SR_CHECKPOINTS:
        raise ValueError("Invalid Model Name %s" % checkpoint_name)

    model_id, filename = AUDIO_SR_CHECKPOINTS[checkpoint_name]
    model_root_path = resolve_audiosr_model_root(model_root)
    target_path = get_audiosr_checkpoint_path(checkpoint_name, model_root_path)
    if target_path.is_file() and not force_download:
        return str(target_path)

    if not allow_download:
        raise FileNotFoundError(
            "AudioSR checkpoint not found. "
            f"model={checkpoint_name}, expected={target_path}, model_root={model_root_path}. "
            "Put the checkpoint there manually, set AUDIOSR_MODEL_ROOT/PRETRAINED_MODELS_ROOT, "
            "or pass allow_download=True to download explicitly."
        )

    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(
        _call_hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=str(target_dir),
            local_files_only=local_files_only,
            force_download=force_download,
        )
    )
    _materialize_downloaded_file(downloaded_path, target_path)
    shutil.rmtree(target_dir / ".cache", ignore_errors=True)

    if not target_path.is_file():
        raise FileNotFoundError(
            f"AudioSR checkpoint download finished but expected file is missing: {target_path}"
        )
    return str(target_path)


def get_basic_config():
    return {
        "preprocessing": {
            "audio": {
                "sampling_rate": 48000,
                "max_wav_value": 32768,
                "duration": 10.24,
            },
            "stft": {"filter_length": 2048, "hop_length": 480, "win_length": 2048},
            "mel": {"n_mel_channels": 256, "mel_fmin": 20, "mel_fmax": 24000},
        },
        "augmentation": {"mixup": 0.5},
        "model": {
            "target": "data_gen.sr.audiosr.latent_diffusion.models.ddpm.LatentDiffusion",
            "params": {
                "first_stage_config": {
                    "base_learning_rate": 0.000008,
                    "target": "data_gen.sr.audiosr.latent_encoder.autoencoder.AutoencoderKL",
                    "params": {
                        "reload_from_ckpt": "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/log/vae/vae_48k_256/ds_8_kl_1/checkpoints/ckpt-checkpoint-484999.ckpt",
                        "sampling_rate": 48000,
                        "batchsize": 4,
                        "monitor": "val/rec_loss",
                        "image_key": "fbank",
                        "subband": 1,
                        "embed_dim": 16,
                        "time_shuffle": 1,
                        "ddconfig": {
                            "double_z": True,
                            "mel_bins": 256,
                            "z_channels": 16,
                            "resolution": 256,
                            "downsample_time": False,
                            "in_channels": 1,
                            "out_ch": 1,
                            "ch": 128,
                            "ch_mult": [1, 2, 4, 8],
                            "num_res_blocks": 2,
                            "attn_resolutions": [],
                            "dropout": 0.1,
                        },
                    },
                },
                "base_learning_rate": 0.0001,
                "warmup_steps": 5000,
                "optimize_ddpm_parameter": True,
                "sampling_rate": 48000,
                "batchsize": 16,
                "beta_schedule": "cosine",
                "linear_start": 0.0015,
                "linear_end": 0.0195,
                "num_timesteps_cond": 1,
                "log_every_t": 200,
                "timesteps": 1000,
                "unconditional_prob_cfg": 0.1,
                "parameterization": "v",
                "first_stage_key": "fbank",
                "latent_t_size": 128,
                "latent_f_size": 32,
                "channels": 16,
                "monitor": "val/loss_simple_ema",
                "scale_by_std": True,
                "unet_config": {
                    "target": "data_gen.sr.audiosr.latent_diffusion.modules.diffusionmodules.openaimodel.UNetModel",
                    "params": {
                        "image_size": 64,
                        "in_channels": 32,
                        "out_channels": 16,
                        "model_channels": 128,
                        "attention_resolutions": [8, 4, 2],
                        "num_res_blocks": 2,
                        "channel_mult": [1, 2, 3, 5],
                        "num_head_channels": 32,
                        "extra_sa_layer": True,
                        "use_spatial_transformer": True,
                        "transformer_depth": 1,
                    },
                },
                "evaluation_params": {
                    "unconditional_guidance_scale": 3.5,
                    "ddim_sampling_steps": 200,
                    "n_candidates_per_samples": 1,
                },
                "cond_stage_config": {
                    "concat_lowpass_cond": {
                        "cond_stage_key": "lowpass_mel",
                        "conditioning_key": "concat",
                        "target": "data_gen.sr.audiosr.latent_diffusion.modules.encoders.modules.VAEFeatureExtract",
                        "params": {
                            "first_stage_config": {
                                "base_learning_rate": 0.000008,
                                "target": "data_gen.sr.audiosr.latent_encoder.autoencoder.AutoencoderKL",
                                "params": {
                                    "sampling_rate": 48000,
                                    "batchsize": 4,
                                    "monitor": "val/rec_loss",
                                    "image_key": "fbank",
                                    "subband": 1,
                                    "embed_dim": 16,
                                    "time_shuffle": 1,
                                    "ddconfig": {
                                        "double_z": True,
                                        "mel_bins": 256,
                                        "z_channels": 16,
                                        "resolution": 256,
                                        "downsample_time": False,
                                        "in_channels": 1,
                                        "out_ch": 1,
                                        "ch": 128,
                                        "ch_mult": [1, 2, 4, 8],
                                        "num_res_blocks": 2,
                                        "attn_resolutions": [],
                                        "dropout": 0.1,
                                    },
                                },
                            }
                        },
                    }
                },
            },
        },
    }
