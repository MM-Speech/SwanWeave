import subprocess
import io
import base64
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy.io import wavfile
import pyloudnorm as pyln
import soundfile as sf
import librosa


def save_wav(wav, path, sr, norm=False):
    wav = wav.astype(float)
    if norm:
        meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -18.0)
        if np.abs(wav).max() >= 1:
            wav = wav / np.abs(wav).max() * 0.95
    wav = wav * 32767
    wavfile.write(path[:-4] + '.wav', sr, wav.astype(np.int16))
    if path[-4:] == '.mp3':
        to_mp3(path[:-4])


def to_mp3(out_path):
    if out_path[-4:] == '.wav':
        out_path = out_path[:-4]
    subprocess.check_call(
        f'ffmpeg -threads 1 -loglevel error -i "{out_path}.wav" -vn -b:a 192k -y -hide_banner -async 1 "{out_path}.mp3"',
        shell=True, stdin=subprocess.PIPE)
    subprocess.check_call(f'rm -f "{out_path}.wav"', shell=True)


def to_wav_bytes(wav, sr, norm=False):
    wav = wav.astype(float)
    if norm:
        meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -18.0)
        if np.abs(wav).max() >= 1:
            wav = wav / np.abs(wav).max() * 0.95
    wav = wav * 32767
    bytes_io = io.BytesIO()
    wavfile.write(bytes_io, sr, wav.astype(np.int16))
    return bytes_io.getvalue()

def wav_bytes_to_mp3_bytes(wav_bytes):
    from pydub import AudioSegment
    wav_io = io.BytesIO(wav_bytes)
    audio = AudioSegment.from_wav(wav_io)
    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3")
    mp3_bytes = mp3_io.getvalue()
    return mp3_bytes


def save_mp3_bytes(wav_bytes, path):
    with open(path[:-4] + '.mp3', 'wb') as file:
        file.write(wav_bytes)


def save_wav_bytes(wav_bytes, path):
    with open(path[:-4] + '.wav', 'wb') as file:
        file.write(wav_bytes)
    if path[-4:] == '.mp3':
        to_mp3(path[:-4])


def is_url(s: str) -> bool:
    try:
        from urllib.parse import urlparse
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def is_probably_base64(s: str) -> bool:
    if s.startswith("data:audio"):
        return True
    if ("/" not in s and "\\" not in s) and len(s) > 256:
        return True
    return False


def decode_base64_bytes(b64: str) -> bytes:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def load_audio_any(x: str) -> Tuple[np.ndarray, int]:
    import urllib.request
    if is_url(x):
        with urllib.request.urlopen(x) as resp:
            audio_bytes = resp.read()
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    elif is_probably_base64(x):
        audio_bytes = decode_base64_bytes(x)
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    else:
        audio, sr = librosa.load(x, sr=None, mono=False)

    audio = np.asarray(audio, dtype=np.float32)
    sr = int(sr)
    return audio, sr
