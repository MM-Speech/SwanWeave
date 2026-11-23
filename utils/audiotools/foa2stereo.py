#!/usr/bin/env python3
"""Convert SwanWeave FOA audio to forward-facing stereo for listening.

Usage:
    python utils/audiotools/foa2stereo.py input_foa.wav
    python utils/audiotools/foa2stereo.py input_foa.wav -o output_stereo.wav
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def foa_to_stereo(foa: np.ndarray) -> np.ndarray:
    """Decode [samples, 4] SwanWeave FOA into [samples, 2] stereo."""
    foa = np.asarray(foa, dtype=np.float32)
    if foa.ndim != 2 or foa.shape[1] != 4:
        raise ValueError(f"Expected four FOA channels in W,Y,Z,X order, got {foa.shape}")
    if foa.shape[0] == 0:
        raise ValueError("Input audio is empty")
    if not np.isfinite(foa).all():
        raise ValueError("Input audio contains NaN or Inf")

    w, y, _z, x = foa.T
    sqrt2 = np.float32(np.sqrt(2.0))
    left = (sqrt2 * w + (x + y) / sqrt2) * 0.5
    right = (sqrt2 * w + (x - y) / sqrt2) * 0.5
    return np.column_stack((left, right))


def convert_file(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Convert a FOA file, scaling both outputs together only to avoid clipping."""
    input_path = Path(input_path)
    output_path = (Path(output_path) if output_path is not None
                   else input_path.with_name(input_path.stem + "_stereo.wav"))
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must be different from the input file")
    if output_path.suffix.lower() != ".wav":
        raise ValueError("Output must have a .wav extension")

    foa, sample_rate = sf.read(input_path, dtype="float32", always_2d=True)
    stereo = foa_to_stereo(foa)
    peak = float(np.max(np.abs(stereo)))
    if peak > 1.0:
        stereo *= 0.99 / peak
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, stereo, sample_rate, subtype="PCM_24")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input four-channel FOA audio")
    parser.add_argument("-o", "--output", type=Path, help="Output WAV; default: <input_stem>_stereo.wav")
    args = parser.parse_args()
    try:
        output_path = convert_file(args.input, args.output)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(output_path)


if __name__ == "__main__":
    main()
