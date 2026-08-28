from __future__ import annotations

import argparse
from pathlib import Path

from data_gen.source_separation.uvr.uvr_api import build_uvr_model, run_uvr_model


def iter_inputs(path: Path):
    if path.is_file():
        yield path
        return
    for item in sorted(path.iterdir()):
        if item.is_file():
            yield item


def main():
    parser = argparse.ArgumentParser(description="Run UVR source separation.")
    parser.add_argument("--input", help="Input audio file or directory.")
    parser.add_argument("--output", help="Output file or directory. Defaults to input location.", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="MDX23C-8KFFT-InstVoc_HQ")
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size-sec", type=float, default=None)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--no-legacy-fallback", action="store_true")
    args = parser.parse_args()

    model = build_uvr_model(
        device=args.device,
        model_name=args.model_name,
        model_root=args.model_root,
        allow_legacy_fallback=not args.no_legacy_fallback,
        precision=args.precision,
    )

    input_items = []
    for raw_input in args.input:
        input_path = Path(raw_input).expanduser()
        input_items.extend(iter_inputs(input_path))

    output_path = Path(args.output).expanduser() if args.output else None

    run_output = output_path
    if len(input_items) > 1 and output_path is not None and output_path.suffix:
        raise ValueError("When multiple inputs are provided, --output must be a directory.")

    result = run_uvr_model(
        input_items[0] if len(input_items) == 1 else input_items,
        model,
        output_path=run_output,
        batch_size=args.batch_size,
        chunk_size_sec=args.chunk_size_sec,
    )

    results = result if isinstance(result, list) else [result]
    for item, item_result in zip(input_items, results):
        print(f"uvr done: {item} -> sr={item_result['sr']}")


if __name__ == "__main__":
    main()
