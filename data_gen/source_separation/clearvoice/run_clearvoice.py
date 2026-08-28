from __future__ import annotations

import argparse
from pathlib import Path

from data_gen.source_separation.clearvoice.clearvoice_api import (
    build_clearvoice_enhancer,
    build_clearvoice_separator,
    run_clearvoice_enhancement,
    run_clearvoice_separation,
)


def iter_inputs(path: Path):
    if path.is_file():
        yield path
        return
    for item in sorted(path.iterdir()):
        if item.is_file():
            yield item


def main():
    parser = argparse.ArgumentParser(description="Run ClearVoice enhancement or separation.")
    parser.add_argument("input", help="Input audio file or directory.")
    parser.add_argument("--output", help="Output file or directory. Defaults to input location.", default=None)
    parser.add_argument(
        "--task",
        choices=["enhancement", "separation"],
        default="enhancement",
        help="ClearVoice task to run.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--no-legacy-fallback", action="store_true")
    parser.add_argument("--no-residual", action="store_true")
    args = parser.parse_args()

    allow_legacy_fallback = not args.no_legacy_fallback
    if args.task == "enhancement":
        model = build_clearvoice_enhancer(
            device=args.device,
            model_name=args.model_name or "MossFormer2_SE_48K",
            model_root=args.model_root,
            allow_legacy_fallback=allow_legacy_fallback,
        )
        runner = lambda item, item_output: run_clearvoice_enhancement(  # noqa: E731
            item,
            model,
            output_path=item_output,
            include_residual=not args.no_residual,
        )
    else:
        model = build_clearvoice_separator(
            device=args.device,
            model_name=args.model_name or "MossFormer2_SS_16K",
            model_root=args.model_root,
            allow_legacy_fallback=allow_legacy_fallback,
        )
        runner = lambda item, item_output: run_clearvoice_separation(item, model, output_path=item_output)  # noqa: E731

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser() if args.output else None

    for item in iter_inputs(input_path):
        item_output = output_path
        if output_path and input_path.is_dir():
            item_output = output_path / item.stem
        result = runner(item, item_output)
        print(f"clearvoice {args.task} done: {item} -> sr={result['sr']}")


if __name__ == "__main__":
    main()
