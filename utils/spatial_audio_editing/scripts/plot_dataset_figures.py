#!/usr/bin/env python3
"""Plot dataset overview figures for the spatial audio editing data.

Expects the default layout produced by the generation scripts under --data-root:
  <domain>/*.jsonl, <domain>_multi/*.jsonl, test_set/<domain>/*.jsonl

Usage:
  python scripts/plot_dataset_figures.py  # defaults from data_paths.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.spatial_audio_editing.data_paths import DPO_SCENE_ROOT, FIGURE_ROOT, SCENE_ROOT
from utils.spatial_audio_editing.src.progress import Progress

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


DOMAIN_LABELS = {
    "audio_only": "Audio only",
    "speech_audio": "Speech + audio",
}
DOMAIN_COLORS = {"audio_only": "#4267AC", "speech_audio": "#44A67C"}

STOPWORDS = set("""
    a all an and are as at away be behind but by directly does doesnt for from front
    had has have he her hers him his in insert into it is its it's keeping left let
    maintaining make moderate move not of on only place put right same scene she
    short sound than that the their them then they this time to was were what when
    where which while who whose will with without would you your yours
""".split())

TEXT_KEYS = {
    "caption",
    "captions",
    "chosen",
    "negative_text",
    "original_text",
    "prompt",
    "rejected",
    "target_text",
    "text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=SCENE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_ROOT)
    parser.add_argument("--dpo-root", type=Path, default=DPO_SCENE_ROOT)
    parser.add_argument(
        "--spatial-lines-per-file",
        type=int,
        default=3000,
        help="Maximum positive JSONL rows to sample from each file for spatial/text plots.",
    )
    parser.add_argument(
        "--max-dpo-files",
        type=int,
        default=0,
        help="Debug option: only parse this many DPO files. 0 means all.",
    )
    parser.add_argument(
        "--wordcloud-max-files",
        type=int,
        default=0,
        help="Debug option: only parse this many JSONL files for the word cloud. 0 means all.",
    )
    return parser.parse_args()


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z_']+")


def count_content_words(text: str, words: Counter):
    for token in TOKEN_RE.findall(text.lower().replace("_", " ")):
        if token not in STOPWORDS and len(token) > 2:
            words[token] += 1


def count_lines(path: Path, progress=None) -> int:
    count = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n_lines = chunk.count(b"\n")
            count += n_lines
            if progress is not None:
                progress.update(n_lines)
    return count


def is_text_file(path: Path) -> bool:
    return path.name.endswith("_text.jsonl")


def is_negative_file(path: Path) -> bool:
    return path.name.endswith("_negative.jsonl")


def is_positive_jsonl(path: Path) -> bool:
    return path.suffix == ".jsonl" and not is_text_file(path) and not is_negative_file(path)


def edit_type_from_file(path: Path) -> str:
    name = path.name.removesuffix(".jsonl")
    return name.removesuffix("_negative").removesuffix("_text")


def iter_jsonl(path: Path, max_rows: int | None = None, progress=None):
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows is not None and idx >= max_rows:
                break
            line = line.strip()
            if line:
                row = json.loads(line)
                if progress is not None:
                    progress.update()
                yield row


def positive_files(root: Path, domain: str) -> list[Path]:
    base = root / domain
    if not base.exists():
        return []
    return sorted(p for p in base.glob("*.jsonl") if is_positive_jsonl(p))


def negative_files(root: Path, domain: str, dpo_root: Path | None = None) -> list[Path]:
    if dpo_root is None:
        # Support datasets that keep negative files beside positive files.
        base = root / domain
        return sorted(base.glob("*_negative.jsonl")) if base.exists() else []
    is_multi = domain.endswith("_multi")
    base_domain = domain.removesuffix("_multi")
    base = dpo_root / base_domain / "train"
    return sorted(
        path for path in base.glob("*_negative.jsonl")
        if path.name.startswith("multi_") == is_multi
    )


def test_positive_files(root: Path, domain: str) -> list[Path]:
    base = root / "test_set" / domain
    if not base.exists():
        return []
    return sorted(p for p in base.glob("*.jsonl") if is_positive_jsonl(p))


def collect_counts(root: Path, dpo_root: Path | None = None, progress=None) -> dict:
    counts: dict[str, object] = {
        "composition": defaultdict(Counter),
        "single_edit": defaultdict(Counter),
        "multi_op": defaultdict(Counter),
    }

    for domain in DOMAIN_LABELS:
        for path in positive_files(root, domain):
            n = count_lines(path, progress=progress)
            counts["composition"][domain]["single_positive"] += n
            counts["single_edit"][domain][edit_type_from_file(path)] += n

        for path in negative_files(root, domain, dpo_root):
            counts["composition"][domain]["single_dpo"] += count_lines(path, progress=progress)

        for path in positive_files(root, f"{domain}_multi"):
            n = count_lines(path, progress=progress)
            counts["composition"][domain]["multi_positive"] += n
            match = re.search(r"multi_(\d+)op", path.name)
            op_count = f"{match.group(1)} operations" if match else edit_type_from_file(path)
            counts["multi_op"][domain][op_count] += n

        for path in negative_files(root, f"{domain}_multi", dpo_root):
            counts["composition"][domain]["multi_dpo"] += count_lines(path, progress=progress)

        for path in test_positive_files(root, domain):
            counts["composition"][domain]["test_positive"] += count_lines(path, progress=progress)

    return counts


def collect_dpo_modes(root: Path, max_files: int = 0, dpo_root: Path | None = None, progress=None) -> tuple[Counter, dict[str, Counter]]:
    totals = Counter()
    by_group: dict[str, Counter] = defaultdict(Counter)
    files: list[tuple[str, Path]] = []

    for domain in DOMAIN_LABELS:
        files += [(f"{domain}:single", p) for p in negative_files(root, domain, dpo_root)]
        files += [(f"{domain}:multi", p) for p in negative_files(root, f"{domain}_multi", dpo_root)]

    if max_files:
        files = files[:max_files]

    for group, path in files:
        for row in iter_jsonl(path, progress=progress):
            mode = row.get("error_mode", "unknown")
            totals[mode] += 1
            by_group[group][mode] += 1

    return totals, by_group


def collect_room_change_targets(root: Path, progress=None) -> Counter:
    targets = Counter()
    for domain in DOMAIN_LABELS:
        path = root / domain / "room_change.jsonl"
        if not path.exists():
            continue
        for row in iter_jsonl(path, progress=progress):
            target = row.get("target_room")
            if target:
                targets[str(target)] += 1
    return targets


def collect_all_operation_counts(root: Path, progress=None) -> Counter:
    operations = Counter()

    for domain in DOMAIN_LABELS:
        for path in positive_files(root, domain):
            operations[edit_type_from_file(path)] += count_lines(path, progress=progress)

        for path in test_positive_files(root, domain):
            operations[edit_type_from_file(path)] += count_lines(path, progress=progress)

        for path in positive_files(root, f"{domain}_multi"):
            for row in iter_jsonl(path, progress=progress):
                edit_info = row.get("edit_info")
                if isinstance(edit_info, list):
                    for step in edit_info:
                        if isinstance(step, dict):
                            op_type = step.get("type") or step.get("edit_type")
                            if op_type:
                                operations[str(op_type)] += 1
                elif isinstance(edit_info, dict):
                    op_type = edit_info.get("type") or edit_info.get("edit_type")
                    if op_type:
                        operations[str(op_type)] += 1

    return operations


def iter_text_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_text_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_text_values(item)


def collect_wordcloud_words(root: Path, max_files: int = 0, dpo_root: Path | None = None, progress=None) -> Counter:
    words = Counter()
    files = []
    for folder in [root, root / "test_set"]:
        if folder.exists():
            files.extend(sorted(folder.glob("*/*.jsonl")))
    if dpo_root is not None:
        files.extend(sorted(dpo_root.glob("*/*/*_negative.jsonl")))
    if max_files:
        files = files[:max_files]

    for path in files:
        for row in iter_jsonl(path, progress=progress):
            for key in TEXT_KEYS:
                if key not in row:
                    continue
                for text in iter_text_values(row[key]):
                    count_content_words(text, words)

    return words


def collect_sampled_features(root: Path, lines_per_file: int, progress=None) -> tuple[np.ndarray, np.ndarray, Counter]:
    xs: list[float] = []
    zs: list[float] = []
    words = Counter()
    files: list[Path] = []
    for domain in DOMAIN_LABELS:
        files += positive_files(root, domain)
        files += positive_files(root, f"{domain}_multi")

    for path in files:
        for row in iter_jsonl(path, lines_per_file, progress=progress):
            captions = row.get("captions") or []
            for caption in captions[:1]:
                count_content_words(caption, words)

            for event in row.get("target_scene") or []:
                pos = event.get("position") or {}
                x = pos.get("x")
                z = pos.get("z")
                if isinstance(x, (int, float)) and isinstance(z, (int, float)):
                    xs.append(float(x))
                    zs.append(float(z))

    return np.asarray(xs), np.asarray(zs), words


def style_axis(ax, grid_axis: str = "y"):
    ax.grid(axis=grid_axis, color="#E6E8EC", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CFD4DC")
    ax.spines["bottom"].set_color("#CFD4DC")
    ax.tick_params(colors="#343A40")


def save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt_k(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(int(value))


def plot_composition(counts: dict, out: Path):
    categories = [
        ("single_positive", "Single positive", "#4267AC"),
        ("single_dpo", "Single DPO negative", "#D65F5F"),
        ("multi_positive", "Multi-op positive", "#44A67C"),
        ("multi_dpo", "Multi-op DPO negative", "#B779C9"),
        ("test_positive", "Test positive", "#E3A33A"),
    ]
    domains = list(DOMAIN_LABELS)
    x = np.arange(len(categories))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for idx, domain in enumerate(domains):
        vals = np.array([counts["composition"][domain][key] for key, _, _ in categories], dtype=float)
        xpos = x + (idx - 0.5) * width
        ax.bar(
            xpos,
            vals,
            width,
            label=DOMAIN_LABELS[domain],
            color=DOMAIN_COLORS[domain],
        )
        for px, val in zip(xpos, vals):
            if val > 0:
                ax.text(
                    px,
                    val * 1.08,
                    fmt_k(val),
                    ha="center",
                    va="bottom",
                    fontsize=8.5 if val >= 1000 else 8,
                    color="#222222",
                )

    ax.set_title("Dataset Composition", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Samples, log scale")
    ax.set_yscale("log")
    all_vals = [v for d in domains for v in counts["composition"][d].values() if v > 0]
    if all_vals:
        ax.set_ylim(min(all_vals) / 3, max(all_vals) * 3)
    ax.set_xticks(x, [label for _, label, _ in categories], rotation=15, ha="right")
    ax.yaxis.set_major_formatter(lambda v, _: fmt_k(v))
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style_axis(ax)
    save(fig, out / "dataset_composition.png")


def plot_single_edit_distribution(counts: dict, out: Path):
    edit_types = sorted(set().union(*(counts["single_edit"][d].keys() for d in DOMAIN_LABELS)))
    y = np.arange(len(edit_types))
    height = 0.38

    fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.34 * len(edit_types) + 1.8)))
    for offset, domain in [(-height / 2, "audio_only"), (height / 2, "speech_audio")]:
        vals = np.array([counts["single_edit"][domain][e] for e in edit_types], dtype=float)
        ax.barh(y + offset, vals, height, label=DOMAIN_LABELS[domain], color=DOMAIN_COLORS[domain])

    ax.set_title("Single-Operation Edit Type Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Samples")
    ax.set_yticks(y, [e.replace("_", " ") for e in edit_types], fontsize=9)
    ax.xaxis.set_major_formatter(lambda v, _: fmt_k(v))
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax, grid_axis="x")
    save(fig, out / "single_edit_type_distribution.png")


def plot_multi_distribution(counts: dict, out: Path):
    op_labels = ["2 operations", "3 operations", "4 operations"]
    x = np.arange(len(op_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for idx, domain in enumerate(DOMAIN_LABELS):
        vals = np.array([counts["multi_op"][domain][label] for label in op_labels], dtype=float)
        xpos = x + (idx - 0.5) * width
        ax.bar(xpos, vals, width, label=DOMAIN_LABELS[domain], color=DOMAIN_COLORS[domain])
        for px, val in zip(xpos, vals):
            ax.text(px, val, fmt_k(val), ha="center", va="bottom", fontsize=9)

    ax.set_title("Multi-Operation Data Scale", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Samples")
    ax.set_xticks(x, op_labels)
    ax.yaxis.set_major_formatter(lambda v, _: fmt_k(v))
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, out / "multi_operation_distribution.png")


def plot_dpo_error_modes(by_group: dict[str, Counter], out: Path):
    groups = ["audio_only:single", "speech_audio:single", "audio_only:multi", "speech_audio:multi"]
    labels = ["Audio single", "Speech+audio single", "Audio multi", "Speech+audio multi"]
    modes = sorted(set().union(*(by_group[g].keys() for g in groups)))
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(modes))))
    y = np.arange(len(groups))
    left = np.zeros(len(groups))

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for color, mode in zip(colors, modes):
        vals = np.array([by_group[g][mode] for g in groups], dtype=float)
        ax.barh(y, vals, left=left, label=mode.replace("_", " "), color=color)
        left += vals

    ax.set_title("DPO Negative Error Mode Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Negative samples")
    ax.set_yticks(y, labels)
    ax.xaxis.set_major_formatter(lambda v, _: fmt_k(v))
    style_axis(ax, grid_axis="x")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save(fig, out / "dpo_error_mode_distribution.png")


def plot_spatial_heatmap(xs: np.ndarray, zs: np.ndarray, out: Path):
    if xs.size == 0:
        return
    q = np.quantile(np.abs(np.concatenate([xs, zs])), 0.995)
    lim = max(4.0, min(20.0, math.ceil(q)))

    fig, ax = plt.subplots(figsize=(6.3, 5.7))
    hist = ax.hist2d(
        xs,
        zs,
        bins=80,
        range=[[-lim, lim], [-lim, lim]],
        cmap="mako" if "mako" in plt.colormaps() else "viridis",
        norm=LogNorm(),
    )
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.75)
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.75)
    ax.scatter([0], [0], marker="x", color="#F3F5F7", s=45, linewidth=1.5)
    ax.set_title("Spatial Event Position Density", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("x position")
    ax.set_ylabel("z position")
    ax.set_aspect("equal")
    cb = fig.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Event count, log scale")
    save(fig, out / "spatial_position_heatmap.png")


def plot_room_change_targets(targets: Counter, out: Path):
    if not targets:
        return
    preferred = ["small", "medium", "large", "huge"]
    labels = [label for label in preferred if targets[label]]
    vals = [targets[label] for label in labels]
    colors = ["#4267AC", "#44A67C", "#E3A33A", "#D65F5F", "#B779C9"]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=(5.9, 4.8))
    wedges, _, autotexts = ax.pie(
        vals,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        colors=colors[: len(vals)],
        wedgeprops={"width": 0.42, "edgecolor": "white"},
    )
    for text in autotexts:
        text.set_fontsize(9)
    legend_labels = [f"{label.title()} ({fmt_k(value)}, {value / total:.1%})" for label, value in zip(labels, vals)]
    ax.legend(wedges, legend_labels, frameon=False, loc="center left", bbox_to_anchor=(0.88, 0.5))
    ax.set_title("Room-Change Target Room Distribution", fontsize=14, fontweight="bold", pad=12)
    save(fig, out / "room_change_target_distribution.png")


def plot_all_operation_pie(operations: Counter, out: Path):
    if not operations:
        return

    ordered = operations.most_common()
    labels, vals = zip(*ordered)
    total = sum(vals)
    colors = [plt.cm.tab20(i % 20) for i in range(len(labels))]

    def pct_label(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= 3.0 else ""

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    wedges, _, autotexts = ax.pie(
        vals,
        autopct=pct_label,
        startangle=90,
        pctdistance=0.78,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 0.8},
    )
    for text in autotexts:
        text.set_fontsize(8.5)

    legend_labels = [
        f"{label.replace('_', ' ')} ({fmt_k(value)}, {value / total:.1%})"
        for label, value in ordered
    ]
    ax.legend(
        wedges,
        legend_labels,
        frameon=False,
        fontsize=8.3,
        loc="center left",
        bbox_to_anchor=(0.94, 0.5),
    )
    ax.set_title("All Operation Type Distribution", fontsize=14, fontweight="bold", pad=12)
    save(fig, out / "all_operation_type_pie.png")


def plot_all_data_wordcloud(words: Counter, out: Path, progress=None):
    if not words:
        return

    try:
        from wordcloud import WordCloud

        wc = WordCloud(
            width=1800,
            height=1000,
            background_color="white",
            colormap="viridis",
            prefer_horizontal=0.92,
            max_words=180,
            collocations=False,
            random_state=7,
            margin=3,
        ).generate_from_frequencies(words)

        fig, ax = plt.subplots(figsize=(11, 6.2))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("All Text Word Cloud", fontsize=14, fontweight="bold", pad=12)
        save(fig, out / "all_data_wordcloud.png")
        return
    except Exception as exc:
        message = f"wordcloud package unavailable, using fallback text layout: {exc}"
        if progress is not None:
            progress.write(message)
        else:
            print(message)

    top = words.most_common(90)
    max_count = top[0][1]
    rng = np.random.default_rng(7)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for word, count in top:
        size = 9 + 33 * math.sqrt(count / max_count)
        ax.text(
            rng.uniform(0.06, 0.94),
            rng.uniform(0.08, 0.92),
            word,
            fontsize=size,
            ha="center",
            va="center",
            color=plt.cm.viridis(0.2 + 0.65 * count / max_count),
            alpha=0.88,
        )
    ax.set_title("All Text Word Cloud", fontsize=14, fontweight="bold", pad=12)
    save(fig, out / "all_data_wordcloud.png")


def plot_caption_words(words: Counter, out: Path):
    top = words.most_common(24)
    if not top:
        return
    labels, vals = zip(*reversed(top))
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(vals)))
    ax.barh(np.arange(len(labels)), vals, color=colors)
    ax.set_title("Frequent Caption Content Words", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Occurrences in sampled captions")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.xaxis.set_major_formatter(lambda v, _: fmt_k(v))
    style_axis(ax, grid_axis="x")
    save(fig, out / "caption_word_frequency.png")


def dump_stats(
    counts: dict,
    dpo_totals: Counter,
    dpo_by_group: dict[str, Counter],
    room_targets: Counter,
    sampled_words: Counter,
    all_text_words: Counter,
    operations: Counter,
    out: Path,
):
    payload = {
        "composition": {k: dict(v) for k, v in counts["composition"].items()},
        "single_edit": {k: dict(v) for k, v in counts["single_edit"].items()},
        "multi_op": {k: dict(v) for k, v in counts["multi_op"].items()},
        "dpo_error_modes": dict(dpo_totals),
        "dpo_error_modes_by_group": {k: dict(v) for k, v in dpo_by_group.items()},
        "all_operation_types": dict(operations),
        "room_change_target_rooms": dict(room_targets),
        "sampled_caption_words_top50": dict(sampled_words.most_common(50)),
        "all_text_wordcloud_words_top100": dict(all_text_words.most_common(100)),
    }
    (out / "stats.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    root = args.data_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    with Progress(desc="Dataset figures", unit="rows") as progress:
        progress.reset(desc='Counting dataset samples', unit="rows")
        counts = collect_counts(root, args.dpo_root, progress=progress)
        progress.reset(desc='Reading DPO error modes', unit="rows")
        dpo_totals, dpo_by_group = collect_dpo_modes(root, args.max_dpo_files, args.dpo_root, progress=progress)
        progress.reset(desc='Reading room-change targets', unit="rows")
        room_targets = collect_room_change_targets(root, progress=progress)
        progress.reset(desc='Counting edit operations', unit="rows")
        operations = collect_all_operation_counts(root, progress=progress)
        progress.reset(desc='Sampling spatial features', unit="rows")
        xs, zs, words = collect_sampled_features(root, args.spatial_lines_per_file, progress=progress)
        progress.reset(desc='Reading text for word cloud', unit="rows")
        all_text_words = collect_wordcloud_words(root, args.wordcloud_max_files, args.dpo_root, progress=progress)

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "axes.titlesize": 14,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
            }
        )

        progress.reset(total=10, desc="Writing dataset figures", unit="outputs")
        plot_composition(counts, out)
        progress.update()
        plot_single_edit_distribution(counts, out)
        progress.update()
        plot_multi_distribution(counts, out)
        progress.update()
        plot_dpo_error_modes(dpo_by_group, out)
        progress.update()
        plot_spatial_heatmap(xs, zs, out)
        progress.update()
        plot_all_operation_pie(operations, out)
        progress.update()
        plot_all_data_wordcloud(all_text_words, out, progress=progress)
        progress.update()
        plot_room_change_targets(room_targets, out)
        progress.update()
        plot_caption_words(words, out)
        progress.update()
        dump_stats(counts, dpo_totals, dpo_by_group, room_targets, words, all_text_words, operations, out)
        progress.update()

    print(f"Wrote figures to {out}")
    for path in sorted(out.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
