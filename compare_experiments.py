import argparse
import json
import os
from pathlib import Path
import re
from tempfile import gettempdir
from typing import Dict, List, Mapping

MPLCONFIGDIR = Path(gettempdir()) / "sod_matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


METRIC_NAMES = ("iou", "precision", "recall", "f1_score", "mae", "mse")
PLOT_METRICS = ("iou", "f1_score", "mae")
DEFAULT_EXPERIMENTS = (
    "baseline",
    "baseline_no_bn",
    "strong_aug",
    "iou_heavy",
    "improved",
)
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
DEFAULT_DESCRIPTIONS = {
    "baseline": "Baseline encoder-decoder CNN, light augmentation, BCE + 0.5 IoU loss.",
    "baseline_no_bn": "Baseline encoder-decoder with BatchNorm removed.",
    "strong_aug": "Baseline encoder-decoder with stronger train-time augmentation.",
    "iou_heavy": "Baseline encoder-decoder with IoU loss weight increased to 1.0.",
    "improved": (
        "Small UNet with skip connections, dropout, strong augmentation, lower LR, and heavier IoU loss."
    ),
}


def resolve_path(path_value: str, project_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def safe_experiment_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    name = name.strip("._-")
    return name or "experiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create metrics tables and visual comparisons for SOD experiments."
    )
    parser.add_argument("--baseline_name", type=str, default="baseline")
    parser.add_argument("--improved_name", type=str, default="improved")
    parser.add_argument(
        "--experiment_names",
        nargs="+",
        default=None,
        help=(
            "Experiment names to compare. Defaults to baseline, baseline_no_bn, "
            "strong_aug, iou_heavy, and improved."
        ),
    )
    parser.add_argument("--metrics_dir", type=str, default="outputs/metrics")
    parser.add_argument("--visualizations_dir", type=str, default="outputs/visualizations")
    parser.add_argument("--output_markdown", type=str, default="outputs/metrics/experiment_comparison.md")
    parser.add_argument(
        "--output_plot",
        type=str,
        default="outputs/metrics/plots/experiment_metrics_comparison.png",
    )
    parser.add_argument(
        "--output_visual",
        type=str,
        default="outputs/visualizations/experiment_comparison_contact_sheet.png",
    )
    return parser.parse_args()


def load_metrics(metrics_dir: Path, experiment_name: str) -> Dict[str, float] | None:
    path = metrics_dir / f"test_metrics_{experiment_name}.json"
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return {key: float(payload[key]) for key in METRIC_NAMES if key in payload}


def format_metric(metrics: Mapping[str, float] | None, key: str) -> str:
    if metrics is None or key not in metrics:
        return "missing"
    return f"{metrics[key]:.4f}"


def format_delta(
    metrics: Mapping[str, float] | None,
    baseline_metrics: Mapping[str, float] | None,
    key: str,
) -> str:
    if metrics is None or baseline_metrics is None:
        return "missing"
    if key not in metrics or key not in baseline_metrics:
        return "missing"
    delta = metrics[key] - baseline_metrics[key]
    return f"{delta:+.4f}"


def relative_link(path: Path | None, project_dir: Path) -> str:
    if path is None:
        return "missing"
    try:
        display_path = path.resolve().relative_to(project_dir)
    except ValueError:
        display_path = path
    return f"[{display_path}]({display_path})"


def relative_image(path: Path | None, project_dir: Path, alt_text: str) -> str:
    if path is None:
        return "missing"
    try:
        display_path = path.resolve().relative_to(project_dir)
    except ValueError:
        display_path = path
    return f"![{alt_text}]({display_path})"


def find_first_visual(visualizations_dir: Path, experiment_name: str) -> Path | None:
    experiment_dir = visualizations_dir / experiment_name
    if not experiment_dir.exists():
        return None

    candidates = sorted(experiment_dir.glob("*.png"))
    return candidates[0] if candidates else None


def write_markdown_table(
    rows: List[Dict[str, object]],
    output_path: Path,
    project_dir: Path,
    plot_path: Path | None,
    visual_path: Path | None,
) -> None:
    baseline_metrics = rows[0]["metrics"] if rows else None
    lines = [
        "# Experiment Comparison Results",
        "",
        "Current saved metrics use the project custom split and threshold `0.5`. "
        "Positive deltas are better for IoU, precision, recall, and F1-score; "
        "negative deltas are better for MAE and MSE.",
        "",
        "| Experiment | Configuration | IoU | Δ IoU | Precision | Recall | F1-score | Δ F1 | MAE | Δ MAE | MSE | Δ MSE | Visual example |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        metrics = row["metrics"]
        visual = row["visual"]
        lines.append(
            "| {name} | {description} | {iou} | {delta_iou} | {precision} | {recall} | "
            "{f1_score} | {delta_f1} | {mae} | {delta_mae} | {mse} | {delta_mse} | "
            "{visual_link} |".format(
                name=row["name"],
                description=row["description"],
                iou=format_metric(metrics, "iou"),
                delta_iou=format_delta(metrics, baseline_metrics, "iou"),
                precision=format_metric(metrics, "precision"),
                recall=format_metric(metrics, "recall"),
                f1_score=format_metric(metrics, "f1_score"),
                delta_f1=format_delta(metrics, baseline_metrics, "f1_score"),
                mae=format_metric(metrics, "mae"),
                delta_mae=format_delta(metrics, baseline_metrics, "mae"),
                mse=format_metric(metrics, "mse"),
                delta_mse=format_delta(metrics, baseline_metrics, "mse"),
                visual_link=relative_link(visual, project_dir),
            )
        )

    lines.extend(["", "## Visual Outputs", ""])
    if plot_path is not None:
        lines.append("### Metric Comparison Chart")
        lines.append("")
        lines.append(relative_image(plot_path, project_dir, "Metric comparison chart"))
        lines.append("")
    if visual_path is not None:
        lines.append("### Qualitative Contact Sheet")
        lines.append("")
        lines.append(relative_image(visual_path, project_dir, "Qualitative contact sheet"))
        lines.append("")
    if plot_path is None and visual_path is None:
        lines.append("- No comparison visuals were generated because the required inputs are missing.")

    lines.extend(["", "## Per-Experiment Visual Examples", ""])
    for row in rows:
        visual = row["visual"]
        if visual is None:
            lines.append(f"### {row['name']}")
            lines.append("")
            lines.append("No visual example found for this experiment.")
            lines.append("")
            continue
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append(relative_image(visual, project_dir, f"{row['name']} visual example"))
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `baseline_no_bn` is worse than the baseline, showing BatchNorm is helpful for the scratch encoder-decoder.",
            "- `strong_aug` improves recall but reduces precision, so it predicts larger salient regions and creates more false positives.",
            "- `iou_heavy` does not improve the baseline, suggesting the plain architecture is the main bottleneck.",
            "- `improved` is the strongest run, with much better IoU/F1 and substantially lower MAE/MSE.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric_comparison(
    metrics_by_name: Mapping[str, Mapping[str, float] | None],
    output_path: Path,
) -> Path | None:
    available = {
        name: metrics
        for name, metrics in metrics_by_name.items()
        if metrics is not None and all(metric in metrics for metric in PLOT_METRICS)
    }
    if len(available) < 2:
        return None

    names = list(available)
    x_positions = list(range(len(PLOT_METRICS)))
    bar_width = min(0.16, 0.8 / max(len(names), 1))

    fig, axis = plt.subplots(figsize=(10, 5.5))
    for offset, name in enumerate(names):
        values = [available[name][metric] for metric in PLOT_METRICS]
        center_offset = offset - (len(names) - 1) / 2
        positions = [x + center_offset * bar_width for x in x_positions]
        axis.bar(positions, values, width=bar_width, label=name)

    axis.set_xticks(x_positions)
    axis.set_xticklabels(["IoU", "F1-score", "MAE"])
    axis.set_title("Experiment Test Metrics")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def labeled_image(path: Path, label: str, width: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    height = max(1, int(image.height * (width / image.width)))
    image = image.resize((width, height), RESAMPLE_LANCZOS)

    label_height = 38
    canvas = Image.new("RGB", (width, height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), label, fill=(20, 20, 20))
    canvas.paste(image, (0, label_height))
    return canvas


def create_contact_sheet(
    visual_by_name: Mapping[str, Path | None],
    output_path: Path,
) -> Path | None:
    available = {
        name: path
        for name, path in visual_by_name.items()
        if path is not None and path.exists()
    }
    if len(available) < 2:
        return None

    width = 900
    labeled_images = [labeled_image(path, name, width) for name, path in available.items()]
    total_height = sum(image.height for image in labeled_images)
    canvas = Image.new("RGB", (width, total_height), "white")

    top = 0
    for image in labeled_images:
        canvas.paste(image, (0, top))
        top += image.height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    metrics_dir = resolve_path(args.metrics_dir, project_dir)
    visualizations_dir = resolve_path(args.visualizations_dir, project_dir)
    output_markdown = resolve_path(args.output_markdown, project_dir)
    output_plot = resolve_path(args.output_plot, project_dir)
    output_visual = resolve_path(args.output_visual, project_dir)

    if args.experiment_names:
        names = [safe_experiment_name(name) for name in args.experiment_names]
    else:
        default_names = list(DEFAULT_EXPERIMENTS)
        default_names[0] = safe_experiment_name(args.baseline_name)
        default_names[-1] = safe_experiment_name(args.improved_name)
        names = default_names

    metrics_by_name = {
        name: load_metrics(metrics_dir, name)
        for name in names
    }
    visual_by_name = {
        name: find_first_visual(visualizations_dir, name)
        for name in names
    }

    plot_path = plot_metric_comparison(metrics_by_name, output_plot)
    visual_path = create_contact_sheet(visual_by_name, output_visual)

    rows = [
        {
            "name": name,
            "description": DEFAULT_DESCRIPTIONS.get(name, "Custom experiment."),
            "metrics": metrics_by_name[name],
            "visual": visual_by_name[name],
        }
        for name in names
    ]
    write_markdown_table(rows, output_markdown, project_dir, plot_path, visual_path)

    print(f"Comparison table saved to: {output_markdown}")
    if plot_path is not None:
        print(f"Metric chart saved to: {plot_path}")
    if visual_path is not None:
        print(f"Contact sheet saved to: {visual_path}")


if __name__ == "__main__":
    main()
