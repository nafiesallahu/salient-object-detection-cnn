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
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
DEFAULT_DESCRIPTIONS = {
    "baseline": "Baseline encoder-decoder CNN, light augmentation, BCE + 0.5 IoU loss.",
    "improved": (
        "Small UNet with deeper convolution blocks, skip connections, and dropout."
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
        description="Create a baseline-vs-improved metrics table and visual comparison."
    )
    parser.add_argument("--baseline_name", type=str, default="baseline")
    parser.add_argument("--improved_name", type=str, default="improved")
    parser.add_argument("--metrics_dir", type=str, default="outputs/metrics")
    parser.add_argument("--visualizations_dir", type=str, default="outputs/visualizations")
    parser.add_argument("--output_markdown", type=str, default="outputs/metrics/experiment_comparison.md")
    parser.add_argument(
        "--output_plot",
        type=str,
        default="outputs/metrics/plots/baseline_vs_improved_metrics.png",
    )
    parser.add_argument(
        "--output_visual",
        type=str,
        default="outputs/visualizations/baseline_vs_improved_contact_sheet.png",
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


def relative_link(path: Path | None, project_dir: Path) -> str:
    if path is None:
        return "missing"
    try:
        display_path = path.resolve().relative_to(project_dir)
    except ValueError:
        display_path = path
    return f"[{display_path}]({display_path})"


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
    lines = [
        "# Baseline vs Improved Results",
        "",
        "| Experiment | Configuration | IoU | Precision | Recall | F1-score | MAE | MSE | Visual example |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        metrics = row["metrics"]
        visual = row["visual"]
        lines.append(
            "| {name} | {description} | {iou} | {precision} | {recall} | "
            "{f1_score} | {mae} | {mse} | {visual_link} |".format(
                name=row["name"],
                description=row["description"],
                iou=format_metric(metrics, "iou"),
                precision=format_metric(metrics, "precision"),
                recall=format_metric(metrics, "recall"),
                f1_score=format_metric(metrics, "f1_score"),
                mae=format_metric(metrics, "mae"),
                mse=format_metric(metrics, "mse"),
                visual_link=relative_link(visual, project_dir),
            )
        )

    lines.extend(["", "## Visual Outputs", ""])
    if plot_path is not None:
        lines.append(f"- Metric comparison chart: {relative_link(plot_path, project_dir)}")
    if visual_path is not None:
        lines.append(f"- Qualitative contact sheet: {relative_link(visual_path, project_dir)}")
    if plot_path is None and visual_path is None:
        lines.append("- No comparison visuals were generated because the required inputs are missing.")

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
    bar_width = 0.35

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for offset, name in enumerate(names):
        values = [available[name][metric] for metric in PLOT_METRICS]
        positions = [x + (offset - 0.5) * bar_width for x in x_positions]
        axis.bar(positions, values, width=bar_width, label=name)

    axis.set_xticks(x_positions)
    axis.set_xticklabels(["IoU", "F1-score", "MAE"])
    axis.set_title("Baseline vs Improved Test Metrics")
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

    baseline_name = safe_experiment_name(args.baseline_name)
    improved_name = safe_experiment_name(args.improved_name)
    names = [baseline_name, improved_name]

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
