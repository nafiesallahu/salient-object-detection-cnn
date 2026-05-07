import argparse
import csv
import os
from pathlib import Path
import re
from tempfile import gettempdir
from typing import Dict, List, Mapping, Sequence

MPLCONFIGDIR = Path(gettempdir()) / "sod_matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from metrics import HISTORY_KEYS, METRIC_NAMES, initialize_history, load_history_json
from sod_model import MODEL_TYPES


METRIC_TITLES = {
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1 Score",
    "iou": "IoU",
    "mae": "MAE",
    "mse": "MSE",
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
    parser = argparse.ArgumentParser(description="Plot saved training metrics history.")
    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to training history JSON or CSV. Defaults to the selected model history.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline",
        choices=list(MODEL_TYPES),
        help="Model type used to choose the default history path.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help=(
            "Optional run name used to choose training_history_<experiment_name>.json "
            "when --history is not provided."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/metrics/plots",
        help="Directory where plot images will be saved.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix for plot outputs. Defaults to the model type.",
    )
    return parser.parse_args()


def parse_csv_value(value: str | None):
    if value is None or value == "":
        return None
    try:
        numeric_value = float(value)
    except ValueError:
        return value
    if numeric_value.is_integer():
        return int(numeric_value)
    return numeric_value


def load_history_csv(path: Path) -> Dict[str, List]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append({key: parse_csv_value(value) for key, value in row.items()})

    history = {key: [] for key in HISTORY_KEYS}
    for row in rows:
        for key in HISTORY_KEYS:
            history[key].append(row.get(key))
    return initialize_history(history)


def load_history(path: Path) -> Dict[str, List]:
    if path.suffix.lower() == ".json":
        return load_history_json(path)
    if path.suffix.lower() == ".csv":
        return load_history_csv(path)
    raise ValueError("History path must end with .json or .csv")


def get_epochs(history: Mapping[str, Sequence]) -> List[int]:
    epochs = history.get("epoch", [])
    if epochs:
        return [int(epoch) for epoch in epochs]
    row_count = max((len(values) for values in history.values()), default=0)
    return list(range(1, row_count + 1))


def valid_series(
    epochs: Sequence[int],
    values: Sequence,
) -> tuple[List[int], List[float]]:
    x_values: List[int] = []
    y_values: List[float] = []
    for index, value in enumerate(values):
        if value is None or value == "":
            continue
        x_values.append(int(epochs[index]) if index < len(epochs) else index + 1)
        y_values.append(float(value))
    return x_values, y_values


def plot_loss_curves(history: Mapping[str, Sequence], output_path: Path) -> None:
    epochs = get_epochs(history)
    fig, axis = plt.subplots(figsize=(8, 4.5))

    for key, label in [("train_loss", "Train"), ("val_loss", "Validation")]:
        x_values, y_values = valid_series(epochs, history.get(key, []))
        if y_values:
            axis.plot(x_values, y_values, marker="o", linewidth=2, label=label)

    axis.set_title("Train vs Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_curves(history: Mapping[str, Sequence], output_path: Path) -> None:
    epochs = get_epochs(history)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    for axis, metric_name in zip(axes.flat, METRIC_NAMES):
        for split_name, label in [("train", "Train"), ("val", "Validation")]:
            key = f"{split_name}_{metric_name}"
            x_values, y_values = valid_series(epochs, history.get(key, []))
            if y_values:
                axis.plot(x_values, y_values, marker="o", linewidth=2, label=label)

        axis.set_title(METRIC_TITLES[metric_name])
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_training_history(
    history: Mapping[str, Sequence],
    output_dir: Path,
    prefix: str,
) -> List[Path]:
    output_paths = [
        output_dir / f"{prefix}_loss_curves.png",
        output_dir / f"{prefix}_metric_curves.png",
    ]
    plot_loss_curves(history, output_paths[0])
    plot_metric_curves(history, output_paths[1])
    return output_paths


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    default_name = (
        safe_experiment_name(args.experiment_name)
        if args.experiment_name
        else args.model_type
    )
    history_path = (
        resolve_path(args.history, project_dir)
        if args.history
        else project_dir / "outputs" / "metrics" / f"training_history_{default_name}.json"
    )
    output_dir = resolve_path(args.output_dir, project_dir)
    prefix = args.prefix or default_name

    history = load_history(history_path)
    output_paths = plot_training_history(history, output_dir, prefix)

    print(f"Loaded history from: {history_path}")
    for output_path in output_paths:
        print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
