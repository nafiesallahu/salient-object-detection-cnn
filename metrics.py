import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping

import torch


EPSILON = 1e-7
METRIC_NAMES = ("precision", "recall", "f1_score", "iou", "mae", "mse")
HISTORY_KEYS = (
    "epoch",
    "train_loss",
    "val_loss",
    "train_precision",
    "train_recall",
    "train_f1_score",
    "train_iou",
    "train_mae",
    "train_mse",
    "val_precision",
    "val_recall",
    "val_f1_score",
    "val_iou",
    "val_mae",
    "val_mse",
)


@torch.no_grad()
def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = EPSILON,
) -> Dict[str, float]:
    """Compute binary saliency metrics for a batch."""
    totals = empty_metric_totals()
    update_metric_totals(totals, predictions, targets, threshold=threshold)
    return finalize_metric_totals(totals, epsilon=epsilon)


def empty_metric_totals() -> Dict[str, float]:
    return {
        "true_positive": 0.0,
        "false_positive": 0.0,
        "false_negative": 0.0,
        "absolute_error": 0.0,
        "squared_error": 0.0,
        "num_pixels": 0.0,
        "num_batches": 0.0,
    }


@torch.no_grad()
def update_metric_totals(
    totals: MutableMapping[str, float],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> None:
    predictions = predictions.detach().float()
    targets = targets.detach().float()

    pred_binary = (predictions >= threshold).float()
    target_binary = (targets >= threshold).float()

    pred_flat = pred_binary.reshape(-1)
    target_flat = target_binary.reshape(-1)

    totals["true_positive"] += float((pred_flat * target_flat).sum().item())
    totals["false_positive"] += float((pred_flat * (1.0 - target_flat)).sum().item())
    totals["false_negative"] += float(((1.0 - pred_flat) * target_flat).sum().item())
    totals["absolute_error"] += float(torch.abs(predictions - targets).sum().item())
    totals["squared_error"] += float(torch.square(predictions - targets).sum().item())
    totals["num_pixels"] += float(targets.numel())
    totals["num_batches"] += 1.0


def finalize_metric_totals(
    totals: Mapping[str, float],
    epsilon: float = EPSILON,
) -> Dict[str, float]:
    true_positive = float(totals.get("true_positive", 0.0))
    false_positive = float(totals.get("false_positive", 0.0))
    false_negative = float(totals.get("false_negative", 0.0))
    num_pixels = max(float(totals.get("num_pixels", 0.0)), 1.0)

    precision = (true_positive + epsilon) / (
        true_positive + false_positive + epsilon
    )
    recall = (true_positive + epsilon) / (
        true_positive + false_negative + epsilon
    )
    f1_score = (2.0 * precision * recall + epsilon) / (
        precision + recall + epsilon
    )
    iou = (true_positive + epsilon) / (
        true_positive + false_positive + false_negative + epsilon
    )
    mae = float(totals.get("absolute_error", 0.0)) / num_pixels
    mse = float(totals.get("squared_error", 0.0)) / num_pixels

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "iou": iou,
        "mae": mae,
        "mse": mse,
    }


def initialize_history(existing_history: Mapping[str, Iterable] | None = None) -> Dict[str, List]:
    history = {key: [] for key in HISTORY_KEYS}
    if existing_history:
        for key, values in existing_history.items():
            if key in history and isinstance(values, list):
                history[key] = list(values)

    max_length = max((len(values) for values in history.values()), default=0)
    if not history["epoch"] and max_length > 0:
        history["epoch"] = list(range(1, max_length + 1))

    target_length = max(len(history["epoch"]), max_length)
    for key in HISTORY_KEYS:
        if len(history[key]) < target_length:
            history[key].extend([None] * (target_length - len(history[key])))
        elif len(history[key]) > target_length:
            history[key] = history[key][:target_length]

    return history


def append_epoch_history(
    history: MutableMapping[str, List],
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
) -> None:
    target_length = len(history.get("epoch", []))
    for key in HISTORY_KEYS:
        history.setdefault(key, [])
        if len(history[key]) < target_length:
            history[key].extend([None] * (target_length - len(history[key])))

    history["epoch"].append(int(epoch))
    history["train_loss"].append(float(train_loss))
    history["val_loss"].append(float(val_loss))

    for metric_name in METRIC_NAMES:
        history[f"train_{metric_name}"].append(float(train_metrics[metric_name]))
        history[f"val_{metric_name}"].append(float(val_metrics[metric_name]))


def history_to_rows(history: Mapping[str, List]) -> List[Dict[str, object]]:
    row_count = max((len(history.get(key, [])) for key in HISTORY_KEYS), default=0)
    rows: List[Dict[str, object]] = []
    for index in range(row_count):
        row = {}
        for key in HISTORY_KEYS:
            values = history.get(key, [])
            row[key] = values[index] if index < len(values) else None
        rows.append(row)
    return rows


def save_history_json(history: Mapping[str, List], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def save_history_csv(history: Mapping[str, List], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(HISTORY_KEYS))
        writer.writeheader()
        writer.writerows(history_to_rows(history))


def save_history(history: Mapping[str, List], json_path: Path, csv_path: Path) -> None:
    save_history_json(history, json_path)
    save_history_csv(history, csv_path)


def load_history_json(path: Path) -> Dict[str, List]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if "history" in payload and isinstance(payload["history"], dict):
        payload = payload["history"]
    return initialize_history(payload)
