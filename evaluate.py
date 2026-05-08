import argparse
import csv
import json
from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import DUTSDataset, PreprocessedDUTSDataset
from device_utils import get_available_device
from metrics import (
    compute_metrics,
    empty_metric_totals,
    finalize_metric_totals,
    update_metric_totals,
)
from sod_model import MODEL_TYPES, get_model


def resolve_path(path_value: str, project_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def safe_experiment_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    name = name.strip("._-")
    return name or "experiment"


def resolve_checkpoint_path(
    checkpoint_value: str | None,
    model_type: str | None,
    experiment_name: str | None,
    project_dir: Path,
) -> Path:
    if checkpoint_value:
        checkpoint_path = resolve_path(checkpoint_value, project_dir)
        if checkpoint_path.exists():
            return checkpoint_path
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}.")

    candidates = []
    if experiment_name:
        candidates.append(
            project_dir
            / "checkpoints"
            / f"best_model_{safe_experiment_name(experiment_name)}.pth"
        )
    if model_type:
        candidates.append(project_dir / "checkpoints" / f"best_model_{model_type}.pth")
    candidates.extend(
        [
            project_dir / "checkpoints" / "best_model_improved.pth",
            project_dir / "checkpoints" / "best_model.pth",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidates_text = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No checkpoint found. Tried: {candidates_text}")


def default_per_image_path(
    metrics_dir: Path,
    experiment_name: str | None,
) -> Path:
    if experiment_name:
        return metrics_dir / f"per_image_metrics_{safe_experiment_name(experiment_name)}.csv"
    return metrics_dir / "per_image_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SOD model on DUTS-TE.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="pre-processed",
        help="Path to preprocessed data made by pre_processing.py.",
    )
    parser.add_argument("--image_size", type=int, default=128, help="Input image size.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=list(MODEL_TYPES),
        help="Model architecture used by the checkpoint. Defaults to checkpoint metadata.",
    )
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument(
        "--use_raw_data",
        action="store_true",
        help="Use raw DUTS folders directly. This is slower and preprocesses during loading.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to model checkpoint. Defaults to best_model_<experiment_name>.pth, "
            "best_model_<model_type>.pth, then the improved/best aliases."
        ),
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help=(
            "Optional run name used to also save outputs/metrics/"
            "test_metrics_<experiment_name>.json."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary threshold used for IoU, precision, recall, and F1-score.",
    )
    parser.add_argument(
        "--per_image_csv",
        type=str,
        default=None,
        help=(
            "Optional path for per-image metrics CSV. If omitted, a run-specific "
            "CSV is saved when --experiment_name is provided."
        ),
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    data_dir = resolve_path(args.data_dir, project_dir)
    metrics_dir = project_dir / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_checkpoint_path(
        args.checkpoint,
        args.model_type,
        args.experiment_name,
        project_dir,
    )
    per_image_path = (
        resolve_path(args.per_image_csv, project_dir)
        if args.per_image_csv
        else default_per_image_path(metrics_dir, args.experiment_name)
    )

    device = get_available_device()

    print("=" * 70)
    print("Salient Object Detection Evaluation")
    print(f"Data directory: {data_dir}")
    print(f"Data mode: {'raw DUTS' if args.use_raw_data else 'preprocessed tensors'}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Requested model type: {args.model_type or 'from checkpoint metadata'}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {device}")
    print("=" * 70)

    dataset_class = DUTSDataset if args.use_raw_data else PreprocessedDUTSDataset
    test_dataset = dataset_class(
        data_dir=str(data_dir),
        split="test",
        image_size=args.image_size,
        augment=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Test samples: {len(test_dataset)}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_model_type = checkpoint.get("model_type")
    model_type = args.model_type or checkpoint_model_type or "baseline"
    if checkpoint_model_type and checkpoint_model_type != model_type:
        print(
            "Warning: checkpoint was trained with "
            f"model_type='{checkpoint_model_type}', but current model_type is "
            f"'{model_type}'."
        )
    print(f"Effective model type: {model_type}")
    model = get_model(model_type).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    totals = empty_metric_totals()
    per_image_rows = []
    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        predictions = model(images)
        update_metric_totals(totals, predictions, masks, threshold=args.threshold)
        for index in range(images.size(0)):
            sample_metrics = compute_metrics(
                predictions[index:index + 1].detach().cpu(),
                masks[index:index + 1].detach().cpu(),
                threshold=args.threshold,
            )
            per_image_rows.append(
                {
                    "image_path": batch["image_path"][index],
                    "mask_path": batch["mask_path"][index],
                    **sample_metrics,
                }
            )

    final_metrics = finalize_metric_totals(totals)
    metrics_path = metrics_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(final_metrics, file, indent=2)

    experiment_metrics_path = None
    if args.experiment_name:
        experiment_name = safe_experiment_name(args.experiment_name)
        experiment_metrics_path = metrics_dir / f"test_metrics_{experiment_name}.json"
        with open(experiment_metrics_path, "w", encoding="utf-8") as file:
            json.dump(final_metrics, file, indent=2)

    if per_image_rows:
        per_image_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "image_path",
            "mask_path",
            "precision",
            "recall",
            "f1_score",
            "iou",
            "mae",
            "mse",
        ]
        with open(per_image_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_image_rows)

    print("\nTest metrics")
    print(f"IoU:       {final_metrics['iou']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"F1-score:  {final_metrics['f1_score']:.4f}")
    print(f"MAE:       {final_metrics['mae']:.4f}")
    print(f"MSE:       {final_metrics['mse']:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    if experiment_metrics_path is not None:
        print(f"Experiment metrics saved to: {experiment_metrics_path}")
    print(f"Per-image metrics saved to: {per_image_path}")


if __name__ == "__main__":
    main()
