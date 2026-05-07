import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import DUTSDataset, PreprocessedDUTSDataset
from device_utils import get_available_device
from metrics import empty_metric_totals, finalize_metric_totals, update_metric_totals
from sod_model import get_model


def resolve_path(path_value: str, project_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


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
        default="baseline",
        choices=["baseline", "unet_small"],
        help="Model architecture used by the checkpoint.",
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
        default="checkpoints/best_model.pth",
        help="Path to best model checkpoint.",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    data_dir = resolve_path(args.data_dir, project_dir)
    checkpoint_path = resolve_path(args.checkpoint, project_dir)
    metrics_dir = project_dir / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = get_available_device()

    print("=" * 70)
    print("Salient Object Detection Evaluation")
    print(f"Data directory: {data_dir}")
    print(f"Data mode: {'raw DUTS' if args.use_raw_data else 'preprocessed tensors'}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {device}")
    print("=" * 70)

    if not checkpoint_path.exists():
        typed_checkpoint = project_dir / "checkpoints" / f"best_model_{args.model_type}.pth"
        if typed_checkpoint.exists():
            checkpoint_path = typed_checkpoint
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path} or {typed_checkpoint}."
            )

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

    model = get_model(args.model_type).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_model_type = checkpoint.get("model_type")
    if checkpoint_model_type and checkpoint_model_type != args.model_type:
        print(
            "Warning: checkpoint was trained with "
            f"model_type='{checkpoint_model_type}', but current model_type is "
            f"'{args.model_type}'."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    totals = empty_metric_totals()
    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        predictions = model(images)
        update_metric_totals(totals, predictions, masks)

    final_metrics = finalize_metric_totals(totals)
    metrics_path = metrics_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(final_metrics, file, indent=2)

    print("\nTest metrics")
    print(f"IoU:       {final_metrics['iou']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"F1-score:  {final_metrics['f1_score']:.4f}")
    print(f"MAE:       {final_metrics['mae']:.4f}")
    print(f"MSE:       {final_metrics['mse']:.4f}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
