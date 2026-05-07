import argparse
from pathlib import Path
import random
import shutil
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from data_loader import create_dataloaders
from device_utils import get_available_device, is_cuda_device
from losses import combined_loss
from metrics import (
    append_epoch_history,
    empty_metric_totals,
    finalize_metric_totals,
    initialize_history,
    save_history,
    update_metric_totals,
)
from sod_model import get_model


def resolve_path(path_value: str, project_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "manual_seed")
    ):
        torch.mps.manual_seed(seed)


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    phase: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    metric_totals = empty_metric_totals()
    non_blocking = is_cuda_device(device)

    progress_bar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]", leave=False)
    for batch in progress_bar:
        images = batch["image"].to(device, non_blocking=non_blocking)
        masks = batch["mask"].to(device, non_blocking=non_blocking)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            predictions = model(images)
            loss = combined_loss(predictions, masks)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        update_metric_totals(metric_totals, predictions.detach(), masks.detach())
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_samples, 1), finalize_metric_totals(metric_totals)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, Dict[str, float]]:
    return run_epoch(model, loader, device, epoch, phase="train", optimizer=optimizer)


def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> Tuple[float, Dict[str, float]]:
    return run_epoch(model, loader, device, epoch, phase="val")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    history: Dict[str, list],
    model_type: str,
    image_size: int,
    epochs_without_improvement: int,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "current_epoch": epoch,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "model_type": model_type,
        "image_size": image_size,
        "epochs_without_improvement": epochs_without_improvement,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, float, Dict[str, list], int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    current_epoch = int(checkpoint.get("current_epoch", checkpoint.get("epoch", 0)))
    start_epoch = current_epoch + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    history = initialize_history(checkpoint.get("history"))
    epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))

    return start_epoch, best_val_loss, history, epochs_without_improvement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scratch CNN for DUTS saliency detection.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="pre-processed",
        help="Path to preprocessed data made by pre_processing.py.",
    )
    parser.add_argument("--image_size", type=int, default=128, help="Input image size.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline",
        choices=["baseline", "unet_small"],
        help="Model architecture to train.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument(
        "--use_raw_data",
        action="store_true",
        help="Use raw DUTS folders directly. This is slower and preprocesses during loading.",
    )
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Stop after this many epochs without validation improvement.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    data_dir = resolve_path(args.data_dir, project_dir)
    checkpoints_dir = project_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = project_dir / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_available_device()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print("Salient Object Detection Training")
    print(f"Data directory: {data_dir}")
    print(f"Data mode: {'raw DUTS' if args.use_raw_data else 'preprocessed tensors'}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {device}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print("=" * 70)

    train_loader, val_loader, _ = create_dataloaders(
        data_dir=str(data_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_preprocessed=not args.use_raw_data,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = get_model(args.model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    latest_path = checkpoints_dir / f"latest_{args.model_type}.pth"
    latest_alias_path = checkpoints_dir / "latest_checkpoint.pth"
    best_path = checkpoints_dir / "best_model.pth"
    best_typed_path = checkpoints_dir / f"best_model_{args.model_type}.pth"
    history_json_path = metrics_dir / f"training_history_{args.model_type}.json"
    history_csv_path = metrics_dir / f"training_history_{args.model_type}.csv"

    start_epoch = 1
    best_val_loss = float("inf")
    history = initialize_history()
    epochs_without_improvement = 0

    if args.resume:
        if latest_path.exists():
            start_epoch, best_val_loss, history, epochs_without_improvement = load_checkpoint(
                latest_path,
                model,
                optimizer,
                device,
            )
            print(f"Resumed training from: {latest_path}")
            print(f"Starting at epoch: {start_epoch}")
            print(f"Best validation loss so far: {best_val_loss:.4f}")
        else:
            print(f"No checkpoint found at {latest_path}. Starting from scratch.")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
        )
        val_loss, val_metrics = validate(model, val_loader, device, epoch)

        append_epoch_history(
            history,
            epoch,
            train_loss,
            val_loss,
            train_metrics,
            val_metrics,
        )
        save_history(history, history_json_path, history_csv_path)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(
            "Validation metrics: "
            f"IoU={val_metrics['iou']:.4f}, "
            f"Precision={val_metrics['precision']:.4f}, "
            f"Recall={val_metrics['recall']:.4f}, "
            f"F1={val_metrics['f1_score']:.4f}, "
            f"MAE={val_metrics['mae']:.4f}, "
            f"MSE={val_metrics['mse']:.4f}"
        )
        print(f"History saved to: {history_json_path} and {history_csv_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
                history,
                args.model_type,
                args.image_size,
                epochs_without_improvement,
            )
            shutil.copyfile(best_path, best_typed_path)
            print(f"Best model saved to: {best_path}")
        else:
            epochs_without_improvement += 1
            print(
                "Validation did not improve "
                f"({epochs_without_improvement}/{args.early_stopping_patience})."
            )

        save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch,
            best_val_loss,
            history,
            args.model_type,
            args.image_size,
            epochs_without_improvement,
        )
        shutil.copyfile(latest_path, latest_alias_path)
        print(f"Latest checkpoint saved to: {latest_path}")

        if epochs_without_improvement >= args.early_stopping_patience:
            print("Early stopping triggered.")
            break

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
