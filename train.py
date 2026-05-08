import argparse
from pathlib import Path
import random
import re
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
    bce_weight: float = 1.0,
    iou_weight: float = 0.5,
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
            loss = combined_loss(
                predictions,
                masks,
                bce_weight=bce_weight,
                iou_weight=iou_weight,
            )
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
    bce_weight: float = 1.0,
    iou_weight: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    return run_epoch(
        model,
        loader,
        device,
        epoch,
        phase="train",
        optimizer=optimizer,
        bce_weight=bce_weight,
        iou_weight=iou_weight,
    )


def validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    bce_weight: float = 1.0,
    iou_weight: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    return run_epoch(
        model,
        loader,
        device,
        epoch,
        phase="val",
        bce_weight=bce_weight,
        iou_weight=iou_weight,
    )


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
    training_config: Dict[str, object],
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
        "training_config": training_config,
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
        "--experiment_name",
        type=str,
        default=None,
        help=(
            "Optional run name used for checkpoints and metric history. "
            "Defaults to the model type."
        ),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline",
        choices=list(MODEL_TYPES),
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability for models that support dropout, such as unet_small.",
    )
    parser.add_argument(
        "--augmentation_strength",
        type=str,
        default="light",
        choices=["none", "light", "strong"],
        help="Train-time augmentation policy.",
    )
    parser.add_argument(
        "--bce_weight",
        type=float,
        default=1.0,
        help="Weight applied to binary cross-entropy loss.",
    )
    parser.add_argument(
        "--iou_weight",
        type=float,
        default=0.5,
        help="Weight applied to soft IoU loss.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Automatically resume from the latest checkpoint if one exists. This is the default.",
    )
    parser.add_argument(
        "--no_resume",
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start training from scratch even if a latest checkpoint exists.",
    )
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
    parser.add_argument(
        "--write_global_aliases",
        action="store_true",
        help=(
            "Also update checkpoints/best_model.pth and latest_checkpoint.pth. "
            "Disabled by default so ablations do not overwrite the demo/default model."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    data_dir = resolve_path(args.data_dir, project_dir)
    run_name = (
        safe_experiment_name(args.experiment_name)
        if args.experiment_name
        else args.model_type
    )
    if args.bce_weight < 0 or args.iou_weight < 0:
        raise ValueError("Loss weights must be non-negative.")
    if args.bce_weight == 0 and args.iou_weight == 0:
        raise ValueError("At least one loss weight must be greater than zero.")

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
    print(f"Experiment name: {run_name}")
    print(f"Model type: {args.model_type}")
    if args.model_type == "unet_small":
        print(f"Dropout: {args.dropout}")
    print(f"Device: {device}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Augmentation strength: {args.augmentation_strength}")
    print(f"Loss: {args.bce_weight:g} * BCE + {args.iou_weight:g} * IoU loss")
    print(f"Update global checkpoint aliases: {args.write_global_aliases}")
    print("=" * 70)

    train_loader, val_loader, _ = create_dataloaders(
        data_dir=str(data_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_preprocessed=not args.use_raw_data,
        augmentation_strength=args.augmentation_strength,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = get_model(args.model_type, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    training_config = {
        "experiment_name": run_name,
        "model_type": args.model_type,
        "dropout": args.dropout if args.model_type == "unet_small" else None,
        "augmentation_strength": args.augmentation_strength,
        "bce_weight": args.bce_weight,
        "iou_weight": args.iou_weight,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "seed": args.seed,
        "write_global_aliases": args.write_global_aliases,
    }

    latest_path = checkpoints_dir / f"latest_{run_name}.pth"
    latest_alias_path = checkpoints_dir / "latest_checkpoint.pth"
    best_alias_path = checkpoints_dir / "best_model.pth"
    best_run_path = checkpoints_dir / f"best_model_{run_name}.pth"
    history_json_path = metrics_dir / f"training_history_{run_name}.json"
    history_csv_path = metrics_dir / f"training_history_{run_name}.csv"

    start_epoch = 1
    best_val_loss = float("inf")
    history = initialize_history()
    epochs_without_improvement = 0
    resumed_from_checkpoint = False

    if args.resume and latest_path.exists():
        start_epoch, best_val_loss, history, epochs_without_improvement = load_checkpoint(
            latest_path,
            model,
            optimizer,
            device,
        )
        resumed_from_checkpoint = True
        print(f"Checkpoint found. Resumed training from: {latest_path}")
        print(f"Continuing from epoch: {start_epoch}")
        print(f"Best validation loss so far: {best_val_loss:.4f}")
    elif args.resume:
        print(f"No checkpoint found at {latest_path}. Starting from scratch.")
    else:
        print("Automatic checkpoint resume disabled. Starting from scratch.")

    if resumed_from_checkpoint and start_epoch > args.epochs:
        print(
            f"Checkpoint already reached epoch {start_epoch - 1}; "
            f"requested epochs: {args.epochs}."
        )
        print("Increase --epochs to continue training, or use --no_resume to start over.")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            bce_weight=args.bce_weight,
            iou_weight=args.iou_weight,
        )
        val_loss, val_metrics = validate(
            model,
            val_loader,
            device,
            epoch,
            bce_weight=args.bce_weight,
            iou_weight=args.iou_weight,
        )

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
                best_run_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
                history,
                args.model_type,
                args.image_size,
                epochs_without_improvement,
                training_config,
            )
            print(f"Experiment best model saved to: {best_run_path}")
            if args.write_global_aliases:
                shutil.copyfile(best_run_path, best_alias_path)
                print(f"Global best alias updated: {best_alias_path}")
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
            training_config,
        )
        print(f"Latest checkpoint saved to: {latest_path}")
        if args.write_global_aliases:
            shutil.copyfile(latest_path, latest_alias_path)
            print(f"Global latest alias updated: {latest_alias_path}")

        if epochs_without_improvement >= args.early_stopping_patience:
            print("Early stopping triggered.")
            break

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
