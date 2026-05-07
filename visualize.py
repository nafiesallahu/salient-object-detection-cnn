import argparse
import os
from pathlib import Path
from tempfile import gettempdir
import re

MPLCONFIGDIR = Path(gettempdir()) / "sod_matplotlib_cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import DUTSDataset, PreprocessedDUTSDataset
from device_utils import get_available_device
from sod_model import MODEL_TYPES, get_model


def resolve_path(path_value: str, project_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def safe_folder_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    name = name.strip("._-")
    return name or "experiment"


def infer_experiment_name(model_type: str, checkpoint_path: Path) -> str:
    checkpoint_stem = checkpoint_path.stem
    for prefix in ("best_model_", "latest_"):
        if checkpoint_stem.startswith(prefix):
            checkpoint_stem = checkpoint_stem[len(prefix):]
            break

    if checkpoint_stem in {"best_model", "latest_checkpoint"}:
        checkpoint_stem = model_type

    return safe_folder_name(checkpoint_stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save saliency visualization examples.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="pre-processed",
        help="Path to preprocessed data made by pre_processing.py.",
    )
    parser.add_argument("--image_size", type=int, default=128, help="Input image size.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline",
        choices=list(MODEL_TYPES),
        help="Model architecture used by the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to best model checkpoint.",
    )
    parser.add_argument("--num_samples", type=int, default=10, help="Number of examples to save.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help=(
            "Name of the experiment subfolder inside outputs/visualizations. "
            "Defaults to the checkpoint name, or model_type for checkpoints/best_model.pth."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/visualizations",
        help="Base directory where experiment visualization folders are saved.",
    )
    parser.add_argument(
        "--use_raw_data",
        action="store_true",
        help="Use raw DUTS folders directly. This is slower and preprocesses during loading.",
    )
    return parser.parse_args()


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    heatmap = np.zeros_like(image)
    heatmap[..., 0] = mask
    overlay = 0.65 * image + 0.35 * heatmap
    return np.clip(overlay, 0.0, 1.0)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    data_dir = resolve_path(args.data_dir, project_dir)
    checkpoint_path = resolve_path(args.checkpoint, project_dir)

    if not checkpoint_path.exists():
        typed_checkpoint = project_dir / "checkpoints" / f"best_model_{args.model_type}.pth"
        if typed_checkpoint.exists():
            checkpoint_path = typed_checkpoint
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path} or {typed_checkpoint}."
            )

    experiment_name = (
        safe_folder_name(args.experiment_name)
        if args.experiment_name
        else infer_experiment_name(args.model_type, checkpoint_path)
    )
    output_base_dir = resolve_path(args.output_dir, project_dir)
    output_dir = output_base_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_available_device()
    print("=" * 70)
    print("Generating Saliency Visualizations")
    print(f"Data directory: {data_dir}")
    print(f"Data mode: {'raw DUTS' if args.use_raw_data else 'preprocessed tensors'}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Experiment: {experiment_name}")
    print(f"Saving to: {output_dir}")
    print("=" * 70)

    dataset_class = DUTSDataset if args.use_raw_data else PreprocessedDUTSDataset
    dataset = dataset_class(
        data_dir=str(data_dir),
        split="test",
        image_size=args.image_size,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = get_model(args.model_type).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    saved_count = 0
    for index, batch in enumerate(tqdm(loader, desc="Visualizing")):
        if saved_count >= args.num_samples:
            break

        image_tensor = batch["image"].to(device)
        mask_tensor = batch["mask"].to(device)
        prediction = model(image_tensor)

        image = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        ground_truth = mask_tensor[0, 0].detach().cpu().numpy()
        predicted_mask = prediction[0, 0].detach().cpu().numpy()
        overlay = make_overlay(image, predicted_mask)

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(image)
        axes[0].set_title("Input")
        axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")
        axes[2].imshow(predicted_mask, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Prediction")
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay")

        for axis in axes:
            axis.axis("off")

        image_name = Path(batch["image_path"][0]).stem
        save_path = output_dir / f"visualization_{index + 1:02d}_{image_name}.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        saved_count += 1

    print(f"Saved {saved_count} visualization files to: {output_dir}")


if __name__ == "__main__":
    main()
