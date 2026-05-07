import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_path(path_value: str, project_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_dir / path).resolve()


def resolve_input_path(path_value: str, project_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    project_candidate = (project_dir / path).resolve()
    if project_candidate.exists():
        return project_candidate

    parent_candidate = (project_dir.parent / path).resolve()
    if parent_candidate.exists():
        return parent_candidate

    return project_candidate


def find_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    mask_paths = {
        p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    }

    pairs = [
        (image_path, mask_paths[image_path.stem])
        for image_path in image_paths
        if image_path.stem in mask_paths
    ]

    missing_masks = len(image_paths) - len(pairs)
    if missing_masks:
        print(f"Warning: skipped {missing_masks} images without matching masks in {image_dir}.")

    return pairs


def collect_all_duts_pairs(raw_data_dir: Path) -> List[Tuple[Path, Path]]:
    subsets = [
        (
            raw_data_dir / "DUTS-TR" / "DUTS-TR-Image",
            raw_data_dir / "DUTS-TR" / "DUTS-TR-Mask",
        ),
        (
            raw_data_dir / "DUTS-TE" / "DUTS-TE-Image",
            raw_data_dir / "DUTS-TE" / "DUTS-TE-Mask",
        ),
    ]

    all_pairs: List[Tuple[Path, Path]] = []
    for image_dir, mask_dir in subsets:
        for folder in [image_dir, mask_dir]:
            if not folder.exists():
                raise FileNotFoundError(f"Required DUTS folder not found: {folder}")
        all_pairs.extend(find_pairs(image_dir, mask_dir))

    return all_pairs


def split_train_val_test(
    pairs: List[Tuple[Path, Path]],
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    split_sum = train_split + val_split + test_split
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(
            "train_split + val_split + test_split must equal 1.0. "
            f"Got {split_sum:.4f}."
        )

    shuffled = pairs[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_count = round(len(shuffled) * val_split)
    test_count = round(len(shuffled) * test_split)
    train_count = len(shuffled) - val_count - test_count

    train_pairs = shuffled[:train_count]
    val_pairs = shuffled[train_count:train_count + val_count]
    test_pairs = shuffled[train_count + val_count:]

    return train_pairs, val_pairs, test_pairs


def preprocess_pair(
    image_path: Path,
    mask_path: Path,
    image_size: int,
) -> Tuple[Image.Image, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image = image.resize((image_size, image_size), Image.BILINEAR)
    mask = mask.resize((image_size, image_size), Image.NEAREST)

    return image, mask


def pair_to_tensors(
    image: Image.Image,
    mask: Image.Image,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    mask_array = np.asarray(mask, dtype=np.float32) / 255.0

    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).contiguous()
    mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)

    return image_tensor, mask_tensor


def save_split(
    split_name: str,
    pairs: List[Tuple[Path, Path]],
    output_dir: Path,
    image_size: int,
    output_format: str,
) -> List[Dict[str, str]]:
    image_output_dir = output_dir / split_name / "images"
    mask_output_dir = output_dir / split_name / "masks"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_samples = []
    for sample_index, (image_path, mask_path) in enumerate(
        tqdm(pairs, desc=f"Preprocessing {split_name}")
    ):
        image, mask = preprocess_pair(image_path, mask_path, image_size)

        tensor_stem = f"{sample_index:06d}_{image_path.stem}"
        if output_format == "tensors":
            image_tensor, mask_tensor = pair_to_tensors(image, mask)
            image_tensor_path = image_output_dir / f"{tensor_stem}.pt"
            mask_tensor_path = mask_output_dir / f"{tensor_stem}.pt"

            torch.save(image_tensor, image_tensor_path)
            torch.save(mask_tensor, mask_tensor_path)

            manifest_samples.append(
                {
                    "image_tensor": str(image_tensor_path.relative_to(output_dir)),
                    "mask_tensor": str(mask_tensor_path.relative_to(output_dir)),
                    "source_image": str(image_path),
                    "source_mask": str(mask_path),
                }
            )
        else:
            image_suffix = image_path.suffix.lower()
            mask_suffix = mask_path.suffix.lower()
            image_file_path = image_output_dir / f"{tensor_stem}{image_suffix}"
            mask_file_path = mask_output_dir / f"{tensor_stem}{mask_suffix}"

            image.save(image_file_path)
            mask.save(mask_file_path)

            manifest_samples.append(
                {
                    "image_path": str(image_file_path.relative_to(output_dir)),
                    "mask_path": str(mask_file_path.relative_to(output_dir)),
                    "source_image": str(image_path),
                    "source_mask": str(mask_path),
                }
            )

    return manifest_samples


def clear_existing_preprocessed_splits(output_dir: Path) -> None:
    for split_name in ["train", "val", "test"]:
        split_dir = output_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess DUTS once into fixed-size tensors for training."
    )
    parser.add_argument("--raw_data_dir", type=str, default="data", help="Raw DUTS folder.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pre-processed",
        help="Folder where preprocessed tensors will be saved.",
    )
    parser.add_argument("--image_size", type=int, default=128, help="Saved tensor size.")
    parser.add_argument(
        "--output_format",
        type=str,
        default="images",
        choices=["images", "tensors"],
        help="Save preprocessed outputs as image files or torch tensors.",
    )
    parser.add_argument("--train_split", type=float, default=0.70, help="Training split ratio.")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test_split", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for train/val/test split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    raw_data_dir = resolve_input_path(args.raw_data_dir, project_dir)
    output_dir = resolve_path(args.output_dir, project_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DUTS Preprocessing")
    print(f"Raw data: {raw_data_dir}")
    print(f"Output: {output_dir}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Output format: {args.output_format}")
    print(
        "Split ratios: "
        f"train={args.train_split:.2f}, "
        f"val={args.val_split:.2f}, "
        f"test={args.test_split:.2f}"
    )
    print("=" * 70)

    all_pairs = collect_all_duts_pairs(raw_data_dir)
    train_pairs, val_pairs, test_pairs = split_train_val_test(
        all_pairs,
        args.train_split,
        args.val_split,
        args.test_split,
        args.seed,
    )

    print(f"Total paired samples: {len(all_pairs)}")
    print(f"Train samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")
    print(f"Test samples: {len(test_pairs)}")

    clear_existing_preprocessed_splits(output_dir)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_data_dir": str(raw_data_dir),
        "image_size": args.image_size,
        "output_format": args.output_format,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "seed": args.seed,
        "splits": {
            "train": save_split(
                "train",
                train_pairs,
                output_dir,
                args.image_size,
                args.output_format,
            ),
            "val": save_split(
                "val",
                val_pairs,
                output_dir,
                args.image_size,
                args.output_format,
            ),
            "test": save_split(
                "test",
                test_pairs,
                output_dir,
                args.image_size,
                args.output_format,
            ),
        },
    }
    manifest["counts"] = {
        split_name: len(samples)
        for split_name, samples in manifest["splits"].items()
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("\nPreprocessing finished.")
    print(f"Manifest saved to: {manifest_path}")
    print("Training can now use: python train.py --data_dir pre-processed")


if __name__ == "__main__":
    main()
