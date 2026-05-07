import json
from pathlib import Path
import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUGMENTATION_STRENGTHS = {
    "none": {
        "padding_divisor": 16,
        "min_padding": 0,
        "rotation_degrees": 0.0,
        "brightness": (1.0, 1.0),
        "contrast": None,
        "scale_range": (1.0, 1.0),
    },
    "light": {
        "padding_divisor": 16,
        "min_padding": 4,
        "rotation_degrees": 10.0,
        "brightness": (0.8, 1.2),
        "contrast": None,
        "scale_range": (1.0, 1.25),
    },
    "strong": {
        "padding_divisor": 8,
        "min_padding": 8,
        "rotation_degrees": 20.0,
        "brightness": (0.7, 1.3),
        "contrast": (0.85, 1.15),
        "scale_range": (1.0, 1.4),
    },
}


def validate_augmentation_strength(value: str) -> str:
    strength = value.lower()
    if strength not in AUGMENTATION_STRENGTHS:
        valid_values = ", ".join(AUGMENTATION_STRENGTHS)
        raise ValueError(f"augmentation_strength must be one of: {valid_values}")
    return strength


def resolve_data_dir(data_dir: str) -> Path:
    path = Path(data_dir)
    if path.is_absolute():
        return path

    project_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / path,
        project_dir / path,
        project_dir.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (project_dir / path).resolve()


def find_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    mask_paths = {
        p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    }
    return [
        (image_path, mask_paths[image_path.stem])
        for image_path in image_paths
        if image_path.stem in mask_paths
    ]


def collect_all_duts_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    subsets = [
        (
            data_dir / "DUTS-TR" / "DUTS-TR-Image",
            data_dir / "DUTS-TR" / "DUTS-TR-Mask",
        ),
        (
            data_dir / "DUTS-TE" / "DUTS-TE-Image",
            data_dir / "DUTS-TE" / "DUTS-TE-Mask",
        ),
    ]

    samples: List[Tuple[Path, Path]] = []
    for image_dir, mask_dir in subsets:
        samples.extend(find_pairs(image_dir, mask_dir))
    return samples


def split_samples(
    samples: List[Tuple[Path, Path]],
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> Dict[str, List[Tuple[Path, Path]]]:
    split_sum = train_split + val_split + test_split
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(
            "train_split + val_split + test_split must equal 1.0. "
            f"Got {split_sum:.4f}."
        )

    shuffled = samples[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_count = round(len(shuffled) * val_split)
    test_count = round(len(shuffled) * test_split)
    train_count = len(shuffled) - val_count - test_count

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:train_count + val_count],
        "test": shuffled[train_count + val_count:],
    }


class PreprocessedDUTSDataset(Dataset):
    """Dataset that reads tensors or image files created by pre_processing.py."""

    def __init__(
        self,
        data_dir: str = "pre-processed",
        split: str = "train",
        image_size: int | None = None,
        augment: bool = False,
        augmentation_strength: str = "light",
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.data_dir = resolve_data_dir(data_dir)
        self.split = split
        self.augmentation_strength = validate_augmentation_strength(augmentation_strength)
        self.augment = (
            augment
            and split == "train"
            and self.augmentation_strength != "none"
        )

        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Preprocessed manifest not found at {manifest_path}. "
                "Run: python pre_processing.py --raw_data_dir data --output_dir pre-processed"
            )

        with open(manifest_path, "r", encoding="utf-8") as file:
            self.manifest: Dict = json.load(file)

        self.image_size = int(self.manifest["image_size"])
        if image_size is not None and int(image_size) != self.image_size:
            raise ValueError(
                f"Requested image_size={image_size}, but preprocessed data was "
                f"created with image_size={self.image_size}. Run pre_processing.py "
                "again with the requested size or pass the matching image_size."
            )

        self.samples = self.manifest["splits"].get(split, [])
        if not self.samples:
            raise RuntimeError(f"No samples found for split='{split}' in {manifest_path}.")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_tensor(path: Path) -> torch.Tensor:
        return torch.load(path, map_location="cpu")

    @staticmethod
    def _load_image_tensor(path: Path, *, is_mask: bool) -> torch.Tensor:
        image = Image.open(path).convert("L" if is_mask else "RGB")
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        if is_mask:
            tensor = torch.from_numpy(image_array).unsqueeze(0)
            return torch.clamp(tensor, 0.0, 1.0)
        return torch.from_numpy(image_array).permute(2, 0, 1)

    def _apply_train_augmentations(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        config = AUGMENTATION_STRENGTHS[self.augmentation_strength]

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        padding = max(
            int(config["min_padding"]),
            self.image_size // int(config["padding_divisor"]),
        )
        image = TF.pad(image, padding, padding_mode="reflect")
        mask = TF.pad(mask, padding, fill=0, padding_mode="constant")

        max_top = image.shape[-2] - self.image_size
        max_left = image.shape[-1] - self.image_size
        top = random.randint(0, max_top)
        left = random.randint(0, max_left)
        image = TF.crop(image, top, left, self.image_size, self.image_size)
        mask = TF.crop(mask, top, left, self.image_size, self.image_size)

        max_angle = float(config["rotation_degrees"])
        angle = random.uniform(-max_angle, max_angle)
        image = TF.rotate(
            image,
            angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        mask = TF.rotate(
            mask,
            angle,
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        )

        brightness_min, brightness_max = config["brightness"]
        brightness_factor = random.uniform(float(brightness_min), float(brightness_max))
        image = TF.adjust_brightness(image, brightness_factor)

        contrast_range = config["contrast"]
        if contrast_range is not None:
            contrast_min, contrast_max = contrast_range
            contrast_factor = random.uniform(float(contrast_min), float(contrast_max))
            image = TF.adjust_contrast(image, contrast_factor)

        return torch.clamp(image, 0.0, 1.0), torch.clamp(mask, 0.0, 1.0)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_rel = sample.get("image_tensor") or sample.get("image_path")
        mask_rel = sample.get("mask_tensor") or sample.get("mask_path")
        if image_rel is None or mask_rel is None:
            raise KeyError(
                "Preprocessed sample is missing image/mask entries. "
                "Expected 'image_tensor'/'mask_tensor' or 'image_path'/'mask_path'."
            )

        image_path = self.data_dir / image_rel
        mask_path = self.data_dir / mask_rel

        if image_path.suffix.lower() == ".pt":
            image = self._load_tensor(image_path)
        else:
            image = self._load_image_tensor(image_path, is_mask=False)

        if mask_path.suffix.lower() == ".pt":
            mask = self._load_tensor(mask_path)
        else:
            mask = self._load_image_tensor(mask_path, is_mask=True)

        if self.augment:
            image, mask = self._apply_train_augmentations(image, mask)

        return {
            "image": image.float(),
            "mask": mask.float(),
            "image_path": sample["source_image"],
            "mask_path": sample["source_mask"],
        }


class DUTSDataset(Dataset):
    """Raw DUTS reader kept for debugging or one-off experiments."""

    def __init__(
        self,
        data_dir: str = "data",
        split: str = "train",
        image_size: int = 128,
        augment: bool = False,
        train_split: float = 0.70,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
        augmentation_strength: str = "light",
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.data_dir = resolve_data_dir(data_dir)
        self.split = split
        self.image_size = image_size
        self.augmentation_strength = validate_augmentation_strength(augmentation_strength)
        self.augment = (
            augment
            and split == "train"
            and self.augmentation_strength != "none"
        )
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(
                f"No image/mask pairs found for split='{split}' in {self.data_dir}."
            )

    def _build_samples(self) -> List[Tuple[Path, Path]]:
        samples = collect_all_duts_pairs(self.data_dir)
        splits = split_samples(
            samples,
            self.train_split,
            self.val_split,
            self.test_split,
            self.seed,
        )
        return splits[self.split]

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_train_transforms(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        config = AUGMENTATION_STRENGTHS[self.augmentation_strength]
        scale_min, scale_max = config["scale_range"]
        random_size = int(
            self.image_size * random.uniform(float(scale_min), float(scale_max))
        )
        image = TF.resize(image, [random_size, random_size], InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [random_size, random_size], InterpolationMode.NEAREST)

        top = random.randint(0, random_size - self.image_size)
        left = random.randint(0, random_size - self.image_size)
        image = TF.crop(image, top, left, self.image_size, self.image_size)
        mask = TF.crop(mask, top, left, self.image_size, self.image_size)

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        max_angle = float(config["rotation_degrees"])
        angle = random.uniform(-max_angle, max_angle)
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        brightness_min, brightness_max = config["brightness"]
        brightness_factor = random.uniform(float(brightness_min), float(brightness_max))
        image = TF.adjust_brightness(image, brightness_factor)

        contrast_range = config["contrast"]
        if contrast_range is not None:
            contrast_min, contrast_max = contrast_range
            contrast_factor = random.uniform(float(contrast_min), float(contrast_max))
            image = TF.adjust_contrast(image, contrast_factor)

        return image, mask

    def _apply_eval_transforms(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        image = TF.resize(image, [self.image_size, self.image_size], InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], InterpolationMode.NEAREST)
        return image, mask

    @staticmethod
    def _to_tensor(image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)

        return image_tensor, mask_tensor

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self._apply_train_transforms(image, mask)
        else:
            image, mask = self._apply_eval_transforms(image, mask)

        image_tensor, mask_tensor = self._to_tensor(image, mask)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def create_datasets(
    data_dir: str = "pre-processed",
    image_size: int = 128,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    use_preprocessed: bool = True,
    augmentation_strength: str = "light",
):
    dataset_class = PreprocessedDUTSDataset if use_preprocessed else DUTSDataset
    common_kwargs = {"data_dir": data_dir, "image_size": image_size}

    if use_preprocessed:
        train_dataset = dataset_class(
            split="train",
            augment=True,
            augmentation_strength=augmentation_strength,
            **common_kwargs,
        )
        val_dataset = dataset_class(split="val", augment=False, **common_kwargs)
        test_dataset = dataset_class(split="test", augment=False, **common_kwargs)
    else:
        train_dataset = dataset_class(
            split="train",
            augment=True,
            augmentation_strength=augmentation_strength,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            **common_kwargs,
        )
        val_dataset = dataset_class(
            split="val",
            augment=False,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            **common_kwargs,
        )
        test_dataset = dataset_class(
            split="test",
            augment=False,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            **common_kwargs,
        )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    data_dir: str = "pre-processed",
    image_size: int = 128,
    batch_size: int = 16,
    num_workers: int = 2,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    use_preprocessed: bool = True,
    augmentation_strength: str = "light",
):
    train_dataset, val_dataset, test_dataset = create_datasets(
        data_dir=data_dir,
        image_size=image_size,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        use_preprocessed=use_preprocessed,
        augmentation_strength=augmentation_strength,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
