from pathlib import Path
import time

import numpy as np
from PIL import Image
import streamlit as st
import torch

from device_utils import get_available_device
from sod_model import get_model


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
PREFERRED_CHECKPOINT_STEMS = (
    "best_model_improved",
    "best_model_baseline",
    "best_model_baseline_no_bn",
    "best_model_strong_aug",
    "best_model_iou_heavy",
)
EXPERIMENT_LABELS = {
    "improved": "Improved",
    "baseline": "Baseline",
    "baseline_no_bn": "Baseline no BatchNorm",
    "strong_aug": "Strong augmentation",
    "iou_heavy": "IoU-heavy loss",
    "best_model": "Best model alias",
}
EXPERIMENT_MODEL_TYPES = {
    "improved": "unet_small",
    "baseline": "baseline",
    "baseline_no_bn": "baseline_no_bn",
    "strong_aug": "baseline",
    "iou_heavy": "baseline",
}
NON_EXPERIMENT_CHECKPOINTS = {"unet_small"}


def get_experiment_name(checkpoint_path: Path) -> str:
    stem = checkpoint_path.stem
    if stem.startswith("best_model_"):
        return stem.removeprefix("best_model_")
    if stem == "best_model":
        return "best_model"
    return stem


def format_checkpoint_label(checkpoint_path: Path) -> str:
    experiment_name = get_experiment_name(checkpoint_path)
    return EXPERIMENT_LABELS.get(
        experiment_name,
        experiment_name.replace("_", " ").title(),
    )


def infer_model_type_from_checkpoint(checkpoint_path: Path) -> str:
    experiment_name = get_experiment_name(checkpoint_path)
    return EXPERIMENT_MODEL_TYPES.get(experiment_name, "unet_small")


def get_available_checkpoints(checkpoints_dir: Path = CHECKPOINTS_DIR) -> list[Path]:
    checkpoints: list[Path] = []
    seen: set[Path] = set()

    for stem in PREFERRED_CHECKPOINT_STEMS:
        path = checkpoints_dir / f"{stem}.pth"
        if path.exists():
            checkpoints.append(path)
            seen.add(path)

    if checkpoints_dir.exists():
        for path in sorted(checkpoints_dir.glob("best_model_*.pth")):
            experiment_name = get_experiment_name(path)
            if experiment_name not in NON_EXPERIMENT_CHECKPOINTS and path not in seen:
                checkpoints.append(path)
                seen.add(path)

        alias_path = checkpoints_dir / "best_model.pth"
        if not checkpoints and alias_path.exists():
            checkpoints.append(alias_path)

    return checkpoints


def get_default_checkpoint_path() -> Path:
    checkpoints = get_available_checkpoints()
    if checkpoints:
        return checkpoints[0]
    return CHECKPOINTS_DIR / "best_model_improved.pth"


def preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    return image_tensor


def make_overlay(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    resized_image = image.convert("RGB").resize(mask.shape[::-1], Image.BILINEAR)
    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    heatmap = np.zeros_like(image_array)
    heatmap[..., 0] = mask
    overlay = 0.65 * image_array + 0.35 * heatmap
    return np.clip(overlay, 0.0, 1.0)


@st.cache_resource
def load_model(checkpoint_path: str):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = get_available_device()
    checkpoint = torch.load(path, map_location=device)
    model_type = checkpoint.get("model_type", infer_model_type_from_checkpoint(path))

    model = get_model(model_type).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_type, device


def main() -> None:
    st.set_page_config(page_title="Salient Object Detection Demo", layout="wide")
    st.title("Salient Object Detection")

    image_size = st.sidebar.number_input("Image size", min_value=64, max_value=512, value=128, step=32)
    available_checkpoints = get_available_checkpoints()
    if available_checkpoints:
        checkpoint_options = {
            format_checkpoint_label(path): path
            for path in available_checkpoints
        }
        checkpoint_label = st.sidebar.selectbox(
            "Experiment checkpoint",
            list(checkpoint_options),
        )
        checkpoint_path = checkpoint_options[checkpoint_label]
        st.sidebar.caption(f"Checkpoint file: {checkpoint_path.name}")
    else:
        checkpoint_path = get_default_checkpoint_path()
        st.sidebar.warning("No checkpoint files found.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded_file is None:
        st.info("Upload an image to predict its saliency mask.")
        return

    try:
        model, model_type, device = load_model(str(checkpoint_path))
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = preprocess_image(image, int(image_size)).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        prediction = model(input_tensor)
    inference_time = time.perf_counter() - start_time

    predicted_mask = prediction[0, 0].detach().cpu().numpy()
    display_image = image.convert("RGB").resize(predicted_mask.shape[::-1], Image.BILINEAR)
    overlay = make_overlay(image, predicted_mask)

    st.caption(
        f"Checkpoint: {checkpoint_path.name} | Model: {model_type} | "
        f"Inference time: {inference_time:.4f}s"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(display_image, caption="Original image", use_container_width=True)
    with col2:
        st.image(predicted_mask, caption="Predicted saliency mask", clamp=True, use_container_width=True)
    with col3:
        st.image(overlay, caption="Overlay", clamp=True, use_container_width=True)


if __name__ == "__main__":
    main()
