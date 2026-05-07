from pathlib import Path
import time

import numpy as np
from PIL import Image
import streamlit as st
import torch

from device_utils import get_available_device
from sod_model import get_model


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "best_model.pth"


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
def load_model(checkpoint_path: str, fallback_model_type: str):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = get_available_device()
    checkpoint = torch.load(path, map_location=device)
    model_type = checkpoint.get("model_type", fallback_model_type)

    model = get_model(model_type).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_type, device


def main() -> None:
    st.set_page_config(page_title="Salient Object Detection Demo", layout="wide")
    st.title("Salient Object Detection")

    image_size = st.sidebar.number_input("Image size", min_value=64, max_value=512, value=128, step=32)
    fallback_model_type = st.sidebar.selectbox("Fallback model type", ["baseline", "unet_small"])

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded_file is None:
        st.info("Upload an image to predict its saliency mask.")
        return

    try:
        model, model_type, device = load_model(str(CHECKPOINT_PATH), fallback_model_type)
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
    overlay = make_overlay(image, predicted_mask)

    st.caption(f"Model: {model_type} | Device: {device} | Inference time: {inference_time:.4f}s")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Original image", use_container_width=True)
    with col2:
        st.image(predicted_mask, caption="Predicted saliency mask", clamp=True, use_container_width=True)
    with col3:
        st.image(overlay, caption="Overlay", clamp=True, use_container_width=True)


if __name__ == "__main__":
    main()
