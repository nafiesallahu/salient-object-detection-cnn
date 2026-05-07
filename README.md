# Salient Object Detection with Scratch-Built CNNs

<!-- Badges: replace placeholders after publishing the repository/package metadata. -->
![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo%20App-ff4b4b)
![Computer Vision](https://img.shields.io/badge/Task-Salient%20Object%20Detection-6f42c1)
![License](https://img.shields.io/badge/License-TBD-lightgrey)

A complete PyTorch pipeline for **Salient Object Detection (SOD)** on the DUTS dataset. The project predicts pixel-level saliency masks that highlight the most visually important object or region in an image.

This repository is intentionally built around **scratch-designed CNN encoder-decoder models**. It does not use pretrained backbones, transfer learning, or torchvision segmentation models, making it a clear, beginner-friendly implementation for learning how image-to-mask deep learning systems are structured end to end.

## Project Highlights

- End-to-end SOD workflow covering preprocessing, training, evaluation, visualization, and interactive inference.
- Two scratch-built model options: a baseline encoder-decoder CNN and a compact UNet-style model with skip connections.
- DUTS image/mask pairing, reshuffling, and reproducible `70% / 15% / 15%` train-validation-test splitting.
- Training history saved after every epoch in both JSON and CSV formats.
- Checkpointing, resume support, validation monitoring, and early stopping.
- Evaluation with standard segmentation and saliency metrics including IoU, precision, recall, F1-score, MAE, and MSE.
- Jupyter notebook workflow preserved for academic review, demonstrations, and step-by-step experimentation.
- Streamlit demo app for uploading an image and viewing the predicted saliency mask and overlay.

## Features

- **Scratch CNN architectures**: no pretrained encoders or external segmentation models.
- **Reusable preprocessing pipeline**: converts DUTS images and masks into fixed-size project-ready samples.
- **Robust data loading**: supports preprocessed image files and optional tensor outputs through the same dataset class.
- **Train-time augmentation**: horizontal flips, padded random crops, brightness variation, and small rotations.
- **Combined saliency loss**: binary cross-entropy plus soft IoU loss.
- **Device-aware execution**: automatically uses CUDA, Apple Silicon MPS, or CPU.
- **Experiment tracking**: per-epoch losses and metrics are persisted for plotting and comparison.
- **Visualization utilities**: saves input, ground-truth mask, predicted mask, and overlay examples.
- **Interactive demo**: Streamlit app for lightweight qualitative testing.

## Dataset

This project uses the **DUTS** dataset:

- `DUTS-TR`: original DUTS training folder, approximately 10,553 RGB images with binary masks.
- `DUTS-TE`: original DUTS test folder, approximately 5,019 RGB images with binary masks.
- Each RGB image is paired with a one-channel saliency mask by filename stem.
- For this project workflow, both DUTS folders are combined and reshuffled into custom `70% / 15% / 15%` splits.

Expected raw dataset layout:

```text
data/
├── DUTS-TR/
│   ├── DUTS-TR-Image/
│   └── DUTS-TR-Mask/
└── DUTS-TE/
    ├── DUTS-TE-Image/
    └── DUTS-TE-Mask/
```

## Repository Structure

```text
salient-object-detection-cnn/
├── data/
│   ├── DUTS-TR/
│   └── DUTS-TE/
├── pre-processed/
├── checkpoints/
├── outputs/
│   ├── visualizations/
│   └── metrics/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_training_baseline.ipynb
│   ├── 04_experiments.ipynb
│   └── 05_demo.ipynb
├── pre_processing.py
├── data_loader.py
├── sod_model.py
├── losses.py
├── metrics.py
├── train.py
├── evaluate.py
├── visualize.py
├── plot_training_history.py
├── demo_app.py
├── requirements.txt
└── README.md
```

> Note: large folders such as `data/`, `pre-processed/`, `checkpoints/`, and generated `outputs/` are usually excluded from public GitHub commits unless intentionally shared.

## Installation

Clone the repository, create a virtual environment, and install the project dependencies:

```bash
git clone <repository-url>
cd salient-object-detection-cnn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Notebook Workflow

The `notebooks/` directory provides a step-by-step version of the full project workflow. The notebooks are useful for learning, academic presentation, exploratory analysis, and documenting experiments.

The core implementation remains in Python scripts such as `pre_processing.py`, `data_loader.py`, `train.py`, `evaluate.py`, `visualize.py`, and `demo_app.py`. The notebooks use normal Python imports for exploration and use Python's `subprocess` library only when launching full project scripts.

Recommended order:

1. `notebooks/01_data_preprocessing.ipynb`
2. `notebooks/02_data_exploration.ipynb`
3. `notebooks/03_training_baseline.ipynb`
4. `notebooks/04_experiments.ipynb`
5. `notebooks/05_demo.ipynb`

## Preprocessing

Run preprocessing once before training:

```bash
python pre_processing.py --raw_data_dir data --output_dir pre-processed --image_size 128
```

If the raw dataset is stored one folder above the project, `--raw_data_dir data` is resolved automatically when `../data` exists.

The default preprocessing command saves resized images and masks. You can also save PyTorch tensors by adding `--output_format tensors`.

Generated layout:

```text
pre-processed/
├── manifest.json
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Preprocessing performs the expensive fixed work once:

- matches images and masks by filename stem;
- combines `DUTS-TR` and `DUTS-TE`;
- reshuffles all paired samples into `70% train`, `15% validation`, and `15% test`;
- resizes images and masks to `128x128` by default;
- normalizes image and mask pixel values to `[0, 1]` during loading;
- writes a `manifest.json` file with split metadata and sample paths.

Training still applies lightweight random augmentation to the preprocessed training split.

## Training

Train the baseline encoder-decoder model:

```bash
python train.py --data_dir pre-processed --image_size 128 --batch_size 16 --epochs 25 --lr 1e-3 --model_type baseline
```

Train the small UNet-style model with skip connections:

```bash
python train.py --data_dir pre-processed --image_size 128 --batch_size 16 --epochs 25 --lr 1e-3 --model_type unet_small
```

The trainer automatically selects the best available device in this order: CUDA, Apple Silicon MPS, then CPU.

Training includes:

- preprocessed data loading instead of raw image resizing every epoch;
- train-time augmentation with horizontal flip, padded random crop, brightness adjustment, and small rotation;
- combined loss: `BCE + 0.5 * IoU loss`;
- Adam optimizer;
- validation after every epoch;
- epoch-level train and validation metrics;
- JSON and CSV training history persistence after every epoch;
- best-model checkpointing by validation loss;
- latest-checkpoint saving after every epoch;
- early stopping with patience `5` by default.

## Resume Training

Resume from the latest checkpoint for the selected model type:

```bash
python train.py --data_dir pre-processed --image_size 128 --batch_size 16 --epochs 25 --lr 1e-3 --model_type unet_small --resume
```

Checkpoints are saved in:

```text
checkpoints/best_model.pth
checkpoints/best_model_<model_type>.pth
checkpoints/latest_<model_type>.pth
checkpoints/latest_checkpoint.pth
```

Training history is saved after every epoch in:

```text
outputs/metrics/training_history_<model_type>.json
outputs/metrics/training_history_<model_type>.csv
```

Each checkpoint stores `model_state_dict`, `optimizer_state_dict`, `current_epoch`, `best_val_loss`, the full `history` dictionary, model metadata, and the current early-stopping counter. Resuming with `--resume` loads this state and continues at `current_epoch + 1`.

## Evaluation

Evaluate the best checkpoint on the preprocessed test split:

```bash
python evaluate.py --data_dir pre-processed --image_size 128 --model_type unet_small
```

Metrics are printed to the terminal and saved to:

```text
outputs/metrics/test_metrics.json
```

## Training Curves

Plot saved loss and metric history:

```bash
python plot_training_history.py --model_type unet_small
```

Plots are saved to:

```text
outputs/metrics/plots/
```

## Visualizations

Create qualitative prediction examples with the input image, ground-truth mask, predicted mask, and overlay:

```bash
python visualize.py --data_dir pre-processed --image_size 128 --model_type unet_small --num_samples 10
```

Visualizations are saved to:

```text
outputs/visualizations/
```

## Streamlit Demo

Run the interactive demo:

```bash
streamlit run demo_app.py
```

The app allows you to upload an image and inspect:

- the original image;
- the predicted saliency mask;
- the saliency overlay;
- inference time.

## Models

### `baseline`

The baseline model is a simple scratch-built encoder-decoder CNN:

- RGB input with shape `[3, 128, 128]`;
- four convolutional encoder stages with BatchNorm, ReLU, and MaxPool;
- four ConvTranspose decoder stages;
- one-channel sigmoid saliency output with the same spatial size as the input.

### `unet_small`

The `unet_small` model is a compact UNet-style architecture implemented from scratch:

- custom ConvBlocks with two convolution layers per block;
- BatchNorm and ReLU activations;
- encoder-to-decoder skip connections;
- bottleneck stage with optional dropout in deeper layers;
- one-channel sigmoid saliency output.

## Metrics

Binary metrics use a threshold of `0.5`.

- **IoU**: overlap between predicted salient pixels and ground-truth salient pixels.
- **Precision**: fraction of predicted salient pixels that are correct.
- **Recall**: fraction of ground-truth salient pixels recovered by the prediction.
- **F1-score**: harmonic mean of precision and recall.
- **MAE**: mean absolute error between the predicted probability mask and ground truth.
- **MSE**: mean squared error between the predicted probability mask and ground truth.

## Outputs

The project writes generated artifacts to:

```text
checkpoints/                 # model checkpoints
outputs/metrics/             # metrics, histories, and plots
outputs/visualizations/      # qualitative prediction images
pre-processed/               # processed train/val/test data
```

## Future Improvements

- Add configurable image sizes and experiment presets for faster comparison.
- Add richer augmentation policies for saliency-specific robustness.
- Add additional scratch-built architectures for model comparison.
- Add automated tests for data loading, metric computation, and checkpoint resume behavior.
- Add clearer experiment reports that compare model variants using saved metrics and visual outputs.
- Add a finalized open-source license file before publishing.

## Conclusion

This project demonstrates a complete, reproducible computer vision workflow for Salient Object Detection using only scratch-built CNN models. It is designed to be approachable for beginners while still showing the structure expected in a professional deep learning repository: clean preprocessing, modular training code, checkpointing, evaluation, visualization, notebooks, and an interactive demo.
