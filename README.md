# Salient Object Detection with Scratch-Built CNNs

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo%20App-ff4b4b)
![Computer Vision](https://img.shields.io/badge/Task-Salient%20Object%20Detection-6f42c1)

A complete PyTorch pipeline for **Salient Object Detection (SOD)** on the DUTS dataset. The project predicts pixel-level saliency masks that highlight the most visually important object or region in an image.

This repository is intentionally built around **scratch-designed CNN encoder-decoder models**. It does not use pretrained backbones, transfer learning, or torchvision segmentation models, making it a clear, beginner-friendly implementation for learning how image-to-mask deep learning systems are structured end to end.

## Project Highlights

- End-to-end SOD workflow covering preprocessing, training, evaluation, visualization, and interactive inference.
- Three scratch-built model options: a baseline encoder-decoder CNN, a no-BatchNorm baseline ablation, and a compact UNet-style model with skip connections.
- DUTS image/mask pairing with an official-protocol split mode that keeps `DUTS-TE` held out for testing.
- Training history saved after every epoch in both JSON and CSV formats.
- Checkpointing, resume support, validation monitoring, and early stopping.
- Evaluation with standard segmentation and saliency metrics including IoU, precision, recall, F1-score, MAE, and MSE, plus optional per-image metric CSV files.
- Jupyter notebook workflow preserved for academic review, demonstrations, and step-by-step experimentation.
- Streamlit demo app for uploading an image and viewing the predicted saliency mask and overlay.
- Markdown project report in `REPORT.md`.

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
- Recommended final workflow: keep `DUTS-TE` as the held-out test set and split only `DUTS-TR` into training and validation.
- Optional custom workflow: reshuffle `DUTS-TR` and `DUTS-TE` together into `70% / 15% / 15%` splits for project-only experiments.

> Current saved local metrics in `outputs/metrics/` were generated from the custom project split. For official DUTS reporting, rerun preprocessing with `--split_strategy official`, retrain, and evaluate on the official `DUTS-TE` test split.

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
│   │   ├── baseline/
│   │   ├── baseline_no_bn/
│   │   └── improved/
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
├── compare_experiments.py
├── REPORT.md
├── requirements.txt
├── tests/
└── README.md
```

> Note: large folders such as `data/`, `pre-processed/`, `checkpoints/`, and generated `outputs/` are usually excluded from public GitHub commits unless intentionally shared.

## Installation

Create a virtual environment from the project root and install the project dependencies:

```bash
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
python3 pre_processing.py --raw_data_dir data --output_dir pre-processed --image_size 128 --split_strategy official
```

If the raw dataset is stored one folder above the project, `--raw_data_dir data` is resolved automatically when `../data` exists.

The default preprocessing command saves resized images and masks. You can also save PyTorch tensors by adding `--output_format tensors`.

The recommended `official` strategy preserves the DUTS protocol:

- `DUTS-TR` is shuffled and split into train/validation using `--val_split` (`0.15` by default).
- `DUTS-TE` is kept as the held-out test set.

For project-only experiments that reproduce the saved local metrics, use the custom split:

```bash
python3 pre_processing.py --raw_data_dir data --output_dir pre-processed --image_size 128 --split_strategy custom --train_split 0.70 --val_split 0.15 --test_split 0.15
```

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
- supports official `DUTS-TE` testing or custom reshuffled splitting;
- resizes images and masks to `128x128` by default;
- normalizes image and mask pixel values to `[0, 1]` during loading;
- writes a `manifest.json` file with split metadata and sample paths.

Training applies configurable random augmentation to the preprocessed training split.

## Training

Train the baseline encoder-decoder model:

```bash
python3 train.py --data_dir pre-processed --image_size 128 --batch_size 16 --epochs 25 --lr 1e-3 --model_type baseline --experiment_name baseline --augmentation_strength light --bce_weight 1.0 --iou_weight 0.5
```

Train the no-BatchNorm baseline ablation for Experiment 2:

```bash
python3 train.py --data_dir pre-processed --image_size 128 --batch_size 16 --epochs 25 --lr 1e-3 --model_type baseline_no_bn --experiment_name baseline_no_bn --augmentation_strength light --bce_weight 1.0 --iou_weight 0.5
```

Train an improved small UNet-style model. This run changes more than two things after the baseline: deeper convolution blocks, skip connections, dropout, stronger augmentation, lower learning rate, and heavier IoU loss weighting.

```bash
python3 train.py --data_dir pre-processed --image_size 128 --batch_size 16 --epochs 25 --lr 5e-4 --model_type unet_small --experiment_name improved --dropout 0.3 --augmentation_strength strong --bce_weight 1.0 --iou_weight 0.75 --write_global_aliases
```

The trainer automatically selects the best available device in this order: CUDA, Apple Silicon MPS, then CPU.

Training includes:

- preprocessed data loading instead of raw image resizing every epoch;
- configurable train-time augmentation with `none`, `light`, or `strong` policies;
- configurable combined loss, defaulting to `BCE + 0.5 * IoU loss`;
- Adam optimizer;
- validation after every epoch;
- epoch-level train and validation metrics;
- JSON and CSV training history persistence after every epoch;
- run-specific best-model checkpointing by validation loss;
- latest-checkpoint saving after every epoch;
- early stopping with patience `5` by default.
- seeded DataLoader shuffling and worker initialization for reproducibility.

By default, training writes run-specific checkpoints such as `checkpoints/best_model_improved.pth`. The generic aliases `checkpoints/best_model.pth` and `checkpoints/latest_checkpoint.pth` are updated only when `--write_global_aliases` is passed, which prevents ablation runs from accidentally replacing the demo/default checkpoint.

## Resume Training

Training automatically resumes from the latest checkpoint for the selected run name when one exists:

```bash
python3 train.py --data_dir pre-processed --image_size 128 --batch_size 16 --epochs 25 --lr 5e-4 --model_type unet_small --experiment_name improved --dropout 0.3 --augmentation_strength strong --bce_weight 1.0 --iou_weight 0.75 --write_global_aliases
```

To ignore an existing checkpoint and start over, add `--no_resume`.

Checkpoints are saved in:

```text
checkpoints/best_model_<experiment_name_or_model_type>.pth
checkpoints/latest_<experiment_name_or_model_type>.pth
checkpoints/best_model.pth          # optional global alias
checkpoints/latest_checkpoint.pth   # optional global alias
```

Training history is saved after every epoch in:

```text
outputs/metrics/training_history_<experiment_name_or_model_type>.json
outputs/metrics/training_history_<experiment_name_or_model_type>.csv
```

Each checkpoint stores `model_state_dict`, `optimizer_state_dict`, `current_epoch`, `best_val_loss`, the full `history` dictionary, model metadata, the training configuration, and the current early-stopping counter. Automatic resume loads this state and continues at `current_epoch + 1`.

## Evaluation

Evaluate the best checkpoint on the preprocessed test split:

```bash
python3 evaluate.py --data_dir pre-processed --image_size 128 --checkpoint checkpoints/best_model_baseline.pth --experiment_name baseline
python3 evaluate.py --data_dir pre-processed --image_size 128 --checkpoint checkpoints/best_model_improved.pth --experiment_name improved
```

`evaluate.py` reads `model_type` from checkpoint metadata when available. If you have not retrained with `--experiment_name improved` yet, evaluate the existing UNet checkpoint with `--checkpoint checkpoints/best_model_unet_small.pth --experiment_name improved`.

Use a different binary threshold when needed:

```bash
python3 evaluate.py --data_dir pre-processed --checkpoint checkpoints/best_model_improved.pth --experiment_name improved --threshold 0.45
```

Metrics are printed to the terminal and saved to:

```text
outputs/metrics/test_metrics.json
outputs/metrics/test_metrics_<experiment_name>.json
outputs/metrics/per_image_metrics_<experiment_name>.csv
```

## Training Curves

Plot saved loss and metric history:

```bash
python3 plot_training_history.py --experiment_name baseline
python3 plot_training_history.py --experiment_name improved
```

If you are using the existing UNet checkpoint, plot its saved history with `python3 plot_training_history.py --history outputs/metrics/training_history_unet_small.json --prefix improved`.

Plots are saved to:

```text
outputs/metrics/plots/
```

## Visualizations

Create qualitative prediction examples with the input image, ground-truth mask, predicted mask, and overlay:

```bash
python3 visualize.py --data_dir pre-processed --image_size 128 --checkpoint checkpoints/best_model_baseline.pth --experiment_name baseline --num_samples 10
python3 visualize.py --data_dir pre-processed --image_size 128 --checkpoint checkpoints/best_model_improved.pth --experiment_name improved --num_samples 10
```

If you are using the existing UNet checkpoint, replace `checkpoints/best_model_improved.pth` with `checkpoints/best_model_unet_small.pth`.

Give custom experiments their own folder with `--experiment_name`:

```bash
python3 visualize.py --data_dir pre-processed --image_size 128 --checkpoint checkpoints/best_model_baseline.pth --experiment_name baseline
```

Visualizations are saved to:

```text
outputs/visualizations/<experiment_name>/
```

## Experiment Comparison

Build the final results table and visual comparison after evaluating the saved runs:

```bash
python3 compare_experiments.py
```

The detailed comparison report is saved to:

```text
outputs/metrics/experiment_comparison.md
outputs/metrics/plots/experiment_metrics_comparison.png
outputs/visualizations/experiment_comparison_contact_sheet.png
```

Current saved test comparison, using the project custom split:

| Experiment | IoU | Δ IoU | Precision | Recall | F1-score | Δ F1 | MAE | Δ MAE | MSE | Δ MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.5805 | +0.0000 | 0.7262 | 0.7431 | 0.7346 | +0.0000 | 0.1555 | +0.0000 | 0.0879 | +0.0000 |
| Baseline no BatchNorm | 0.5393 | -0.0411 | 0.6760 | 0.7273 | 0.7007 | -0.0338 | 0.1794 | +0.0239 | 0.0997 | +0.0117 |
| Strong augmentation | 0.5795 | -0.0010 | 0.6824 | 0.7935 | 0.7337 | -0.0008 | 0.1707 | +0.0152 | 0.0923 | +0.0044 |
| IoU-heavy loss | 0.5742 | -0.0062 | 0.6822 | 0.7840 | 0.7295 | -0.0050 | 0.1573 | +0.0018 | 0.0978 | +0.0098 |
| Improved | 0.7872 | +0.2067 | 0.8657 | 0.8967 | 0.8809 | +0.1464 | 0.0664 | -0.0891 | 0.0428 | -0.0451 |

The detailed report in `REPORT.md` embeds the metric chart, all-experiment contact sheet, and representative visual examples for each experiment.

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

When run-specific checkpoints are available, the sidebar lets you choose the experiment checkpoint to use, including baseline, BatchNorm ablation, strong augmentation, IoU-heavy loss, and improved runs. The app defaults to the improved checkpoint and falls back to `checkpoints/best_model.pth` only when no run-specific best checkpoint exists.

## Models

### `baseline`

The baseline model is a simple scratch-built encoder-decoder CNN:

- approximately `563,713` trainable parameters;
- RGB input with shape `[3, 128, 128]`;
- four convolutional encoder stages with BatchNorm, ReLU, and MaxPool;
- four ConvTranspose decoder stages;
- one-channel sigmoid saliency output with the same spatial size as the input.

### `baseline_no_bn`

The `baseline_no_bn` model keeps the same baseline encoder-decoder layout but removes every BatchNorm layer. It has approximately `562,753` trainable parameters and is used for the BatchNorm ablation.

### `unet_small`

The `unet_small` model is a compact UNet-style architecture implemented from scratch:

- approximately `7,763,041` trainable parameters;
- custom ConvBlocks with two convolution layers per block;
- BatchNorm and ReLU activations;
- encoder-to-decoder skip connections;
- bottleneck and decoder stages with configurable dropout in deeper layers;
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
outputs/visualizations/      # qualitative prediction images grouped by experiment
pre-processed/               # processed train/val/test data
```

## Tests

Run the lightweight regression tests with:

```bash
python3 -m pytest
```

The current tests cover model output shapes, saliency metric arithmetic, combined loss validation, and preprocessing split reproducibility.

## Report

The Markdown report in `REPORT.md` contains the project objective, dataset protocol, architecture summary, training setup, experiment table, metric analysis, visualization evidence, limitations, and next steps. PDF and slide exports are intentionally not included yet.

## Future Improvements

- Add experiment presets for faster comparison.
- Add additional scratch-built architectures for model comparison.
- Add tests for data loading with temporary image/mask folders and checkpoint resume behavior.
- Add official DUTS-TE retraining artifacts after preprocessing with `--split_strategy official`.
- Add a finalized open-source license file before publishing.

## Conclusion

This project demonstrates a complete, reproducible computer vision workflow for Salient Object Detection using only scratch-built CNN models. It is designed to be approachable for beginners while still showing the structure expected in a professional deep learning repository: clean preprocessing, modular training code, checkpointing, evaluation, visualization, notebooks, and an interactive demo.
