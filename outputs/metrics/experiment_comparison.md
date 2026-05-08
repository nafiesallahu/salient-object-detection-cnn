# Experiment Comparison Results

Current saved metrics use the project custom split and threshold `0.5`. Positive deltas are better for IoU, precision, recall, and F1-score; negative deltas are better for MAE and MSE.

| Experiment | Configuration | IoU | Δ IoU | Precision | Recall | F1-score | Δ F1 | MAE | Δ MAE | MSE | Δ MSE | Visual example |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | Baseline encoder-decoder CNN, light augmentation, BCE + 0.5 IoU loss. | 0.5805 | +0.0000 | 0.7262 | 0.7431 | 0.7346 | +0.0000 | 0.1555 | +0.0000 | 0.0879 | +0.0000 | [outputs/visualizations/baseline/visualization_01_ILSVRC2012_test_00036927.png](outputs/visualizations/baseline/visualization_01_ILSVRC2012_test_00036927.png) |
| baseline_no_bn | Baseline encoder-decoder with BatchNorm removed. | 0.5393 | -0.0411 | 0.6760 | 0.7273 | 0.7007 | -0.0338 | 0.1794 | +0.0239 | 0.0997 | +0.0117 | [outputs/visualizations/baseline_no_bn/visualization_01_ILSVRC2012_test_00036927.png](outputs/visualizations/baseline_no_bn/visualization_01_ILSVRC2012_test_00036927.png) |
| strong_aug | Baseline encoder-decoder with stronger train-time augmentation. | 0.5795 | -0.0010 | 0.6824 | 0.7935 | 0.7337 | -0.0008 | 0.1707 | +0.0152 | 0.0923 | +0.0044 | [outputs/visualizations/strong_aug/visualization_01_ILSVRC2012_test_00036927.png](outputs/visualizations/strong_aug/visualization_01_ILSVRC2012_test_00036927.png) |
| iou_heavy | Baseline encoder-decoder with IoU loss weight increased to 1.0. | 0.5742 | -0.0062 | 0.6822 | 0.7840 | 0.7295 | -0.0050 | 0.1573 | +0.0018 | 0.0978 | +0.0098 | [outputs/visualizations/iou_heavy/visualization_01_ILSVRC2012_test_00036927.png](outputs/visualizations/iou_heavy/visualization_01_ILSVRC2012_test_00036927.png) |
| improved | Small UNet with skip connections, dropout, strong augmentation, lower LR, and heavier IoU loss. | 0.7872 | +0.2067 | 0.8657 | 0.8967 | 0.8809 | +0.1464 | 0.0664 | -0.0891 | 0.0428 | -0.0451 | [outputs/visualizations/improved/visualization_01_ILSVRC2012_test_00036927.png](outputs/visualizations/improved/visualization_01_ILSVRC2012_test_00036927.png) |

## Visual Outputs

### Metric Comparison Chart

![Metric comparison chart](outputs/metrics/plots/experiment_metrics_comparison.png)

### Qualitative Contact Sheet

![Qualitative contact sheet](outputs/visualizations/experiment_comparison_contact_sheet.png)


## Per-Experiment Visual Examples

### baseline

![baseline visual example](outputs/visualizations/baseline/visualization_01_ILSVRC2012_test_00036927.png)

### baseline_no_bn

![baseline_no_bn visual example](outputs/visualizations/baseline_no_bn/visualization_01_ILSVRC2012_test_00036927.png)

### strong_aug

![strong_aug visual example](outputs/visualizations/strong_aug/visualization_01_ILSVRC2012_test_00036927.png)

### iou_heavy

![iou_heavy visual example](outputs/visualizations/iou_heavy/visualization_01_ILSVRC2012_test_00036927.png)

### improved

![improved visual example](outputs/visualizations/improved/visualization_01_ILSVRC2012_test_00036927.png)

## Interpretation

- `baseline_no_bn` is worse than the baseline, showing BatchNorm is helpful for the scratch encoder-decoder.
- `strong_aug` improves recall but reduces precision, so it predicts larger salient regions and creates more false positives.
- `iou_heavy` does not improve the baseline, suggesting the plain architecture is the main bottleneck.
- `improved` is the strongest run, with much better IoU/F1 and substantially lower MAE/MSE.

