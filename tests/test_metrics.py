import torch

from metrics import compute_metrics


def test_compute_metrics_perfect_prediction():
    targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    predictions = targets.clone()

    metrics = compute_metrics(predictions, targets)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0


def test_compute_metrics_counts_false_positives_and_false_negatives():
    targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    predictions = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    metrics = compute_metrics(predictions, targets)

    assert round(metrics["precision"], 4) == 0.5
    assert round(metrics["recall"], 4) == 0.5
    assert round(metrics["f1_score"], 4) == 0.5
    assert round(metrics["iou"], 4) == 0.3333

