import torch

from losses import combined_loss


def test_combined_loss_is_low_for_perfect_predictions():
    targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    predictions = targets.clone()

    loss = combined_loss(predictions, targets, bce_weight=1.0, iou_weight=0.5)

    assert loss.item() < 1e-5


def test_combined_loss_rejects_zero_total_weight():
    targets = torch.zeros(1, 1, 2, 2)
    predictions = torch.zeros(1, 1, 2, 2)

    try:
        combined_loss(predictions, targets, bce_weight=0.0, iou_weight=0.0)
    except ValueError as error:
        assert "At least one loss weight" in str(error)
    else:
        raise AssertionError("combined_loss should reject zero total loss weight")

