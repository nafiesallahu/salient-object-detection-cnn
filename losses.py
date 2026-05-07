import torch
import torch.nn.functional as F


EPSILON = 1e-7


def bce_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary Cross Entropy for predictions that already passed through sigmoid."""
    return F.binary_cross_entropy(predictions, targets)


def soft_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """Soft IoU using continuous mask probabilities in [0, 1]."""
    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    return ((intersection + epsilon) / (union + epsilon)).mean()


def iou_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return 1.0 - soft_iou(predictions, targets)


def combined_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return bce_loss(predictions, targets) + 0.5 * iou_loss(predictions, targets)
