import torch

from sod_model import MODEL_TYPES, get_model


def test_all_models_preserve_saliency_mask_shape():
    inputs = torch.randn(2, 3, 128, 128)

    for model_type in MODEL_TYPES:
        model = get_model(model_type, dropout=0.2)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)

        assert outputs.shape == (2, 1, 128, 128)
        assert torch.all(outputs >= 0)
        assert torch.all(outputs <= 1)

