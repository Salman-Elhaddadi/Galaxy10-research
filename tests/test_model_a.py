import torch

from galaxy10.model.CNN import ModelA


def test_model_a_parameter_count() -> None:
    model = ModelA()
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == 391_466, f"expected 391466 parameters, got {count}"


def test_model_a_forward_shape() -> None:
    model = ModelA()
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 10), y.shape
