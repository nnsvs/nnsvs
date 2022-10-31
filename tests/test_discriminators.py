import pytest
import torch
from nnsvs.discriminators import Conv2dD
from nnsvs.util import init_seed


def _test_model_impl(model, in_dim):
    B = 4
    T = 100
    init_seed(B * T)
    x = torch.rand(B, T, in_dim)
    lengths = torch.Tensor([T] * B).long()

    # warmup forward pass
    with torch.no_grad():
        y = model(x, lengths=lengths)
    # should contain multiple outputs
    assert isinstance(y, list)
    # should contain intermediate outputs to compute feature matching loss
    assert isinstance(y[-1], list)


@pytest.mark.parametrize("padding_mode", ["reflect", "zeros"])
def test_conv2d(padding_mode):
    params = {
        "in_dim": 60,
        "channels": 8,
        "kernel_size": (3, 3),
        "padding": (0, 0),
        "padding_mode": padding_mode,
        "last_sigmoid": False,
        "init_type": "none",
    }
    model = Conv2dD(**params)
    _test_model_impl(model, params["in_dim"])
