import pytest
import torch
from nnsvs.postfilters import Conv2dPostFilter, MultistreamPostFilter
from nnsvs.util import init_seed


def _test_model_impl(model, in_dim):
    B = 4
    T = 100
    init_seed(B * T)
    x = torch.rand(B, T, in_dim)
    lengths = torch.Tensor([T] * B).long()

    # warmup forward pass
    with torch.no_grad():
        y = model(x, lengths)
        y_inf = model.inference(x, lengths)

    assert y.shape == (B, T, in_dim)
    assert y.shape == y_inf.shape


@pytest.mark.parametrize("noise_type", ["bin_wise", "frame_wise"])
def test_conv2d_postfilter(noise_type):
    params = {
        "in_dim": 60,
        "channels": 8,
        "kernel_size": (3, 3),
        "padding_mode": "zeros",
        "noise_type": noise_type,
        "init_type": "none",
    }
    model = Conv2dPostFilter(**params)
    _test_model_impl(model, params["in_dim"])


@pytest.mark.parametrize("mgc_offset", [0, 1, 2])
@pytest.mark.parametrize("bap_offset", [0, 1])
def test_multistream_postfilter(mgc_offset, bap_offset):
    params = {
        "channels": 8,
        "kernel_size": (3, 3),
        "padding_mode": "zeros",
        "noise_type": "frame_wise",
        "init_type": "none",
    }
    mgc_postfilter = Conv2dPostFilter(**{**params, "in_dim": 60 - mgc_offset})
    bap_postfilter = Conv2dPostFilter(**{**params, "in_dim": 5 - bap_offset})

    # (mgc, lf0, vuv, bap)
    stream_sizes = [60, 1, 1, 5]

    model = MultistreamPostFilter(
        mgc_postfilter=mgc_postfilter,
        bap_postfilter=bap_postfilter,
        lf0_postfilter=None,
        stream_sizes=stream_sizes,
        mgc_offset=mgc_offset,
        bap_offset=bap_offset,
    )
    _test_model_impl(model, sum(stream_sizes))
