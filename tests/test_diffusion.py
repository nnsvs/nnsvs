import pytest
import torch
from nnsvs.base import PredictionType
from nnsvs.diffsinger.denoiser import DiffNet
from nnsvs.diffsinger.diffusion import GaussianDiffusion
from nnsvs.diffsinger.fs2 import FFTBlocks, FFTBlocksEncoder
from nnsvs.model import LSTMEncoder
from nnsvs.util import init_seed

from .util import _test_model_impl


@pytest.mark.parametrize("use_pos_embed", [False, True])
def test_fftblocks(use_pos_embed):
    model = FFTBlocks(16, 2, use_pos_embed=use_pos_embed)
    B = 4
    T = 100
    init_seed(B * T)
    x = torch.rand(B, T, 16)
    lengths = torch.Tensor([T] * B).long()
    y_hat = model(x, lengths)
    assert x.shape[1] == y_hat.shape[1]


@pytest.mark.parametrize("reduction_factor", [1, 4])
@pytest.mark.parametrize("downsample_by_conv", [False, True])
def test_fs2(reduction_factor, downsample_by_conv):
    params = {
        "in_dim": 86,
        "hidden_dim": 16,
        "out_dim": 16,
        "num_layers": 2,
        "ffn_kernel_size": 3,
        "reduction_factor": reduction_factor,
        "downsample_by_conv": downsample_by_conv,
    }
    model = FFTBlocksEncoder(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])


def test_denoiser():
    model = DiffNet(
        in_dim=80,
        encoder_hidden_dim=12,
        residual_layers=2,
        residual_channels=4,
        dilation_cycle_length=4,
    )
    x = torch.rand(2, 1, 80, 100)
    cond = torch.rand(2, 12, 100)
    step = torch.randint(0, 100, (2,))
    y = model(x, step, cond)
    assert x.shape == y.shape


@pytest.mark.parametrize("pndm_speedup", [None])
def test_gaussian_diffusion(pndm_speedup):
    encoder = LSTMEncoder(
        in_dim=60,
        hidden_dim=2,
        out_dim=16,
        num_layers=2,
        dropout=0.5,
        init_type="none",
    )
    params = {
        "in_dim": 60,
        "out_dim": 80,
        "denoise_fn": DiffNet(
            in_dim=80,
            encoder_hidden_dim=16,
            residual_layers=2,
            residual_channels=4,
            dilation_cycle_length=4,
        ),
        "K_step": 100,
        "betas": None,
        "pndm_speedup": pndm_speedup,
    }
    model = GaussianDiffusion(**params)
    B = 4
    T = 100
    init_seed(B * T)
    x = torch.rand(B, T, model.in_dim)
    y = torch.rand(B, T, model.out_dim)
    lengths = torch.Tensor([T] * B).long()
    encoder_outs = encoder(x, lengths)
    noise, x_recon = model(encoder_outs, lengths, y)
    assert noise.shape == y.shape
    assert x_recon.shape == y.shape

    y_hat = model.inference(encoder_outs, lengths)
    assert y_hat.shape == y.shape


@pytest.mark.parametrize("pndm_speedup", [None])
def test_gaussian_diffusion_with_encoder(pndm_speedup):
    params = {
        "in_dim": 60,
        "out_dim": 80,
        "denoise_fn": DiffNet(
            in_dim=80,
            encoder_hidden_dim=16,
            residual_layers=2,
            residual_channels=4,
            dilation_cycle_length=4,
        ),
        "encoder": LSTMEncoder(
            in_dim=60,
            hidden_dim=2,
            out_dim=16,
            num_layers=2,
            dropout=0.5,
            init_type="none",
        ),
        "K_step": 100,
        "betas": None,
        "pndm_speedup": pndm_speedup,
    }
    model = GaussianDiffusion(**params)
    B = 4
    T = 100
    init_seed(B * T)
    x = torch.rand(B, T, model.encoder.in_dim)
    y = torch.rand(B, T, model.out_dim)
    lengths = torch.Tensor([T] * B).long()
    noise, x_recon = model(x, lengths, y)
    assert noise.shape == y.shape
    assert x_recon.shape == y.shape

    y_hat = model.inference(x, lengths)
    assert y_hat.shape == y.shape
