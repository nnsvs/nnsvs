from pathlib import Path

import hydra
import nnsvs.bin.train
import nnsvs.bin.train_acoustic
import nnsvs.bin.train_postfilter
import pytest
import torch
from nnsvs.base import PredictionType
from nnsvs.util import init_seed
from omegaconf import OmegaConf

RECIPE_DIR = Path(__file__).parent.parent / "recipes"


def _test_model_impl(model, in_dim, out_dim):
    B = 4
    T = 100
    init_seed(B * T)
    x = torch.rand(B, T, in_dim)
    y = torch.rand(B, T, out_dim)
    lengths = torch.Tensor([T] * B).long()

    # warmup forward pass
    with torch.no_grad():
        outs = model(x, lengths, y)
        if model.has_residual_lf0_prediction():
            y, lf0_residual = outs
        else:
            y, lf0_residual = outs, None
        y_inf = model.inference(x, lengths)

    # Hybrid (MDN + non-MDN)
    if model.prediction_type() == PredictionType.MULTISTREAM_HYBRID:
        # TODO
        pass
    # MDN case
    elif model.prediction_type() == PredictionType.PROBABILISTIC:
        log_pi, log_sigma, mu = y
        num_gaussian = log_pi.shape[2]
        assert mu.shape == (B, T, num_gaussian, out_dim)
        assert log_sigma.shape == (B, T, num_gaussian, out_dim)
        if lf0_residual is not None:
            assert lf0_residual.shape == (B, T, num_gaussian)
        # NOTE: infernece output shouldn't have num_gaussian axis
        mu_inf, sigma_inf = y_inf
        assert mu_inf.shape == (B, T, out_dim)
        assert sigma_inf.shape == (B, T, out_dim)
    else:
        if lf0_residual is not None:
            if isinstance(lf0_residual, list):
                lf0_residual = lf0_residual[-1]
            assert lf0_residual.shape == (B, T, 1)
        # NOTE: some models have multiple outputs (e.g. Tacotron)
        if isinstance(y, list):
            y = y[-1]
        assert y.shape == (B, T, out_dim)
        assert y.shape == y_inf.shape


def _test_postfilter_impl(model, model_config):
    B = 4
    T = 100
    init_seed(B * T)

    in_dim = sum(model_config.stream_sizes)
    x = torch.rand(B, T, in_dim)
    lengths = torch.Tensor([T] * B).long()

    # warmup forward pass
    with torch.no_grad():
        y = model(x, lengths)
        y_inf = model.inference(x, lengths)

    assert x.shape == y.shape
    assert y_inf.shape == y.shape


@pytest.mark.parametrize(
    "model_config",
    (Path(nnsvs.bin.train.__file__).parent / "conf" / "train" / "model").glob("*.yaml"),
)
def test_model_config(model_config):
    model_config = OmegaConf.load(model_config)
    model = hydra.utils.instantiate(model_config.netG)
    _test_model_impl(model, model_config.netG.in_dim, model_config.netG.out_dim)


@pytest.mark.parametrize(
    "model_config",
    (
        Path(nnsvs.bin.train_acoustic.__file__).parent
        / "conf"
        / "train_acoustic"
        / "model"
    ).glob("*.yaml"),
)
def test_acoustic_model_config(model_config):
    print(model_config)
    model_config = OmegaConf.load(model_config)

    # Dummy
    model_config.netG.in_lf0_idx = 10
    model_config.netG.in_lf0_min = 5.3936276
    model_config.netG.in_lf0_max = 6.491111
    model_config.netG.out_lf0_idx = 60
    model_config.netG.out_lf0_mean = 5.953093881972361
    model_config.netG.out_lf0_scale = 0.23435173188961034

    model = hydra.utils.instantiate(model_config.netG)

    _test_model_impl(model, model_config.netG.in_dim, model_config.netG.out_dim)


@pytest.mark.parametrize(
    "model_config",
    (
        Path(nnsvs.bin.train_postfilter.__file__).parent
        / "conf"
        / "train_postfilter"
        / "model"
    ).glob("*.yaml"),
)
def test_postfilter_model_config(model_config):
    model_config = OmegaConf.load(model_config)
    if "stream_sizes" in model_config.netG:
        model_config.netG.stream_sizes = model_config.stream_sizes
    # Post-filter config should have netD
    hydra.utils.instantiate(model_config.netD)
    model = hydra.utils.instantiate(model_config.netG)
    _test_postfilter_impl(model, model_config)


@pytest.mark.parametrize(
    "model_config", RECIPE_DIR.glob("**/_common/conf/**/train/timelag/model/*.yaml")
)
def test_timelag_model_config_recipes(model_config):
    model_config = OmegaConf.load(model_config)
    model = hydra.utils.instantiate(model_config.netG)
    _test_model_impl(model, model_config.netG.in_dim, model_config.netG.out_dim)


@pytest.mark.parametrize(
    "model_config", RECIPE_DIR.glob("**/_common/conf/**/train/duration/model/*.yaml")
)
def test_duration_model_config_recipes(model_config):
    model_config = OmegaConf.load(model_config)
    model = hydra.utils.instantiate(model_config.netG)
    _test_model_impl(model, model_config.netG.in_dim, model_config.netG.out_dim)


@pytest.mark.parametrize(
    "model_config", RECIPE_DIR.glob("**/_common/conf/**/train_acoustic/model/*.yaml")
)
def test_train_acoustic_model_config_recipes(model_config):
    print(model_config)
    model_config = OmegaConf.load(model_config)

    # Dummy
    model_config.netG.in_lf0_idx = 10
    model_config.netG.in_lf0_min = 5.3936276
    model_config.netG.in_lf0_max = 6.491111
    model_config.netG.out_lf0_idx = 60
    model_config.netG.out_lf0_mean = 5.953093881972361
    model_config.netG.out_lf0_scale = 0.23435173188961034

    # NOTE: phoneme embedding impl requires careful specifications of
    # in_ph_start_idx and in_ph_end_idx, which is diffifult to set property
    # without hed files. So we skip phoneme embedding impl for now.
    if "embed_dim" in model_config.netG:
        model_config.netG.embed_dim = None

    keys = ["lf0_model", "mgc_model", "bap_model", "vuv_model", "encoder", "mel_model"]
    for key in keys:
        if key in model_config.netG and "embed_dim" in model_config.netG[key]:
            model_config.netG[key].embed_dim = None

        # For GaussianDiffusion that includes encoder in its implementation
        if (
            key in model_config.netG
            and "encoder" in model_config.netG[key]
            and "embed_dim" in model_config.netG[key]["encoder"]
        ):
            model_config.netG[key]["encoder"].embed_dim = None

    model = hydra.utils.instantiate(model_config.netG)
    _test_model_impl(model, model_config.netG.in_dim, model_config.netG.out_dim)


@pytest.mark.parametrize(
    "model_config",
    RECIPE_DIR.glob("**/icassp2023-24k-world/conf/train_acoustic/model/*.yaml"),
)
def test_train_acoustic_model_config_recipes_icassp2023(model_config):
    model_config = OmegaConf.load(model_config)

    # Dummy
    model_config.netG.in_lf0_idx = 10
    model_config.netG.in_lf0_min = 5.3936276
    model_config.netG.in_lf0_max = 6.491111
    model_config.netG.out_lf0_idx = 60
    model_config.netG.out_lf0_mean = 5.953093881972361
    model_config.netG.out_lf0_scale = 0.23435173188961034

    # NOTE: phoneme embedding impl requires careful specifications of
    # in_ph_start_idx and in_ph_end_idx, which is diffifult to set property
    # without hed files. So we skip phoneme embedding impl for now.
    if "embed_dim" in model_config.netG:
        model_config.netG.embed_dim = None

    keys = ["lf0_model", "mgc_model", "bap_model", "vuv_model", "encoder", "mel_model"]
    for key in keys:
        if key in model_config.netG and "embed_dim" in model_config.netG[key]:
            model_config.netG[key].embed_dim = None

    model = hydra.utils.instantiate(model_config.netG)
    _test_model_impl(model, model_config.netG.in_dim, model_config.netG.out_dim)


@pytest.mark.parametrize(
    "model_config", RECIPE_DIR.glob("**/_common/conf/**/train_postfilter/model/*.yaml")
)
def test_postfilter_config_recipes(model_config):
    model_config = OmegaConf.load(model_config)
    if "stream_sizes" in model_config.netG:
        model_config.netG.stream_sizes = model_config.stream_sizes
    # Post-filter config should have netD
    hydra.utils.instantiate(model_config.netD)
    model = hydra.utils.instantiate(model_config.netG)
    _test_postfilter_impl(model, model_config)
