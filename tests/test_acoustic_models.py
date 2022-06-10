import torch
from nnsvs.acoustic_models import (
    ResF0Conv1dResnet,
    ResF0Conv1dResnetMDN,
    ResF0VariancePredictor,
    ResSkipF0FFConvLSTM,
)
from nnsvs.base import PredictionType
from nnsvs.util import init_seed


def _test_resf0_model_impl(model, in_dim, out_dim):
    B = 4
    T = 100
    init_seed(B * T)
    x = torch.rand(B, T, in_dim)
    lengths = torch.Tensor([T] * B).long()

    # warmup forward pass
    with torch.no_grad():
        y, lf0_residual = model(x, lengths)
        y_inf = model.inference(x, lengths)

    # MDN case
    if model.prediction_type() == PredictionType.PROBABILISTIC:
        log_pi, log_sigma, mu = y
        num_gaussian = log_pi.shape[2]
        assert mu.shape == (B, T, num_gaussian, out_dim)
        assert log_sigma.shape == (B, T, num_gaussian, out_dim)
        assert lf0_residual.shape == (B, T, num_gaussian)

        # NOTE: infernece output shouldn't have num_gaussian axis
        mu_inf, sigma_inf = y_inf
        assert mu_inf.shape == (B, T, out_dim)
        assert sigma_inf.shape == (B, T, out_dim)
    else:
        assert lf0_residual.shape == (B, T, 1)
        assert y.shape == (B, T, out_dim)
        assert y.shape == y_inf.shape


def test_resf0_conv1d_resnet():
    params = {
        "in_dim": 300,
        "hidden_dim": 8,
        "out_dim": 200,
        "num_layers": 2,
        "num_gaussians": 2,
        "dim_wise": True,
        "init_type": "none",
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 180,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
    }
    model = ResF0Conv1dResnet(**{**params, "use_mdn": False})
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    _test_resf0_model_impl(model, params["in_dim"], params["out_dim"])

    model = ResF0Conv1dResnet(**{**params, "use_mdn": True})
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    _test_resf0_model_impl(model, params["in_dim"], params["out_dim"])

    # Deprecated
    model = ResF0Conv1dResnetMDN(**params)
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    _test_resf0_model_impl(model, params["in_dim"], params["out_dim"])


def test_reskipf0_ff_conv_lstm():
    params = {
        "in_dim": 300,
        "ff_hidden_dim": 8,
        "conv_hidden_dim": 8,
        "lstm_hidden_dim": 8,
        "dropout": 0.1,
        "num_lstm_layers": 2,
        "bidirectional": True,
        "out_dim": 200,
        "num_gaussians": 2,
        "dim_wise": True,
        "init_type": "none",
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 180,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
    }
    model = ResSkipF0FFConvLSTM(**{**params, "use_mdn": False})
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    _test_resf0_model_impl(model, params["in_dim"], params["out_dim"])

    model = ResSkipF0FFConvLSTM(**{**params, "use_mdn": True})
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    _test_resf0_model_impl(model, params["in_dim"], params["out_dim"])


def test_resf0_variance_predictor():
    params = {
        "in_dim": 300,
        "out_dim": 200,
        "num_layers": 2,
        "hidden_dim": 8,
        "kernel_size": 5,
        "dropout": 0.5,
        "num_gaussians": 2,
        "dim_wise": True,
        "init_type": "none",
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 180,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
    }
    model = ResF0VariancePredictor(**{**params, "use_mdn": False})
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    _test_resf0_model_impl(model, params["in_dim"], params["out_dim"])

    model = ResF0VariancePredictor(**{**params, "use_mdn": True})
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    _test_resf0_model_impl(model, params["in_dim"], params["out_dim"])
