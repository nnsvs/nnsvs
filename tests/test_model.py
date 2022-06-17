import torch
from nnsvs.base import PredictionType
from nnsvs.model import (
    FFN,
    LSTMRNN,
    LSTMRNNSAR,
    MDN,
    RMDN,
    Conv1dResnet,
    Conv1dResnetMDN,
    Conv1dResnetSAR,
    FFConvLSTM,
    MDNv2,
    VariancePredictor,
)
from nnsvs.util import init_seed


def test_deprecated_imports():
    from nnsvs.model import ResF0Conv1dResnet  # noqa: F401
    from nnsvs.model import ResF0Conv1dResnetMDN  # noqa: F401
    from nnsvs.model import ResF0VariancePredictor  # noqa: F401
    from nnsvs.model import ResSkipF0FFConvLSTM  # noqa: F401


def _test_model_impl(model, in_dim, out_dim):
    B = 4
    T = 100
    init_seed(B * T)
    x = torch.rand(B, T, in_dim)
    lengths = torch.Tensor([T] * B).long()

    # warmup forward pass
    with torch.no_grad():
        y = model(x, lengths)
        y_inf = model.inference(x, lengths)

    # MDN case
    if model.prediction_type() == PredictionType.PROBABILISTIC:
        log_pi, log_sigma, mu = y
        num_gaussian = log_pi.shape[2]
        assert mu.shape == (B, T, num_gaussian, out_dim)
        assert log_sigma.shape == (B, T, num_gaussian, out_dim)

        # NOTE: infernece output shouldn't have num_gaussian axis
        mu_inf, sigma_inf = y_inf
        assert mu_inf.shape == (B, T, out_dim)
        assert sigma_inf.shape == (B, T, out_dim)
    else:
        assert y.shape == (B, T, out_dim)
        assert y.shape == y_inf.shape


def test_ffn():
    params = {
        "in_dim": 300,
        "hidden_dim": 8,
        "out_dim": 20,
        "num_layers": 2,
        "dropout": 0.1,
        "init_type": "none",
    }
    model = FFN(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])


def test_lstmrnn():
    params = {
        "in_dim": 300,
        "hidden_dim": 8,
        "out_dim": 20,
        "num_layers": 2,
        "dropout": 0.1,
        "init_type": "none",
    }
    model = LSTMRNN(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])


def test_conv1d_resnet():
    params = {
        "in_dim": 300,
        "hidden_dim": 8,
        "out_dim": 180,
        "num_layers": 2,
        "num_gaussians": 2,
        "dim_wise": True,
        "init_type": "none",
    }
    model = Conv1dResnet(**{**params, "use_mdn": False})
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])

    model = Conv1dResnet(**{**params, "use_mdn": True})
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])

    # Deprecated
    model = Conv1dResnetMDN(**params)
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])


def test_conv1d_resnet_sar():
    params = {
        "in_dim": 300,
        "hidden_dim": 8,
        "out_dim": 20,
        "num_layers": 2,
        "stream_sizes": [10, 10],
        "ar_orders": [2, 2],
        "init_type": "none",
    }
    model = Conv1dResnetSAR(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])

    y = torch.rand(8, 100, params["out_dim"])
    y_hat = model.preprocess_target(y)
    assert y.shape == y_hat.shape


def test_lstmrnn_sar():
    params = {
        "in_dim": 300,
        "hidden_dim": 8,
        "out_dim": 20,
        "num_layers": 2,
        "dropout": 0.1,
        "stream_sizes": [10, 10],
        "ar_orders": [2, 2],
        "init_type": "none",
    }
    model = LSTMRNNSAR(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])

    y = torch.rand(8, 100, params["out_dim"])
    y_hat = model.preprocess_target(y)
    assert y.shape == y_hat.shape


def test_mdn():
    params = {
        "in_dim": 300,
        "hidden_dim": 8,
        "out_dim": 180,
        "num_layers": 2,
        "num_gaussians": 2,
        "dim_wise": True,
        "init_type": "none",
    }
    model = MDN(**params)
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])


def test_mdnv2():
    params = {
        "in_dim": 300,
        "hidden_dim": 8,
        "out_dim": 180,
        "num_layers": 2,
        "dropout": 0.5,
        "num_gaussians": 2,
        "dim_wise": True,
        "init_type": "none",
    }
    model = MDNv2(**params)
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])


def test_rmdn():
    params = {
        "in_dim": 300,
        "hidden_dim": 8,
        "out_dim": 180,
        "num_layers": 2,
        "num_gaussians": 2,
        "dim_wise": True,
        "init_type": "none",
    }
    model = RMDN(**params)
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])


def test_ff_conv_lstm():
    params = {
        "in_dim": 300,
        "ff_hidden_dim": 8,
        "conv_hidden_dim": 8,
        "lstm_hidden_dim": 8,
        "dropout": 0.1,
        "num_lstm_layers": 2,
        "bidirectional": True,
        "out_dim": 180,
        "init_type": "none",
    }
    model = FFConvLSTM(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])


def test_variance_predictor():
    params = {
        "in_dim": 300,
        "out_dim": 180,
        "num_layers": 2,
        "hidden_dim": 8,
        "kernel_size": 5,
        "dropout": 0.5,
        "init_type": "none",
    }
    model = VariancePredictor(**{**params, "use_mdn": False})
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])

    model = VariancePredictor(**{**params, "use_mdn": True})
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])
