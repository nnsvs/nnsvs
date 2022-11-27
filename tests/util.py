import torch
from nnsvs.base import PredictionType
from nnsvs.util import init_seed


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
