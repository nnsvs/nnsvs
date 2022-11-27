import pytest
from nnsvs.acoustic_models import (
    BiLSTMMDNNonAttentiveDecoder,
    BiLSTMNonAttentiveDecoder,
    BiLSTMResF0NonAttentiveDecoder,
    MDNMultistreamSeparateF0MelModel,
    MDNNonAttentiveDecoder,
    MDNResF0NonAttentiveDecoder,
    MultistreamSeparateF0MelModel,
    MultistreamSeparateF0ParametricModel,
    NonAttentiveDecoder,
    NPSSMDNMultistreamParametricModel,
    NPSSMultistreamParametricModel,
    ResF0Conv1dResnet,
    ResF0Conv1dResnetMDN,
    ResF0NonAttentiveDecoder,
    ResF0TransformerEncoder,
    ResF0VariancePredictor,
    ResSkipF0FFConvLSTM,
)
from nnsvs.base import PredictionType
from nnsvs.model import FFN, MDN, FFConvLSTM, LSTMEncoder, TransformerEncoder

from .util import _test_model_impl


def test_deprecated_imports():
    from nnsvs.acoustic_models import LSTMEncoder  # noqa: F401


@pytest.mark.parametrize("num_gaussians", [1, 2, 4])
def test_resf0_conv1d_resnet(num_gaussians):
    params = {
        "in_dim": 300,
        "hidden_dim": 8,
        "out_dim": 200,
        "num_layers": 2,
        "num_gaussians": num_gaussians,
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
    _test_model_impl(model, params["in_dim"], params["out_dim"])

    model = ResF0Conv1dResnet(**{**params, "use_mdn": True})
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])

    # Deprecated
    model = ResF0Conv1dResnetMDN(**params)
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("num_gaussians", [1, 2, 4])
def test_reskipf0_ff_conv_lstm(num_gaussians):
    params = {
        "in_dim": 300,
        "ff_hidden_dim": 8,
        "conv_hidden_dim": 8,
        "lstm_hidden_dim": 8,
        "dropout": 0.1,
        "num_lstm_layers": 2,
        "bidirectional": True,
        "out_dim": 200,
        "num_gaussians": num_gaussians,
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
    _test_model_impl(model, params["in_dim"], params["out_dim"])

    model = ResSkipF0FFConvLSTM(**{**params, "use_mdn": True})
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("num_gaussians", [1, 2, 4])
def test_resf0_variance_predictor(num_gaussians):
    params = {
        "in_dim": 300,
        "out_dim": 200,
        "num_layers": 2,
        "hidden_dim": 8,
        "kernel_size": 5,
        "dropout": 0.5,
        "num_gaussians": num_gaussians,
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
    _test_model_impl(model, params["in_dim"], params["out_dim"])

    model = ResF0VariancePredictor(**{**params, "use_mdn": True})
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("reduction_factor", [1, 2])
@pytest.mark.parametrize("num_gaussians", [1, 2, 4])
@pytest.mark.parametrize("use_mdn_lf0", [False, True])
@pytest.mark.parametrize("vuv_model_lf0_conditioning", [False, True])
@pytest.mark.parametrize("vuv_model_mel_conditioning", [False, True])
def test_hybrid_multistream_mel_model_vuv_pred_from_mel(
    reduction_factor,
    num_gaussians,
    use_mdn_lf0,
    vuv_model_lf0_conditioning,
    vuv_model_mel_conditioning,
):
    vuv_in_dim = 300
    if vuv_model_lf0_conditioning:
        vuv_in_dim += 1
    if vuv_model_mel_conditioning:
        vuv_in_dim += 80
    params = {
        "in_dim": 300,
        "out_dim": 82,
        "stream_sizes": [80, 1, 1],
        "reduction_factor": reduction_factor,
        # Separate f0 model
        "lf0_model": ResF0Conv1dResnet(
            in_dim=300,
            hidden_dim=5,
            out_dim=1,
            num_layers=1,
            in_lf0_idx=-1,
            out_lf0_idx=0,
            use_mdn=use_mdn_lf0,
            num_gaussians=num_gaussians,
        ),
        # Decoders
        "mel_model": MDN(
            in_dim=301,
            hidden_dim=5,
            out_dim=80,
            dim_wise=True,
            num_gaussians=num_gaussians,
        ),
        "vuv_model": FFN(in_dim=vuv_in_dim, hidden_dim=5, out_dim=1),
        "vuv_model_lf0_conditioning": vuv_model_lf0_conditioning,
        "vuv_model_mel_conditioning": vuv_model_mel_conditioning,
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 180,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
    }
    model = MDNMultistreamSeparateF0MelModel(**params)
    assert model.prediction_type() == PredictionType.MULTISTREAM_HYBRID
    assert not model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("reduction_factor", [1, 2])
def test_multistream_mel_model(reduction_factor):
    params = {
        "in_dim": 300,
        "out_dim": 82,
        "stream_sizes": [80, 1, 1],
        "reduction_factor": reduction_factor,
        # Separate f0 model
        "lf0_model": ResF0Conv1dResnet(
            in_dim=300,
            hidden_dim=5,
            out_dim=1,
            num_layers=1,
            in_lf0_idx=-1,
            out_lf0_idx=0,
        ),
        # Encoder
        "encoder": LSTMEncoder(
            in_dim=300,
            hidden_dim=2,
            out_dim=4,
            num_layers=2,
            dropout=0.5,
            init_type="none",
        ),
        # Decoders
        "mel_model": FFN(in_dim=6, hidden_dim=5, out_dim=80),
        "vuv_model": FFN(in_dim=6, hidden_dim=5, out_dim=1),
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 180,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
    }
    model = MultistreamSeparateF0MelModel(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    assert not model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("num_gaussians", [1, 2, 4])
@pytest.mark.parametrize("vuv_model_bap0_conditioning", [False, True])
@pytest.mark.parametrize("use_mdn_lf0", [False, True])
@pytest.mark.parametrize("vuv_model_bap_conditioning", [False, True])
@pytest.mark.parametrize("vuv_model_lf0_conditioning", [False, True])
@pytest.mark.parametrize("vuv_model_mgc_conditioning", [False, True])
def test_npss_mdn_multistream_parametric_model(
    num_gaussians,
    vuv_model_bap0_conditioning,
    use_mdn_lf0,
    vuv_model_bap_conditioning,
    vuv_model_lf0_conditioning,
    vuv_model_mgc_conditioning,
):
    vuv_in_dim = 300
    if vuv_model_lf0_conditioning:
        vuv_in_dim += 1
    if vuv_model_mgc_conditioning:
        vuv_in_dim += 60
    if vuv_model_bap_conditioning:
        if vuv_model_bap0_conditioning:
            vuv_in_dim += 1
        else:
            vuv_in_dim += 5
    params = {
        "in_dim": 300,
        "out_dim": 67,
        "stream_sizes": [60, 1, 1, 5],
        "reduction_factor": 1,
        # Separate f0 model
        "lf0_model": ResF0Conv1dResnet(
            in_dim=300,
            hidden_dim=1,
            out_dim=1,
            num_layers=1,
            in_lf0_idx=-1,
            out_lf0_idx=0,
            use_mdn=use_mdn_lf0,
            num_gaussians=num_gaussians,
        ),
        # Decoders
        "mgc_model": MDN(
            in_dim=301,
            hidden_dim=5,
            out_dim=60,
            dim_wise=True,
            num_gaussians=num_gaussians,
        ),
        "bap_model": MDN(
            in_dim=301,
            hidden_dim=5,
            out_dim=5,
            dim_wise=True,
            num_gaussians=num_gaussians,
        ),
        "vuv_model": FFN(
            in_dim=vuv_in_dim,
            hidden_dim=5,
            out_dim=1,
        ),
        "vuv_model_bap_conditioning": vuv_model_bap_conditioning,
        "vuv_model_bap0_conditioning": vuv_model_bap0_conditioning,
        "vuv_model_lf0_conditioning": vuv_model_lf0_conditioning,
        "vuv_model_mgc_conditioning": vuv_model_mgc_conditioning,
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 180,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
    }
    model = NPSSMDNMultistreamParametricModel(**params)
    assert model.prediction_type() == PredictionType.MULTISTREAM_HYBRID
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("vuv_model_bap0_conditioning", [False, True])
@pytest.mark.parametrize("vuv_model_bap_conditioning", [False, True])
@pytest.mark.parametrize("vuv_model_lf0_conditioning", [False, True])
@pytest.mark.parametrize("vuv_model_mgc_conditioning", [False, True])
def test_nonmdn_npss_multistream_parametric_model(
    vuv_model_bap0_conditioning,
    vuv_model_bap_conditioning,
    vuv_model_lf0_conditioning,
    vuv_model_mgc_conditioning,
):
    vuv_in_dim = 300
    if vuv_model_lf0_conditioning:
        vuv_in_dim += 1
    if vuv_model_mgc_conditioning:
        vuv_in_dim += 60
    if vuv_model_bap_conditioning:
        if vuv_model_bap0_conditioning:
            vuv_in_dim += 1
        else:
            vuv_in_dim += 5
    params = {
        "in_dim": 300,
        "out_dim": 67,
        "stream_sizes": [60, 1, 1, 5],
        "reduction_factor": 1,
        # Separate f0 model
        "lf0_model": ResF0Conv1dResnet(
            in_dim=300,
            hidden_dim=1,
            out_dim=1,
            num_layers=1,
            in_lf0_idx=-1,
            out_lf0_idx=0,
            use_mdn=False,
        ),
        # Decoders
        "mgc_model": FFN(
            in_dim=301,
            hidden_dim=5,
            out_dim=60,
        ),
        "bap_model": FFN(
            in_dim=301,
            hidden_dim=5,
            out_dim=5,
        ),
        "vuv_model": FFN(
            in_dim=vuv_in_dim,
            hidden_dim=5,
            out_dim=1,
        ),
        "vuv_model_bap_conditioning": vuv_model_bap_conditioning,
        "vuv_model_bap0_conditioning": vuv_model_bap0_conditioning,
        "vuv_model_lf0_conditioning": vuv_model_lf0_conditioning,
        "vuv_model_mgc_conditioning": vuv_model_mgc_conditioning,
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 180,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
    }
    model = NPSSMultistreamParametricModel(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("reduction_factor", [1, 2])
@pytest.mark.parametrize("downsample_by_conv", [False, True])
@pytest.mark.parametrize("num_gaussians", [1, 2, 4])
def test_bilstm_mdn_tacotron_decoder(
    reduction_factor, downsample_by_conv, num_gaussians
):
    params = {
        "in_dim": 300,
        "ff_hidden_dim": 4,
        "conv_hidden_dim": 4,
        "lstm_hidden_dim": 4,
        "num_lstm_layers": 1,
        "out_dim": 60,
        "decoder_hidden_dim": 4,
        "decoder_layers": 1,
        "prenet_layers": 0,
        "prenet_hidden_dim": 4,
        "prenet_dropout": 0.5,
        "zoneout": 0.1,
        "reduction_factor": reduction_factor,
        "downsample_by_conv": downsample_by_conv,
        "num_gaussians": num_gaussians,
    }
    model = BiLSTMMDNNonAttentiveDecoder(**params)
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    assert model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("use_mdn", [False, True])
def test_ff_conv_lstm(use_mdn):
    params = {
        "in_dim": 300,
        "ff_hidden_dim": 8,
        "conv_hidden_dim": 8,
        "lstm_hidden_dim": 8,
        "dropout": 0.1,
        "num_lstm_layers": 2,
        "bidirectional": True,
        "out_dim": 200,
        "init_type": "none",
        "use_mdn": use_mdn,
        "num_gaussians": 2,
        "dim_wise": True,
    }
    model = FFConvLSTM(**params)
    if use_mdn:
        assert model.prediction_type() == PredictionType.PROBABILISTIC
    else:
        assert model.prediction_type() == PredictionType.DETERMINISTIC
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("reduction_factor", [1, 2])
@pytest.mark.parametrize("downsample_by_conv", [False, True])
@pytest.mark.parametrize("prenet_layers", [0, 1])
def test_resf0_tacotron_decoder_mdn(
    reduction_factor, downsample_by_conv, prenet_layers
):
    params = {
        "in_dim": 300,
        "out_dim": 80,
        "hidden_dim": 4,
        "layers": 1,
        "prenet_layers": prenet_layers,
        "prenet_hidden_dim": 4,
        "prenet_dropout": 0.5,
        "zoneout": 0.1,
        "reduction_factor": reduction_factor,
        "downsample_by_conv": downsample_by_conv,
        "num_gaussians": 4,
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 0,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
    }
    model = MDNResF0NonAttentiveDecoder(**params)
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    assert model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("reduction_factor", [1, 2])
@pytest.mark.parametrize("downsample_by_conv", [False, True])
@pytest.mark.parametrize("prenet_layers", [0, 1, 2])
@pytest.mark.parametrize("sampling_mode", ["mean", "random"])
def test_tacotron_decoder_mdn(
    reduction_factor, downsample_by_conv, prenet_layers, sampling_mode
):
    params = {
        "in_dim": 300,
        "out_dim": 206,
        "hidden_dim": 4,
        "layers": 1,
        "prenet_layers": prenet_layers,
        "prenet_hidden_dim": 4,
        "prenet_dropout": 0.5,
        "zoneout": 0.1,
        "reduction_factor": reduction_factor,
        "downsample_by_conv": downsample_by_conv,
        "num_gaussians": 8,
        "sampling_mode": sampling_mode,
        "initial_value": 0.0,
    }
    model = MDNNonAttentiveDecoder(**params)
    assert model.prediction_type() == PredictionType.PROBABILISTIC
    assert model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("num_heads", [1, 2, 4])
@pytest.mark.parametrize("reduction_factor", [1, 2])
@pytest.mark.parametrize("downsample_by_conv", [False, True])
def test_transformer_encoder(num_heads, reduction_factor, downsample_by_conv):
    params = {
        "in_dim": 300,
        "out_dim": 206,
        "hidden_dim": 4,
        "attention_dim": 4 * num_heads,
        "num_heads": num_heads,
        "num_layers": 2,
        "kernel_size": 3,
        "dropout": 0.1,
        "reduction_factor": reduction_factor,
        "init_type": "none",
        "downsample_by_conv": downsample_by_conv,
    }
    model = TransformerEncoder(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    assert not model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("reduction_factor", [1, 2])
@pytest.mark.parametrize("downsample_by_conv", [False, True])
@pytest.mark.parametrize("prenet_layers", [0, 1, 2])
def test_tacotron_decoder(reduction_factor, downsample_by_conv, prenet_layers):
    params = {
        "in_dim": 300,
        "out_dim": 206,
        "hidden_dim": 4,
        "layers": 1,
        "prenet_layers": prenet_layers,
        "prenet_hidden_dim": 4,
        "prenet_dropout": 0.5,
        "zoneout": 0.1,
        "reduction_factor": reduction_factor,
        "downsample_by_conv": downsample_by_conv,
        "initial_value": 0.0,
    }
    model = NonAttentiveDecoder(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    assert model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("reduction_factor", [1, 2])
@pytest.mark.parametrize("downsample_by_conv", [False, True])
@pytest.mark.parametrize("prenet_layers", [0, 1])
def test_bilstm_tacotron_decoder(reduction_factor, downsample_by_conv, prenet_layers):
    params = {
        "in_dim": 300,
        "ff_hidden_dim": 4,
        "conv_hidden_dim": 4,
        "lstm_hidden_dim": 4,
        "num_lstm_layers": 1,
        "out_dim": 60,
        "decoder_hidden_dim": 4,
        "decoder_layers": 1,
        "prenet_layers": prenet_layers,
        "prenet_hidden_dim": 4,
        "prenet_dropout": 0.5,
        "zoneout": 0.1,
        "reduction_factor": reduction_factor,
        "downsample_by_conv": downsample_by_conv,
    }
    model = BiLSTMNonAttentiveDecoder(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    assert model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("reduction_factor", [1, 2])
@pytest.mark.parametrize("downsample_by_conv", [False, True])
@pytest.mark.parametrize("prenet_layers", [0, 1])
def test_resf0_tacotron_decoder(reduction_factor, downsample_by_conv, prenet_layers):
    params = {
        "in_dim": 300,
        "out_dim": 3,
        "hidden_dim": 4,
        "layers": 1,
        "prenet_layers": prenet_layers,
        "prenet_hidden_dim": 4,
        "prenet_dropout": 0.5,
        "zoneout": 0.1,
        "reduction_factor": reduction_factor,
        "downsample_by_conv": downsample_by_conv,
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 0,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
    }
    model = ResF0NonAttentiveDecoder(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    assert model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("reduction_factor", [1, 2])
@pytest.mark.parametrize("downsample_by_conv", [False, True])
@pytest.mark.parametrize("prenet_layers", [0, 1])
@pytest.mark.parametrize("use_mdn", [False, True])
def test_bilstm_resf0_tacotron_decoder(
    reduction_factor, downsample_by_conv, prenet_layers, use_mdn
):
    params = {
        "in_dim": 300,
        "ff_hidden_dim": 4,
        "conv_hidden_dim": 4,
        "lstm_hidden_dim": 4,
        "num_lstm_layers": 1,
        "out_dim": 3,
        "decoder_hidden_dim": 4,
        "decoder_layers": 1,
        "prenet_layers": prenet_layers,
        "prenet_hidden_dim": 4,
        "prenet_dropout": 0.5,
        "zoneout": 0.1,
        "reduction_factor": reduction_factor,
        "downsample_by_conv": downsample_by_conv,
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 0,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
        "use_mdn": use_mdn,
    }
    model = BiLSTMResF0NonAttentiveDecoder(**params)
    if use_mdn:
        assert model.prediction_type() == PredictionType.PROBABILISTIC
    else:
        assert model.prediction_type() == PredictionType.DETERMINISTIC
    assert model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("num_heads", [1, 2, 4])
@pytest.mark.parametrize("reduction_factor", [1, 2])
def test_resf0_transformer_encoder(num_heads, reduction_factor):
    params = {
        "in_dim": 300,
        "out_dim": 206,
        "hidden_dim": 4,
        "attention_dim": 4 * num_heads,
        "num_heads": num_heads,
        "num_layers": 2,
        "kernel_size": 3,
        "dropout": 0.1,
        "reduction_factor": reduction_factor,
        "init_type": "none",
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 180,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
    }
    model = ResF0TransformerEncoder(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    assert not model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


def test_lstm_encoder():
    params = {
        "in_dim": 300,
        "out_dim": 206,
        "hidden_dim": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "init_type": "none",
    }
    model = LSTMEncoder(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    assert not model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("reduction_factor", [1, 2])
def test_nonar_multistream_parametric_model(reduction_factor):
    params = {
        "in_dim": 300,
        "out_dim": 199,
        "stream_sizes": [180, 3, 1, 15],
        "reduction_factor": reduction_factor,
        # Separate f0 model
        "lf0_model": ResF0Conv1dResnet(
            in_dim=300,
            hidden_dim=5,
            out_dim=3,
            num_layers=1,
            in_lf0_idx=-1,
            out_lf0_idx=0,
        ),
        # Encoder
        "encoder": LSTMEncoder(
            in_dim=300,
            hidden_dim=2,
            out_dim=4,
            num_layers=2,
            dropout=0.5,
            init_type="none",
        ),
        # Decoders
        "mgc_model": FFN(in_dim=8, hidden_dim=5, out_dim=180),
        "vuv_model": FFN(in_dim=8, hidden_dim=5, out_dim=1),
        "bap_model": FFN(in_dim=8, hidden_dim=5, out_dim=15),
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 180,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
    }
    model = MultistreamSeparateF0ParametricModel(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    assert not model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("reduction_factor", [1, 2])
def test_ar_multistream_parametric_model(reduction_factor):
    params = {
        "in_dim": 300,
        "out_dim": 199,
        "stream_sizes": [180, 3, 1, 15],
        "reduction_factor": reduction_factor,
        # Separate f0 model
        "lf0_model": ResF0Conv1dResnet(
            in_dim=300,
            hidden_dim=5,
            out_dim=3,
            num_layers=1,
            in_lf0_idx=-1,
            out_lf0_idx=0,
        ),
        # Encoder
        "encoder": LSTMEncoder(
            in_dim=300,
            hidden_dim=2,
            out_dim=4,
            num_layers=2,
            dropout=0.5,
            init_type="none",
        ),
        # Decoders
        "mgc_model": NonAttentiveDecoder(
            in_dim=8,
            hidden_dim=5,
            out_dim=180,
            layers=2,
            reduction_factor=reduction_factor,
        ),
        "vuv_model": NonAttentiveDecoder(
            in_dim=8,
            hidden_dim=5,
            out_dim=1,
            layers=1,
            reduction_factor=reduction_factor,
        ),
        "bap_model": NonAttentiveDecoder(
            in_dim=8,
            hidden_dim=5,
            out_dim=15,
            layers=1,
            reduction_factor=reduction_factor,
        ),
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 180,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
    }
    model = MultistreamSeparateF0ParametricModel(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    assert model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


@pytest.mark.parametrize("reduction_factor", [1, 2])
@pytest.mark.parametrize("postnet_layers", [0, 1])
def test_ar_multistream_parametric_model_with_postnet(reduction_factor, postnet_layers):
    params = {
        "in_dim": 300,
        "out_dim": 67,
        "stream_sizes": [60, 1, 1, 5],
        "reduction_factor": reduction_factor,
        # Separate f0 model
        "lf0_model": ResF0Conv1dResnet(
            in_dim=300,
            hidden_dim=5,
            out_dim=1,
            num_layers=1,
            in_lf0_idx=-1,
            out_lf0_idx=0,
        ),
        # Encoder
        "encoder": LSTMEncoder(
            in_dim=300,
            hidden_dim=2,
            out_dim=4,
            num_layers=2,
            dropout=0.5,
            init_type="none",
        ),
        # Decoders
        "mgc_model": BiLSTMNonAttentiveDecoder(
            in_dim=6,
            out_dim=60,
            ff_hidden_dim=8,
            conv_hidden_dim=8,
            lstm_hidden_dim=8,
            num_lstm_layers=1,
            decoder_layers=1,
            decoder_hidden_dim=8,
            prenet_layers=1,
            prenet_hidden_dim=8,
            postnet_layers=postnet_layers,
            postnet_channels=8,
            postnet_kernel_size=3,
            reduction_factor=reduction_factor,
        ),
        "vuv_model": BiLSTMNonAttentiveDecoder(
            in_dim=6,
            out_dim=1,
            ff_hidden_dim=8,
            conv_hidden_dim=8,
            lstm_hidden_dim=8,
            num_lstm_layers=1,
            decoder_layers=1,
            decoder_hidden_dim=8,
            prenet_layers=1,
            prenet_hidden_dim=8,
            postnet_layers=postnet_layers,
            postnet_channels=8,
            postnet_kernel_size=3,
            reduction_factor=reduction_factor,
        ),
        "bap_model": BiLSTMNonAttentiveDecoder(
            in_dim=6,
            out_dim=5,
            ff_hidden_dim=8,
            conv_hidden_dim=8,
            lstm_hidden_dim=8,
            num_lstm_layers=1,
            decoder_layers=1,
            decoder_hidden_dim=8,
            prenet_layers=1,
            prenet_hidden_dim=8,
            postnet_layers=postnet_layers,
            postnet_channels=8,
            postnet_kernel_size=3,
            reduction_factor=reduction_factor,
        ),
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 60,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
    }
    model = MultistreamSeparateF0ParametricModel(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    assert model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])


def test_nonar_multistream_parametric_model_no_encoder():
    params = {
        "in_dim": 300,
        "out_dim": 67,
        "stream_sizes": [60, 1, 1, 5],
        "reduction_factor": 1,
        # Separate f0 model
        "lf0_model": ResF0Conv1dResnet(
            in_dim=300,
            hidden_dim=5,
            out_dim=1,
            num_layers=1,
            in_lf0_idx=-1,
            out_lf0_idx=0,
        ),
        # No encoder
        "encoder": None,
        # Decoders
        "mgc_model": FFN(in_dim=300, hidden_dim=5, out_dim=60),
        "vuv_model": FFN(in_dim=300, hidden_dim=5, out_dim=1),
        "bap_model": FFN(in_dim=300, hidden_dim=5, out_dim=5),
        # dummy
        "in_lf0_idx": 0,
        "in_lf0_min": 5.3936276,
        "in_lf0_max": 6.491111,
        "out_lf0_idx": 180,
        "out_lf0_mean": 5.953093881972361,
        "out_lf0_scale": 0.23435173188961034,
    }
    model = MultistreamSeparateF0ParametricModel(**params)
    assert model.prediction_type() == PredictionType.DETERMINISTIC
    assert not model.is_autoregressive()
    _test_model_impl(model, params["in_dim"], params["out_dim"])
