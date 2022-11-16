from functools import partial

from nnsvs.acoustic_models.multistream import (
    MDNMultistreamSeparateF0MelModel,
    MultistreamSeparateF0MelModel,
    MultistreamSeparateF0ParametricModel,
    NPSSMDNMultistreamParametricModel,
    NPSSMultistreamParametricModel,
)
from nnsvs.acoustic_models.sinsy import ResSkipF0FFConvLSTM
from nnsvs.acoustic_models.tacotron import (
    BiLSTMMDNNonAttentiveDecoder,
    BiLSTMNonAttentiveDecoder,
    MDNNonAttentiveDecoder,
    NonAttentiveDecoder,
)
from nnsvs.acoustic_models.tacotron_f0 import (
    BiLSTMResF0NonAttentiveDecoder,
    MDNResF0NonAttentiveDecoder,
    ResF0NonAttentiveDecoder,
)
from nnsvs.acoustic_models.util import predict_lf0_with_residual
from nnsvs.base import BaseModel, PredictionType
from nnsvs.layers.conv import ResnetBlock, WNConv1d
from nnsvs.mdn import MDNLayer, mdn_get_most_probable_sigma_and_mu
from nnsvs.model import TransformerEncoder, VariancePredictor
from nnsvs.util import init_weights
from torch import nn

__all__ = [
    # Non-autoregressive models
    "ResF0Conv1dResnet",
    "ResSkipF0FFConvLSTM",
    "ResF0VariancePredictor",
    "ResF0TransformerEncoder",
    # Autoregressive models
    "NonAttentiveDecoder",
    "MDNNonAttentiveDecoder",
    "BiLSTMNonAttentiveDecoder",
    "BiLSTMMDNNonAttentiveDecoder",
    "ResF0NonAttentiveDecoder",
    "MDNResF0NonAttentiveDecoder",
    "BiLSTMResF0NonAttentiveDecoder",
    # Multi-stream models
    "MultistreamSeparateF0ParametricModel",
    "NPSSMDNMultistreamParametricModel",
    "NPSSMultistreamParametricModel",
    "MultistreamSeparateF0MelModel",
    "MDNMultistreamSeparateF0MelModel",
]


class ResF0Conv1dResnet(BaseModel):
    """Conv1d + Resnet + Residual F0 prediction

    Residual F0 prediction is inspired by  :cite:t:`hono2021sinsy`.

    Args:
        in_dim (int): input dimension
        hidden_dim (int): hidden dimension
        out_dim (int): output dimension
        num_layers (int): number of layers
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum value of lf0 in the training data of input features
        in_lf0_max (float): maximum value of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features. Typically 180.
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        init_type (str): initialization type
        use_mdn (bool): whether to use MDN or not
        num_gaussians (int): number of gaussians in MDN
        dim_wise (bool): whether to use dimension-wise MDN or not
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=4,
        # NOTE: you must carefully set the following parameters
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        init_type="none",
        use_mdn=False,
        num_gaussians=8,
        dim_wise=False,
    ):
        super().__init__()
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale
        self.use_mdn = use_mdn

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(in_dim, hidden_dim, kernel_size=7, padding=0),
        ]
        for n in range(num_layers):
            model.append(ResnetBlock(hidden_dim, dilation=2 ** n))

        last_conv_out_dim = hidden_dim if use_mdn else out_dim
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(hidden_dim, last_conv_out_dim, kernel_size=7, padding=0),
        ]
        self.model = nn.Sequential(*model)

        if self.use_mdn:
            self.mdn_layer = MDNLayer(
                in_dim=hidden_dim,
                out_dim=out_dim,
                num_gaussians=num_gaussians,
                dim_wise=dim_wise,
            )
        else:
            self.mdn_layer = None

        init_weights(self, init_type)

    def prediction_type(self):
        return (
            PredictionType.PROBABILISTIC
            if self.use_mdn
            else PredictionType.DETERMINISTIC
        )

    def has_residual_lf0_prediction(self):
        return True

    def forward(self, x, lengths=None, y=None):
        """Forward step

        Args:
            x (torch.Tensor): input features
            lengths (torch.Tensor): lengths of input features
            y (torch.Tensor): output features

        Returns:
            tuple: (output features, residual log-F0)
        """
        out = self.model(x.transpose(1, 2)).transpose(1, 2)

        if self.use_mdn:
            log_pi, log_sigma, mu = self.mdn_layer(out)
        else:
            mu = out

        lf0_pred, lf0_residual = predict_lf0_with_residual(
            x,
            mu,
            self.in_lf0_idx,
            self.in_lf0_min,
            self.in_lf0_max,
            self.out_lf0_idx,
            self.out_lf0_mean,
            self.out_lf0_scale,
        )

        # Inject the predicted lf0 into the output features
        if self.use_mdn:
            mu[:, :, :, self.out_lf0_idx] = lf0_pred
        else:
            mu[:, :, self.out_lf0_idx] = lf0_pred.squeeze(-1)

        if self.use_mdn:
            return (log_pi, log_sigma, mu), lf0_residual
        else:
            return mu, lf0_residual

    def inference(self, x, lengths=None):
        """Inference step

        Args:
            x (torch.Tensor): input features
            lengths (torch.Tensor): lengths of input features

        Returns:
            tuple: (mu, sigma) if use_mdn, (output, ) otherwise
        """
        if self.use_mdn:
            (log_pi, log_sigma, mu), _ = self(x, lengths)
            sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
            return mu, sigma
        else:
            return self(x, lengths)[0]


# Will be removed in v0.1.0
ResF0Conv1dResnetMDN = partial(ResF0Conv1dResnet, use_mdn=True)


class ResF0VariancePredictor(VariancePredictor):
    """Variance predictor in :cite:t:`ren2020fastspeech` with residual F0 prediction

    Args:
        in_dim (int): the input dimension
        out_dim (int): the output dimension
        num_layers (int): the number of layers
        hidden_dim (int): the hidden dimension
        kernel_size (int): the kernel size
        dropout (float): the dropout rate
        in_lf0_idx (int): the index of the input LF0
        in_lf0_min (float): the minimum value of the input LF0
        in_lf0_max (float): the maximum value of the input LF0
        out_lf0_idx (int): the index of the output LF0
        out_lf0_mean (float): the mean value of the output LF0
        out_lf0_scale (float): the scale value of the output LF0
        init_type (str): the initialization type
        use_mdn (bool): whether to use MDN or not
        num_gaussians (int): the number of gaussians
        dim_wise (bool): whether to use dim-wise or not
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        num_layers=5,
        hidden_dim=256,
        kernel_size=5,
        dropout=0.5,
        init_type="none",
        use_mdn=False,
        num_gaussians=1,
        dim_wise=False,
        # NOTE: you must carefully set the following parameters
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            init_type=init_type,
            use_mdn=use_mdn,
            num_gaussians=num_gaussians,
            dim_wise=dim_wise,
        )
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

    def has_residual_lf0_prediction(self):
        return True

    def forward(self, x, lengths=None, y=None):
        """Forward step

        Args:
            x (torch.Tensor): input features
            lengths (torch.Tensor): lengths of input features
            y (torch.Tensor): output features

        Returns:
            tuple: (output features, residual log-F0)
        """
        out = super().forward(x, lengths, y)
        if self.use_mdn:
            log_pi, log_sigma, mu = out
        else:
            mu = out

        lf0_pred, lf0_residual = predict_lf0_with_residual(
            x,
            mu,
            self.in_lf0_idx,
            self.in_lf0_min,
            self.in_lf0_max,
            self.out_lf0_idx,
            self.out_lf0_mean,
            self.out_lf0_scale,
        )

        # Inject the predicted lf0 into the output features
        if self.use_mdn:
            mu[:, :, :, self.out_lf0_idx] = lf0_pred
        else:
            mu[:, :, self.out_lf0_idx] = lf0_pred.squeeze(-1)

        if self.use_mdn:
            return (log_pi, log_sigma, mu), lf0_residual
        else:
            return mu, lf0_residual

    def inference(self, x, lengths=None):
        """Inference step

        Args:
            x (torch.Tensor): input features
            lengths (torch.Tensor): lengths of input features

        Returns:
            tuple: (mu, sigma) if use_mdn, (output, ) otherwise
        """
        if self.use_mdn:
            (log_pi, log_sigma, mu), _ = self(x, lengths)
            sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
            return mu, sigma
        else:
            return self(x, lengths)[0]


class ResF0TransformerEncoder(BaseModel):
    """Transformer encoder with residual f0 prediction"""

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        attention_dim,
        num_heads=2,
        num_layers=2,
        kernel_size=3,
        dropout=0.1,
        reduction_factor=1,
        init_type="none",
        downsample_by_conv=False,
        # NOTE: you must carefully set the following parameters
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
    ):
        super().__init__()
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale
        self.reduction_factor = reduction_factor
        self.encoder = TransformerEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            reduction_factor=reduction_factor,
            init_type=init_type,
            downsample_by_conv=downsample_by_conv,
        )

    def has_residual_lf0_prediction(self):
        return True

    def forward(self, x, lengths=None, y=None):
        """Forward pass

        Args:
            x (torch.Tensor): input tensor
            lengths (torch.Tensor): input sequence lengths
            y (torch.Tensor): target tensor (optional)

        Returns:
            torch.Tensor: output tensor
        """
        outs = self.encoder(x, lengths)

        lf0_pred, lf0_residual = predict_lf0_with_residual(
            x,
            outs,
            self.in_lf0_idx,
            self.in_lf0_min,
            self.in_lf0_max,
            self.out_lf0_idx,
            self.out_lf0_mean,
            self.out_lf0_scale,
        )
        outs[:, :, self.out_lf0_idx] = lf0_pred.squeeze(-1)

        return outs, lf0_residual

    def inference(self, x, lengths):
        return self(x, lengths)[0]


def LSTMEncoder(*args, **kwargs):
    from nnsvs.model import LSTMEncoder

    return LSTMEncoder(*args, **kwargs)
