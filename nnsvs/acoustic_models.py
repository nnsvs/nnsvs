from functools import partial

import numpy as np
import torch
from nnsvs.base import BaseModel, PredictionType
from nnsvs.layers.conv import ResnetBlock, WNConv1d
from nnsvs.mdn import MDNLayer, mdn_get_most_probable_sigma_and_mu
from nnsvs.model import VariancePredictor
from nnsvs.util import init_weights
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = [
    "ResF0Conv1dResnet",
    "ResSkipF0FFConvLSTM",
    "ResF0VariancePredictor",
    "predict_lf0_with_residual",
]


def predict_lf0_with_residual(
    in_feats,
    out_feats,
    in_lf0_idx=300,
    in_lf0_min=5.3936276,
    in_lf0_max=6.491111,
    out_lf0_idx=180,
    out_lf0_mean=5.953093881972361,
    out_lf0_scale=0.23435173188961034,
    residual_f0_max_cent=600,
):
    """Predict log-F0 with residual.

    Args:
        in_feats (np.ndarray): input features
        out_feats (np.ndarray): output of an acoustic model
        in_lf0_idx (int): index of LF0 in input features
        in_lf0_min (float): minimum value of LF0 in the training data of input features
        in_lf0_max (float): maximum value of LF0 in the training data of input features
        out_lf0_idx (int): index of LF0 in output features
        out_lf0_mean (float): mean of LF0 in the training data of output features
        out_lf0_scale (float): scale of LF0 in the training data of output features
        residual_f0_max_cent (int): maximum value of residual LF0 in cent

    Returns:
        tuple: (predicted log-F0, residual log-F0)
    """
    # Denormalize lf0 from input musical score
    lf0_score = in_feats[:, :, in_lf0_idx].unsqueeze(-1)
    lf0_score_denorm = lf0_score * (in_lf0_max - in_lf0_min) + in_lf0_min

    # To avoid unbounded residual f0 that would potentially cause artifacts,
    # let's constrain the residual F0 to be in a certain range by the scaled tanh
    max_lf0_ratio = residual_f0_max_cent * np.log(2) / 1200

    if len(out_feats.shape) == 4:
        # MDN case (B, T, num_gaussians, C) -> (B, T, num_gaussians)
        lf0_residual = out_feats[:, :, :, out_lf0_idx]
    else:
        # (B, T, C) -> (B, T, 1)
        lf0_residual = out_feats[:, :, out_lf0_idx].unsqueeze(-1)
    lf0_residual = max_lf0_ratio * torch.tanh(lf0_residual)

    # Residual connection in the denormalized f0 domain
    lf0_pred_denorm = lf0_score_denorm + lf0_residual

    # Back to normalized f0
    lf0_pred = (lf0_pred_denorm - out_lf0_mean) / out_lf0_scale

    return lf0_pred, lf0_residual


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
            mu[:, :, :, self.out_lf0_idx] = lf0_pred.squeeze(-1)
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


class ResSkipF0FFConvLSTM(BaseModel):
    """FFN + Conv1d + LSTM + residual/skip connections

    A model proposed in :cite:t:`hono2021sinsy`.

    Args:
        in_dim (int): input dimension
        ff_hidden_dim (int): hidden dimension of feed-forward layer
        conv_hidden_dim (int): hidden dimension of convolutional layer
        lstm_hidden_dim (int): hidden dimension of LSTM layer
        out_dim (int): output dimension
        dropout (float): dropout rate
        num_ls (int): number of layers of LSTM
        bidirectional (bool): whether to use bidirectional LSTM or not
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum of lf0 in the training data of input features
        in_lf0_max (float): maximum of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        skip_inputs (bool): whether to use skip connection for the input features
        init_type (str): initialization type
        use_mdn (bool): whether to use MDN or not
        num_gaussians (int): number of gaussians in MDN
        dim_wise (bool): whether to use MDN with dim-wise or not
    """

    def __init__(
        self,
        in_dim,
        ff_hidden_dim=2048,
        conv_hidden_dim=1024,
        lstm_hidden_dim=256,
        out_dim=199,
        dropout=0.0,
        num_lstm_layers=2,
        bidirectional=True,
        # NOTE: you must carefully set the following parameters
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        skip_inputs=False,
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
        self.skip_inputs = skip_inputs
        self.use_mdn = use_mdn

        self.ff = nn.Sequential(
            nn.Linear(in_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(ff_hidden_dim + 1, conv_hidden_dim, kernel_size=7, padding=0),
            nn.BatchNorm1d(conv_hidden_dim),
            nn.ReLU(),
            nn.ReflectionPad1d(3),
            nn.Conv1d(conv_hidden_dim, conv_hidden_dim, kernel_size=7, padding=0),
            nn.BatchNorm1d(conv_hidden_dim),
            nn.ReLU(),
            nn.ReflectionPad1d(3),
            nn.Conv1d(conv_hidden_dim, conv_hidden_dim, kernel_size=7, padding=0),
            nn.BatchNorm1d(conv_hidden_dim),
            nn.ReLU(),
        )

        num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            conv_hidden_dim,
            lstm_hidden_dim,
            num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        if self.skip_inputs:
            last_in_dim = num_direction * lstm_hidden_dim + in_dim
        else:
            last_in_dim = num_direction * lstm_hidden_dim

        if self.use_mdn:
            self.mdn_layer = MDNLayer(
                last_in_dim, out_dim, num_gaussians=num_gaussians, dim_wise=dim_wise
            )
        else:
            self.fc = nn.Linear(last_in_dim, out_dim)

        init_weights(self, init_type)

    def prediction_type(self):
        return (
            PredictionType.PROBABILISTIC
            if self.use_mdn
            else PredictionType.DETERMINISTIC
        )

    def forward(self, x, lengths=None, y=None):
        """Forward step

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor
            y (torch.Tensor): the optional target tensor

        Returns:
            torch.Tensor: the output tensor
        """
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.to("cpu")

        lf0_score = x[:, :, self.in_lf0_idx].unsqueeze(-1)

        out = self.ff(x)
        out = torch.cat([out, lf0_score], dim=-1)

        out = self.conv(out.transpose(1, 2)).transpose(1, 2)
        sequence = pack_padded_sequence(out, lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = torch.cat([out, x], dim=-1) if self.skip_inputs else out

        if self.use_mdn:
            log_pi, log_sigma, mu = self.mdn_layer(out)
        else:
            mu = self.fc(out)

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
            mu[:, :, :, self.out_lf0_idx] = lf0_pred.squeeze(-1)
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
            mu[:, :, :, self.out_lf0_idx] = lf0_pred.squeeze(-1)
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
