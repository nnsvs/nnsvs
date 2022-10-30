from functools import partial

import numpy as np
import torch
from nnsvs.acoustic_models.multistream import (
    HybridMultistreamSeparateF0MelModel,
    MultistreamSeparateF0MelModel,
    MultistreamSeparateF0ParametricModel,
    NPSSMDNMultistreamParametricModel,
    NPSSMultistreamParametricModel,
)
from nnsvs.acoustic_models.util import pad_inference, predict_lf0_with_residual
from nnsvs.base import BaseModel, PredictionType
from nnsvs.layers.conv import ResnetBlock, WNConv1d
from nnsvs.mdn import MDNLayer, mdn_get_most_probable_sigma_and_mu, mdn_get_sample
from nnsvs.model import TransformerEncoder, VariancePredictor
from nnsvs.tacotron.decoder import MDNNonAttentiveDecoder
from nnsvs.tacotron.decoder import NonAttentiveDecoder as TacotronNonAttentiveDecoder
from nnsvs.tacotron.decoder import Prenet, ZoneOutCell
from nnsvs.tacotron.postnet import Postnet as TacotronPostnet
from nnsvs.util import init_weights
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
    "HybridMultistreamSeparateF0MelModel",
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


class NonAttentiveDecoder(TacotronNonAttentiveDecoder):
    """Non-attentive autoregressive decoder based on the duration-informed Tacotron."""

    def __init__(
        self,
        in_dim=512,
        out_dim=80,
        layers=2,
        hidden_dim=1024,
        prenet_layers=2,
        prenet_hidden_dim=256,
        prenet_dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        downsample_by_conv=False,
        postnet_layers=0,
        postnet_channels=512,
        postnet_kernel_size=5,
        postnet_dropout=0.0,
        init_type="none",
        eval_dropout=True,
        prenet_noise_std=0.0,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            layers=layers,
            hidden_dim=hidden_dim,
            prenet_layers=prenet_layers,
            prenet_hidden_dim=prenet_hidden_dim,
            prenet_dropout=prenet_dropout,
            zoneout=zoneout,
            reduction_factor=reduction_factor,
            downsample_by_conv=downsample_by_conv,
            eval_dropout=eval_dropout,
            prenet_noise_std=prenet_noise_std,
        )
        if postnet_layers > 0:
            self.postnet = TacotronPostnet(
                out_dim,
                layers=postnet_layers,
                channels=postnet_channels,
                kernel_size=postnet_kernel_size,
                dropout=postnet_dropout,
            )
        else:
            self.postnet = None
        init_weights(self, init_type)

    def forward(self, x, lengths=None, y=None):
        outs = super().forward(x, lengths, y)

        if self.postnet is not None:
            # NOTE: `outs.clone()`` is necessary to compute grad on both outs and outs_fine
            outs_fine = outs + self.postnet(outs.transpose(1, 2).clone()).transpose(
                1, 2
            )
            return [outs, outs_fine]
        else:
            return outs

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self, x=x, lengths=lengths, reduction_factor=self.reduction_factor
        )


class BiLSTMNonAttentiveDecoder(BaseModel):
    """NonAttentiveDecoder + BiLSTM"""

    def __init__(
        self,
        in_dim=512,
        ff_hidden_dim=2048,
        conv_hidden_dim=1024,
        lstm_hidden_dim=256,
        num_lstm_layers=2,
        out_dim=80,
        decoder_layers=2,
        decoder_hidden_dim=1024,
        prenet_layers=2,
        prenet_hidden_dim=256,
        prenet_dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        downsample_by_conv=False,
        postnet_layers=0,
        postnet_channels=512,
        postnet_kernel_size=5,
        postnet_dropout=0.0,
        in_ph_start_idx: int = 1,
        in_ph_end_idx: int = 50,
        embed_dim=None,
        init_type="none",
        eval_dropout=True,
        prenet_noise_std=0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim
        self.reduction_factor = reduction_factor

        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            ff_in_dim = embed_dim
        else:
            ff_in_dim = in_dim

        # Encoder
        # NOTE: can be simply replaced by a BiLSTM?
        # so far I use sinsy like architecture
        self.ff = nn.Sequential(
            nn.Linear(ff_in_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(ff_hidden_dim, conv_hidden_dim, kernel_size=7, padding=0),
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

        self.lstm = nn.LSTM(
            conv_hidden_dim,
            lstm_hidden_dim,
            num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.0,
        )

        # Autoregressive decoder
        decoder_in_dim = 2 * lstm_hidden_dim
        self.decoder = TacotronNonAttentiveDecoder(
            in_dim=decoder_in_dim,
            out_dim=out_dim,
            layers=decoder_layers,
            hidden_dim=decoder_hidden_dim,
            prenet_layers=prenet_layers,
            prenet_hidden_dim=prenet_hidden_dim,
            prenet_dropout=prenet_dropout,
            zoneout=zoneout,
            reduction_factor=reduction_factor,
            downsample_by_conv=downsample_by_conv,
            eval_dropout=eval_dropout,
            prenet_noise_std=prenet_noise_std,
        )

        if postnet_layers > 0:
            self.postnet = TacotronPostnet(
                out_dim,
                layers=postnet_layers,
                channels=postnet_channels,
                kernel_size=postnet_kernel_size,
                dropout=postnet_dropout,
            )
        else:
            self.postnet = None
        init_weights(self, init_type)

    def is_autoregressive(self):
        return self.decoder.is_autoregressive()

    def forward(self, x, lengths=None, y=None):
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.to("cpu")

        if self.embed_dim is not None:
            x_first, x_ph_onehot, x_last = torch.split(
                x,
                [
                    self.in_ph_start_idx,
                    self.num_vocab,
                    self.in_dim - self.num_vocab - self.in_ph_start_idx,
                ],
                dim=-1,
            )
            x_ph = torch.argmax(x_ph_onehot, dim=-1)
            # Make sure to have one-hot vector
            assert (x_ph_onehot.sum(-1) <= 1).all()
            x = self.emb(x_ph) + self.fc_in(torch.cat([x_first, x_last], dim=-1))

        out = self.ff(x)
        out = self.conv(out.transpose(1, 2)).transpose(1, 2)
        sequence = pack_padded_sequence(out, lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        outs = self.decoder(out, lengths, y)

        if self.postnet is not None:
            # NOTE: `outs.clone()`` is necessary to compute grad on both outs and outs_fine
            outs_fine = outs + self.postnet(outs.transpose(1, 2).clone()).transpose(
                1, 2
            )
            return [outs, outs_fine]
        else:
            return outs

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self, x=x, lengths=lengths, reduction_factor=self.reduction_factor
        )


class BiLSTMMDNNonAttentiveDecoder(BaseModel):
    """NonAttentiveDecoder + BiLSTM (MDN version)"""

    def __init__(
        self,
        in_dim=512,
        ff_hidden_dim=2048,
        conv_hidden_dim=1024,
        lstm_hidden_dim=256,
        num_lstm_layers=2,
        out_dim=80,
        decoder_layers=2,
        decoder_hidden_dim=1024,
        prenet_layers=2,
        prenet_hidden_dim=256,
        prenet_dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        downsample_by_conv=False,
        num_gaussians=8,
        sampling_mode="mean",
        in_ph_start_idx: int = 1,
        in_ph_end_idx: int = 50,
        embed_dim=None,
        init_type="none",
        eval_dropout=True,
        prenet_noise_std=0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim
        self.reduction_factor = reduction_factor

        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            ff_in_dim = embed_dim
        else:
            ff_in_dim = in_dim

        # Encoder
        # NOTE: can be simply replaced by a BiLSTM?
        # so far I use sinsy like architecture
        self.ff = nn.Sequential(
            nn.Linear(ff_in_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(ff_hidden_dim, conv_hidden_dim, kernel_size=7, padding=0),
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

        self.lstm = nn.LSTM(
            conv_hidden_dim,
            lstm_hidden_dim,
            num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.0,
        )

        # Autoregressive decoder
        decoder_in_dim = 2 * lstm_hidden_dim
        self.decoder = MDNNonAttentiveDecoder(
            in_dim=decoder_in_dim,
            out_dim=out_dim,
            layers=decoder_layers,
            hidden_dim=decoder_hidden_dim,
            prenet_layers=prenet_layers,
            prenet_hidden_dim=prenet_hidden_dim,
            prenet_dropout=prenet_dropout,
            zoneout=zoneout,
            reduction_factor=reduction_factor,
            downsample_by_conv=downsample_by_conv,
            num_gaussians=num_gaussians,
            sampling_mode=sampling_mode,
            eval_dropout=eval_dropout,
            prenet_noise_std=prenet_noise_std,
        )
        init_weights(self, init_type)

    def is_autoregressive(self):
        return self.decoder.is_autoregressive()

    def prediction_type(self):
        return self.decoder.prediction_type()

    def forward(self, x, lengths=None, y=None):
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.to("cpu")

        if self.embed_dim is not None:
            x_first, x_ph_onehot, x_last = torch.split(
                x,
                [
                    self.in_ph_start_idx,
                    self.num_vocab,
                    self.in_dim - self.num_vocab - self.in_ph_start_idx,
                ],
                dim=-1,
            )
            x_ph = torch.argmax(x_ph_onehot, dim=-1)
            # Make sure to have one-hot vector
            assert (x_ph_onehot.sum(-1) <= 1).all()
            x = self.emb(x_ph) + self.fc_in(torch.cat([x_first, x_last], dim=-1))

        out = self.ff(x)
        out = self.conv(out.transpose(1, 2)).transpose(1, 2)
        sequence = pack_padded_sequence(out, lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        outs = self.decoder(out, lengths, y)

        return outs

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self,
            x=x,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            mdn=True,
        )


class ResF0NonAttentiveDecoder(BaseModel):
    """Duration-informed Tacotron with residual f0 prediction.

    Single-stream architecture.

    Args:
        in_dim (int) : dimension of encoder hidden layer
        out_dim (int) : dimension of output
        layers (int) : number of LSTM layers
        hidden_dim (int) : dimension of hidden layer
        prenet_layers (int) : number of pre-net layers
        prenet_hidden_dim (int) : dimension of pre-net hidden layer
        prenet_dropout (float) : dropout rate of pre-net
        zoneout (float) : zoneout rate
        reduction_factor (int) : reduction factor
        attention_hidden_dim (int) : dimension of attention hidden layer
        attention_conv_channel (int) : number of attention convolution channels
        attention_conv_kernel_size (int) : kernel size of attention convolution
        downsample_by_conv (bool) : if True, downsample by convolution
    """

    def __init__(
        self,
        in_dim=512,
        out_dim=80,
        layers=2,
        hidden_dim=1024,
        prenet_layers=2,
        prenet_hidden_dim=256,
        prenet_dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        downsample_by_conv=False,
        scaled_tanh=True,
        # NOTE: you must carefully set the following parameters
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        init_type="none",
        eval_dropout=True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.reduction_factor = reduction_factor
        self.prenet_dropout = prenet_dropout
        self.scaled_tanh = scaled_tanh
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

        if prenet_layers > 0:
            self.prenet = Prenet(
                out_dim,
                prenet_layers,
                prenet_hidden_dim,
                prenet_dropout,
                eval_dropout=eval_dropout,
            )
            lstm_in_dim = in_dim + prenet_hidden_dim
        else:
            self.prenet = None
            prenet_hidden_dim = 0
            lstm_in_dim = in_dim + out_dim

        self.lstm = nn.ModuleList()
        for layer in range(layers):
            lstm = nn.LSTMCell(
                lstm_in_dim if layer == 0 else hidden_dim,
                hidden_dim,
            )
            self.lstm += [ZoneOutCell(lstm, zoneout)]

        proj_in_dim = in_dim + hidden_dim
        self.feat_out = nn.Linear(proj_in_dim, out_dim * reduction_factor, bias=False)

        if reduction_factor > 1 and downsample_by_conv:
            self.conv_downsample = nn.Conv1d(
                in_dim,
                in_dim,
                kernel_size=reduction_factor,
                stride=reduction_factor,
                groups=in_dim,
            )
        else:
            self.conv_downsample = None
        init_weights(self, init_type)

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def is_autoregressive(self):
        return True

    def forward(self, encoder_outs, in_lens, decoder_targets=None):
        """Forward step

        Args:
            encoder_outs (torch.Tensor): encoder outputs (B, T, C)
            in_lens (torch.Tensor): input lengths
            decoder_targets (torch.Tensor): decoder targets for teacher-forcing. (B, T, C)

        Returns:
            torch.Tensor: the output (B, C, T)
        """
        is_inference = decoder_targets is None
        if not is_inference:
            assert encoder_outs.shape[1] == decoder_targets.shape[1]

        # Denormalize lf0 from input musical score
        lf0_score = encoder_outs[:, :, self.in_lf0_idx].unsqueeze(-1)
        lf0_score_denorm = (
            lf0_score * (self.in_lf0_max - self.in_lf0_min) + self.in_lf0_min
        )
        # (B, T, C) -> (B, C, T)
        lf0_score_denorm = lf0_score_denorm.transpose(1, 2)

        # To avoid unbounded residual f0 that would potentially cause artifacts,
        # let's constrain the residual F0 to be in a certain range by the scaled tanh
        residual_f0_max_cent = 600
        max_lf0_ratio = residual_f0_max_cent * np.log(2) / 1200

        # Adjust number of frames according to the reduction factor
        # (B, Lmax, out_dim) ->  (B, Lmax/r, out_dim)
        if self.reduction_factor > 1 and not is_inference:
            decoder_targets = decoder_targets[
                :, self.reduction_factor - 1 :: self.reduction_factor
            ]
        if self.reduction_factor > 1:
            if self.conv_downsample is not None:
                encoder_outs = self.conv_downsample(
                    encoder_outs.transpose(1, 2)
                ).transpose(1, 2)
            else:
                encoder_outs = encoder_outs[
                    :, self.reduction_factor - 1 :: self.reduction_factor
                ]

        h_list, c_list = [], []
        for _ in range(len(self.lstm)):
            h_list.append(self._zero_state(encoder_outs))
            c_list.append(self._zero_state(encoder_outs))

        go_frame = encoder_outs.new_zeros(encoder_outs.size(0), self.out_dim)
        prev_out = go_frame

        if not is_inference and self.prenet is not None:
            prenet_outs = self.prenet(decoder_targets)

        outs = []
        lf0_residuals = []
        for t in range(encoder_outs.shape[1]):
            # Pre-Net
            if self.prenet is not None:
                if is_inference:
                    prenet_out = self.prenet(prev_out)
                else:
                    prenet_out = prenet_outs[:, t, :]
            else:
                prenet_out = F.dropout(prev_out, self.prenet_dropout, training=True)

            # LSTM
            xs = torch.cat([encoder_outs[:, t], prenet_out], dim=1)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](
                    h_list[i - 1], (h_list[i], c_list[i])
                )
            # Output
            hcs = torch.cat([h_list[-1], encoder_outs[:, t]], dim=1)
            out = self.feat_out(hcs).view(encoder_outs.size(0), self.out_dim, -1)
            # Residual F0
            if self.scaled_tanh:
                lf0_residual = max_lf0_ratio * torch.tanh(
                    out[:, self.out_lf0_idx, :]
                ).unsqueeze(1)
            else:
                lf0_residual = out[:, self.out_lf0_idx, :].unsqueeze(1)

            # Residual connection in the denormalized f0 domain
            lf0_score_denorm_t = lf0_score_denorm[
                :, :, t * self.reduction_factor : (t + 1) * self.reduction_factor
            ]
            lf0_pred_denorm = lf0_score_denorm_t + lf0_residual
            # Back to normalized f0
            lf0_pred = (lf0_pred_denorm - self.out_lf0_mean) / self.out_lf0_scale
            out[:, self.out_lf0_idx, :] = lf0_pred.squeeze(1)

            outs.append(out)
            lf0_residuals.append(lf0_residual)

            # Update decoder input for the next time step
            if is_inference:
                prev_out = outs[-1][:, :, -1]  # (1, out_dim)
            else:
                # Teacher forcing
                prev_out = decoder_targets[:, t, :]

        outs = torch.cat(outs, dim=2)  # (B, out_dim, Lmax)
        lf0_residuals = torch.cat(lf0_residuals, dim=2)  # (B, out_dim, Lmax)

        # (B, C, T) -> (B, T, C)
        outs = outs.transpose(1, 2)
        lf0_residuals = lf0_residuals.transpose(1, 2)

        return outs, lf0_residuals

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self, x=x, lengths=lengths, reduction_factor=self.reduction_factor
        )


class MDNResF0NonAttentiveDecoder(BaseModel):
    def __init__(
        self,
        in_dim=512,
        out_dim=80,
        layers=2,
        hidden_dim=1024,
        prenet_layers=2,
        prenet_hidden_dim=256,
        prenet_dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        downsample_by_conv=False,
        scaled_tanh=True,
        num_gaussians=4,
        sampling_mode="mean",
        # NOTE: you must carefully set the following parameters
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        init_type="none",
        eval_dropout=True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.reduction_factor = reduction_factor
        self.prenet_dropout = prenet_dropout
        self.scaled_tanh = scaled_tanh
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale
        self.num_gaussians = num_gaussians
        self.sampling_mode = sampling_mode
        assert sampling_mode in ["mean", "random"]

        if prenet_layers > 0:
            self.prenet = Prenet(
                out_dim,
                prenet_layers,
                prenet_hidden_dim,
                prenet_dropout,
                eval_dropout=eval_dropout,
            )
            lstm_in_dim = in_dim + prenet_hidden_dim
        else:
            self.prenet = None
            prenet_hidden_dim = 0
            lstm_in_dim = in_dim + out_dim

        self.lstm = nn.ModuleList()
        for layer in range(layers):
            lstm = nn.LSTMCell(
                lstm_in_dim if layer == 0 else hidden_dim,
                hidden_dim,
            )
            self.lstm += [ZoneOutCell(lstm, zoneout)]

        proj_in_dim = in_dim + hidden_dim
        self.feat_out = MDNLayer(
            proj_in_dim,
            out_dim * reduction_factor,
            num_gaussians=num_gaussians,
            dim_wise=True,
        )

        if reduction_factor > 1 and downsample_by_conv:
            self.conv_downsample = nn.Conv1d(
                in_dim,
                in_dim,
                kernel_size=reduction_factor,
                stride=reduction_factor,
                groups=in_dim,
            )
        else:
            self.conv_downsample = None
        init_weights(self, init_type)

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def is_autoregressive(self):
        return True

    def forward(self, encoder_outs, in_lens, decoder_targets=None):
        """Forward step

        Args:
            encoder_outs (torch.Tensor): encoder outputs (B, T, C)
            in_lens (torch.Tensor): input lengths
            decoder_targets (torch.Tensor): decoder targets for teacher-forcing. (B, T, C)

        Returns:
            torch.Tensor: the output (B, C, T)
        """
        is_inference = decoder_targets is None
        if not is_inference:
            assert encoder_outs.shape[1] == decoder_targets.shape[1]

        # Denormalize lf0 from input musical score
        lf0_score = encoder_outs[:, :, self.in_lf0_idx].unsqueeze(-1)
        lf0_score_denorm = (
            lf0_score * (self.in_lf0_max - self.in_lf0_min) + self.in_lf0_min
        )
        # (B, T, C) -> (B, C, T)
        # (B, T, C)
        # lf0_score_denorm = lf0_score_denorm.transpose(1, 2)

        # To avoid unbounded residual f0 that would potentially cause artifacts,
        # let's constrain the residual F0 to be in a certain range by the scaled tanh
        residual_f0_max_cent = 600
        max_lf0_ratio = residual_f0_max_cent * np.log(2) / 1200

        # Adjust number of frames according to the reduction factor
        # (B, Lmax, out_dim) ->  (B, Lmax/r, out_dim)
        if self.reduction_factor > 1 and not is_inference:
            decoder_targets = decoder_targets[
                :, self.reduction_factor - 1 :: self.reduction_factor
            ]
        if self.reduction_factor > 1:
            if self.conv_downsample is not None:
                encoder_outs = self.conv_downsample(
                    encoder_outs.transpose(1, 2)
                ).transpose(1, 2)
            else:
                encoder_outs = encoder_outs[
                    :, self.reduction_factor - 1 :: self.reduction_factor
                ]

        h_list, c_list = [], []
        for _ in range(len(self.lstm)):
            h_list.append(self._zero_state(encoder_outs))
            c_list.append(self._zero_state(encoder_outs))

        go_frame = encoder_outs.new_zeros(encoder_outs.size(0), self.out_dim)
        prev_out = go_frame

        if not is_inference and self.prenet is not None:
            prenet_outs = self.prenet(decoder_targets)

        mus = []
        log_pis = []
        log_sigmas = []
        lf0_residuals = []
        mus_inf = []
        for t in range(encoder_outs.shape[1]):
            # Pre-Net
            if self.prenet is not None:
                if is_inference:
                    prenet_out = self.prenet(prev_out)
                else:
                    prenet_out = prenet_outs[:, t, :]
            else:
                prenet_out = F.dropout(prev_out, self.prenet_dropout, training=True)

            # LSTM
            xs = torch.cat([encoder_outs[:, t], prenet_out], dim=1)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](
                    h_list[i - 1], (h_list[i], c_list[i])
                )
            # Output
            hcs = torch.cat([h_list[-1], encoder_outs[:, t]], dim=1)
            log_pi, log_sigma, mu = self.feat_out(hcs.unsqueeze(1))

            # (B, reduction_factor, num_gaussians, out_dim)
            log_pi = (
                log_pi.transpose(1, 2)
                .view(encoder_outs.size(0), self.num_gaussians, -1, self.out_dim)
                .transpose(1, 2)
            )
            log_sigma = (
                log_sigma.transpose(1, 2)
                .view(encoder_outs.size(0), self.num_gaussians, -1, self.out_dim)
                .transpose(1, 2)
            )
            mu = (
                mu.transpose(1, 2)
                .view(encoder_outs.size(0), self.num_gaussians, -1, self.out_dim)
                .transpose(1, 2)
            )

            # Residual F0
            # (B, reduction_factor, num_gaussians)
            if self.scaled_tanh:
                lf0_residual = max_lf0_ratio * torch.tanh(mu[:, :, :, self.out_lf0_idx])
            else:
                lf0_residual = mu[:, :, :, self.out_lf0_idx]

            # Residual connection in the denormalized f0 domain
            lf0_score_denorm_t = lf0_score_denorm[
                :, t * self.reduction_factor : (t + 1) * self.reduction_factor, :
            ]
            # NOTE: broadcast against num_gaussians axis
            # (B, 1, 1) + (B, 1, num_gaussians) -> (B, 1, num_gaussians)
            lf0_pred_denorm = lf0_score_denorm_t + lf0_residual
            # Back to normalized f0
            lf0_pred = (lf0_pred_denorm - self.out_lf0_mean) / self.out_lf0_scale
            mu[:, :, :, self.out_lf0_idx] = lf0_pred

            mus.append(mu)
            log_pis.append(log_pi)
            log_sigmas.append(log_sigma)
            lf0_residuals.append(lf0_residual)

            # Update decoder input for the next time step
            if is_inference:
                # (B, reduction_factor, out_dim)
                if self.sampling_mode == "mean":
                    _, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
                elif self.sampling_mode == "random":
                    mu = mdn_get_sample(log_pi, log_sigma, mu)
                # Feed last sample for the feedback loop
                prev_out = mu[:, -1]

                mus_inf.append(mu)
            else:
                # Teacher forcing
                prev_out = decoder_targets[:, t, :]

        # (B, T, G, out_dim)
        mus = torch.cat(mus, dim=1)
        log_pis = torch.cat(log_pis, dim=1)
        log_sigmas = torch.cat(log_sigmas, dim=1)
        # (B, T, num_gaussians)
        lf0_residuals = torch.cat(lf0_residuals, dim=1)

        if is_inference:
            mu = torch.cat(mus_inf, dim=1)
            # TODO: may need to track sigma. For now we only use mu
            return mu, mu
        else:
            return (log_pis, log_sigmas, mus), lf0_residuals

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self,
            x=x,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            mdn=True,
        )


class BiLSTMResF0NonAttentiveDecoder(BaseModel):
    def __init__(
        self,
        in_dim=512,
        ff_hidden_dim=2048,
        conv_hidden_dim=1024,
        lstm_hidden_dim=256,
        num_lstm_layers=2,
        dropout=0.0,
        out_dim=80,
        decoder_layers=2,
        decoder_hidden_dim=1024,
        prenet_layers=2,
        prenet_hidden_dim=256,
        prenet_dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        downsample_by_conv=False,
        scaled_tanh=True,
        # NOTE: you must carefully set the following parameters
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        use_mdn=False,
        num_gaussians=4,
        sampling_mode="mean",
        in_ph_start_idx: int = 1,
        in_ph_end_idx: int = 50,
        embed_dim=None,
        init_type="none",
    ):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale
        self.use_mdn = use_mdn
        self.in_dim = in_dim
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim

        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            in_ff_dim = embed_dim
        else:
            in_ff_dim = in_dim

        # Encoder
        # NOTE: can be simply replaced by a BiLSTM?
        # so far I use sinsy like architecture
        self.ff = nn.Sequential(
            nn.Linear(in_ff_dim, ff_hidden_dim),
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

        self.lstm = nn.LSTM(
            conv_hidden_dim,
            lstm_hidden_dim,
            num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        # Autoregressive decoder
        decoder_in_dim = 2 * lstm_hidden_dim + 1
        # NOTE: hard code in_lf0_idx to -1
        cls = MDNResF0NonAttentiveDecoder if use_mdn else ResF0NonAttentiveDecoder
        if use_mdn:
            ex_kwargs = {"num_gaussians": num_gaussians, "sampling_mode": sampling_mode}
        else:
            ex_kwargs = {}
        self.decoder = cls(
            in_dim=decoder_in_dim,
            out_dim=out_dim,
            layers=decoder_layers,
            hidden_dim=decoder_hidden_dim,
            prenet_layers=prenet_layers,
            prenet_hidden_dim=prenet_hidden_dim,
            prenet_dropout=prenet_dropout,
            zoneout=zoneout,
            reduction_factor=reduction_factor,
            downsample_by_conv=downsample_by_conv,
            scaled_tanh=scaled_tanh,
            in_lf0_idx=-1,
            in_lf0_min=in_lf0_min,
            in_lf0_max=in_lf0_max,
            out_lf0_idx=out_lf0_idx,
            out_lf0_mean=out_lf0_mean,
            out_lf0_scale=out_lf0_scale,
            **ex_kwargs,
        )
        init_weights(self, init_type)

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.decoder, "out_lf0_mean"):
            self.decoder.in_lf0_min = self.in_lf0_min
            self.decoder.in_lf0_max = self.in_lf0_max
            self.decoder.out_lf0_mean = self.out_lf0_mean
            self.decoder.out_lf0_scale = self.out_lf0_scale

    def is_autoregressive(self):
        return self.decoder.is_autoregressive()

    def prediction_type(self):
        return (
            PredictionType.PROBABILISTIC
            if self.use_mdn
            else PredictionType.DETERMINISTIC
        )

    def forward(self, x, lengths=None, y=None):
        self._set_lf0_params()
        lf0_score = x[:, :, self.in_lf0_idx].unsqueeze(-1)

        if self.embed_dim is not None:
            x_first, x_ph_onehot, x_last = torch.split(
                x,
                [
                    self.in_ph_start_idx,
                    self.num_vocab,
                    self.in_dim - self.num_vocab - self.in_ph_start_idx,
                ],
                dim=-1,
            )
            x_ph = torch.argmax(x_ph_onehot, dim=-1)
            # Make sure to have one-hot vector
            assert (x_ph_onehot.sum(-1) <= 1).all()
            x = self.emb(x_ph) + self.fc_in(torch.cat([x_first, x_last], dim=-1))

        if isinstance(lengths, torch.Tensor):
            lengths = lengths.to("cpu")

        out = self.ff(x)
        out = torch.cat([out, lf0_score], dim=-1)

        out = self.conv(out.transpose(1, 2)).transpose(1, 2)
        sequence = pack_padded_sequence(out, lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)

        # NOTE: need to concat the lf0 score to the output of the lstm to tell
        # the decoder the lf0
        out = torch.cat([out, lf0_score], dim=-1)

        outs, lf0_residual = self.decoder(out, lengths, y)

        return outs, lf0_residual

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self,
            x=x,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            mdn=self.use_mdn,
        )


def LSTMEncoder(*args, **kwargs):
    from nnsvs.model import LSTMEncoder

    return LSTMEncoder(*args, **kwargs)
