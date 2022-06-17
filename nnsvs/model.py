from warnings import warn

import torch
from nnsvs.base import BaseModel, PredictionType
from nnsvs.dsp import TrTimeInvFIRFilter
from nnsvs.layers.conv import ResnetBlock, WNConv1d
from nnsvs.layers.layer_norm import LayerNorm
from nnsvs.mdn import MDNLayer, mdn_get_most_probable_sigma_and_mu
from nnsvs.multistream import split_streams
from nnsvs.util import init_weights
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = [
    "FFN",
    "LSTMRNN",
    "LSTMRNNSAR",
    "MDN",
    "MDNv2",
    "RMDN",
    "Conv1dResnet",
    "Conv1dResnetMDN",
    "Conv1dResnetSAR",
    "FFConvLSTM",
    "VariancePredictor",
]


class Conv1dResnet(BaseModel):
    """Conv1d + Resnet

    The model is inspired by the MelGAN's model architecture (:cite:t:`kumar2019melgan`).
    MDN layer is added if use_mdn is True.

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        init_type (str): the type of weight initialization
        use_mdn (bool): whether to use MDN or not
        num_gaussians (int): the number of gaussians in MDN
        dim_wise (bool): whether to use dim-wise or not
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=4,
        init_type="none",
        use_mdn=False,
        num_gaussians=8,
        dim_wise=False,
        **kwargs,
    ):
        super().__init__()
        self.use_mdn = use_mdn

        if "dropout" in kwargs:
            warn(
                "dropout argument in Conv1dResnet is deprecated"
                " and will be removed in future versions"
            )

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
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor
            y (torch.Tensor): the optional target tensor

        Returns:
            torch.Tensor: the output tensor
        """
        out = self.model(x.transpose(1, 2)).transpose(1, 2)

        if self.use_mdn:
            return self.mdn_layer(out)
        else:
            return out

    def inference(self, x, lengths=None):
        """Inference step

        Find the most likely mean and variance if use_mdn is True

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor

        Returns:
            tuple: mean and variance of the output features
        """
        if self.use_mdn:
            log_pi, log_sigma, mu = self(x, lengths)
            sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
            return mu, sigma
        else:
            return self(x, lengths)


@torch.no_grad()
def _shallow_ar_inference(out, stream_sizes, analysis_filts):
    from torchaudio.functional import lfilter

    out_streams = split_streams(out, stream_sizes)
    # back to conv1d friendly (B, C, T) format
    out_streams = map(lambda x: x.transpose(1, 2), out_streams)

    out_syn = []
    for sidx, os in enumerate(out_streams):
        out_stream_syn = torch.zeros_like(os)
        a = analysis_filts[sidx].get_filt_coefs()
        # apply IIR filter for each dimiesion
        for idx in range(os.shape[1]):
            # NOTE: scipy.signal.lfilter accespts b, a in order,
            # but torchaudio expect the oppsite; a, b in order
            ai = a[idx].view(-1).flip(0)
            bi = torch.zeros_like(ai)
            bi[0] = 1
            out_stream_syn[:, idx, :] = lfilter(os[:, idx, :], ai, bi, clamp=False)
        out_syn += [out_stream_syn]

    out_syn = torch.cat(out_syn, 1)
    return out_syn.transpose(1, 2)


class Conv1dResnetSAR(Conv1dResnet):
    """Conv1dResnet with shallow AR structure

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        stream_sizes (list): Stream sizes
        ar_orders (list): Filter dimensions for each stream.
        init_type (str): the type of weight initialization
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=4,
        stream_sizes=None,
        ar_orders=None,
        init_type="none",
        **kwargs,
    ):
        super().__init__(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=num_layers
        )

        if "dropout" in kwargs:
            warn(
                "dropout argument in Conv1dResnetSAR is deprecated"
                " and will be removed in future versions"
            )

        if stream_sizes is None:
            stream_sizes = [180, 3, 1, 15]
        if ar_orders is None:
            ar_orders = [20, 200, 20, 20]
        self.stream_sizes = stream_sizes

        init_weights(self, init_type)

        self.analysis_filts = nn.ModuleList()
        for s, K in zip(stream_sizes, ar_orders):
            self.analysis_filts += [TrTimeInvFIRFilter(s, K + 1)]

    def preprocess_target(self, y):
        assert sum(self.stream_sizes) == y.shape[-1]
        ys = split_streams(y, self.stream_sizes)
        for idx, yi in enumerate(ys):
            ys[idx] = self.analysis_filts[idx](yi.transpose(1, 2)).transpose(1, 2)
        return torch.cat(ys, -1)

    def inference(self, x, lengths=None, y=None):
        out = self.model(x.transpose(1, 2)).transpose(1, 2)
        return _shallow_ar_inference(out, self.stream_sizes, self.analysis_filts)


class FFN(BaseModel):
    """Feed-forward network

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        dropout (float): dropout rate
        init_type (str): the type of weight initialization
        last_sigmoid (bool): whether to apply sigmoid on the output
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=2,
        dropout=0.0,
        init_type="none",
        last_sigmoid=False,
    ):
        super(FFN, self).__init__()
        self.first_linear = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.last_linear = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.last_sigmoid = last_sigmoid
        init_weights(self, init_type)

    def forward(self, x, lengths=None, y=None):
        """Forward step

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor
            y (torch.Tensor): the optional target tensor

        Returns:
            torch.Tensor: the output tensor
        """
        h = self.relu(self.first_linear(x))
        for hl in self.hidden_layers:
            h = self.dropout(self.relu(hl(h)))
        out = self.last_linear(h)
        out = torch.sigmoid(out) if self.last_sigmoid else out
        return out


# For compatibility
FeedForwardNet = FFN


class LSTMRNN(BaseModel):
    """LSTM-based recurrent neural network

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        bidirectional (bool): whether to use bidirectional LSTM
        dropout (float): dropout rate
        init_type (str): the type of weight initialization
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=1,
        bidirectional=True,
        dropout=0.0,
        init_type="none",
    ):
        super(LSTMRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            in_dim,
            hidden_dim,
            num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )
        self.hidden2out = nn.Linear(self.num_direction * self.hidden_dim, out_dim)
        init_weights(self, init_type)

    def forward(self, x, lengths, y=None):
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
        x = pack_padded_sequence(x, lengths, batch_first=True)
        out, _ = self.lstm(x)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.hidden2out(out)
        return out


class LSTMRNNSAR(LSTMRNN):
    """LSTM-RNN with shallow AR structure

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        bidirectional (bool): whether to use bidirectional LSTM
        dropout (float): dropout rate
        stream_sizes (list): Stream sizes
        ar_orders (list): Filter dimensions for each stream.
        init_type (str): the type of weight initialization
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=1,
        bidirectional=True,
        dropout=0.0,
        stream_sizes=None,
        ar_orders=None,
        init_type="none",
    ):
        super().__init__(
            in_dim, hidden_dim, out_dim, num_layers, bidirectional, dropout, init_type
        )
        if stream_sizes is None:
            stream_sizes = [180, 3, 1, 15]
        if ar_orders is None:
            ar_orders = [20, 200, 20, 20]

        self.stream_sizes = stream_sizes
        self.analysis_filts = nn.ModuleList()
        for s, K in zip(stream_sizes, ar_orders):
            self.analysis_filts += [TrTimeInvFIRFilter(s, K + 1)]

    def preprocess_target(self, y):
        assert sum(self.stream_sizes) == y.shape[-1]
        ys = split_streams(y, self.stream_sizes)
        for idx, yi in enumerate(ys):
            ys[idx] = self.analysis_filts[idx](yi.transpose(1, 2)).transpose(1, 2)
        return torch.cat(ys, -1)

    def inference(self, x, lengths=None, y=None):
        out = self.forward(x, lengths)
        return _shallow_ar_inference(out, self.stream_sizes, self.analysis_filts)


class RMDN(BaseModel):
    """RNN-based mixture density networks (MDN)

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        bidirectional (bool): whether to use bidirectional LSTM
        dropout (float): dropout rate
        num_gaussians (int): the number of gaussians
        dim_wise (bool): whether to use dimension-wise or not
        init_type (str): the type of weight initialization
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=1,
        bidirectional=True,
        dropout=0.0,
        num_gaussians=8,
        dim_wise=False,
        init_type="none",
    ):
        super(RMDN, self).__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )
        self.mdn = MDNLayer(
            in_dim=self.num_direction * hidden_dim,
            out_dim=out_dim,
            num_gaussians=num_gaussians,
            dim_wise=dim_wise,
        )
        init_weights(self, init_type)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, x, lengths, y=None):
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
        out = self.linear(x)
        sequence = pack_padded_sequence(self.relu(out), lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.mdn(out)
        return out

    def inference(self, x, lengths=None):
        """Inference step

        Find the most likely mean and variance

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor

        Returns:
            tuple: mean and variance of the output features
        """
        log_pi, log_sigma, mu = self.forward(x, lengths)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        return mu, sigma


class MDN(BaseModel):
    """Mixture density networks (MDN) with FFN

    .. warning::

        It is recommended to use MDNv2 instead, unless you want to
        fine-turn from a old checkpoint of MDN.

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        num_gaussians (int): the number of gaussians
        dim_wise (bool): whether to use dimension-wise or not
        init_type (str): the type of weight initialization
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=1,
        num_gaussians=8,
        dim_wise=False,
        init_type="none",
        **kwargs,
    ):
        super(MDN, self).__init__()
        if "dropout" in kwargs:
            warn(
                "dropout argument in MDN is deprecated"
                " and will be removed in future versions"
            )
        model = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        if num_layers > 1:
            for _ in range(num_layers - 1):
                model += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        model += [
            MDNLayer(
                in_dim=hidden_dim,
                out_dim=out_dim,
                num_gaussians=num_gaussians,
                dim_wise=dim_wise,
            )
        ]
        self.model = nn.Sequential(*model)
        init_weights(self, init_type)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, x, lengths=None, y=None):
        """Forward step

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor
            y (torch.Tensor): the optional target tensor

        Returns:
            torch.Tensor: the output tensor
        """
        return self.model(x)

    def inference(self, x, lengths=None):
        """Inference step

        Find the most likely mean and variance

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor

        Returns:
            tuple: mean and variance of the output features
        """
        log_pi, log_sigma, mu = self.forward(x, lengths)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        return mu, sigma


class MDNv2(BaseModel):
    """Mixture density networks (MDN) with FFN

    MDN (v1) + Dropout

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        dropout (float): dropout rate
        num_gaussians (int): the number of gaussians
        dim_wise (bool): whether to use dimension-wise or not
        init_type (str): the type of weight initialization
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=1,
        dropout=0.5,
        num_gaussians=8,
        dim_wise=False,
        init_type="none",
    ):
        super(MDNv2, self).__init__()
        model = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        if num_layers > 1:
            for _ in range(num_layers - 1):
                model += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
        model += [
            MDNLayer(
                in_dim=hidden_dim,
                out_dim=out_dim,
                num_gaussians=num_gaussians,
                dim_wise=dim_wise,
            )
        ]
        self.model = nn.Sequential(*model)
        init_weights(self, init_type)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, x, lengths=None, y=None):
        """Forward step

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor
            y (torch.Tensor): the optional target tensor

        Returns:
            torch.Tensor: the output tensor
        """
        return self.model(x)

    def inference(self, x, lengths=None):
        """Inference step

        Find the most likely mean and variance

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor

        Returns:
            tuple: mean and variance of the output features
        """
        log_pi, log_sigma, mu = self.forward(x, lengths)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        return mu, sigma


class Conv1dResnetMDN(BaseModel):
    """Conv1dResnet with MDN output layer

    .. warning::

        Will be removed in v0.1.0. Use Conv1dResNet with ``use_mdn=True`` instead.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=4,
        num_gaussians=8,
        dim_wise=False,
        init_type="none",
        **kwargs,
    ):
        super().__init__()

        if "dropout" in kwargs:
            warn(
                "dropout argument in Conv1dResnet is deprecated"
                " and will be removed in future versions"
            )

        model = [
            Conv1dResnet(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                num_layers=num_layers,
            ),
            nn.ReLU(),
            MDNLayer(
                in_dim=hidden_dim,
                out_dim=out_dim,
                num_gaussians=num_gaussians,
                dim_wise=dim_wise,
            ),
        ]
        self.model = nn.Sequential(*model)
        init_weights(self, init_type)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, x, lengths=None, y=None):
        """Forward step

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor
            y (torch.Tensor): the optional target tensor

        Returns:
            torch.Tensor: the output tensor
        """
        return self.model(x)

    def inference(self, x, lengths=None):
        """Inference step

        Find the most likely mean and variance

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor

        Returns:
            tuple: mean and variance of the output features
        """
        log_pi, log_sigma, mu = self.forward(x, lengths)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        return mu, sigma


class FFConvLSTM(BaseModel):
    """FFN + Conv1d + LSTM

    A model proposed in :cite:t:`hono2021sinsy` without residual F0 prediction.

    Args:
        in_dim (int): the dimension of the input
        ff_hidden_dim (int): the dimension of the hidden state of the FFN
        conv_hidden_dim (int): the dimension of the hidden state of the conv1d
        lstm_hidden_dim (int): the dimension of the hidden state of the LSTM
        out_dim (int): the dimension of the output
        dropout (float): dropout rate
        num_lstm_layers (int): the number of layers of the LSTM
        bidirectional (bool): whether to use bidirectional LSTM
        init_type (str): the type of weight initialization
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
        init_type="none",
    ):
        super().__init__()

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

        num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            conv_hidden_dim,
            lstm_hidden_dim,
            num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        last_in_dim = num_direction * lstm_hidden_dim
        self.fc = nn.Linear(last_in_dim, out_dim)
        init_weights(self, init_type)

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

        out = self.ff(x)
        out = self.conv(out.transpose(1, 2)).transpose(1, 2)
        sequence = pack_padded_sequence(out, lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.fc(out)

        return out


class VariancePredictor(BaseModel):
    """Variance predictor in :cite:t:`ren2020fastspeech`.

    The model is composed of stacks of Conv1d + ReLU + LayerNorm layers.
    The model can be used for duration or pitch prediction.

    Args:
        in_dim (int): the input dimension
        out_dim (int): the output dimension
        num_layers (int): the number of layers
        hidden_dim (int): the hidden dimension
        kernel_size (int): the kernel size
        dropout (float): the dropout rate
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
    ):
        super().__init__()
        self.use_mdn = use_mdn

        conv = nn.ModuleList()
        for idx in range(num_layers):
            in_channels = in_dim if idx == 0 else hidden_dim
            conv += [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        hidden_dim,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    nn.ReLU(),
                    LayerNorm(hidden_dim, dim=1),
                    nn.Dropout(dropout),
                )
            ]
        self.conv = nn.Sequential(*conv)
        if self.use_mdn:
            self.mdn_layer = MDNLayer(
                hidden_dim, out_dim, num_gaussians=num_gaussians, dim_wise=dim_wise
            )
        else:
            self.fc = nn.Linear(hidden_dim, out_dim)

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
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)

        if self.use_mdn:
            return self.mdn_layer(out)
        else:
            return self.fc(out)

    def inference(self, x, lengths=None):
        """Inference step

        Find the most likely mean and variance if use_mdn is True

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor

        Returns:
            tuple: mean and variance of the output features
        """
        if self.use_mdn:
            log_pi, log_sigma, mu = self(x, lengths)
            sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
            return mu, sigma
        else:
            return self(x, lengths)


# For backward compatibility
# Will be removed in v0.1.0


def ResF0Conv1dResnet(*args, **kwargs):
    from nnsvs.acoustic_models import ResF0Conv1dResnet

    return ResF0Conv1dResnet(*args, **kwargs)


def ResF0Conv1dResnetMDN(*args, **kwargs):
    from nnsvs.acoustic_models import ResF0Conv1dResnetMDN

    return ResF0Conv1dResnetMDN(*args, **kwargs)


def ResF0VariancePredictor(*args, **kwargs):
    from nnsvs.acoustic_models import ResF0VariancePredictor

    return ResF0VariancePredictor(*args, **kwargs)


def ResSkipF0FFConvLSTM(*args, **kwargs):
    from nnsvs.acoustic_models import ResSkipF0FFConvLSTM

    return ResSkipF0FFConvLSTM(*args, **kwargs)
