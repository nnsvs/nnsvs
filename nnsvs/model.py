from warnings import warn

import torch
from nnsvs.base import BaseModel, PredictionType
from nnsvs.dsp import TrTimeInvFIRFilter
from nnsvs.layers.conv import ResnetBlock, WNConv1d
from nnsvs.layers.layer_norm import LayerNorm
from nnsvs.mdn import MDNLayer, mdn_get_most_probable_sigma_and_mu
from nnsvs.multistream import split_streams
from nnsvs.transformer.attentions import sequence_mask
from nnsvs.transformer.encoder import Encoder as _TransformerEncoder
from nnsvs.util import init_weights
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = [
    "ExtractFromInput",
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
    "LSTMEncoder",
    "VariancePredictor",
    "TransformerEncoder",
]


class ExtractFromInput(BaseModel):
    """Extract a part of input

    This class doesn't have any learnable parameters.

    Args:
        start_idx (int): the start index of the input
        end_idx (int): the end index of the input

    """

    def __init__(self, start_idx, end_idx):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx

    def forward(self, x, lengths=None, y=None):
        """Forward pass"""
        # x: (B, T, C)
        return x[:, :, self.start_idx : self.end_idx]


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
        in_ph_start_idx (int): the start index of phoneme identity in a hed file
        in_ph_end_idx (int): the end index of phoneme identity in a hed file
        embed_dim (int): the dimension of the phoneme embedding
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
        in_ph_start_idx: int = 1,
        in_ph_end_idx: int = 50,
        embed_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_mdn = use_mdn
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim

        if "dropout" in kwargs:
            warn(
                "dropout argument in Conv1dResnet is deprecated"
                " and will be removed in future versions"
            )

        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            conv_in_dim = embed_dim
        else:
            conv_in_dim = in_dim

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(conv_in_dim, hidden_dim, kernel_size=7, padding=0),
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
    """Conv1dResnet with MDN output layer"""

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
            warn("dropout argument in Conv1dResnet is deprecated")

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
        use_mdn (bool): whether to use MDN or not
        dim_wise (bool): whether to use dimension-wise or not
        num_gaussians (int): the number of gaussians
        in_ph_start_idx (int): the start index of phoneme identity in a hed file
        in_ph_end_idx (int): the end index of phoneme identity in a hed file
        embed_dim (int): the dimension of the phoneme embedding
    """

    def __init__(
        self,
        in_dim,
        ff_hidden_dim=2048,
        conv_hidden_dim=1024,
        lstm_hidden_dim=256,
        out_dim=67,
        dropout=0.0,
        num_lstm_layers=2,
        bidirectional=True,
        init_type="none",
        use_mdn=False,
        dim_wise=True,
        num_gaussians=4,
        in_ph_start_idx: int = 1,
        in_ph_end_idx: int = 50,
        embed_dim=None,
        enforce_sorted=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim
        self.use_mdn = use_mdn
        self.enforce_sorted = enforce_sorted

        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            ff_in_dim = embed_dim
        else:
            ff_in_dim = in_dim

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
        if self.use_mdn:
            assert dim_wise
            self.fc = MDNLayer(
                in_dim=last_in_dim,
                out_dim=out_dim,
                num_gaussians=num_gaussians,
                dim_wise=dim_wise,
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
        sequence = pack_padded_sequence(
            out, lengths, batch_first=True, enforce_sorted=self.enforce_sorted
        )
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))

        return self.fc(out)

    def inference(self, x, lengths=None):
        if self.use_mdn:
            log_pi, log_sigma, mu = self(x, lengths)
            sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
            return mu, sigma
        else:
            return self(x, lengths)


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
        in_ph_start_idx (int): the start index of phoneme identity in a hed file
        in_ph_end_idx (int): the end index of phoneme identity in a hed file
        embed_dim (int): the dimension of the phoneme embedding
        mask_indices (list): the input feature indices to be masked.
            e.g., specify pitch_idx to mask pitch features.
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
        in_ph_start_idx: int = 1,
        in_ph_end_idx: int = 50,
        embed_dim=None,
        mask_indices=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_mdn = use_mdn
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim
        self.use_mdn = use_mdn
        self.mask_indices = mask_indices

        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            in_dim = embed_dim

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
        # Masking specified features
        if self.mask_indices is not None:
            for idx in self.mask_indices:
                x[:, :, idx] *= 0.0

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


class LSTMEncoder(BaseModel):
    """LSTM encoder

    A simple LSTM-based encoder

    Args:
        in_dim (int): the input dimension
        hidden_dim (int): the hidden dimension
        out_dim (int): the output dimension
        num_layers (int): the number of layers
        bidirectional (bool): whether to use bidirectional or not
        dropout (float): the dropout rate
        init_type (str): the initialization type
        in_ph_start_idx (int): the start index of phonetic context in a hed file
        in_ph_end_idx (int): the end index of phonetic context in a hed file
        embed_dim (int): the embedding dimension
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
        init_type: str = "none",
        in_ph_start_idx: int = 1,
        in_ph_end_idx: int = 50,
        embed_dim=None,
        enforce_sorted=True,
    ):
        super(LSTMEncoder, self).__init__()
        self.in_dim = in_dim
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim
        self.enforce_sorted = enforce_sorted

        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            lstm_in_dim = embed_dim
        else:
            lstm_in_dim = in_dim

        self.num_layers = num_layers
        num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            lstm_in_dim,
            hidden_dim,
            num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )
        self.hidden2out = nn.Linear(num_direction * hidden_dim, out_dim)
        init_weights(self, init_type)

    def forward(self, x, lengths, y=None):
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
        total_length = x.size(1)
        x = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=self.enforce_sorted
        )
        out, _ = self.lstm(x)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=total_length)
        out = self.hidden2out(out)
        return out


class TransformerEncoder(BaseModel):
    """Transformer encoder


    .. warning::

        So far this is not well tested. Maybe be removed in the future.

    Args:
        in_dim (int): the input dimension
        out_dim (int): the output dimension
        hidden_dim (int): the hidden dimension
        attention_dim (int): the attention dimension
        num_heads (int): the number of heads
        num_layers (int): the number of layers
        kernel_size (int): the kernel size
        dropout (float): the dropout rate
        reduction_factor (int): the reduction factor
        init_type (str): the initialization type
        downsample_by_conv (bool): whether to use convolutional downsampling or not
        in_ph_start_idx (int): the start index of phonetic context in a hed file
        in_ph_end_idx (int): the end index of phonetic context in a hed file
        embed_dim (int): the embedding dimension
    """

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
        in_ph_start_idx: int = 1,
        in_ph_end_idx: int = 50,
        embed_dim=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim

        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            self.fc = nn.Linear(embed_dim, hidden_dim)
        else:
            self.emb = None
            self.fc_in = None
            self.fc = nn.Linear(in_dim, hidden_dim)
        self.reduction_factor = reduction_factor
        self.encoder = _TransformerEncoder(
            hidden_channels=hidden_dim,
            filter_channels=attention_dim,
            n_heads=num_heads,
            n_layers=num_layers,
            kernel_size=kernel_size,
            p_dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_dim, out_dim * reduction_factor)

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

        for f in [self.fc_in, self.emb, self.fc, self.fc_out]:
            if f is not None:
                init_weights(f, init_type)

    def forward(self, x, lengths=None, y=None):
        """Forward pass

        Args:
            x (torch.Tensor): input tensor
            lengths (torch.Tensor): input sequence lengths
            y (torch.Tensor): target tensor (optional)

        Returns:
            torch.Tensor: output tensor
        """
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths).to(x.device)

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

        # Adjust lengths based on the reduction factor
        if self.reduction_factor > 1:
            lengths = (lengths / self.reduction_factor).long()
            if self.conv_downsample is not None:
                x = self.conv_downsample(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = x[:, self.reduction_factor - 1 :: self.reduction_factor]

        x = self.fc(x)
        # (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x_mask = sequence_mask(lengths, x.shape[2]).unsqueeze(1).to(x.device)
        x = self.encoder(x * x_mask, x_mask)

        # (B, C, T) -> (B, T, C)
        x = self.fc_out(x.transpose(1, 2)).view(x.shape[0], -1, self.out_dim)

        return x


# For backward compatibility


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
