from warnings import warn

import numpy as np
import torch
from nnsvs.base import BaseModel, PredictionType
from nnsvs.dsp import TrTimeInvFIRFilter
from nnsvs.layer_norm import LayerNorm
from nnsvs.mdn import MDNLayer, mdn_get_most_probable_sigma_and_mu
from nnsvs.multistream import split_streams
from nnsvs.tacotron.postnet import Postnet
from nnsvs.util import init_weights
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class Conv1dResnet(BaseModel):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=4,
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
            nn.ReflectionPad1d(3),
            WNConv1d(in_dim, hidden_dim, kernel_size=7, padding=0),
        ]
        for n in range(num_layers):
            model.append(ResnetBlock(hidden_dim, dilation=2 ** n))
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(hidden_dim, out_dim, kernel_size=7, padding=0),
        ]

        self.model = nn.Sequential(*model)
        init_weights(self, init_type)

    def forward(self, x, lengths=None):
        return self.model(x.transpose(1, 2)).transpose(1, 2)


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
        stream_sizes (list): Stream sizes
        ar_orders (list): Filter dimensions for each stream.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=4,
        dropout=0.0,
        stream_sizes=None,
        ar_orders=None,
        init_type="none",
    ):
        super().__init__(in_dim, hidden_dim, out_dim, num_layers, dropout, init_type)
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

    def inference(self, x, lengths=None):
        out = self.model(x.transpose(1, 2)).transpose(1, 2)
        return _shallow_ar_inference(out, self.stream_sizes, self.analysis_filts)


class FFN(BaseModel):
    def __init__(
        self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.0, init_type="none"
    ):
        super(FFN, self).__init__()
        self.first_linear = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.last_linear = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        init_weights(self, init_type)

    def forward(self, x, lengths=None):
        h = self.relu(self.first_linear(x))
        for hl in self.hidden_layers:
            h = self.dropout(self.relu(hl(h)))
        return self.last_linear(h)


# For compatibility
FeedForwardNet = FFN


class LSTMRNN(BaseModel):
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

    def forward(self, sequence, lengths):
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.to("cpu")
        sequence = pack_padded_sequence(sequence, lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.hidden2out(out)
        return out


class LSTMRNNSAR(LSTMRNN):
    """LSTM-RNN with shallow AR structure

    Args:
        stream_sizes (list): Stream sizes
        ar_orders (list): Filter dimensions for each stream.
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

    def inference(self, x, lengths=None):
        out = self.forward(x, lengths)
        return _shallow_ar_inference(out, self.stream_sizes, self.analysis_filts)


class RMDN(BaseModel):
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
            self.num_direction * hidden_dim, out_dim, num_gaussians, dim_wise
        )
        init_weights(self, init_type)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, x, lengths):
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.to("cpu")
        out = self.linear(x)
        sequence = pack_padded_sequence(self.relu(out), lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.mdn(out)
        return out

    def inference(self, x, lengths=None):
        log_pi, log_sigma, mu = self.forward(x, lengths)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        return mu, sigma


class MDN(BaseModel):
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
        model += [MDNLayer(hidden_dim, out_dim, num_gaussians, dim_wise)]
        self.model = nn.Sequential(*model)
        init_weights(self, init_type)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, x, lengths=None):
        return self.model(x)

    def inference(self, x, lengths=None):
        log_pi, log_sigma, mu = self.forward(x, lengths)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        return mu, sigma


class Conv1dResnetMDN(BaseModel):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=4,
        dropout=0.0,
        num_gaussians=8,
        dim_wise=False,
        init_type="none",
    ):
        super().__init__()
        model = [
            Conv1dResnet(in_dim, hidden_dim, hidden_dim, num_layers, dropout),
            nn.ReLU(),
            MDNLayer(hidden_dim, out_dim, num_gaussians, dim_wise),
        ]
        self.model = nn.Sequential(*model)
        init_weights(self, init_type)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, x, lengths=None):
        return self.model(x)

    def inference(self, x, lengths=None):
        log_pi, log_sigma, mu = self.forward(x, lengths)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        return mu, sigma


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
    ):
        super().__init__()
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(in_dim, hidden_dim, kernel_size=7, padding=0),
        ]
        for n in range(num_layers):
            model.append(ResnetBlock(hidden_dim, dilation=2 ** n))
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(hidden_dim, out_dim, kernel_size=7, padding=0),
        ]
        self.model = nn.Sequential(*model)
        init_weights(self, init_type)

    def forward(self, x, lengths=None):
        out = self.model(x.transpose(1, 2)).transpose(1, 2)

        lf0_pred, lf0_residual = predict_lf0_with_residual(
            x,
            out,
            self.in_lf0_idx,
            self.in_lf0_min,
            self.in_lf0_max,
            self.out_lf0_idx,
            self.out_lf0_mean,
            self.out_lf0_scale,
        )

        # Inject the predicted lf0 into the output features
        out[:, :, self.out_lf0_idx] = lf0_pred.squeeze(-1)

        return out, lf0_residual

    def inference(self, x, lengths=None):
        return self(x, lengths)[0]


class ResF0Conv1dResnetMDN(BaseModel):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=4,
        num_gaussians=2,
        dim_wise=False,
        # NOTE: you must carefully set the following parameters
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        init_type="none",
    ):
        super().__init__()
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(in_dim, hidden_dim, kernel_size=7, padding=0),
        ]
        for n in range(num_layers):
            model.append(ResnetBlock(hidden_dim, dilation=2 ** n))
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(hidden_dim, hidden_dim, kernel_size=7, padding=0),
            nn.ReLU(),
        ]
        self.model = nn.Sequential(*model)
        self.mdn_layer = MDNLayer(hidden_dim, out_dim, num_gaussians, dim_wise)
        init_weights(self, init_type)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, x, lengths=None):
        out = self.model(x.transpose(1, 2)).transpose(1, 2)
        log_pi, log_sigma, mu = self.mdn_layer(out)

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
        mu[:, :, :, self.out_lf0_idx] = lf0_pred.squeeze(-1)

        return (log_pi, log_sigma, mu), lf0_residual

    def inference(self, x, lengths=None):
        (log_pi, log_sigma, mu), _ = self.forward(x, lengths)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        return mu, sigma


class ResSkipF0FFConvLSTM(BaseModel):
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
    ):
        super().__init__()
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale
        self.skip_inputs = skip_inputs

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

        self.fc = nn.Linear(last_in_dim, out_dim)
        init_weights(self, init_type)

    def forward(self, x, lengths=None):
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
        out = self.fc(out)

        lf0_pred, lf0_residual = predict_lf0_with_residual(
            x,
            out,
            self.in_lf0_idx,
            self.in_lf0_min,
            self.in_lf0_max,
            self.out_lf0_idx,
            self.out_lf0_mean,
            self.out_lf0_scale,
        )

        # Inject the predicted lf0 into the output features
        out[:, :, self.out_lf0_idx] = lf0_pred.squeeze(-1)

        return out, lf0_residual

    def inference(self, x, lengths=None):
        return self(x, lengths)[0]


class FFConvLSTM(BaseModel):
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

    def forward(self, x, lengths=None):
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.to("cpu")

        out = self.ff(x)
        out = self.conv(out.transpose(1, 2)).transpose(1, 2)
        sequence = pack_padded_sequence(out, lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.fc(out)

        return out


class ResF0Conv1dResnetWithPostnet(ResF0Conv1dResnet):
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
        postnet_layers=2,
        postnet_dropout=0.1,
        postnet_kernel_size=5,
        postnet_channels=16,
    ):
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            in_lf0_idx=in_lf0_idx,
            in_lf0_min=in_lf0_min,
            in_lf0_max=in_lf0_max,
            out_lf0_idx=out_lf0_idx,
            out_lf0_mean=out_lf0_mean,
            out_lf0_scale=out_lf0_scale,
            init_type=init_type,
        )
        self.postnet = Postnet(
            out_dim,
            layers=postnet_layers,
            channels=postnet_channels,
            kernel_size=postnet_kernel_size,
            dropout=postnet_dropout,
        )
        init_weights(self, init_type)

    def forward(self, x, lengths=None):
        out, lf0_residual = super().forward(x, lengths)
        # NOTE: use detach here so that gradients from the postnet outputs only propagate to
        # the parameters of the postnet
        out_det = out.detach()
        post_out = self.postnet(out_det.transpose(1, 2)).transpose(1, 2)
        out_fine = out_det + post_out

        # special treatment for the lf0 prediction
        lf0_residual_fine = post_out[:, :, self.out_lf0_idx].unsqueeze(-1)
        # To avoid unbounded residual f0 that would potentially cause artifacts,
        # let's constrain the residual F0 to be in a certain range by the scaled tanh
        max_lf0_ratio = 600 * np.log(2) / 1200 / self.out_lf0_scale
        lf0_residual_fine = max_lf0_ratio * torch.tanh(lf0_residual_fine)
        out_fine[:, :, self.out_lf0_idx] = out_det[
            :, :, self.out_lf0_idx
        ] + lf0_residual_fine.squeeze(-1)

        outs = [out, out_fine]
        lf0_residuals = [lf0_residual, lf0_residual_fine]

        return outs, lf0_residuals

    def inference(self, x, lengths=None):
        return self(x, lengths)[0][-1]


class ResSkipF0FFConvLSTMWithPostnet(ResSkipF0FFConvLSTM):
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
        postnet_layers=5,
        postnet_dropout=0.1,
        postnet_kernel_size=5,
        postnet_channels=512,
    ):
        super().__init__(
            in_dim=in_dim,
            ff_hidden_dim=ff_hidden_dim,
            conv_hidden_dim=conv_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            num_lstm_layers=num_lstm_layers,
            bidirectional=bidirectional,
            in_lf0_idx=in_lf0_idx,
            in_lf0_min=in_lf0_min,
            in_lf0_max=in_lf0_max,
            out_lf0_idx=out_lf0_idx,
            out_lf0_mean=out_lf0_mean,
            out_lf0_scale=out_lf0_scale,
            skip_inputs=skip_inputs,
            init_type=init_type,
        )
        self.postnet = Postnet(
            out_dim,
            layers=postnet_layers,
            channels=postnet_channels,
            kernel_size=postnet_kernel_size,
            dropout=postnet_dropout,
        )
        init_weights(self, init_type)

    def forward(self, x, lengths=None):
        out, lf0_residual = super().forward(x, lengths)
        # NOTE: use detach here so that gradients from the postnet outputs only propagate to
        # the parameters of the postnet
        out_det = out.detach()
        post_out = self.postnet(out_det.transpose(1, 2)).transpose(1, 2)
        out_fine = out_det + post_out

        # special treatment for the lf0 prediction
        lf0_residual_fine = post_out[:, :, self.out_lf0_idx].unsqueeze(-1)
        # To avoid unbounded residual f0 that would potentially cause artifacts,
        # let's constrain the residual F0 to be in a certain range by the scaled tanh
        max_lf0_ratio = 600 * np.log(2) / 1200 / self.out_lf0_scale
        lf0_residual_fine = max_lf0_ratio * torch.tanh(lf0_residual_fine)
        out_fine[:, :, self.out_lf0_idx] = out_det[
            :, :, self.out_lf0_idx
        ] + lf0_residual_fine.squeeze(-1)

        outs = [out, out_fine]
        lf0_residuals = [lf0_residual, lf0_residual_fine]

        return outs, lf0_residuals

    def inference(self, x, lengths=None):
        return self(x, lengths)[0][-1]


class VariancePredictor(nn.Module):
    def __init__(
        self, in_dim, out_dim, num_layers=5, hidden_dim=256, kernel_size=5, dropout=0.5
    ):
        super().__init__()
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
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, lengths=None):
        x = self.conv(x.transpose(1, 2))
        return self.fc(x.transpose(1, 2))


class ResF0VariancePredictor(VariancePredictor):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_layers=5,
        hidden_dim=256,
        kernel_size=5,
        dropout=0.5,
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
        )
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

    def forward(self, x, lengths=None):
        out = self.conv(x.transpose(1, 2))
        out = self.fc(out.transpose(1, 2))

        lf0_pred, lf0_residual = predict_lf0_with_residual(
            x,
            out,
            self.in_lf0_idx,
            self.in_lf0_min,
            self.in_lf0_max,
            self.out_lf0_idx,
            self.out_lf0_mean,
            self.out_lf0_scale,
        )

        # Inject the predicted lf0 into the output features
        out[:, :, self.out_lf0_idx] = lf0_pred.squeeze(-1)

        return out, lf0_residual


class MultistreamParametricModel(BaseModel):
    def __init__(
        self,
        energy_model: nn.Module,
        energy_stream_sizes: list,
        energy_has_dynamic_features: list,
        energy_num_windows: int,
        pitch_model: nn.Module,
        pitch_stream_sizes: list,
        pitch_has_dynamic_features: list,
        pitch_num_windows: int,
        timbre_model: nn.Module,
        timbre_stream_sizes: list,
        timbre_has_dynamic_features: list,
        timbre_num_windows: int,
        timbre_postnet: nn.Module,
        # NOTE: you must carefully set the following parameters
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
    ):
        super().__init__()
        self.energy_stream_sizes = energy_stream_sizes
        self.energy_has_dynamic_features = energy_has_dynamic_features
        self.energy_num_windows = energy_num_windows
        self.energy_model = energy_model
        if hasattr(self.energy_model, "out_dim"):
            assert self.energy_model.out_dim == sum(
                self.energy_stream_sizes
            ), "Energy model output dimension is not consistent with the stream sizes"

        self.pitch_stream_sizes = pitch_stream_sizes
        self.pitch_has_dynamic_features = pitch_has_dynamic_features
        self.pitch_num_windows = pitch_num_windows
        self.pitch_model = pitch_model
        if hasattr(self.pitch_model, "out_dim"):
            assert self.pitch_model.out_dim == sum(
                self.pitch_stream_sizes
            ), "Pitch model output dimension is not consistent with the stream sizes"

        self.timbre_stream_sizes = timbre_stream_sizes
        self.timbre_has_dynamic_features = timbre_has_dynamic_features
        self.timbre_num_windows = timbre_num_windows
        self.timbre_model = timbre_model
        if hasattr(self.timbre_model, "out_dim"):
            assert self.timbre_model.out_dim == sum(
                self.timbre_stream_sizes
            ), "Timbre model output dimension is not consistent with the stream sizes"
        self.timbre_postnet = timbre_postnet

        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx
        if hasattr(self.pitch_model, "out_lf0_mean"):
            self.pitch_model.in_lf0_idx = self.in_lf0_idx
            self.pitch_model.in_lf0_min = self.in_lf0_min
            self.pitch_model.in_lf0_max = self.in_lf0_max
            self.pitch_model.out_lf0_mean = self.out_lf0_mean
            self.pitch_model.out_lf0_scale = self.out_lf0_scale

    def forward(self, x, lengths=None):
        self._set_lf0_params()

        out = self.energy_model(x, lengths)
        erg = split_streams(out, self.energy_stream_sizes)[0]

        # NOTE: so far assuming residual F0 prediction models
        out, lf0_residual = self.pitch_model(x, lengths)
        if len(self.pitch_stream_sizes) == 2:
            lf0, vuv = split_streams(out, self.pitch_stream_sizes)
        elif len(self.pitch_stream_sizes) == 3:
            lf0, vuv, vib = split_streams(out, self.pitch_stream_sizes)
        else:
            lf0, vuv, vib, vib_flags = split_streams(out, self.pitch_stream_sizes)

        out = self.timbre_model(x, lengths)
        if self.timbre_postnet is not None:
            noise = torch.randn_like(out)
            out = self.timbre_postnet(out + noise)
        mgc, bap = split_streams(out, self.timbre_stream_sizes)

        # concat mgcs' 0-th and rest dims
        mgc = torch.cat([erg, mgc], dim=-1)

        # make a concatenated stream
        if len(self.pitch_stream_sizes) == 2:
            out = torch.cat([mgc, lf0, vuv, bap], dim=-1)
        elif len(self.pitch_stream_sizes) == 3:
            out = torch.cat([mgc, lf0, vuv, bap, vib], dim=-1)
        else:
            out = torch.cat([mgc, lf0, vuv, bap, vib, vib_flags], dim=-1)

        return out, lf0_residual

    def inference(self, x, lengths=None):
        return self(x, lengths)[0][-1]
