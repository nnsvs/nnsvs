# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.nn.utils import weight_norm

from nnsvs.base import BaseModel, PredictionType
from nnsvs.mdn import MDNLayer, mdn_get_most_probable_sigma_and_mu
from nnsvs.dsp import TrTimeInvFIRFilter
from nnsvs.multistream import split_streams


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
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4, dropout=0.0):
        super().__init__()
        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(in_dim, hidden_dim, kernel_size=7, padding=0),
        ]
        for n in range(num_layers):
            model.append(ResnetBlock(hidden_dim, dilation=2**n))
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(hidden_dim, out_dim, kernel_size=7, padding=0),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x, lengths=None):
        return self.model(x.transpose(1,2)).transpose(1,2)


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
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4, dropout=0.0,
            stream_sizes=[180, 3, 1, 15], ar_orders=[20, 200, 20, 20]):
        super().__init__(in_dim, hidden_dim, out_dim, num_layers, dropout)
        self.stream_sizes = stream_sizes

        self.analysis_filts = nn.ModuleList()
        for s, K in zip(stream_sizes, ar_orders):
            self.analysis_filts += [TrTimeInvFIRFilter(s, K+1)]

    def preprocess_target(self, y):
        assert sum(self.stream_sizes) == y.shape[-1]
        ys = split_streams(y, self.stream_sizes)
        for idx, yi in enumerate(ys):
            ys[idx] = self.analysis_filts[idx](yi.transpose(1,2)).transpose(1,2)
        return torch.cat(ys, -1)

    def inference(self, x, lengths=None):
        out = self.model(x.transpose(1, 2)).transpose(1, 2)
        return _shallow_ar_inference(out, self.stream_sizes, self.analysis_filts)


class FeedForwardNet(BaseModel):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.0):
        super(FeedForwardNet, self).__init__()
        self.first_linear = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.last_linear = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        h = self.relu(self.first_linear(x))
        for hl in self.hidden_layers:
            h = self.dropout(self.relu(hl(h)))
        return self.last_linear(h)


class LSTMRNN(BaseModel):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, bidirectional=True,
            dropout=0.0):
        super(LSTMRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_direction =  2 if bidirectional else 1
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers,
            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.hidden2out = nn.Linear(self.num_direction*self.hidden_dim, out_dim)

    def forward(self, sequence, lengths):
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
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, bidirectional=True,
            dropout=0.0, stream_sizes=[180, 3, 1, 15], ar_orders=[20, 200, 20, 20]):
        super().__init__(in_dim, hidden_dim, out_dim, num_layers,
            bidirectional, dropout)

        self.stream_sizes = stream_sizes
        self.analysis_filts = nn.ModuleList()
        for s, K in zip(stream_sizes, ar_orders):
            self.analysis_filts += [TrTimeInvFIRFilter(s, K+1)]

    def preprocess_target(self, y):
        assert sum(self.stream_sizes) == y.shape[-1]
        ys = split_streams(y, self.stream_sizes)
        for idx, yi in enumerate(ys):
            ys[idx] = self.analysis_filts[idx](yi.transpose(1,2)).transpose(1,2)
        return torch.cat(ys, -1)

    def inference(self, x, lengths=None):
        out = self.forward(x, lengths)
        return _shallow_ar_inference(out, self.stream_sizes, self.analysis_filts)


class RMDN(BaseModel):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, bidirectional=True,
            dropout=0.0, num_gaussians=8, dim_wise=False):
        super(RMDN, self).__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.num_direction=2 if bidirectional else 1
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            bidirectional=bidirectional, batch_first=True,
                            dropout=dropout)
        self.mdn = MDNLayer(self.num_direction*hidden_dim, out_dim, num_gaussians, dim_wise)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, x, lengths):
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
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, dropout=0.0,
            num_gaussians=8, dim_wise=False):
        super(MDN, self).__init__()
        model = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        if num_layers > 1:
            for _ in range(num_layers - 1):
                model += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        model += [MDNLayer(hidden_dim, out_dim, num_gaussians, dim_wise)]
        self.model = nn.Sequential(*model)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, x, lengths=None):
        return self.model(x)

    def inference(self, x, lengths=None):
        log_pi, log_sigma, mu = self.forward(x, lengths)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        return mu, sigma


class Conv1dResnetMDN(BaseModel):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4, dropout=0.0,
            num_gaussians=8, dim_wise=False):
        super().__init__()
        model = [Conv1dResnet(in_dim, hidden_dim, hidden_dim, num_layers, dropout),
                 nn.ReLU(),
                 MDNLayer(hidden_dim, out_dim, num_gaussians, dim_wise)]
        self.model = nn.Sequential(*model)

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, x, lengths=None):
        return self.model(x)

    def inference(self, x, lengths=None):
        log_pi, log_sigma, mu = self.forward(x, lengths)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        return mu, sigma