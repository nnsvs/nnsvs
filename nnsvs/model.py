# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.nn.utils import weight_norm
from nnsvs.mdn import MDNLayer

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


class Conv1dResnet(nn.Module):
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
        self.prediction_type="deterministic"
    def forward(self, x, lengths=None):
        return self.model(x.transpose(1,2)).transpose(1,2)


class FeedForwardNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.0):
        super(FeedForwardNet, self).__init__()
        self.first_linear = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.last_linear = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.prediction_type="deterministic"
    def forward(self, x, lengths=None):
        h = self.relu(self.first_linear(x))
        for hl in self.hidden_layers:
            h = self.dropout(self.relu(hl(h)))
        return self.last_linear(h)


class LSTMRNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, bidirectional=True,
            dropout=0.0):
        super(LSTMRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_direction =  2 if bidirectional else 1
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers,
            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.hidden2out = nn.Linear(self.num_direction*self.hidden_dim, out_dim)
        self.prediction_type="deterministic"
    def forward(self, sequence, lengths):
        sequence = pack_padded_sequence(sequence, lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.hidden2out(out)
        return out

class RMDN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, bidirectional=True, dropout=0.0, num_gaussians=8):
        super(RMDN, self).__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.num_direction=2 if bidirectional else 1
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                            bidirectional=bidirectional, batch_first=True, 
                            dropout=dropout)
        self.mdn = MDNLayer(self.num_direction*hidden_dim, out_dim, num_gaussians=num_gaussians)
        self.prediction_type="probabilistic"
    def forward(self, x, lengths):
        out = self.linear(x)
        sequence = pack_padded_sequence(out, lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.mdn(out)
        return out

class MDN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=1, dropout=0.0, num_gaussians=8):
        super(MDN, self).__init__()
        model = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        if num_layers > 1:
            for _ in range(num_layers - 1):
                model += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        model += [MDNLayer(hidden_dim, out_dim, num_gaussians=num_gaussians)]
        self.model = nn.Sequential(*model)
        self.prediction_type="probabilistic"
    def forward(self, x, lengths=None):
        return self.model(x)
