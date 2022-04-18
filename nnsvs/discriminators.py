"""Discriminator implementations

All the discriminators must returns list of tensors.
The last tensor of the list is regarded as the output of the discrminator.
The others are used as intermedieate feature maps.

Multi-scale architecture is not supported yet.
"""

import torch
from nnsvs.model import ResnetBlock, WNConv1d
from nnsvs.util import init_weights
from torch import nn


class FFN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=2,
        dropout=0.0,
        init_type="normal",
        cin_dim=-1,
        last_sigmoid=False,
    ):
        super(FFN, self).__init__()
        self.first_linear = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.last_linear = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.last_sigmoid = last_sigmoid

        if cin_dim > 0:
            self.cond = nn.Linear(cin_dim, hidden_dim)
        else:
            self.cond = None

        init_weights(self, init_type)

    def forward(self, x, c, lengths=None):
        outs = []
        h = self.relu(self.first_linear(x))
        outs.append(h)
        for hl in self.hidden_layers:
            h = self.dropout(self.relu(hl(h)))
            outs.append(h)
        out = self.last_linear(h)
        if self.cond is not None:
            # NOTE: sum against the last feature-axis (B, T, C)
            inner_product = (h * self.cond(c)).sum(dim=-1, keepdim=True)
            out = out + inner_product

        out = torch.sigmoid(out) if self.last_sigmoid else out
        outs.append(out)

        return outs


class Conv1dResnet(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=2,
        dropout=0.0,
        init_type="normal",
        cin_dim=-1,
        last_sigmoid=False,
    ):
        super().__init__()
        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(in_dim, hidden_dim, kernel_size=7, padding=0),
        ]
        for n in range(num_layers):
            model.append(ResnetBlock(hidden_dim, dilation=2 ** n))
        model += [
            nn.LeakyReLU(0.2),
        ]
        self.model = nn.ModuleList(model)
        self.last_conv = WNConv1d(hidden_dim, out_dim, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.last_sigmoid = last_sigmoid

        if cin_dim > 0:
            self.cond = WNConv1d(cin_dim, hidden_dim, kernel_size=1, padding=0)
        else:
            self.cond = None

        init_weights(self, init_type)

    def forward(self, x, c, lengths=None):
        # (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        outs = []
        for f in self.model:
            x = self.dropout(f(x))
            outs.append(x)
        out = self.last_conv(x)

        if self.cond is not None:
            # NOTE: sum against the feature-axis (B, C, T)
            inner_product = (x * self.cond(c.transpose(1, 2))).sum(dim=1, keepdim=True)
            out = out + inner_product

        out = torch.sigmoid(out) if self.last_sigmoid else out

        # (B, C, T) -> (B, T, C)
        out = out.transpose(1, 2)
        outs.append(out)

        return outs


class Conv2dGLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=None,
        dilation=(1, 1),
        norm_layer=True,
        *args,
        **kwargs,
    ):
        super(Conv2dGLU, self).__init__()
        if padding is None:
            padding = []
            for idx, ks in enumerate(kernel_size):
                assert ks % 2 == 1
                p = (ks - 1) // 2 * dilation[idx]
                padding.append(p)
                padding.append(p)
            padding.reverse()
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=0,
            dilation=dilation,
            *args,
            **kwargs,
        )
        if norm_layer:
            self.norm = nn.InstanceNorm2d(out_channels * 2, affine=True)
        else:
            self.norm = None

    def forward(self, x):
        x = self.conv(self.pad(x))
        x = self.norm(x) if self.norm is not None else x
        a, b = x.split(x.size(1) // 2, dim=1)
        x = a * torch.sigmoid(b)
        return x


class CycleGANVC2D(nn.Module):
    def __init__(
        self,
        in_dim=None,
        hidden_dim=64,
        padding=None,
        init_type="normal",
        last_sigmoid=False,
    ):
        super().__init__()
        C = hidden_dim
        self.conv_in = Conv2dGLU(
            1, C, (3, 3), stride=(1, 1), padding=padding, norm_layer=None
        )
        self.downsample = nn.ModuleList(
            [
                Conv2dGLU(C, 2 * C, (3, 3), stride=(2, 2), padding=padding),
                Conv2dGLU(2 * C, 4 * C, (3, 3), stride=(2, 2), padding=padding),
                Conv2dGLU(4 * C, 8 * C, (3, 3), stride=(2, 2), padding=padding),
                Conv2dGLU(8 * C, 8 * C, (1, 5), stride=(1, 1), padding=padding),
            ]
        )
        # NOTE: 8x smaller time lengths for the output
        # depends on the stride
        self.downsample_scale = 8
        if padding is None:
            padding_ = (1, 1, 0, 0)
            self.conv_out = nn.Sequential(
                nn.ReflectionPad1d(padding_),
                nn.Conv2d(8 * C, 1, (1, 3), padding=0),
            )
        else:
            self.conv_out = nn.Conv2d(8 * C, 1, (1, 3), padding=padding)
        self.last_sigmoid = last_sigmoid
        init_weights(self, init_type)

    def forward(self, x, c, lengths):
        # W: frame-axis
        # H: feature-axis
        # (B, W, H) -> (B, H, W) -> (B, 1, H, W)
        outs = []
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.conv_in(x)
        outs.append(x)
        for f in self.downsample:
            x = f(x)
            outs.append(x)
        x = self.conv_out(x)
        x = x.squeeze(1).transpose(1, 2)
        x = torch.sigmoid(x) if self.last_sigmoid else x
        outs.append(x)
        return outs
