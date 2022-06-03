"""Discriminator implementations mostly used for GAN-based post-filters.

All the discriminators must returns list of tensors.
The last tensor of the list is regarded as the output of the discrminator.
The others are used as intermedieate feature maps.
"""

import numpy as np
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

        return [outs]


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

        return [outs]


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
        return [outs]


class NUGANDImpl(nn.Module):
    def __init__(
        self,
        in_dim,
        groups,
        n_layers=3,
        kernel_size=3,
        stride=2,
        init_type="normal",
        last_sigmoid=False,
    ):
        super().__init__()
        model = nn.ModuleDict()

        for n in range(0, n_layers):
            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    in_dim,
                    in_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=groups,
                ),
                nn.LeakyReLU(0.2),
            )

        model["layer_%d" % (n_layers)] = nn.Sequential(
            WNConv1d(in_dim, groups, kernel_size=kernel_size, stride=1),
            nn.LeakyReLU(0.2),
        )
        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            groups, 1, kernel_size=kernel_size, stride=1
        )
        self.last_sigmoid = last_sigmoid
        self.model = model
        init_weights(self, init_type)

    def forward(self, x, c, lengths):
        outs = []
        x = x.transpose(1, 2)
        for _, layer in self.model.items():
            x = layer(x)
            outs.append(x.transpose(1, 2))
        if self.last_sigmoid:
            outs[-1] = torch.sigmoid(outs[-1])
        return outs


class NUGAND(nn.Module):
    def __init__(
        self,
        in_dim,
        groups,
        **kwargs,
    ):
        super().__init__()
        self.models = nn.ModuleList()
        for group in groups:
            self.models.append(NUGANDImpl(in_dim, group, **kwargs))

    def forward(self, x, c, lengths):
        outs = []
        for model in self.models:
            outs.append(model(x, c, lengths))
        return outs


class Conv2dD(nn.Module):
    """Conv2d-based discriminator


    The implementation follows the discrimiantor of the GAN-based post-filters
    in :cite:t:`Kaneko2017Interspeech`.

    Args:
        in_dim (int): Input feature dim
        channels (int): Number of channels
        kernel_size (tuple): Kernel size for 2d-convolution
        padding (tuple): Padding for 2d-convolution
        last_sigmoid (bool): If True, apply sigmoid on the output
        init_type (str): Initialization type
        padding_mode (str): Padding mode
    """

    def __init__(
        self,
        in_dim=None,
        channels=64,
        kernel_size=(5, 3),
        padding=(0, 0),
        last_sigmoid=False,
        init_type="kaiming_normal",
        padding_mode="zeros",
    ):
        super().__init__()
        self.last_sigmoid = last_sigmoid
        C = channels
        ks = np.asarray(list(kernel_size))
        if padding is None:
            padding = (ks - 1) // 2

        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(
                    1,
                    C,
                    kernel_size=ks,
                    padding=padding,
                    stride=(1, 1),
                    padding_mode=padding_mode,
                ),
                nn.LeakyReLU(0.2),
            )
        )
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(
                    C,
                    2 * C,
                    kernel_size=ks,
                    padding=padding,
                    stride=(2, 1),
                    padding_mode=padding_mode,
                ),
                nn.LeakyReLU(0.2),
            )
        )
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(
                    2 * C,
                    4 * C,
                    kernel_size=ks,
                    padding=padding,
                    stride=(2, 1),
                    padding_mode=padding_mode,
                ),
                nn.LeakyReLU(0.2),
            )
        )
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(
                    4 * C,
                    2 * C,
                    kernel_size=ks,
                    padding=padding,
                    stride=(2, 1),
                    padding_mode=padding_mode,
                ),
                nn.LeakyReLU(0.2),
            )
        )
        self.last_conv = nn.Conv2d(
            2 * C,
            1,
            kernel_size=ks,
            padding=padding,
            stride=(1, 1),
            padding_mode=padding_mode,
        )
        init_weights(self, init_type)

    def forward(self, x, c=None, lengths=None):
        """Forward step

        Args:
            x (torch.Tensor): Input tensor
            c (torch.Tensor): Optional conditional features
            lengths (torch.Tensor): Optional lengths of the input

        Returns:
            list: List of output tensors
        """
        outs = []
        # (B, T, C) -> (B, 1, T, C):
        x = x.unsqueeze(1)
        for conv in self.convs:
            x = conv(x)
            outs.append(x)
        y = self.last_conv(x)
        y = torch.sigmoid(y) if self.last_sigmoid else y
        # (B, 1, T, C) -> (B, T, C)
        y = y.squeeze(1)
        outs.append(y)

        return [outs]


class MultiscaleConv2d(nn.Module):
    def __init__(
        self,
        in_dim=None,
        channels=64,
        kernel_size=(5, 3),
        padding=None,
        last_sigmoid=False,
        init_type="kaiming_normal",
        padding_mode="reflect",
        stream_sizes=(8, 20, 30),
        overlap=0,
    ):
        super().__init__()
        self.stream_sizes = stream_sizes
        self.overlap = overlap

        self.conv1 = Conv2dD(
            in_dim=stream_sizes[0],
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            last_sigmoid=last_sigmoid,
            init_type=init_type,
            padding_mode=padding_mode,
        )
        self.conv2 = Conv2dD(
            in_dim=stream_sizes[1],
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            last_sigmoid=last_sigmoid,
            init_type=init_type,
            padding_mode=padding_mode,
        )
        self.conv3 = Conv2dD(
            in_dim=stream_sizes[2],
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            last_sigmoid=last_sigmoid,
            init_type=init_type,
            padding_mode=padding_mode,
        )

    def forward(self, x, c=None, lengths=None):
        assert x.shape[-1] == sum(self.stream_sizes)

        outs = []
        # (B, T, C)
        outs.append(
            self.conv1(x[:, :, : self.stream_sizes[0] + self.overlap], c, lengths)[0]
        )
        outs.append(
            self.conv2(
                x[
                    :,
                    :,
                    self.stream_sizes[0]
                    - self.overlap : sum(self.stream_sizes[:2])
                    + self.overlap,
                ],
                c,
                lengths,
            )[0]
        )
        outs.append(
            self.conv3(
                x[:, :, sum(self.stream_sizes[:2]) - self.overlap :], c, lengths
            )[0]
        )
        return outs
