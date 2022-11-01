# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Residual block modules.

References:
    - https://github.com/bigpon/QPPWG
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/r9y9/wavenet_vocoder

"""

import math
import sys
from logging import getLogger

import torch
import torch.nn as nn
from nnsvs.usfgan.utils import pd_indexing

# A logger for this file
logger = getLogger(__name__)


class Conv1d(nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias=True):
        """Initialize 1x1 Conv1d module."""
        super(Conv1d1x1, self).__init__(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )


class Conv2d(nn.Conv2d):
    """Conv2d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv2d module."""
        super(Conv2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Conv2d1x1(Conv2d):
    """1x1 Conv2d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias=True):
        """Initialize 1x1 Conv2d module."""
        super(Conv2d1x1, self).__init__(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )


class FixedBlock(nn.Module):
    """Fixed block module in QPPWG."""

    def __init__(
        self,
        residual_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=80,
        kernel_size=3,
        dilation=1,
        bias=True,
    ):
        """Initialize Fixed ResidualBlock module.

        Args:
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Local conditioning channels i.e. auxiliary input dimension.
            dilation (int): Dilation size.
            bias (bool): Whether to add bias parameter in convolution layers.

        """
        super(FixedBlock, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation

        # dilation conv
        self.conv = Conv1d(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=padding,
            padding_mode="reflect",
            dilation=dilation,
            bias=bias,
        )

        # local conditioning
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_channels, bias=bias)

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, residual_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, aux_channels, T).

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        residual = x
        x = self.conv(x)

        # split into two part for gated activation
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

        # local conditioning
        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb

        x = torch.tanh(xa) * torch.sigmoid(xb)

        # for skip connection
        s = self.conv1x1_skip(x)

        # for residual connection
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)

        return x, s


class AdaptiveBlock(nn.Module):
    """Adaptive block module in QPPWG."""

    def __init__(
        self,
        residual_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=80,
        bias=True,
    ):
        """Initialize Adaptive ResidualBlock module.

        Args:
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Local conditioning channels i.e. auxiliary input dimension.
            bias (bool): Whether to add bias parameter in convolution layers.

        """
        super(AdaptiveBlock, self).__init__()

        # pitch-dependent dilation conv
        self.convP = Conv1d1x1(residual_channels, gate_channels, bias=bias)  # past
        self.convC = Conv1d1x1(residual_channels, gate_channels, bias=bias)  # current
        self.convF = Conv1d1x1(residual_channels, gate_channels, bias=bias)  # future

        # local conditioning
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_channels, bias=bias)

    def forward(self, xC, xP, xF, c):
        """Calculate forward propagation.

        Args:
            xC (Tensor): Current input tensor (B, residual_channels, T).
            xP (Tensor): Past input tensor (B, residual_channels, T).
            xF (Tensor): Future input tensor (B, residual_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, aux_channels, T).

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        residual = xC
        x = self.convC(xC) + self.convP(xP) + self.convF(xF)

        # split into two part for gated activation
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

        # local conditioning
        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb

        x = torch.tanh(xa) * torch.sigmoid(xb)

        # for skip connection
        s = self.conv1x1_skip(x)

        # for residual connection
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)

        return x, s


class ResidualBlocks(nn.Module):
    """Multiple residual blocks stacking module."""

    def __init__(
        self,
        blockA,
        cycleA,
        blockF,
        cycleF,
        cascade_mode=0,
        residual_channels=64,
        gate_channels=128,
        skip_channels=64,
        aux_channels=80,
    ):
        """Initialize ResidualBlocks module.

        Args:
            blockA (int): Number of adaptive residual blocks.
            cycleA (int): Number of dilation cycles of adaptive residual blocks.
            blockF (int): Number of fixed residual blocks.
            cycleF (int): Number of dilation cycles of fixed residual blocks.
            cascade_mode (int): Cascaded mode (0: Adaptive->Fixed; 1: Fixed->Adaptive).
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.

        """
        super(ResidualBlocks, self).__init__()

        # check the number of blocks and cycles
        cycleA = max(cycleA, 1)
        cycleF = max(cycleF, 1)
        assert blockA % cycleA == 0
        self.blockA_per_cycle = blockA // cycleA
        assert blockF % cycleF == 0
        blockF_per_cycle = blockF // cycleF

        # define adaptive residual blocks
        adaptive_blocks = nn.ModuleList()
        for block in range(blockA):
            conv = AdaptiveBlock(
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
            )
            adaptive_blocks += [conv]

        # define fixed residual blocks
        fixed_blocks = nn.ModuleList()
        for block in range(blockF):
            dilation = 2 ** (block % blockF_per_cycle)
            conv = FixedBlock(
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
            )
            fixed_blocks += [conv]

        # define cascaded structure
        if cascade_mode == 0:  # adaptive->fixed
            self.conv_dilated = adaptive_blocks.extend(fixed_blocks)
            self.block_modes = [True] * blockA + [False] * blockF
        elif cascade_mode == 1:  # fixed->adaptive
            self.conv_dilated = fixed_blocks.extend(adaptive_blocks)
            self.block_modes = [False] * blockF + [True] * blockA
        else:
            logger.error(f"Cascaded mode {cascade_mode} is not supported!")
            sys.exit(0)

    def forward(self, x, c, d, batch_index, ch_index):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T).
            d (Tensor): Input pitch-dependent dilated factors (B, 1, T).

        Returns:
            Tensor: Output tensor (B, residual_channels, T).

        """
        skips = 0
        blockA_idx = 0
        for f, mode in zip(self.conv_dilated, self.block_modes):
            if mode:  # adaptive block
                dilation = 2 ** (blockA_idx % self.blockA_per_cycle)
                xP, xF = pd_indexing(x, d, dilation, batch_index, ch_index)
                x, h = f(x, xP, xF, c)
                blockA_idx += 1
            else:  # fixed block
                x, h = f(x, c)
            skips = h + skips
        skips *= math.sqrt(1.0 / len(self.conv_dilated))

        return x


class PeriodicityEstimator(nn.Module):
    """Periodicity estimator module."""

    def __init__(
        self,
        in_channels,
        residual_channels=64,
        conv_layers=3,
        kernel_size=5,
        dilation=1,
        padding_mode="replicate",
    ):
        """Initialize USFGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            residual_channels (int): Number of channels in residual conv.
            conv_layers (int):  # Number of convolution layers.
            kernel_size (int): Kernel size.
            dilation (int): Dilation size.
            padding_mode (str): Padding mode.

        """
        super(PeriodicityEstimator, self).__init__()

        modules = []
        for idx in range(conv_layers):
            conv1d = Conv1d(
                in_channels,
                residual_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=kernel_size // 2 * dilation,
                padding_mode=padding_mode,
            )

            # initialize the initial outputs sigmoid(0)=0.5 to stabilize training
            if idx != conv_layers - 1:
                nonlinear = nn.ReLU(inplace=True)
            else:
                # NOTE: zero init induces nan or inf if weight normalization is used
                # nn.init.zeros_(conv1d.weight)
                nn.init.normal_(conv1d.weight, std=1e-4)
                nonlinear = nn.Sigmoid()

            modules += [conv1d, nonlinear]
            in_channels = residual_channels

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input auxiliary features (B, C ,T).

        Returns:
            Tensor: Output tensor (B, residual_channels, T).

        """
        return self.layers(x)
