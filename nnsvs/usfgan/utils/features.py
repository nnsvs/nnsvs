# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Feature-related functions.

References:
    - https://github.com/bigpon/QPPWG

"""

import sys
from logging import getLogger

import numpy as np
import torch
from torch.nn.functional import interpolate

# A logger for this file
logger = getLogger(__name__)


def validate_length(x, y, hop_size=None):
    """Validate length

    Args:
        x (ndarray): numpy array with x.shape[0] = len_x
        y (ndarray): numpy array with y.shape[0] = len_y
        hop_size (int): upsampling factor

    Returns:
        (ndarray): length adjusted x with same length y
        (ndarray): length adjusted y with same length x

    """
    if hop_size is None:
        if x.shape[0] < y.shape[0]:
            y = y[: x.shape[0]]
        if x.shape[0] > y.shape[0]:
            x = x[: y.shape[0]]
        assert len(x) == len(y)
    else:
        if x.shape[0] > y.shape[0] * hop_size:
            x = x[: y.shape[0] * hop_size]
        if x.shape[0] < y.shape[0] * hop_size:
            mod_y = y.shape[0] * hop_size - x.shape[0]
            mod_y_frame = mod_y // hop_size + 1
            y = y[:-mod_y_frame]
            x = x[: y.shape[0] * hop_size]
        assert len(x) == len(y) * hop_size

    return x, y


def dilated_factor(batch_f0, fs, dense_factor):
    """Pitch-dependent dilated factor

    Args:
        batch_f0 (ndarray): the f0 sequence (T)
        fs (int): sampling rate
        dense_factor (int): the number of taps in one cycle

    Return:
        dilated_factors(np array):
            float array of the pitch-dependent dilated factors (T)

    """
    batch_f0[batch_f0 == 0] = fs / dense_factor
    dilated_factors = np.ones(batch_f0.shape) * fs
    dilated_factors /= batch_f0
    dilated_factors /= dense_factor
    assert np.all(dilated_factors > 0)

    return dilated_factors


class SignalGenerator:
    """Input signal generator module."""

    def __init__(
        self,
        sample_rate=24000,
        hop_size=120,
        sine_amp=0.1,
        noise_amp=0.003,
        signal_types=["sine", "noise"],
    ):
        """Initialize WaveNetResidualBlock module.

        Args:
            sample_rate (int): Sampling rate.
            hop_size (int): Hop size of input F0.
            sine_amp (float): Sine amplitude for NSF-based sine generation.
            noise_amp (float): Noise amplitude for NSF-based sine generation.
            signal_types (list): List of input signal types for generator.

        """
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.signal_types = signal_types
        self.sine_amp = sine_amp
        self.noise_amp = noise_amp

        # for signal_type in signal_types:
        #     if not signal_type in ["noise", "sine", "uv"]:
        #         logger.info(f"{signal_type} is not supported type for generator input.")
        #         sys.exit(0)
        # logger.info(f"Use {signal_types} for generator input signals.")

    @torch.no_grad()
    def __call__(self, f0):
        signals = []
        for typ in self.signal_types:
            if "noise" == typ:
                signals.append(self.random_noise(f0))
            if "sine" == typ:
                signals.append(self.sinusoid(f0))
            if "uv" == typ:
                signals.append(self.vuv_binary(f0))

        input_batch = signals[0]
        for signal in signals[1:]:
            input_batch = torch.cat([input_batch, signal], axis=1)

        return input_batch

    @torch.no_grad()
    def random_noise(self, f0):
        """Calculate noise signals.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: Gaussian noise signals (B, 1, T).

        """
        B, _, T = f0.size()
        noise = torch.randn((B, 1, T * self.hop_size), device=f0.device)

        return noise

    @torch.no_grad()
    def sinusoid(self, f0):
        """Calculate sine signals.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: Sines generated following NSF (B, 1, T).

        """
        B, _, T = f0.size()
        vuv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)
        radious = (interpolate(f0, T * self.hop_size) / self.sample_rate) % 1
        sine = vuv * torch.sin(torch.cumsum(radious, dim=2) * 2 * np.pi) * self.sine_amp
        if self.noise_amp > 0:
            noise_amp = vuv * self.noise_amp + (1.0 - vuv) * self.noise_amp / 3.0
            noise = torch.randn((B, 1, T * self.hop_size), device=f0.device) * noise_amp
            sine = sine + noise

        return sine

    @torch.no_grad()
    def vuv_binary(self, f0):
        """Calculate V/UV binary sequences.

        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).

        Returns:
            Tensor: V/UV binary sequences (B, 1, T).

        """
        _, _, T = f0.size()
        uv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)

        return uv
