# -*- coding: utf-8 -*-

# Copyright 2020 Yi-Chiao Wu (Nagoya University)
# based on a WaveNet script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/PytorchWaveNetVocoder)
# based on sprocket-vc script by Kazuhiro Kobayashi (Nagoya University)
# (https://github.com/k2kobayashi/sprocket)
#  MIT License (https://opensource.org/licenses/MIT)

"""Filters."""

import numpy as np
from scipy.signal import firwin, lfilter

NUMTAPS = 255


def low_cut_filter(x, fs, cutoff=70):
    """Low-cut filter

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence

    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    numtaps = NUMTAPS
    fil = firwin(numtaps, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def low_pass_filter(x, fs, cutoff=70):
    """Low-pass filter

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence

    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    numtaps = NUMTAPS
    fil = firwin(numtaps, norm_cutoff, pass_zero=True)
    x_pad = np.pad(x, (numtaps, numtaps), "edge")
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2 : -numtaps // 2]

    return lpf_x
