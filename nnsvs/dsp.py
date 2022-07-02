import torch
from scipy import signal
from torch import nn
from torch.nn import functional as F

# Part of code was adapted from:
# https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts


def lowpass_filter(x, fs, cutoff=5, N=5):
    """Lowpass filter

    Args:
        x (np.ndarray): input signal
        fs (int): sampling rate
        cutoff (int): cutoff frequency

    Returns:
        np.ndarray: filtered signal
    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    Wn = [norm_cutoff]

    b, a = signal.butter(N, Wn, "lowpass")
    if len(x) <= max(len(a), len(b)) * (N // 2 + 1):
        # NOTE: input signal is too short
        return x

    # NOTE: use zero-phase filter
    y = signal.filtfilt(b, a, x)

    return y


def bandpass_filter(x, sr, cutoff=70):
    """Band-pass filter

    Args:
        x (np.ndarray): input signal
        fs (int): sampling rate
        cutoff (int): cutoff frequency

    Returns:
        np.ndarray: filtered signal
    """
    nyquist = sr // 2
    norm_cutoff = cutoff / nyquist
    Wn = [norm_cutoff, 0.999]

    b, a = signal.butter(5, Wn, "bandpass")
    y = signal.filtfilt(b, a, x)

    return y


class TimeInvFIRFilter(nn.Conv1d):
    """Time-invatiant FIR filter implementation

    Args:
        channels (int): input channels
        filt_coef (torch.Tensor): FIR filter coefficients
        causal (bool): causal
        requires_grad (bool): trainable kernel or not
    """

    def __init__(self, channels, filt_coef, causal=True, requires_grad=False):
        # assuming 1-D filter coef vector and odd num taps
        assert len(filt_coef.shape) == 1
        # assert len(filt_coef) % 2 == 1
        kernel_size = len(filt_coef)
        self.causal = causal
        if causal:
            padding = (kernel_size - 1) * 1
        else:
            padding = (kernel_size - 1) // 2 * 1
        # channel-wise filtering (groups=channels)
        super(TimeInvFIRFilter, self).__init__(
            channels, channels, kernel_size, padding=padding, groups=channels, bias=None
        )
        self.weight.data[:, :, :] = filt_coef.flip(-1)
        self.weight.requires_grad = requires_grad

    def forward(self, x):
        out = super(TimeInvFIRFilter, self).forward(x)
        out = out[:, :, : -self.padding[0]] if self.causal else out
        return out


class TrTimeInvFIRFilter(nn.Conv1d):
    """Trainable Time-invatiant FIR filter implementation

    H(z) = \\sigma_{k=0}^{filt_dim} b_{k}z_{-k}

    Note that b_{0} is fixed to 1 if fixed_0th is True.

    Args:
        channels (int): input channels
        filt_dim (int): FIR filter dimension
        causal (bool): causal
        tanh (bool): apply tanh to filter coef or not.
        fixed_0th (bool): fix the first filt coef to 1 or not.
    """

    def __init__(self, channels, filt_dim, causal=True, tanh=True, fixed_0th=True):
        # Initialize filt coef with small random values
        init_filt_coef = torch.randn(filt_dim) * (1 / filt_dim)
        # assert len(filt_coef) % 2 == 1
        kernel_size = len(init_filt_coef)
        self.causal = causal
        if causal:
            padding = (kernel_size - 1) * 1
        else:
            padding = (kernel_size - 1) // 2 * 1
        # channel-wise filtering (groups=channels)
        super(TrTimeInvFIRFilter, self).__init__(
            channels, channels, kernel_size, padding=padding, groups=channels, bias=None
        )
        self.weight.data[:, :, :] = init_filt_coef.flip(-1)
        self.weight.requires_grad = True
        self.tanh = tanh
        self.fixed_0th = fixed_0th

    def get_filt_coefs(self):
        # apply tanh for filtter stability
        b = torch.tanh(self.weight) if self.tanh else self.weight
        b = b.clone()
        if self.fixed_0th:
            b[:, :, -1] = 1
        return b

    def forward(self, x):
        b = self.get_filt_coefs()
        out = F.conv1d(
            x, b, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.padding[0] > 0:
            out = out[:, :, : -self.padding[0]] if self.causal else out
        return out
