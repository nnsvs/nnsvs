import numpy as np
import torch
from nnsvs.base import BaseModel
from nnsvs.multistream import split_streams
from nnsvs.util import init_weights
from torch import nn


class SimplifiedTADN(nn.Module):
    """Simplified temporal adaptive de-normalization for Gaussian noise

    Args:
        channels (int): number of channels
        kernel_size (int): kernel size. Default is 7.
    """

    def __init__(self, channels, kernel_size=7):
        super().__init__()
        C = channels
        padding = (kernel_size - 1) // 2
        # NOTE: process each channel independently by setting groups=C
        self.conv_gamma = nn.Conv1d(
            C, C, kernel_size=kernel_size, padding=padding, groups=C
        )

    def forward(self, z, c):
        """Forward pass

        Args:
            z (torch.Tensor): input Gaussian noise of shape (B, C, T)
            c (torch.Tensor): input 2d feature of shape (B, C, T)

        Returns:
            torch.Tensor: output 2d feature of shape (B, C, T)
        """
        # NOTE: assuming z ~ N(0, I)
        # (B, C, T)
        gamma = torch.sigmoid(self.conv_gamma(c))
        # N(0, I) -> N(0, I*gamma) where gamma is a learned parameter in [0, 1]
        return z * gamma


class Conv2dPostFilter(BaseModel):
    """A post-filter based on Conv2d

    A model proposed in the paper :cite:t:`kaneko2017generative`.

    Args:
        channels (int): number of channels
        kernel_size (tuple): kernel sizes for Conv2d
        use_noise (bool): whether to use noise
        use_tadn (bool): whether to use temporal adaptive de-normalization
        init_type (str): type of initialization
    """

    def __init__(
        self,
        in_dim=None,
        channels=128,
        kernel_size=(5, 5),
        use_noise=True,
        use_tadn=False,
        init_type="kaiming_normal",
    ):
        super().__init__()
        self.use_noise = use_noise
        C = channels
        assert len(kernel_size) == 2
        ks = np.asarray(list(kernel_size))
        padding = (ks - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 if use_noise else 1, C, kernel_size=ks, padding=padding),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(C + 1, C * 2, kernel_size=ks, padding=padding),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(C * 2 + 1, C, kernel_size=ks, padding=padding),
            nn.ReLU(),
        )
        self.conv4 = nn.Conv2d(C + 1, 1, kernel_size=ks, padding=padding)

        if use_tadn:
            assert in_dim is not None, "in_dim must be provided"
            self.tadn = SimplifiedTADN(in_dim)
        else:
            self.tadn = None

        init_weights(self, init_type)

    def forward(self, x, lengths=None):
        """Forward step

        Args:
            x (torch.Tensor): input tensor of shape (B, T, C)
            lengths (torch.Tensor): lengths of shape (B,)

        Returns:
            torch.Tensor: output tensor of shape (B, T, C)
        """
        # (B, T, C) -> (B, 1, T, C):
        x = x.unsqueeze(1)

        if self.use_noise:
            z = torch.randn_like(x)
            # adaptively scale z
            if self.tadn is not None:
                z = (
                    self.tadn(
                        z.squeeze(1).transpose(1, 2), x.squeeze(1).transpose(1, 2)
                    )
                    .transpose(1, 2)
                    .unsqueeze(1)
                )

        x_syn = x

        if self.use_noise:
            y = self.conv1(torch.cat([x_syn, z], dim=1))
        else:
            y = self.conv1(x_syn)

        y = self.conv2(torch.cat([x_syn, y], dim=1))
        y = self.conv3(torch.cat([x_syn, y], dim=1))
        residual = self.conv4(torch.cat([x_syn, y], dim=1))

        out = x_syn + residual

        # (B, 1, T, C) -> (B, T, C)
        out = out.squeeze(1)

        return out


class Conv1dPostFilter(nn.Module):
    """A post-filter based on 1-d convolutions

    Args:
        channels (int): number of channels
        kernel_size (int): kernel size
        use_noise (bool): whether to use noise
        use_tadn (bool): whether to use temporal adaptive de-normalization
        init_type (str): type of initialization
    """

    def __init__(
        self,
        in_dim=None,
        channels=16,
        kernel_size=5,
        use_noise=False,
        use_tadn=False,
        init_type="kaiming_normal",
    ):
        super().__init__()
        self.use_noise = use_noise
        C = channels
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                2 if use_noise else 1, C, kernel_size=kernel_size, padding=padding
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(C + 1, C * 2, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(C * 2 + 1, C, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )
        self.conv4 = nn.Conv1d(C + 1, 1, kernel_size=kernel_size, padding=padding)

        if use_tadn:
            assert in_dim is not None, "in_dim must be provided"
            assert use_noise, "use_noise must be True"
            self.tadn = SimplifiedTADN(in_dim)
        else:
            self.tadn = None

        init_weights(self, init_type)

    def forward(self, x, lengths=None):
        # (B, T, C) -> (B, C, T):
        x = x.transpose(1, 2)
        x_syn = x

        if self.use_noise:
            z = torch.randn_like(x)
            if self.tadn is not None:
                z = self.tadn(z, x)
            y = self.conv1(torch.cat([x_syn, z], dim=1))
        else:
            y = self.conv1(x_syn)
        y = self.conv2(torch.cat([x_syn, y], dim=1))
        y = self.conv3(torch.cat([x_syn, y], dim=1))
        residual = self.conv4(torch.cat([x_syn, y], dim=1))

        out = x_syn + residual

        # (B, C, T) -> (B, T, C)
        out = out.transpose(1, 2)

        return out


class MultistreamPostFilter(BaseModel):
    """A multi-stream post-filter that applies post-filtering for each feature stream

    Currently, post-filtering for MGC, BAP and log-F0 are supported.
    Note that it doesn't make much sense to apply post-filtering for other features.

    Args:
        mgc_postfilter (nn.Module): post-filter for MGC
        bap_postfilter (nn.Module): post-filter for BAP
        lf0_postfilter (nn.Module): post-filter for log-F0
        stream_sizes (list): sizes of each feature stream
        mgc_offset (int): offset for MGC. Defaults to 2.
    """

    def __init__(
        self,
        mgc_postfilter: nn.Module,
        bap_postfilter: nn.Module,
        lf0_postfilter: nn.Module,
        stream_sizes: list,
        mgc_offset: int = 2,
    ):
        super().__init__()
        self.mgc_postfilter = mgc_postfilter
        self.bap_postfilter = bap_postfilter
        self.lf0_postfilter = lf0_postfilter
        self.stream_sizes = stream_sizes
        self.mgc_offset = mgc_offset

    def forward(self, x, lengths=None):
        """Forward step

        Each feature stream is processed independently.

        Args:
            x (torch.Tensor): input tensor of shape (B, T, C)
            lengths (torch.Tensor): lengths of shape (B,)

        Returns:
            torch.Tensor: output tensor of shape (B, T, C)
        """
        streams = split_streams(x, self.stream_sizes)
        if len(streams) == 4:
            mgc, lf0, vuv, bap = streams
        elif len(streams) == 5:
            mgc, lf0, vuv, bap, vuv = streams
        elif len(streams) == 6:
            mgc, lf0, vuv, bap, vib, vib_flags = streams
        else:
            raise ValueError("Invalid number of streams")

        if self.mgc_postfilter is not None:
            if self.mgc_offset > 0:
                # keep unchanged for the 0-to-${mgc_offset}-th dim of mgc
                mgc0 = mgc[:, :, : self.mgc_offset]
                mgc_pf = self.mgc_postfilter(mgc[:, :, self.mgc_offset :], lengths)
                mgc_pf = torch.cat([mgc0, mgc_pf], dim=-1)
            else:
                mgc_pf = self.mgc_postfilter(mgc, lengths)
            mgc = mgc_pf

        if self.bap_postfilter is not None:
            bap = self.bap_postfilter(bap, lengths)

        if self.lf0_postfilter is not None:
            lf0 = self.lf0_postfilter(lf0, lengths)

        if len(streams) == 4:
            out = torch.cat([mgc, lf0, vuv, bap], dim=-1)
        elif len(streams) == 5:
            out = torch.cat([mgc, lf0, vuv, bap, vib], dim=-1)
        elif len(streams) == 6:
            out = torch.cat([mgc, lf0, vuv, bap, vib, vib_flags], dim=-1)

        return out
