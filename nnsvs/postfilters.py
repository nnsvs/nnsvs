import numpy as np
import torch
from nnsvs.base import BaseModel
from nnsvs.multistream import split_streams
from nnsvs.util import init_weights
from torch import nn
from torch.nn import functional as F


class Conv2dPostFilter(BaseModel):
    """A post-filter based on Conv2d

    A model proposed in the paper :cite:t:`kaneko2017generative`.

    Args:
        channels (int): number of channels
        kernel_size (tuple): kernel sizes for Conv2d
        init_type (str): type of initialization
    """

    def __init__(self, channels=128, kernel_size=(5, 5), init_type="kaiming_normal"):
        super().__init__()
        C = channels
        assert len(kernel_size) == 2
        ks = np.asarray(list(kernel_size))
        padding = (ks - 1) // 2
        self.conv1 = nn.Conv2d(2, C, kernel_size=ks, padding=padding)
        self.conv2 = nn.Conv2d(C + 1, C * 2, kernel_size=ks, padding=padding)
        self.conv3 = nn.Conv2d(C * 2 + 1, C, kernel_size=ks, padding=padding)
        self.conv4 = nn.Conv2d(C + 1, 1, kernel_size=ks, padding=padding)
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
        z = torch.randn_like(x)
        x_syn = x

        y = F.relu(self.conv1(torch.cat([x_syn, z], dim=1)))
        y = F.relu(self.conv2(torch.cat([x_syn, y], dim=1)))
        y = F.relu(self.conv3(torch.cat([x_syn, y], dim=1)))
        residual = self.conv4(torch.cat([x_syn, y], dim=1))

        out = x_syn + residual

        # (B, 1, T, C) -> (B, T, C)
        out = out.squeeze(1)

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
