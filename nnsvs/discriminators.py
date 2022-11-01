"""Discriminator implementations mostly used for GAN-based post-filters.

All the discriminators must returns list of tensors.
The last tensor of the list is regarded as the output of the discrminator.
The others are used as intermedieate feature maps.
"""

import numpy as np
import torch
from nnsvs.util import init_weights
from torch import nn


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
