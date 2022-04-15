import torch
from nnsvs.util import init_weights
from torch import nn


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
    def __init__(self, in_dim=None, hidden_dim=64, padding=None, init_type="normal"):
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
        init_weights(self, init_type)

    def forward(self, x, lengths):
        # W: frame-axis
        # H: feature-axis
        # (B, W, H) -> (B, H, W) -> (B, 1, H, W)
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.conv_in(x)
        for f in self.downsample:
            x = f(x)
        x = self.conv_out(x)
        x = x.squeeze(1).transpose(1, 2)
        return x
