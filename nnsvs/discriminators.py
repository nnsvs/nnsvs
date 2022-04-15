import torch
from torch import nn


# Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
def init_weights(net, init_type="normal", init_gain=0.02):
    if init_type == "none":
        return

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


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
