import torch
from torch import nn
from torch.nn import functional as F


class Conv2dPostFilter(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        C = channels
        self.conv1 = nn.Conv2d(1, C, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = nn.Conv2d(C, C * 2, kernel_size=(5, 5), padding=(2, 2))
        self.conv3 = nn.Conv2d(C * 2, C, kernel_size=(5, 5), padding=(2, 2))
        self.conv4 = nn.Conv2d(C, 1, kernel_size=(5, 5), padding=(2, 2))

    def forward(self, x, lengths=None):
        # (B, T, C) -> (B, 1, T, C):
        x = x.unsqueeze(1)
        z = torch.rand_like(x)
        x_syn = x

        y = F.relu(self.conv1(x_syn + z))
        y = F.relu(self.conv2(x_syn + y))
        y = F.relu(self.conv3(x_syn + y))
        residual = F.relu(self.conv4(x_syn + y))

        out = x_syn + residual

        # (B, 1, T, C) -> (B, T, C)
        out = out.squeeze(1)

        return out
