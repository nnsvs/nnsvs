import torch
from nnsvs.base import BaseModel
from nnsvs.multistream import split_streams
from torch import nn
from torch.nn import functional as F


class Conv2dPostFilter(BaseModel):
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


class MGCPostFilter(BaseModel):
    def __init__(
        self,
        timbre_model: nn.Module,
        stream_sizes: list,
    ):
        super().__init__()
        self.timbre_model = timbre_model
        self.stream_sizes = stream_sizes

    def forward(self, x, lengths=None):
        mgc, lf0, vuv, bap, vib, vib_flags = split_streams(x, self.stream_sizes)

        mgc_pf = self.timbre_model(mgc, lengths)

        out = torch.cat([mgc_pf, lf0, vuv, bap, vib, vib_flags], dim=-1)

        return out
