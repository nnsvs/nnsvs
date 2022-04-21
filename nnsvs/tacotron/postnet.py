# Acknowledgement: some of the code was adapted from ESPnet
#  Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from torch import nn


class Postnet(nn.Module):
    """Post-Net of Tacotron 2

    Args:
        in_dim (int): dimension of input
        layers (int): number of layers
        channels (int): number of channels
        kernel_size (int): kernel size
        dropout (float): dropout rate
    """

    def __init__(
        self,
        in_dim,
        layers=5,
        channels=512,
        kernel_size=5,
        dropout=0.5,
    ):
        super().__init__()
        postnet = nn.ModuleList()
        for layer in range(layers):
            in_channels = in_dim if layer == 0 else channels
            out_channels = in_dim if layer == layers - 1 else channels
            postnet += [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            ]
            if layer != layers - 1:
                postnet += [nn.Tanh()]
            postnet += [nn.Dropout(dropout)]
        self.postnet = nn.Sequential(*postnet)

    def forward(self, xs):
        """Forward step

        Args:
            xs (torch.Tensor): input sequence

        Returns:
            torch.Tensor: output sequence
        """
        return self.postnet(xs)
