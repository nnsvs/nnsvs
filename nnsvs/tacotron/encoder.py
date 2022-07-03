# The code was adapted from ttslearn https://github.com/r9y9/ttslearn
# Acknowledgement: some of the code was adapted from ESPnet
#  Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def encoder_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain("relu"))


class Encoder(nn.Module):
    """Encoder of Tacotron 2

    Args:
        in_dim (int): dimension of embeddings
        hidden_dim (int): dimension of hidden units
        conv_layers (int): number of convolutional layers
        conv_channels (int): number of convolutional channels
        conv_kernel_size (int): size of convolutional kernel
        dropout (float): dropout rate
    """

    def __init__(
        self,
        in_dim=512,
        hidden_dim=512,
        conv_layers=3,
        conv_channels=512,
        conv_kernel_size=5,
        dropout=0.5,
    ):
        super(Encoder, self).__init__()
        convs = nn.ModuleList()
        for layer in range(conv_layers):
            in_channels = in_dim if layer == 0 else conv_channels
            convs += [
                nn.Conv1d(
                    in_channels,
                    conv_channels,
                    conv_kernel_size,
                    padding=(conv_kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.convs = nn.Sequential(*convs)
        self.blstm = nn.LSTM(
            conv_channels, hidden_dim // 2, 1, batch_first=True, bidirectional=True
        )
        self.apply(encoder_init)

    def forward(self, seqs, in_lens):
        """Forward step

        Args:
            seqs (torch.Tensor): input sequences (B, T, C)
            in_lens (torch.Tensor): input sequence lengths

        Returns:
            torch.Tensor: encoded sequences
        """
        out = self.convs(seqs.transpose(1, 2)).transpose(1, 2)

        if not isinstance(in_lens, list):
            in_lens = in_lens.to("cpu")

        out = pack_padded_sequence(out, in_lens, batch_first=True)
        out, _ = self.blstm(out)
        out, _ = pad_packed_sequence(out, batch_first=True)

        return out
