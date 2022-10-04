import torch
from nnsvs.wavenet import conv
from torch import nn


def Conv1d(in_channels, out_channels, kernel_size, *args, **kwargs):
    """Weight-normalized Conv1d layer."""
    m = conv.Conv1d(in_channels, out_channels, kernel_size, *args, **kwargs)
    return nn.utils.weight_norm(m)


def Conv1d1x1(in_channels, out_channels, bias=True):
    """1x1 Weight-normalized Conv1d layer."""
    return Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)


class ResSkipBlock(nn.Module):
    """Convolution block with residual and skip connections.

    Args:
        residual_channels (int): Residual connection channels.
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        skip_out_channels (int): Skip connection channels.
        dilation (int): Dilation factor.
        cin_channels (int): Local conditioning channels.
        args (list): Additional arguments for Conv1d.
        kwargs (dict): Additional arguments for Conv1d.
    """

    def __init__(
        self,
        residual_channels,  # 残差結合のチャネル数
        gate_channels,  # ゲートのチャネル数
        kernel_size,  # カーネルサイズ
        skip_out_channels,  # スキップ結合のチャネル数
        dilation=1,  # dilation factor
        cin_channels=80,  # 条件付特徴量のチャネル数
        *args,
        **kwargs,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        # 1 次元膨張畳み込み (dilation == 1 のときは、通常の1 次元畳み込み)
        self.conv = Conv1d(
            residual_channels,
            gate_channels,
            kernel_size,
            *args,
            padding=self.padding,
            dilation=dilation,
            **kwargs,
        )

        # local conditioning 用の 1x1 convolution
        self.conv1x1c = Conv1d1x1(cin_channels, gate_channels, bias=False)

        # ゲート付き活性化関数のために、1 次元畳み込みの出力は2 分割されることに注意
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels)

    def forward(self, x, c):
        """Forward step

        Args:
            x (torch.Tensor): Input signal.
            c (torch.Tensor): Local conditioning signal.

        Returns:
            tuple: Tuple of output signal and skip connection signal
        """
        return self._forward(x, c, False)

    def incremental_forward(self, x, c):
        """Incremental forward

        Args:
            x (torch.Tensor): Input signal.
            c (torch.Tensor): Local conditioning signal.

        Returns:
            tuple: Tuple of output signal and skip connection signal
        """
        return self._forward(x, c, True)

    def _forward(self, x, c, is_incremental):
        # 残差接続用に入力を保持
        residual = x

        # メインの dilated convolutionの計算
        # 推論時と学習時で入力のテンソルのshapeが異なるのに注意
        if is_incremental:
            splitdim = -1  # (B, T, C)
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1  # (B, C, T)
            x = self.conv(x)
            # 因果性を保証するために、出力をシフトする
            x = x[:, :, : -self.padding]

        # チャンネル方向で出力を分割
        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)

        # local conditioning
        c = self._conv1x1_forward(self.conv1x1c, c, is_incremental)
        ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
        a, b = a + ca, b + cb

        # ゲート付き活性化関数
        x = torch.tanh(a) * torch.sigmoid(b)

        # スキップ接続用の出力を計算
        s = self._conv1x1_forward(self.conv1x1_skip, x, is_incremental)

        # 残差接続の要素和行う前に、次元数を合わせる
        x = self._conv1x1_forward(self.conv1x1_out, x, is_incremental)

        x = x + residual

        return x, s

    def _conv1x1_forward(self, conv, x, is_incremental):
        if is_incremental:
            x = conv.incremental_forward(x)
        else:
            x = conv(x)
        return x

    def clear_buffer(self):
        """Clear input buffer."""
        for c in [
            self.conv,
            self.conv1x1_out,
            self.conv1x1_skip,
            self.conv1x1c,
        ]:
            if c is not None:
                c.clear_buffer()
