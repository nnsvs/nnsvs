import torch
from nnsvs.wavenet.modules import Conv1d1x1, ResSkipBlock
from torch import nn
from torch.nn import functional as F


class WaveNet(nn.Module):
    """WaveNet

    Args:
        in_dim (int): the dimension of the input
        out_dim (int): the dimension of the output
        layers (int): the number of layers
        stacks (int): the number of residual stacks
        residual_channels (int): the number of residual channels
        gate_channels (int): the number of channels for the gating function
        skip_out_channels (int): the number of channels in the skip output
        kernel_size (int): the size of the convolutional kernel
    """

    def __init__(
        self,
        in_dim=334,
        out_dim=206,
        layers=10,
        stacks=1,
        residual_channels=64,
        gate_channels=128,
        skip_out_channels=64,
        kernel_size=3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.first_conv = Conv1d1x1(out_dim, residual_channels)

        self.main_conv_layers = nn.ModuleList()
        layers_per_stack = layers // stacks
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResSkipBlock(
                residual_channels,
                gate_channels,
                kernel_size,
                skip_out_channels,
                dilation=dilation,
                cin_channels=in_dim,
            )
            self.main_conv_layers.append(conv)

        self.last_conv_layers = nn.ModuleList(
            [
                nn.ReLU(),
                Conv1d1x1(skip_out_channels, skip_out_channels),
                nn.ReLU(),
                Conv1d1x1(skip_out_channels, out_dim),
            ]
        )

    def forward(self, c, x, lengths=None):
        """Forward step

        Args:
            c (torch.Tensor): the conditional features (B, T, C)
            x (torch.Tensor): the target features (B, T, C)

        Returns:
            torch.Tensor: the output waveform
        """
        x = x.transpose(1, 2)
        c = c.transpose(1, 2)

        x = self.first_conv(x)

        skips = 0
        for f in self.main_conv_layers:
            x, h = f(x, c)
            skips += h

        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)

        return x

    def inference(self, c, num_time_steps=100, tqdm=lambda x: x):
        """Inference step

        Args:
            c (torch.Tensor): the local conditioning feature (B, T, C)
            num_time_steps (int): the number of time steps to generate
            tqdm (lambda): a tqdm function to track progress

        Returns:
            torch.Tensor: the output waveform
        """
        self.clear_buffer()

        # Local conditioning
        B = c.shape[0]

        outputs = []

        # 自己回帰生成における初期値
        current_input = torch.zeros(B, 1, self.out_dim).to(c.device)

        if tqdm is None:
            ts = range(num_time_steps)
        else:
            ts = tqdm(range(num_time_steps))

        # 逐次的に生成
        for t in ts:
            # 時刻 t における入力は、時刻 t-1 における出力
            if t > 0:
                current_input = outputs[-1]

            # 時刻 t における条件付け特徴量
            ct = c[:, t, :].unsqueeze(1)

            x = current_input
            x = self.first_conv.incremental_forward(x)
            skips = 0
            for f in self.main_conv_layers:
                x, h = f.incremental_forward(x, ct)
                skips += h
            x = skips
            for f in self.last_conv_layers:
                if hasattr(f, "incremental_forward"):
                    x = f.incremental_forward(x)
                else:
                    x = f(x)
            # Softmax によって、出力をカテゴリカル分布のパラメータに変換
            x = F.softmax(x.view(B, -1), dim=1)
            # カテゴリカル分布からサンプリング
            x = torch.distributions.OneHotCategorical(x).sample()
            outputs += [x.data]

        # T x B x C
        # 各時刻における出力を結合
        outputs = torch.stack(outputs)
        # B x T x C
        outputs = outputs.transpose(0, 1).contiguous()

        self.clear_buffer()
        return outputs

    def clear_buffer(self):
        """Clear the internal buffer."""
        self.first_conv.clear_buffer()
        for f in self.main_conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass

    def remove_weight_norm_(self):
        """Remove weight normalization of the model"""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)
