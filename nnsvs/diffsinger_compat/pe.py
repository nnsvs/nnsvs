import math

import torch
from torch import nn


def denorm_f0(
    f0,
    uv,
    pitch_padding=None,
    min=None,
    max=None,
    pitch_norm="log",
    use_uv=True,
    f0_std=1.0,
    f0_mean=0.0,
):
    assert use_uv

    if pitch_norm == "standard":
        f0 = f0 * f0_std + f0_mean
    elif pitch_norm == "log":
        f0 = 2 ** f0

    if min is not None:
        f0 = f0.clamp(min=min)
    if max is not None:
        f0 = f0.clamp(max=max)
    if uv is not None and use_uv:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self, input, incremental_state=None, timestep=None, positions=None, **kwargs
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = (
            make_positions(input, self.padding_idx) if positions is None else positions
        )
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class FSLayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(FSLayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(FSLayerNorm, self).forward(x)
        return super(FSLayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class PitchPredictor(torch.nn.Module):
    def __init__(
        self,
        idim,
        n_layers=5,
        n_chans=384,
        odim=2,
        kernel_size=5,
        dropout_rate=0.1,
        padding="SAME",
    ):
        """Initialize pitch predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.ConstantPad1d(
                        ((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                        if padding == "SAME"
                        else (kernel_size - 1, 0),
                        0,
                    ),
                    torch.nn.Conv1d(
                        in_chans, n_chans, kernel_size, stride=1, padding=0
                    ),
                    torch.nn.ReLU(),
                    FSLayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, H)
        return xs


class Prenet(nn.Module):
    def __init__(self, in_dim=80, out_dim=256, kernel=5, n_layers=3, strides=None):
        super(Prenet, self).__init__()
        padding = kernel // 2
        self.layers = []
        self.strides = strides if strides is not None else [1] * n_layers
        for idx in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_dim,
                        out_dim,
                        kernel_size=kernel,
                        padding=padding,
                        stride=self.strides[idx],
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_dim),
                )
            )
            in_dim = out_dim
        self.layers = nn.ModuleList(self.layers)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        """

        :param x: [B, T, 80]
        :return: [L, B, T, H], [B, T, H]
        """
        padding_mask = x.abs().sum(-1).eq(0).data  # [B, T]
        nonpadding_mask_TB = 1 - padding_mask.float()[:, None, :]  # [B, 1, T]
        x = x.transpose(1, 2)
        hiddens = []
        for i, l in enumerate(self.layers):
            nonpadding_mask_TB = nonpadding_mask_TB[:, :, :: self.strides[i]]
            x = l(x) * nonpadding_mask_TB
        hiddens.append(x)
        hiddens = torch.stack(hiddens, 0)  # [L, B, H, T]
        hiddens = hiddens.transpose(2, 3)  # [L, B, T, H]
        x = self.out_proj(x.transpose(1, 2))  # [B, T, H]
        x = x * nonpadding_mask_TB.transpose(1, 2)
        return hiddens, x


class ConvBlock(nn.Module):
    def __init__(
        self, idim=80, n_chans=256, kernel_size=3, stride=1, norm="gn", dropout=0
    ):
        super().__init__()
        self.conv = ConvNorm(idim, n_chans, kernel_size, stride=stride)
        self.norm = norm
        if self.norm == "bn":
            self.norm = nn.BatchNorm1d(n_chans)
        elif self.norm == "in":
            self.norm = nn.InstanceNorm1d(n_chans, affine=True)
        elif self.norm == "gn":
            self.norm = nn.GroupNorm(n_chans // 16, n_chans)
        elif self.norm == "ln":
            self.norm = LayerNorm(n_chans // 16, n_chans)
        elif self.norm == "wn":
            self.conv = torch.nn.utils.weight_norm(self.conv.conv)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        :param x: [B, C, T]
        :return: [B, C, T]
        """
        x = self.conv(x)
        if not isinstance(self.norm, str):
            if self.norm == "none":
                pass
            elif self.norm == "ln":
                x = self.norm(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ConvStacks(nn.Module):
    def __init__(
        self,
        idim=80,
        n_layers=5,
        n_chans=256,
        odim=32,
        kernel_size=5,
        norm="gn",
        dropout=0,
        strides=None,
        res=True,
    ):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.res = res
        self.in_proj = Linear(idim, n_chans)
        if strides is None:
            strides = [1] * n_layers
        else:
            assert len(strides) == n_layers
        for idx in range(n_layers):
            self.conv.append(
                ConvBlock(
                    n_chans,
                    n_chans,
                    kernel_size,
                    stride=strides[idx],
                    norm=norm,
                    dropout=dropout,
                )
            )
        self.out_proj = Linear(n_chans, odim)

    def forward(self, x, return_hiddens=False):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x = self.in_proj(x)
        x = x.transpose(1, -1)  # (B, idim, Tmax)
        hiddens = []
        for f in self.conv:
            x_ = f(x)
            x = x + x_ if self.res else x_  # (B, C, Tmax)
            hiddens.append(x)
        x = x.transpose(1, -1)
        x = self.out_proj(x)  # (B, Tmax, H)
        if return_hiddens:
            hiddens = torch.stack(hiddens, 1)  # [B, L, C, T]
            return x, hiddens
        return x


class PitchExtractor(nn.Module):
    def __init__(
        self,
        n_mel_bins=80,
        conv_layers=2,
        hidden_size=256,
        predictor_hidden=-1,
        ffn_padding="SAME",
        predictor_kernel=5,
        pitch_type="frame",
        use_uv=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pitch_type = pitch_type
        assert pitch_type == "log"
        self.use_uv = use_uv
        self.predictor_hidden = (
            predictor_hidden if predictor_hidden > 0 else self.hidden_size
        )
        self.conv_layers = conv_layers

        self.mel_prenet = Prenet(n_mel_bins, self.hidden_size, strides=[1, 1, 1])
        if self.conv_layers > 0:
            self.mel_encoder = ConvStacks(
                idim=self.hidden_size,
                n_chans=self.hidden_size,
                odim=self.hidden_size,
                n_layers=self.conv_layers,
            )
        self.pitch_predictor = PitchPredictor(
            self.hidden_size,
            n_chans=self.predictor_hidden,
            n_layers=5,
            dropout_rate=0.1,
            odim=2,
            padding=ffn_padding,
            kernel_size=predictor_kernel,
        )

    def forward(self, mel_input=None):
        mel_hidden = self.mel_prenet(mel_input)[1]
        if self.conv_layers > 0:
            mel_hidden = self.mel_encoder(mel_hidden)

        # log2(f0), uv
        pitch_pred = self.pitch_predictor(mel_hidden)
        lf0, uv = pitch_pred[:, :, 0], pitch_pred[:, :, 1]

        # f0
        f0 = 2 ** lf0

        # log(f0)
        lf0 = torch.log(f0)

        lf0[uv > 0] = 0

        return lf0


class PitchExtractorWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = PitchExtractor(**kwargs)

    def forward(self, x, lengths=None, y=None):
        return self.pitch_extractor(x)
