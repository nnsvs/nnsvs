# The code was adapted from ttslearn https://github.com/r9y9/ttslearn
# NonAttentiveDecoder is added to the original code.
# Acknowledgement: some of the code was adapted from ESPnet
#  Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn.functional as F
from nnsvs.base import BaseModel, PredictionType
from nnsvs.mdn import MDNLayer, mdn_get_most_probable_sigma_and_mu, mdn_get_sample
from nnsvs.util import init_weights
from torch import nn


def decoder_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain("tanh"))


class ZoneOutCell(nn.Module):
    def __init__(self, cell, zoneout=0.1):
        super().__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout = zoneout

    def forward(self, inputs, hidden):
        next_hidden = self.cell(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout)
        return next_hidden

    def _zoneout(self, h, next_h, prob):
        h_0, c_0 = h
        h_1, c_1 = next_h
        h_1 = self._apply_zoneout(h_0, h_1, prob)
        c_1 = self._apply_zoneout(c_0, c_1, prob)
        return h_1, c_1

    def _apply_zoneout(self, h, next_h, prob):
        if self.training:
            if prob > 0.0:
                mask = h.new(*h.size()).bernoulli_(prob)
            else:
                mask = 0
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h


class Prenet(nn.Module):
    """Pre-Net of Tacotron.

    Args:
        in_dim (int) : dimension of input
        layers (int) : number of pre-net layers
        hidden_dim (int) : dimension of hidden layer
        dropout (float) : dropout rate
    """

    def __init__(
        self, in_dim, layers=2, hidden_dim=256, dropout=0.5, eval_dropout=True
    ):
        super().__init__()
        self.dropout = dropout
        self.eval_dropout = eval_dropout
        prenet = nn.ModuleList()
        for layer in range(layers):
            prenet += [
                nn.Linear(in_dim if layer == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
            ]
        self.prenet = nn.Sequential(*prenet)

    def forward(self, x):
        """Forward step

        Args:
            x (torch.Tensor) : input tensor

        Returns:
            torch.Tensor : output tensor
        """
        for layer in self.prenet:
            if self.eval_dropout:
                x = F.dropout(layer(x), self.dropout, training=True)
            else:
                x = F.dropout(layer(x), self.dropout, training=self.training)
        return x


class NonAttentiveDecoder(BaseModel):
    """Decoder of Tacotron w/o attention mechanism

    Args:
        in_dim (int) : dimension of encoder hidden layer
        out_dim (int) : dimension of output
        layers (int) : number of LSTM layers
        hidden_dim (int) : dimension of hidden layer
        prenet_layers (int) : number of pre-net layers
        prenet_hidden_dim (int) : dimension of pre-net hidden layer
        prenet_dropout (float) : dropout rate of pre-net
        zoneout (float) : zoneout rate
        reduction_factor (int) : reduction factor
        attention_hidden_dim (int) : dimension of attention hidden layer
        attention_conv_channel (int) : number of attention convolution channels
        attention_conv_kernel_size (int) : kernel size of attention convolution
        downsample_by_conv (bool) : if True, downsample by convolution
        initial_value (float) : initial value for the autoregressive decoder.
    """

    def __init__(
        self,
        in_dim=512,
        out_dim=80,
        layers=2,
        hidden_dim=1024,
        prenet_layers=2,
        prenet_hidden_dim=256,
        prenet_dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        downsample_by_conv=False,
        init_type="none",
        eval_dropout=True,
        prenet_noise_std=0.0,
        initial_value=0.0,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.reduction_factor = reduction_factor
        self.prenet_dropout = prenet_dropout
        self.prenet_noise_std = prenet_noise_std
        self.initial_value = initial_value

        if prenet_layers > 0:
            self.prenet = Prenet(
                out_dim,
                prenet_layers,
                prenet_hidden_dim,
                prenet_dropout,
                eval_dropout=eval_dropout,
            )
            lstm_in_dim = in_dim + prenet_hidden_dim
        else:
            self.prenet = None
            prenet_hidden_dim = 0
            lstm_in_dim = in_dim + out_dim

        self.lstm = nn.ModuleList()
        for layer in range(layers):
            lstm = nn.LSTMCell(
                lstm_in_dim if layer == 0 else hidden_dim,
                hidden_dim,
            )
            self.lstm += [ZoneOutCell(lstm, zoneout)]

        proj_in_dim = in_dim + hidden_dim
        self.feat_out = nn.Linear(proj_in_dim, out_dim * reduction_factor, bias=False)

        if reduction_factor > 1 and downsample_by_conv:
            self.conv_downsample = nn.Conv1d(
                in_dim,
                in_dim,
                kernel_size=reduction_factor,
                stride=reduction_factor,
                groups=in_dim,
            )
        else:
            self.conv_downsample = None

        init_weights(self, init_type)

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def is_autoregressive(self):
        return True

    def forward(self, encoder_outs, in_lens, decoder_targets=None):
        """Forward step

        Args:
            encoder_outs (torch.Tensor): encoder outputs (B, T, C)
            in_lens (torch.Tensor): input lengths
            decoder_targets (torch.Tensor): decoder targets for teacher-forcing. (B, T, C)

        Returns:
            torch.Tensor: the output (B, C, T)
        """
        is_inference = decoder_targets is None
        if not is_inference:
            assert encoder_outs.shape[1] == decoder_targets.shape[1]

        # Adjust number of frames according to the reduction factor
        # (B, Lmax, out_dim) ->  (B, Lmax/r, out_dim)
        if self.reduction_factor > 1 and not is_inference:
            decoder_targets = decoder_targets[
                :, self.reduction_factor - 1 :: self.reduction_factor
            ]
        if self.reduction_factor > 1:
            if self.conv_downsample is not None:
                encoder_outs = self.conv_downsample(
                    encoder_outs.transpose(1, 2)
                ).transpose(1, 2)
            else:
                encoder_outs = encoder_outs[
                    :, self.reduction_factor - 1 :: self.reduction_factor
                ]

        h_list, c_list = [], []
        for _ in range(len(self.lstm)):
            h_list.append(self._zero_state(encoder_outs))
            c_list.append(self._zero_state(encoder_outs))

        go_frame = (
            encoder_outs.new_zeros(encoder_outs.size(0), self.out_dim)
            + self.initial_value
        )
        prev_out = go_frame

        if not is_inference and self.prenet is not None:
            prenet_outs = self.prenet(decoder_targets)

        outs = []
        for t in range(encoder_outs.shape[1]):
            # Pre-Net
            if self.prenet is not None:
                if is_inference:
                    prenet_out = self.prenet(prev_out)
                else:
                    prenet_out = prenet_outs[:, t, :]
            elif self.prenet_noise_std > 0:
                prenet_out = (
                    prev_out + torch.randn_like(prev_out) * self.prenet_noise_std
                )
            else:
                prenet_out = F.dropout(prev_out, self.prenet_dropout, training=True)

            # LSTM
            xs = torch.cat([encoder_outs[:, t], prenet_out], dim=1)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](
                    h_list[i - 1], (h_list[i], c_list[i])
                )
            # Output
            hcs = torch.cat([h_list[-1], encoder_outs[:, t]], dim=1)
            outs.append(self.feat_out(hcs).view(encoder_outs.size(0), self.out_dim, -1))

            # Update decoder input for the next time step
            if is_inference:
                prev_out = outs[-1][:, :, -1]  # (1, out_dim)
            else:
                # Teacher forcing
                prev_out = decoder_targets[:, t, :]

        outs = torch.cat(outs, dim=2)  # (B, out_dim, Lmax)

        if self.reduction_factor > 1:
            outs = outs.view(outs.size(0), self.out_dim, -1)  # (B, out_dim, Lmax)

        # (B, C, T) -> (B, T, C)
        return outs.transpose(1, 2)


class MDNNonAttentiveDecoder(BaseModel):
    """Non-atteive decoder with MDN

    Each decoder step outputs the parameters of MDN.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        layers (int): number of LSTM layers
        hidden_dim (int): hidden dimension
        prenet_layers (int): number of prenet layers
        prenet_hidden_dim (int): prenet hidden dimension
        prenet_dropout (float): prenet dropout rate
        zoneout (float): zoneout rate
        reduction_factor (int): reduction factor
        downsample_by_conv (bool): if True, use conv1d to downsample the input
        num_gaussians (int): number of Gaussians
        sampling_mode (str): sampling mode
        init_type (str): initialization type
        eval_dropout (bool): if True, use dropout in evaluation
        initial_value (float) : initial value for the autoregressive decoder.
    """

    def __init__(
        self,
        in_dim=512,
        out_dim=80,
        layers=2,
        hidden_dim=1024,
        prenet_layers=2,
        prenet_hidden_dim=256,
        prenet_dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        downsample_by_conv=False,
        num_gaussians=8,
        sampling_mode="mean",
        init_type="none",
        eval_dropout=True,
        prenet_noise_std=0.0,
        initial_value=0.0,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.reduction_factor = reduction_factor
        self.prenet_dropout = prenet_dropout
        self.prenet_noise_std = prenet_noise_std
        self.num_gaussians = num_gaussians
        self.sampling_mode = sampling_mode
        assert sampling_mode in ["mean", "random"]
        self.initial_value = initial_value

        if prenet_layers > 0:
            self.prenet = Prenet(
                out_dim,
                prenet_layers,
                prenet_hidden_dim,
                prenet_dropout,
                eval_dropout=eval_dropout,
            )
            lstm_in_dim = in_dim + prenet_hidden_dim
        else:
            self.prenet = None
            prenet_hidden_dim = 0
            lstm_in_dim = in_dim + out_dim

        self.lstm = nn.ModuleList()
        for layer in range(layers):
            lstm = nn.LSTMCell(
                lstm_in_dim if layer == 0 else hidden_dim,
                hidden_dim,
            )
            self.lstm += [ZoneOutCell(lstm, zoneout)]

        proj_in_dim = in_dim + hidden_dim
        self.feat_out = MDNLayer(
            proj_in_dim,
            out_dim * reduction_factor,
            num_gaussians=num_gaussians,
            dim_wise=True,
        )

        if reduction_factor > 1 and downsample_by_conv:
            self.conv_downsample = nn.Conv1d(
                in_dim,
                in_dim,
                kernel_size=reduction_factor,
                stride=reduction_factor,
                groups=in_dim,
            )
        else:
            self.conv_downsample = None

        init_weights(self, init_type)

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def is_autoregressive(self):
        return True

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def forward(self, encoder_outs, in_lens, decoder_targets=None):
        is_inference = decoder_targets is None
        if not is_inference:
            assert encoder_outs.shape[1] == decoder_targets.shape[1]

        # Adjust number of frames according to the reduction factor
        # (B, Lmax, out_dim) ->  (B, Lmax/r, out_dim)
        if self.reduction_factor > 1 and not is_inference:
            decoder_targets = decoder_targets[
                :, self.reduction_factor - 1 :: self.reduction_factor
            ]
        if self.reduction_factor > 1:
            if self.conv_downsample is not None:
                encoder_outs = self.conv_downsample(
                    encoder_outs.transpose(1, 2)
                ).transpose(1, 2)
            else:
                encoder_outs = encoder_outs[
                    :, self.reduction_factor - 1 :: self.reduction_factor
                ]

        h_list, c_list = [], []
        for _ in range(len(self.lstm)):
            h_list.append(self._zero_state(encoder_outs))
            c_list.append(self._zero_state(encoder_outs))

        go_frame = (
            encoder_outs.new_zeros(encoder_outs.size(0), self.out_dim)
            + self.initial_value
        )
        prev_out = go_frame

        if not is_inference and self.prenet is not None:
            prenet_outs = self.prenet(decoder_targets)

        mus = []
        log_pis = []
        log_sigmas = []
        # NOTE: only used for inference
        mus_inf = []

        for t in range(encoder_outs.shape[1]):
            # Pre-Net
            if self.prenet is not None:
                if is_inference:
                    prenet_out = self.prenet(prev_out)
                else:
                    prenet_out = prenet_outs[:, t, :]
            elif self.prenet_noise_std > 0:
                prenet_out = (
                    prev_out + torch.randn_like(prev_out) * self.prenet_noise_std
                )
            else:
                prenet_out = F.dropout(prev_out, self.prenet_dropout, training=True)

            # LSTM
            xs = torch.cat([encoder_outs[:, t], prenet_out], dim=1)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](
                    h_list[i - 1], (h_list[i], c_list[i])
                )
            # Output
            hcs = torch.cat([h_list[-1], encoder_outs[:, t]], dim=1)
            log_pi, log_sigma, mu = self.feat_out(hcs.unsqueeze(1))

            # (B, 1, num_gaussians, out_dim*reduction_factor)
            # -> (B, reduction_factor, num_gaussians, out_dim)
            log_pi = (
                log_pi.transpose(1, 2)
                .view(encoder_outs.size(0), self.num_gaussians, -1, self.out_dim)
                .transpose(1, 2)
            )
            log_sigma = (
                log_sigma.transpose(1, 2)
                .view(encoder_outs.size(0), self.num_gaussians, -1, self.out_dim)
                .transpose(1, 2)
            )
            mu = (
                mu.transpose(1, 2)
                .view(encoder_outs.size(0), self.num_gaussians, -1, self.out_dim)
                .transpose(1, 2)
            )

            mus.append(mu)
            log_pis.append(log_pi)
            log_sigmas.append(log_sigma)

            # Update decoder input for the next time step
            if is_inference:
                # (B, reduction_factor, out_dim)
                if self.sampling_mode == "mean":
                    _, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
                elif self.sampling_mode == "random":
                    mu = mdn_get_sample(log_pi, log_sigma, mu)

                # Feed last sample for the feedback loop
                prev_out = mu[:, -1]

                mus_inf.append(mu)
            else:
                # Teacher forcing
                prev_out = decoder_targets[:, t, :]

        mus = torch.cat(mus, dim=1)  # (B, out_dim, Lmax)
        log_pis = torch.cat(log_pis, dim=1)  # (B, out_dim, Lmax)
        log_sigmas = torch.cat(log_sigmas, dim=1)  # (B, out_dim, Lmax)

        if is_inference:
            mu = torch.cat(mus_inf, dim=1)
            # TODO: may need to track sigma. For now we only use mu
            return mu, mu
        else:
            return log_pis, log_sigmas, mus
