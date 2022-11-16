import numpy as np
import torch
from nnsvs.acoustic_models.util import pad_inference
from nnsvs.base import BaseModel, PredictionType
from nnsvs.mdn import MDNLayer, mdn_get_most_probable_sigma_and_mu, mdn_get_sample
from nnsvs.tacotron.decoder import Prenet, ZoneOutCell
from nnsvs.util import init_weights
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = [
    "ResF0NonAttentiveDecoder",
    "MDNResF0NonAttentiveDecoder",
    "BiLSTMResF0NonAttentiveDecoder",
]


class ResF0NonAttentiveDecoder(BaseModel):
    """Duration-informed Tacotron with residual F0 prediction.

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
        downsample_by_conv (bool) : if True, downsample by convolution
        scaled_tanh (bool) : if True, use scaled tanh for residual F0 prediction
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum of lf0 in the training data of input features
        in_lf0_max (float): maximum of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        init_type (str): initialization type
        eval_dropout (bool): if True, use dropout in evaluation
    """

    def __init__(
        self,
        in_dim=512,
        out_dim=1,
        layers=2,
        hidden_dim=1024,
        prenet_layers=2,
        prenet_hidden_dim=256,
        prenet_dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        downsample_by_conv=False,
        scaled_tanh=True,
        # NOTE: you must carefully set the following parameters
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        init_type="none",
        eval_dropout=True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.reduction_factor = reduction_factor
        self.prenet_dropout = prenet_dropout
        self.scaled_tanh = scaled_tanh
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

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

    def has_residual_lf0_prediction(self):
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

        # Denormalize lf0 from input musical score
        lf0_score = encoder_outs[:, :, self.in_lf0_idx].unsqueeze(-1)
        lf0_score_denorm = (
            lf0_score * (self.in_lf0_max - self.in_lf0_min) + self.in_lf0_min
        )
        # (B, T, C) -> (B, C, T)
        lf0_score_denorm = lf0_score_denorm.transpose(1, 2)

        # To avoid unbounded residual f0 that would potentially cause artifacts,
        # let's constrain the residual F0 to be in a certain range by the scaled tanh
        residual_f0_max_cent = 600
        max_lf0_ratio = residual_f0_max_cent * np.log(2) / 1200

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

        go_frame = encoder_outs.new_zeros(encoder_outs.size(0), self.out_dim)
        prev_out = go_frame

        if not is_inference and self.prenet is not None:
            prenet_outs = self.prenet(decoder_targets)

        outs = []
        lf0_residuals = []
        for t in range(encoder_outs.shape[1]):
            # Pre-Net
            if self.prenet is not None:
                if is_inference:
                    prenet_out = self.prenet(prev_out)
                else:
                    prenet_out = prenet_outs[:, t, :]
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
            out = self.feat_out(hcs).view(encoder_outs.size(0), self.out_dim, -1)
            # Residual F0
            if self.scaled_tanh:
                lf0_residual = max_lf0_ratio * torch.tanh(
                    out[:, self.out_lf0_idx, :]
                ).unsqueeze(1)
            else:
                lf0_residual = out[:, self.out_lf0_idx, :].unsqueeze(1)

            # Residual connection in the denormalized f0 domain
            lf0_score_denorm_t = lf0_score_denorm[
                :, :, t * self.reduction_factor : (t + 1) * self.reduction_factor
            ]
            lf0_pred_denorm = lf0_score_denorm_t + lf0_residual
            # Back to normalized f0
            lf0_pred = (lf0_pred_denorm - self.out_lf0_mean) / self.out_lf0_scale
            out[:, self.out_lf0_idx, :] = lf0_pred.squeeze(1)

            outs.append(out)
            lf0_residuals.append(lf0_residual)

            # Update decoder input for the next time step
            if is_inference:
                prev_out = outs[-1][:, :, -1]  # (1, out_dim)
            else:
                # Teacher forcing
                prev_out = decoder_targets[:, t, :]

        outs = torch.cat(outs, dim=2)  # (B, out_dim, Lmax)
        lf0_residuals = torch.cat(lf0_residuals, dim=2)  # (B, out_dim, Lmax)

        # (B, C, T) -> (B, T, C)
        outs = outs.transpose(1, 2)
        lf0_residuals = lf0_residuals.transpose(1, 2)

        return outs, lf0_residuals

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self, x=x, lengths=lengths, reduction_factor=self.reduction_factor
        )


class MDNResF0NonAttentiveDecoder(BaseModel):
    """Duration-informed Tacotron with residual F0 prediction (MDN-version)

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
        downsample_by_conv (bool) : if True, downsample by convolution
        scaled_tanh (bool) : if True, use scaled tanh for residual F0 prediction
        num_gaussians (int) : number of Gaussian
        sampling_mode (str) : sampling mode
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum of lf0 in the training data of input features
        in_lf0_max (float): maximum of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        init_type (str): initialization type
        eval_dropout (bool): if True, use dropout in evaluation
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
        scaled_tanh=True,
        num_gaussians=4,
        sampling_mode="mean",
        # NOTE: you must carefully set the following parameters
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        init_type="none",
        eval_dropout=True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.reduction_factor = reduction_factor
        self.prenet_dropout = prenet_dropout
        self.scaled_tanh = scaled_tanh
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale
        self.num_gaussians = num_gaussians
        self.sampling_mode = sampling_mode
        assert sampling_mode in ["mean", "random"]

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

    def prediction_type(self):
        return PredictionType.PROBABILISTIC

    def is_autoregressive(self):
        return True

    def has_residual_lf0_prediction(self):
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

        # Denormalize lf0 from input musical score
        lf0_score = encoder_outs[:, :, self.in_lf0_idx].unsqueeze(-1)
        lf0_score_denorm = (
            lf0_score * (self.in_lf0_max - self.in_lf0_min) + self.in_lf0_min
        )
        # (B, T, C) -> (B, C, T)
        # (B, T, C)
        # lf0_score_denorm = lf0_score_denorm.transpose(1, 2)

        # To avoid unbounded residual f0 that would potentially cause artifacts,
        # let's constrain the residual F0 to be in a certain range by the scaled tanh
        residual_f0_max_cent = 600
        max_lf0_ratio = residual_f0_max_cent * np.log(2) / 1200

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

        go_frame = encoder_outs.new_zeros(encoder_outs.size(0), self.out_dim)
        prev_out = go_frame

        if not is_inference and self.prenet is not None:
            prenet_outs = self.prenet(decoder_targets)

        mus = []
        log_pis = []
        log_sigmas = []
        lf0_residuals = []
        mus_inf = []
        for t in range(encoder_outs.shape[1]):
            # Pre-Net
            if self.prenet is not None:
                if is_inference:
                    prenet_out = self.prenet(prev_out)
                else:
                    prenet_out = prenet_outs[:, t, :]
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

            # (B, reduction_factor, num_gaussians, out_dim)
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

            # Residual F0
            # (B, reduction_factor, num_gaussians)
            if self.scaled_tanh:
                lf0_residual = max_lf0_ratio * torch.tanh(mu[:, :, :, self.out_lf0_idx])
            else:
                lf0_residual = mu[:, :, :, self.out_lf0_idx]

            # Residual connection in the denormalized f0 domain
            lf0_score_denorm_t = lf0_score_denorm[
                :, t * self.reduction_factor : (t + 1) * self.reduction_factor, :
            ]
            # NOTE: broadcast against num_gaussians axis
            # (B, 1, 1) + (B, 1, num_gaussians) -> (B, 1, num_gaussians)
            lf0_pred_denorm = lf0_score_denorm_t + lf0_residual
            # Back to normalized f0
            lf0_pred = (lf0_pred_denorm - self.out_lf0_mean) / self.out_lf0_scale
            mu[:, :, :, self.out_lf0_idx] = lf0_pred

            mus.append(mu)
            log_pis.append(log_pi)
            log_sigmas.append(log_sigma)
            lf0_residuals.append(lf0_residual)

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

        # (B, T, G, out_dim)
        mus = torch.cat(mus, dim=1)
        log_pis = torch.cat(log_pis, dim=1)
        log_sigmas = torch.cat(log_sigmas, dim=1)
        # (B, T, num_gaussians)
        lf0_residuals = torch.cat(lf0_residuals, dim=1)

        if is_inference:
            mu = torch.cat(mus_inf, dim=1)
            # TODO: may need to track sigma. For now we only use mu
            return mu, mu
        else:
            return (log_pis, log_sigmas, mus), lf0_residuals

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self,
            x=x,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            mdn=True,
        )


class BiLSTMResF0NonAttentiveDecoder(BaseModel):
    """BiLSTM-based encoder + duration-informed Tacotron with residual F0 prediction.

    Args:
        in_dim (int) : dimension of encoder hidden layer
        ff_hidden_dim (int): Hidden dimension of feed-forward layers in the encoder.
        conv_hidden_dim (int): Hidden dimension of convolution layers in the encoder.
        lstm_hidden_dim (int): Hidden dimension of LSTM layers in the encoder.
        num_lstm_layers (int): Number of LSTM layers in the encoder.
        out_dim (int) : dimension of output
        layers (int) : number of LSTM layers
        hidden_dim (int) : dimension of hidden layer
        prenet_layers (int) : number of pre-net layers
        prenet_hidden_dim (int) : dimension of pre-net hidden layer
        prenet_dropout (float) : dropout rate of pre-net
        zoneout (float) : zoneout rate
        reduction_factor (int) : reduction factor
        downsample_by_conv (bool) : if True, downsample by convolution
        scaled_tanh (bool) : if True, use scaled tanh for residual F0 prediction
        in_lf0_idx (int): index of lf0 in input features
        in_lf0_min (float): minimum of lf0 in the training data of input features
        in_lf0_max (float): maximum of lf0 in the training data of input features
        out_lf0_idx (int): index of lf0 in output features
        out_lf0_mean (float): mean of lf0 in the training data of output features
        out_lf0_scale (float): scale of lf0 in the training data of output features
        use_mdn (bool): if True, use mixture density network for F0 prediction
        num_gaussians (int): number of gaussians in MDN
        sampling_mode (str): sampling mode in inference. "mean" or "random"
        in_ph_start_idx (int): Start index of phoneme features.
        in_ph_end_idx (int): End index of phoneme features.
        embed_dim (int): Embedding dimension.
        init_type (str): initialization type
    """

    def __init__(
        self,
        in_dim=512,
        ff_hidden_dim=2048,
        conv_hidden_dim=1024,
        lstm_hidden_dim=256,
        num_lstm_layers=2,
        dropout=0.0,
        out_dim=80,
        decoder_layers=2,
        decoder_hidden_dim=1024,
        prenet_layers=2,
        prenet_hidden_dim=256,
        prenet_dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        downsample_by_conv=False,
        scaled_tanh=True,
        # NOTE: you must carefully set the following parameters
        in_lf0_idx=300,
        in_lf0_min=5.3936276,
        in_lf0_max=6.491111,
        out_lf0_idx=180,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        use_mdn=False,
        num_gaussians=4,
        sampling_mode="mean",
        in_ph_start_idx: int = 1,
        in_ph_end_idx: int = 50,
        embed_dim=None,
        init_type="none",
    ):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_min = in_lf0_min
        self.in_lf0_max = in_lf0_max
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale
        self.use_mdn = use_mdn
        self.in_dim = in_dim
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim

        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            in_ff_dim = embed_dim
        else:
            in_ff_dim = in_dim

        # Encoder
        # NOTE: can be simply replaced by a BiLSTM?
        # so far I use sinsy like architecture
        self.ff = nn.Sequential(
            nn.Linear(in_ff_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(ff_hidden_dim + 1, conv_hidden_dim, kernel_size=7, padding=0),
            nn.BatchNorm1d(conv_hidden_dim),
            nn.ReLU(),
            nn.ReflectionPad1d(3),
            nn.Conv1d(conv_hidden_dim, conv_hidden_dim, kernel_size=7, padding=0),
            nn.BatchNorm1d(conv_hidden_dim),
            nn.ReLU(),
            nn.ReflectionPad1d(3),
            nn.Conv1d(conv_hidden_dim, conv_hidden_dim, kernel_size=7, padding=0),
            nn.BatchNorm1d(conv_hidden_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            conv_hidden_dim,
            lstm_hidden_dim,
            num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        # Autoregressive decoder
        decoder_in_dim = 2 * lstm_hidden_dim + 1
        # NOTE: hard code in_lf0_idx to -1
        cls = MDNResF0NonAttentiveDecoder if use_mdn else ResF0NonAttentiveDecoder
        if use_mdn:
            ex_kwargs = {"num_gaussians": num_gaussians, "sampling_mode": sampling_mode}
        else:
            ex_kwargs = {}
        self.decoder = cls(
            in_dim=decoder_in_dim,
            out_dim=out_dim,
            layers=decoder_layers,
            hidden_dim=decoder_hidden_dim,
            prenet_layers=prenet_layers,
            prenet_hidden_dim=prenet_hidden_dim,
            prenet_dropout=prenet_dropout,
            zoneout=zoneout,
            reduction_factor=reduction_factor,
            downsample_by_conv=downsample_by_conv,
            scaled_tanh=scaled_tanh,
            in_lf0_idx=-1,
            in_lf0_min=in_lf0_min,
            in_lf0_max=in_lf0_max,
            out_lf0_idx=out_lf0_idx,
            out_lf0_mean=out_lf0_mean,
            out_lf0_scale=out_lf0_scale,
            **ex_kwargs,
        )
        init_weights(self, init_type)

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.decoder, "out_lf0_mean"):
            self.decoder.in_lf0_min = self.in_lf0_min
            self.decoder.in_lf0_max = self.in_lf0_max
            self.decoder.out_lf0_mean = self.out_lf0_mean
            self.decoder.out_lf0_scale = self.out_lf0_scale

    def is_autoregressive(self):
        return self.decoder.is_autoregressive()

    def prediction_type(self):
        return (
            PredictionType.PROBABILISTIC
            if self.use_mdn
            else PredictionType.DETERMINISTIC
        )

    def has_residual_lf0_prediction(self):
        return True

    def forward(self, x, lengths=None, y=None):
        self._set_lf0_params()
        lf0_score = x[:, :, self.in_lf0_idx].unsqueeze(-1)

        if self.embed_dim is not None:
            x_first, x_ph_onehot, x_last = torch.split(
                x,
                [
                    self.in_ph_start_idx,
                    self.num_vocab,
                    self.in_dim - self.num_vocab - self.in_ph_start_idx,
                ],
                dim=-1,
            )
            x_ph = torch.argmax(x_ph_onehot, dim=-1)
            # Make sure to have one-hot vector
            assert (x_ph_onehot.sum(-1) <= 1).all()
            x = self.emb(x_ph) + self.fc_in(torch.cat([x_first, x_last], dim=-1))

        if isinstance(lengths, torch.Tensor):
            lengths = lengths.to("cpu")

        out = self.ff(x)
        out = torch.cat([out, lf0_score], dim=-1)

        out = self.conv(out.transpose(1, 2)).transpose(1, 2)
        sequence = pack_padded_sequence(out, lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)

        # NOTE: need to concat the lf0 score to the output of the lstm to tell
        # the decoder the lf0
        out = torch.cat([out, lf0_score], dim=-1)

        outs, lf0_residual = self.decoder(out, lengths, y)

        return outs, lf0_residual

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self,
            x=x,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            mdn=self.use_mdn,
        )
