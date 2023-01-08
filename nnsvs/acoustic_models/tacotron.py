import torch
from nnsvs.acoustic_models.util import pad_inference
from nnsvs.base import BaseModel
from nnsvs.tacotron.decoder import MDNNonAttentiveDecoder
from nnsvs.tacotron.decoder import NonAttentiveDecoder as TacotronNonAttentiveDecoder
from nnsvs.tacotron.postnet import Postnet as TacotronPostnet
from nnsvs.util import init_weights
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = [
    "NonAttentiveDecoder",
    "MDNNonAttentiveDecoder",
    "BiLSTMNonAttentiveDecoder",
    "BiLSTMMDNNonAttentiveDecoder",
]


class NonAttentiveDecoder(TacotronNonAttentiveDecoder):
    """Non-attentive autoregresive model based on the duration-informed Tacotron

    Duration-informed Tacotron :cite:t:`okamoto2019tacotron`.

    .. note::

        if the target features of the decoder is normalized to N(0, 1), consider
        setting the initial value carefully so that it roughly matches the value of
        silence. e.g., -4 to -10.
        ``initial_value=0`` works okay for large databases but I found that -4 or
        lower worked better for smaller databases such as nit-song070.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        layers (int): Number of LSTM layers.
        hidden_dim (int): Hidden dimension of LSTM.
        prenet_layers (int): Number of prenet layers.
        prenet_hidden_dim (int): Hidden dimension of prenet.
        prenet_dropout (float): Dropout rate of prenet.
        zoneout (float): Zoneout rate.
        reduction_factor (int): Reduction factor.
        downsample_by_conv (bool): If True, downsampling is performed by convolution.
        postnet_layers (int): Number of postnet layers.
        postnet_channels (int): Number of postnet channels.
        postnet_kernel_size (int): Kernel size of postnet.
        postnet_dropout (float): Dropout rate of postnet.
        init_type (str): Initialization type.
        eval_dropout (bool): If True, dropout is applied in evaluation.
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
        postnet_layers=0,
        postnet_channels=512,
        postnet_kernel_size=5,
        postnet_dropout=0.0,
        init_type="none",
        eval_dropout=True,
        prenet_noise_std=0.0,
        initial_value=0.0,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            layers=layers,
            hidden_dim=hidden_dim,
            prenet_layers=prenet_layers,
            prenet_hidden_dim=prenet_hidden_dim,
            prenet_dropout=prenet_dropout,
            zoneout=zoneout,
            reduction_factor=reduction_factor,
            downsample_by_conv=downsample_by_conv,
            eval_dropout=eval_dropout,
            prenet_noise_std=prenet_noise_std,
            initial_value=initial_value,
        )
        if postnet_layers > 0:
            self.postnet = TacotronPostnet(
                out_dim,
                layers=postnet_layers,
                channels=postnet_channels,
                kernel_size=postnet_kernel_size,
                dropout=postnet_dropout,
            )
        else:
            self.postnet = None
        init_weights(self, init_type)

    def forward(self, x, lengths=None, y=None):
        outs = super().forward(x, lengths, y)

        if self.postnet is not None:
            # NOTE: `outs.clone()`` is necessary to compute grad on both outs and outs_fine
            outs_fine = outs + self.postnet(outs.transpose(1, 2).clone()).transpose(
                1, 2
            )
            return [outs, outs_fine]
        else:
            return outs

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self, x=x, lengths=lengths, reduction_factor=self.reduction_factor
        )


class BiLSTMNonAttentiveDecoder(BaseModel):
    """BiLSTM-based encoder + NonAttentiveDecoder

    The encoder is based on the arthitecture of the Sinsy acoustic model.

    Args:
        in_dim (int): Input dimension.
        ff_hidden_dim (int): Hidden dimension of feed-forward layers in the encoder.
        conv_hidden_dim (int): Hidden dimension of convolution layers in the encoder.
        lstm_hidden_dim (int): Hidden dimension of LSTM layers in the encoder.
        num_lstm_layers (int): Number of LSTM layers in the encoder.
        out_dim (int): Output dimension.
        layers (int): Number of LSTM layers.
        hidden_dim (int): Hidden dimension of LSTM.
        prenet_layers (int): Number of prenet layers.
        prenet_hidden_dim (int): Hidden dimension of prenet.
        prenet_dropout (float): Dropout rate of prenet.
        zoneout (float): Zoneout rate.
        reduction_factor (int): Reduction factor.
        downsample_by_conv (bool): If True, downsampling is performed by convolution.
        postnet_layers (int): Number of postnet layers.
        postnet_channels (int): Number of postnet channels.
        postnet_kernel_size (int): Kernel size of postnet.
        postnet_dropout (float): Dropout rate of postnet.
        in_ph_start_idx (int): Start index of phoneme features.
        in_ph_end_idx (int): End index of phoneme features.
        embed_dim (int): Embedding dimension.
        init_type (str): Initialization type.
        eval_dropout (bool): If True, dropout is applied in evaluation.
        initial_value (float) : initial value for the autoregressive decoder.
    """

    def __init__(
        self,
        in_dim=512,
        ff_hidden_dim=2048,
        conv_hidden_dim=1024,
        lstm_hidden_dim=256,
        num_lstm_layers=2,
        out_dim=80,
        decoder_layers=2,
        decoder_hidden_dim=1024,
        prenet_layers=2,
        prenet_hidden_dim=256,
        prenet_dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        downsample_by_conv=False,
        postnet_layers=0,
        postnet_channels=512,
        postnet_kernel_size=5,
        postnet_dropout=0.0,
        in_ph_start_idx: int = 1,
        in_ph_end_idx: int = 50,
        embed_dim=None,
        init_type="none",
        eval_dropout=True,
        prenet_noise_std=0.0,
        initial_value=0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim
        self.reduction_factor = reduction_factor

        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            ff_in_dim = embed_dim
        else:
            ff_in_dim = in_dim

        # Encoder
        # NOTE: can be simply replaced by a BiLSTM?
        # so far I use sinsy like architecture
        self.ff = nn.Sequential(
            nn.Linear(ff_in_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(ff_hidden_dim, conv_hidden_dim, kernel_size=7, padding=0),
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
            dropout=0.0,
        )

        # Autoregressive decoder
        decoder_in_dim = 2 * lstm_hidden_dim
        self.decoder = TacotronNonAttentiveDecoder(
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
            eval_dropout=eval_dropout,
            prenet_noise_std=prenet_noise_std,
            initial_value=initial_value,
        )

        if postnet_layers > 0:
            self.postnet = TacotronPostnet(
                out_dim,
                layers=postnet_layers,
                channels=postnet_channels,
                kernel_size=postnet_kernel_size,
                dropout=postnet_dropout,
            )
        else:
            self.postnet = None
        init_weights(self, init_type)

    def is_autoregressive(self):
        return self.decoder.is_autoregressive()

    def forward(self, x, lengths=None, y=None):
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.to("cpu")

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

        out = self.ff(x)
        out = self.conv(out.transpose(1, 2)).transpose(1, 2)
        sequence = pack_padded_sequence(out, lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        outs = self.decoder(out, lengths, y)

        if self.postnet is not None:
            # NOTE: `outs.clone()`` is necessary to compute grad on both outs and outs_fine
            outs_fine = outs + self.postnet(outs.transpose(1, 2).clone()).transpose(
                1, 2
            )
            return [outs, outs_fine]
        else:
            return outs

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self, x=x, lengths=lengths, reduction_factor=self.reduction_factor
        )


class BiLSTMMDNNonAttentiveDecoder(BaseModel):
    """BiLSTM-based encoder + NonAttentiveDecoder (MDN version)


    The encoder is based on the arthitecture of the Sinsy acoustic model.

    Args:
        in_dim (int): Input dimension.
        ff_hidden_dim (int): Hidden dimension of feed-forward layers in the encoder.
        conv_hidden_dim (int): Hidden dimension of convolution layers in the encoder.
        lstm_hidden_dim (int): Hidden dimension of LSTM layers in the encoder.
        num_lstm_layers (int): Number of LSTM layers in the encoder.
        out_dim (int): Output dimension.
        layers (int): Number of LSTM layers.
        hidden_dim (int): Hidden dimension of LSTM.
        prenet_layers (int): Number of prenet layers.
        prenet_hidden_dim (int): Hidden dimension of prenet.
        prenet_dropout (float): Dropout rate of prenet.
        zoneout (float): Zoneout rate.
        reduction_factor (int): Reduction factor.
        downsample_by_conv (bool): If True, downsampling is performed by convolution.
        num_gaussians (int): Number of Gaussians.
        sampling_mode (str): Sampling mode.
        postnet_layers (int): Number of postnet layers.
        postnet_channels (int): Number of postnet channels.
        postnet_kernel_size (int): Kernel size of postnet.
        postnet_dropout (float): Dropout rate of postnet.
        in_ph_start_idx (int): Start index of phoneme features.
        in_ph_end_idx (int): End index of phoneme features.
        embed_dim (int): Embedding dimension.
        init_type (str): Initialization type.
        eval_dropout (bool): If True, dropout is applied in evaluation.
        initial_value (float) : initial value for the autoregressive decoder.
    """

    def __init__(
        self,
        in_dim=512,
        ff_hidden_dim=2048,
        conv_hidden_dim=1024,
        lstm_hidden_dim=256,
        num_lstm_layers=2,
        out_dim=80,
        decoder_layers=2,
        decoder_hidden_dim=1024,
        prenet_layers=2,
        prenet_hidden_dim=256,
        prenet_dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        downsample_by_conv=False,
        num_gaussians=8,
        sampling_mode="mean",
        in_ph_start_idx: int = 1,
        in_ph_end_idx: int = 50,
        embed_dim=None,
        init_type="none",
        eval_dropout=True,
        prenet_noise_std=0,
        initial_value=0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim
        self.reduction_factor = reduction_factor

        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            ff_in_dim = embed_dim
        else:
            ff_in_dim = in_dim

        # Encoder
        # NOTE: can be simply replaced by a BiLSTM?
        # so far I use sinsy like architecture
        self.ff = nn.Sequential(
            nn.Linear(ff_in_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(ff_hidden_dim, conv_hidden_dim, kernel_size=7, padding=0),
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
            dropout=0.0,
        )

        # Autoregressive decoder
        decoder_in_dim = 2 * lstm_hidden_dim
        self.decoder = MDNNonAttentiveDecoder(
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
            num_gaussians=num_gaussians,
            sampling_mode=sampling_mode,
            eval_dropout=eval_dropout,
            prenet_noise_std=prenet_noise_std,
            initial_value=initial_value,
        )
        init_weights(self, init_type)

    def is_autoregressive(self):
        return self.decoder.is_autoregressive()

    def prediction_type(self):
        return self.decoder.prediction_type()

    def forward(self, x, lengths=None, y=None):
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.to("cpu")

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

        out = self.ff(x)
        out = self.conv(out.transpose(1, 2)).transpose(1, 2)
        sequence = pack_padded_sequence(out, lengths, batch_first=True)
        out, _ = self.lstm(sequence)
        out, _ = pad_packed_sequence(out, batch_first=True)
        outs = self.decoder(out, lengths, y)

        return outs

    def inference(self, x, lengths=None):
        return pad_inference(
            model=self,
            x=x,
            lengths=lengths,
            reduction_factor=self.reduction_factor,
            mdn=True,
        )
