from nnsvs.base import BaseModel
from nnsvs.tacotron.decoder import Decoder
from nnsvs.tacotron.decoder import NoAttDecoder as NoAttTacotron2Decoder
from nnsvs.tacotron.encoder import Encoder as Tacotron2Encoder
from nnsvs.tacotron.postnet import Postnet as Tacotron2Postnet


class Tacotron2(BaseModel):
    def __init__(
        self,
        in_dim=512,
        out_dim=80,
        encoder_hidden_dim=512,
        encoder_conv_layers=3,
        encoder_conv_channels=512,
        encoder_conv_kernel_size=5,
        encoder_dropout=0.5,
        decoder_layers=2,
        decoder_hidden_dim=1024,
        decoder_prenet_layers=2,
        decoder_prenet_hidden_dim=256,
        decoder_prenet_dropout=0.5,
        decoder_zoneout=0.1,
        decoder_attention_hidden_dim=129,
        decoder_attention_conv_channels=32,
        decoder_attention_conv_kernel_size=31,
        postnet_layers=5,
        postnet_channels=512,
        postnet_kernel_size=5,
        postnet_dropout=0.5,
        reduction_factor=1,
    ):
        super().__init__()
        self.encoder = Tacotron2Encoder(
            in_dim,
            encoder_hidden_dim,
            encoder_conv_layers,
            encoder_conv_channels,
            encoder_conv_kernel_size,
            encoder_dropout,
        )
        self.decoder = Decoder(
            encoder_hidden_dim,
            out_dim,
            decoder_layers,
            decoder_hidden_dim,
            decoder_prenet_layers,
            decoder_prenet_hidden_dim,
            decoder_prenet_dropout,
            decoder_zoneout,
            reduction_factor,
            decoder_attention_hidden_dim,
            decoder_attention_conv_channels,
            decoder_attention_conv_kernel_size,
        )
        self.postnet = Tacotron2Postnet(
            out_dim,
            postnet_layers,
            postnet_channels,
            postnet_kernel_size,
            postnet_dropout,
        )

    def is_autoregressive(self):
        return True

    def forward(self, inputs, in_lens, decoder_targets):
        encoder_outs = self.encoder(inputs, in_lens)

        outs = self.decoder(encoder_outs, in_lens, decoder_targets)

        outs_fine = outs + self.postnet(outs)

        # (B, C, T) -> (B, T, C)
        outs = outs.transpose(2, 1)
        outs_fine = outs_fine.transpose(2, 1)

        return outs_fine

    def inference(self, seq, lengths=None):
        """Inference step

        Args:
            seq (torch.Tensor): input sequence

        Returns:
            torch.Tensor: the output
        """
        return self(seq, lengths, None)


class NoAttTacotron2(BaseModel):
    def __init__(
        self,
        in_dim=512,
        out_dim=80,
        encoder_hidden_dim=512,
        encoder_conv_layers=3,
        encoder_conv_channels=512,
        encoder_conv_kernel_size=5,
        encoder_dropout=0.5,
        decoder_layers=2,
        decoder_hidden_dim=1024,
        decoder_prenet_layers=2,
        decoder_prenet_hidden_dim=256,
        decoder_prenet_dropout=0.5,
        decoder_zoneout=0.1,
        postnet_layers=5,
        postnet_channels=512,
        postnet_kernel_size=5,
        postnet_dropout=0.5,
        reduction_factor=1,
    ):
        super().__init__()
        self.encoder = Tacotron2Encoder(
            in_dim,
            encoder_hidden_dim,
            encoder_conv_layers,
            encoder_conv_channels,
            encoder_conv_kernel_size,
            encoder_dropout,
        )
        self.decoder = NoAttTacotron2Decoder(
            encoder_hidden_dim,
            out_dim,
            decoder_layers,
            decoder_hidden_dim,
            decoder_prenet_layers,
            decoder_prenet_hidden_dim,
            decoder_prenet_dropout,
            decoder_zoneout,
            reduction_factor,
        )
        self.postnet = Tacotron2Postnet(
            out_dim,
            postnet_layers,
            postnet_channels,
            postnet_kernel_size,
            postnet_dropout,
        )

    def is_autoregressive(self):
        return True

    def forward(self, inputs, in_lens, decoder_targets):
        encoder_outs = self.encoder(inputs, in_lens)

        outs = self.decoder(encoder_outs, in_lens, decoder_targets)

        outs_fine = outs + self.postnet(outs)

        # (B, C, T) -> (B, T, C)
        outs = outs.transpose(2, 1)
        outs_fine = outs_fine.transpose(2, 1)

        return outs_fine

    def inference(self, seq, lengths=None):
        """Inference step

        Args:
            seq (torch.Tensor): input sequence

        Returns:
            torch.Tensor: the output
        """
        return self(seq, lengths, None)
