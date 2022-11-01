import numpy as np
import torch
from nnsvs.usfgan.utils import SignalGenerator, dilated_factor
from torch import nn


class USFGANWrapper(nn.Module):
    def __init__(self, config, generator):
        super().__init__()
        self.generator = generator
        self.config = config

    def inference(self, f0, aux_feats):
        """Inference for USFGAN

        Args:
            f0 (numpy.ndarray): F0 (T, 1)
            aux_feats (Tensor): Auxiliary features (T, C)

        """
        signal_generator = SignalGenerator(
            sample_rate=self.config.data.sample_rate,
            hop_size=self.config.data.hop_size,
            sine_amp=self.config.data.sine_amp,
            noise_amp=self.config.data.noise_amp,
            signal_types=self.config.data.signal_types,
        )
        pad_fn = nn.ReplicationPad1d(self.config.generator.aux_context_window)
        assert self.config.data.sine_f0_type in ["contf0", "f0"]
        assert self.config.data.df_f0_type in ["contf0", "f0"]
        df = dilated_factor(
            np.squeeze(f0.copy()),
            self.config.data.sample_rate,
            self.config.data.dense_factor,
        )
        df = df.repeat(self.config.data.hop_size, axis=0)

        device = aux_feats.device

        c = pad_fn(aux_feats.unsqueeze(0).transpose(2, 1)).to(device)
        df = torch.FloatTensor(df).view(1, 1, -1).to(device)
        f0 = torch.FloatTensor(f0).unsqueeze(0).transpose(2, 1).to(device)

        in_signal = signal_generator(f0)
        y = self.generator(in_signal, c, df)[0]

        return y
