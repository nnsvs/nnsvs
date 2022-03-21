import json
from pathlib import Path

import numpy as np
import pyworld
import torch
from hydra.utils import instantiate
from nnmnkwii.io import hts
from nnsvs.gen import (
    gen_world_params,
    postprocess_duration,
    predict_acoustic,
    predict_duration,
    predict_timelag,
)
from nnsvs.util import MinMaxScaler, StandardScaler
from omegaconf import OmegaConf


class SPSVS(object):
    """Statistical parametric singing voice synthesis

    .. note::
        This class is designed to be language-independent. Therefore,
        frontend functionality such as converting musicXML to HTS labels
        is not included.

    Args:
        model_dir (str): directory of the model
        device (str): cpu or cuda
    """

    def __init__(self, model_dir, device="cpu"):
        self.device = device

        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        # search for config.yaml
        assert model_dir / "config.yaml"
        self.config = OmegaConf.load(model_dir / "config.yaml")

        # qst
        self.binary_dict, self.numeric_dict = hts.load_question_set(
            model_dir / "qst.hed"
        )

        self.pitch_idx = len(self.binary_dict) + 1
        self.pitch_indices = np.arange(len(self.binary_dict), len(self.binary_dict) + 3)

        # Time-lag model
        self.timelag_config = OmegaConf.load(model_dir / "timelag_model.yaml")
        self.timelag_model = instantiate(self.timelag_config.netG).to(device)
        checkpoint = torch.load(
            model_dir / "timelag_model.pth",
            map_location=device,
        )
        self.timelag_model.load_state_dict(checkpoint["state_dict"])

        self.timelag_in_scaler = MinMaxScaler(
            np.load(model_dir / "in_timelag_scaler_min.npy"),
            np.load(model_dir / "in_timelag_scaler_scale.npy"),
        )
        self.timelag_out_scaler = StandardScaler(
            np.load(model_dir / "out_timelag_scaler_mean.npy"),
            np.load(model_dir / "out_timelag_scaler_var.npy"),
            np.load(model_dir / "out_timelag_scaler_scale.npy"),
        )
        self.timelag_model.eval()

        # Duration model
        self.duration_config = OmegaConf.load(model_dir / "duration_model.yaml")
        self.duration_model = instantiate(self.duration_config.netG).to(device)
        checkpoint = torch.load(
            model_dir / "duration_model.pth",
            map_location=device,
        )
        self.duration_model.load_state_dict(checkpoint["state_dict"])

        self.duration_in_scaler = MinMaxScaler(
            np.load(model_dir / "in_duration_scaler_min.npy"),
            np.load(model_dir / "in_duration_scaler_scale.npy"),
        )
        self.duration_out_scaler = StandardScaler(
            np.load(model_dir / "out_duration_scaler_mean.npy"),
            np.load(model_dir / "out_duration_scaler_var.npy"),
            np.load(model_dir / "out_duration_scaler_scale.npy"),
        )
        self.duration_model.eval()

        # Acoustic model
        self.acoustic_config = OmegaConf.load(model_dir / "acoustic_model.yaml")
        self.acoustic_model = instantiate(self.acoustic_config.netG).to(device)
        checkpoint = torch.load(
            model_dir / "acoustic_model.pth",
            map_location=device,
        )
        self.acoustic_model.load_state_dict(checkpoint["state_dict"])
        self.acoustic_in_scaler = MinMaxScaler(
            np.load(model_dir / "in_acoustic_scaler_min.npy"),
            np.load(model_dir / "in_acoustic_scaler_scale.npy"),
        )
        self.acoustic_out_scaler = StandardScaler(
            np.load(model_dir / "out_acoustic_scaler_mean.npy"),
            np.load(model_dir / "out_acoustic_scaler_var.npy"),
            np.load(model_dir / "out_acoustic_scaler_scale.npy"),
        )
        self.acoustic_model.eval()

    def __repr__(self):
        timelag_str = json.dumps(
            OmegaConf.to_container(self.timelag_config.netG),
            sort_keys=False,
            indent=4,
        )
        duration_str = json.dumps(
            OmegaConf.to_container(self.duration_config.netG),
            sort_keys=False,
            indent=4,
        )
        acoustic_str = json.dumps(
            OmegaConf.to_container(self.acoustic_config.netG),
            sort_keys=False,
            indent=4,
        )

        return f"""Statistical parametric SVS (sampling rate: {self.config.sample_rate})
Time-lag model: {timelag_str}
Duration model: {duration_str}
Acoustic model: {acoustic_str}
"""

    def set_device(self, device):
        """Set device for the TTS models
        Args:
            device (str): cpu or cuda.
        """
        self.device = device
        self.duration_model.to(device)
        self.acoustic_model.to(device)

    @torch.no_grad()
    def svs(self, labels, return_states=False):
        """Synthesize waveform given HTS-style labels

        Args:
            labels (nnmnkwii.io.HTSLabelFile): HTS-style labels

        Returns:
            tuple: (synthesized waveform, sampling rate)
        """
        # Time-lag
        lag = predict_timelag(
            self.device,
            labels,
            self.timelag_model,
            self.timelag_config,
            self.timelag_in_scaler,
            self.timelag_out_scaler,
            self.binary_dict,
            self.numeric_dict,
            self.pitch_indices,
            self.config.log_f0_conditioning,
            self.config.timelag.allowed_range,
            self.config.timelag.allowed_range_rest,
            self.config.timelag.force_clip_input_features,
        )
        # Duration predictions
        durations = predict_duration(
            self.device,
            labels,
            self.duration_model,
            self.duration_config,
            self.duration_in_scaler,
            self.duration_out_scaler,
            self.binary_dict,
            self.numeric_dict,
            self.pitch_indices,
            self.config.log_f0_conditioning,
            self.config.duration.force_clip_input_features,
        )

        # Normalize phoneme durations
        duration_modified_labels = postprocess_duration(labels, durations, lag)

        # Predict acoustic features
        acoustic_features = predict_acoustic(
            self.device,
            duration_modified_labels,
            self.acoustic_model,
            self.acoustic_config,
            self.acoustic_in_scaler,
            self.acoustic_out_scaler,
            self.binary_dict,
            self.numeric_dict,
            self.config.acoustic.subphone_features,
            self.pitch_indices,
            self.config.log_f0_conditioning,
            self.config.acoustic.force_clip_input_features,
        )

        # Generate WORLD parameters
        f0, spectrogram, aperiodicity = gen_world_params(
            duration_modified_labels,
            acoustic_features,
            self.binary_dict,
            self.numeric_dict,
            self.acoustic_config.stream_sizes,
            self.acoustic_config.has_dynamic_features,
            self.config.acoustic.subphone_features,
            self.pitch_idx,
            self.acoustic_config.num_windows,
            self.config.acoustic.post_filter,
            self.config.sample_rate,
            self.config.frame_period,
            self.config.acoustic.relative_f0,
        )

        wav = pyworld.synthesize(
            f0,
            spectrogram,
            aperiodicity,
            self.config.sample_rate,
            self.config.frame_period,
        )

        wav = self.post_process(wav)

        if return_states:
            states = {
                "f0": f0,
                "spectrogram": spectrogram,
                "aperiodicity": aperiodicity,
            }
            return wav, self.config.sample_rate, states

        return wav, self.config.sample_rate

    def post_process(self, wav):
        if np.max(wav) > 10:
            # data is likely already in [-32768, 32767]
            wav = wav.astype(np.int16)
        else:
            wav = np.clip(wav, -1.0, 1.0)
            wav = (wav * 32767.0).astype(np.int16)
        return wav
