import json
from pathlib import Path
from warnings import warn

import numpy as np
import pysptk
import pyworld
import torch
from hydra.utils import instantiate
from nnmnkwii.io import hts
from nnmnkwii.postfilters import merlin_post_filter
from nnsvs.dsp import bandpass_filter
from nnsvs.gen import (
    gen_spsvs_static_features,
    gen_world_params,
    postprocess_duration,
    predict_acoustic,
    predict_duration,
    predict_timelag,
)
from nnsvs.multistream import get_static_stream_sizes
from nnsvs.pitch import lowpass_filter
from nnsvs.postfilters import variance_scaling
from nnsvs.util import MinMaxScaler, StandardScaler
from omegaconf import OmegaConf
from parallel_wavegan.utils import load_model


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

        # Post-filter
        if (model_dir / "postfilter_model.yaml").exists():
            self.postfilter_config = OmegaConf.load(model_dir / "postfilter_model.yaml")
            self.postfilter_model = instantiate(self.postfilter_config.netG).to(device)
            checkpoint = torch.load(
                model_dir / "postfilter_model.pth",
                map_location=device,
            )
            self.postfilter_model.load_state_dict(checkpoint["state_dict"])
            self.postfilter_model.eval()
            self.postfilter_out_scaler = StandardScaler(
                np.load(model_dir / "out_postfilter_scaler_mean.npy"),
                np.load(model_dir / "out_postfilter_scaler_var.npy"),
                np.load(model_dir / "out_postfilter_scaler_scale.npy"),
            )
        else:
            self.postfilter_model = None

        # Vocoder model
        if (model_dir / "vocoder_model.yaml").exists():
            self.vocoder_config = OmegaConf.load(model_dir / "vocoder_model.yaml")
            self.vocoder = load_model(
                model_dir / "vocoder_model.pth", config=self.vocoder_config
            ).to(device)
            self.vocoder.eval()
            self.vocoder.remove_weight_norm()
            self.vocoder_in_scaler = StandardScaler(
                np.load(model_dir / "in_vocoder_scaler_mean.npy"),
                np.load(model_dir / "in_vocoder_scaler_var.npy"),
                np.load(model_dir / "in_vocoder_scaler_scale.npy"),
            )
        else:
            self.vocoder = None

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

        repr = f"""Statistical parametric SVS (sampling rate: {self.config.sample_rate})
Time-lag model: {timelag_str}
Duration model: {duration_str}
Acoustic model: {acoustic_str}
"""
        if self.postfilter_model is not None:
            postfilter_str = json.dumps(
                OmegaConf.to_container(self.postfilter_config.netG),
                sort_keys=False,
                indent=4,
            )
            repr += f"Post-filter model: {postfilter_str}\n"
        else:
            repr += "Post-filter model: None\n"

        if self.vocoder is not None:
            vocoder_params = {
                "generator_type": self.vocoder_config.get(
                    "generator_type", "ParallelWaveGANGenerator"  # type: ignore
                ),
                "generator_params": OmegaConf.to_container(
                    self.vocoder_config.generator_params
                ),
            }
            vocoder_str = json.dumps(
                vocoder_params,
                sort_keys=False,
                indent=4,
            )
            repr += f"Vocoder model: {vocoder_str}\n"
        else:
            repr += "Vocoder model: WORLD\n"

        return repr

    def set_device(self, device):
        """Set device for the TTS models
        Args:
            device (str): cpu or cuda.
        """
        self.device = device
        self.duration_model.to(device)
        self.acoustic_model.to(device)

    @torch.no_grad()
    def svs(
        self,
        labels,
        vocoder_type="world",
        post_filter_type="merlin",
        trajectory_smoothing=True,
        trajectory_smoothing_cutoff=50,
        vuv_threshold=0.1,
        vibrato_scale=1.0,
        return_states=False,
        force_fix_vuv=True,
        post_filter=None,
    ):
        """Synthesize waveform given HTS-style labels

        Args:
            labels (nnmnkwii.io.HTSLabelFile): HTS-style labels
            vocoder_type (str): Vocoder type. world or pwg
            post_filter_type (str): Post-filter type. merlin or nnsvs.

        Returns:
            tuple: (synthesized waveform, sampling rate)
        """
        vocoder_type = vocoder_type.lower()
        if vocoder_type not in ["world", "pwg"]:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")
        if post_filter_type not in ["merlin", "nnsvs", "none"]:
            raise ValueError(f"Unknown post-filter type: {post_filter_type}")

        if vocoder_type == "pwg" and self.vocoder is None:
            raise ValueError(
                """Pre-trained vocodr model is not found.
WORLD is only supported for waveform generation"""
            )
        if post_filter is not None:
            warn("post_filter is deprecated. Use post_filter_type instead.")
            post_filter_type = "merlin" if post_filter else "none"

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

        # Learned post-filter using nnsvs
        if post_filter_type == "nnsvs" and self.postfilter_model is not None:
            # Apply GV post-filtering as the pre-processing step
            static_stream_sizes = get_static_stream_sizes(
                self.acoustic_config.stream_sizes,
                self.acoustic_config.has_dynamic_features,
                self.acoustic_config.num_windows,
            )
            mgc_end_dim = static_stream_sizes[0]
            acoustic_features[:, :mgc_end_dim] = variance_scaling(
                self.postfilter_out_scaler.var_.reshape(-1)[:mgc_end_dim],
                acoustic_features[:, :mgc_end_dim],
                offset=2,
            )
            # bap
            bap_start_dim = sum(static_stream_sizes[:3])
            bap_end_dim = sum(static_stream_sizes[:4])
            acoustic_features[:, bap_start_dim:bap_end_dim] = variance_scaling(
                self.postfilter_out_scaler.var_.reshape(-1)[bap_start_dim:bap_end_dim],
                acoustic_features[:, bap_start_dim:bap_end_dim],
                offset=0,
            )

            # Apply learned post-filter
            in_feats = (
                torch.from_numpy(acoustic_features).float().to(self.device).unsqueeze(0)
            )
            in_feats = self.postfilter_out_scaler.transform(in_feats).float()
            out_feats = self.postfilter_model.inference(in_feats, [in_feats.shape[1]])
            acoustic_features = (
                self.postfilter_out_scaler.inverse_transform(out_feats)
                .squeeze(0)
                .cpu()
                .numpy()
            )

        # Generate WORLD parameters
        mgc, lf0, vuv, bap = gen_spsvs_static_features(
            duration_modified_labels,
            acoustic_features,
            self.binary_dict,
            self.numeric_dict,
            self.acoustic_config.stream_sizes,
            self.acoustic_config.has_dynamic_features,
            self.config.acoustic.subphone_features,
            self.pitch_idx,
            self.acoustic_config.num_windows,
            self.config.frame_period,
            self.config.acoustic.relative_f0,
            vibrato_scale=vibrato_scale,
            vuv_threshold=vuv_threshold,
            force_fix_vuv=force_fix_vuv,
        )

        # NOTE: spectral enhancement based on the Merlin's post-filter implementation
        if post_filter_type == "merlin":
            alpha = pysptk.util.mcepalpha(self.config.sample_rate)
            mgc = merlin_post_filter(mgc, alpha)

        # Remove high-frequency components of mgc/bap
        # NOTE: It seems to be effective to suppress artifacts of GAN-based post-filtering
        if trajectory_smoothing:
            modfs = int(1 / 0.005)
            for d in range(mgc.shape[1]):
                mgc[:, d] = lowpass_filter(
                    mgc[:, d], modfs, cutoff=trajectory_smoothing_cutoff
                )
            for d in range(bap.shape[1]):
                bap[:, d] = lowpass_filter(
                    bap[:, d], modfs, cutoff=trajectory_smoothing_cutoff
                )

        # Waveform generation by (1) WORLD or (2) neural vocoder
        if vocoder_type == "world":
            f0, spectrogram, aperiodicity = gen_world_params(
                mgc, lf0, vuv, bap, self.config.sample_rate, vuv_threshold=vuv_threshold
            )

            wav = pyworld.synthesize(
                f0,
                spectrogram,
                aperiodicity,
                self.config.sample_rate,
                self.config.frame_period,
            )
        elif vocoder_type == "pwg":
            # NOTE: So far vocoder models are trained on binary V/UV features
            vuv = (vuv > vuv_threshold).astype(np.float32)
            voc_inp = (
                torch.from_numpy(
                    self.vocoder_in_scaler.transform(
                        np.concatenate([mgc, lf0, vuv, bap], axis=-1)
                    )
                )
                .float()
                .to(self.device)
            )
            wav = self.vocoder.inference(voc_inp).view(-1).to("cpu").numpy()

        wav = self.post_process(wav)

        if return_states:
            states = {
                "mgc": mgc,
                "lf0": lf0,
                "vuv": vuv,
                "bap": bap,
            }
            if vocoder_type == "world":
                states.update(
                    {
                        "f0": f0,
                        "spectrogram": spectrogram,
                        "aperiodicity": aperiodicity,
                    }
                )

            return wav, self.config.sample_rate, states

        return wav, self.config.sample_rate

    def post_process(self, wav):
        wav = bandpass_filter(wav, self.config.sample_rate)

        if np.max(wav) > 10:
            # data is likely already in [-32768, 32767]
            wav = wav.astype(np.int16)
        else:
            wav = np.clip(wav, -1.0, 1.0)
            wav = (wav * 32767.0).astype(np.int16)
        return wav
