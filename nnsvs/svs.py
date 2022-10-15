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
from nnsvs.io.hts import segment_labels
from nnsvs.multistream import get_static_features, get_static_stream_sizes
from nnsvs.pitch import lowpass_filter
from nnsvs.postfilters import variance_scaling
from nnsvs.util import MinMaxScaler, StandardScaler
from omegaconf import OmegaConf

try:
    from parallel_wavegan.utils import load_model

    _pwg_available = True
except ImportError:
    _pwg_available = False


def extract_static_scaler(out_scaler, model_config):
    mean_ = get_static_features(
        out_scaler.mean_.reshape(1, 1, out_scaler.mean_.shape[-1]),
        model_config.num_windows,
        model_config.stream_sizes,
        model_config.has_dynamic_features,
    )
    mean_ = np.concatenate(mean_, -1).reshape(1, -1)
    var_ = get_static_features(
        out_scaler.var_.reshape(1, 1, out_scaler.var_.shape[-1]),
        model_config.num_windows,
        model_config.stream_sizes,
        model_config.has_dynamic_features,
    )
    var_ = np.concatenate(var_, -1).reshape(1, -1)
    scale_ = get_static_features(
        out_scaler.scale_.reshape(1, 1, out_scaler.scale_.shape[-1]),
        model_config.num_windows,
        model_config.stream_sizes,
        model_config.has_dynamic_features,
    )
    scale_ = np.concatenate(scale_, -1).reshape(1, -1)
    static_scaler = StandardScaler(mean_, var_, scale_)
    return static_scaler


class SPSVS(object):
    """Statistical parametric singing voice synthesis

    .. note::
        This class is designed to be language-independent. Therefore,
        frontend functionality such as converting musicXML/UST to HTS labels
        is not included.

    Args:
        model_dir (str): directory of the model
        device (str): cpu or cuda

    Examples:

    Synthesize wavefrom from a musicxml file

    .. plot::

        import numpy as np
        import pysinsy
        from nnmnkwii.io import hts
        from nnsvs.pretrained import retrieve_pretrained_model
        from nnsvs.svs import SPSVS
        from nnsvs.util import example_xml_file
        import matplotlib.pyplot as plt

        model_dir = retrieve_pretrained_model("r9y9/yoko_latest")
        engine = SPSVS(model_dir)

        contexts = pysinsy.extract_fullcontext(example_xml_file(key="get_over"))
        labels = hts.HTSLabelFile.create_from_contexts(contexts)

        wav, sr = engine.svs(labels)

        fig, ax = plt.subplots(figsize=(8,2))
        librosa.display.waveshow(wav.astype(np.float32), sr, ax=ax)


    With a trained post-filter:

    >>> wav, sr = engine.svs(labels, posft_filter_type="nnsvs")

    With a trained neural vocoder:

    >>> wav, sr = engine.svs(labels, vocoder_type="pwg")

    With a global variance enhancement filter and a neural vocoder:

    >>> wav, sr = engine.svs(labels, post_filter_type="gv", vocoder_type="pwg")

    Default of the NNSVS v0.0.2 or earlier:

    >>> wav, sr = engine.svs(labels, post_filter_type="merlin", vocoder_type="world")
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
        # NOTE: this is used for GV post-filtering
        self.acoustic_out_static_scaler = extract_static_scaler(
            self.acoustic_out_scaler, self.acoustic_config
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
            if not _pwg_available:
                warn("parallel_wavegan is not installed. Vocoder model is disabled.")
                self.vocoder = None
            else:
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
            self.vocoder_config = None
            self.vocoder_in_scaler = None

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
        """Set device for the SVS model

        Args:
            device (str): cpu or cuda.
        """
        self.device = device
        self.timelag_model.to(device)
        self.duration_model.to(device)
        self.acoustic_model.to(device)
        self.postfilter_model.to(device) if self.postfilter_model is not None else None
        self.vocoder.to(device) if self.vocoder is not None else None

    def predict_timings(self, labels):
        """Predict timings for the given HTS labels.

        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels.

        Returns:
            nnmnkwii.io.hts.HTSLabelFile: HTS labels with predicted timings.
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
        return duration_modified_labels

    def synthesis_from_timings(
        self,
        duration_modified_labels,
        vocoder_type="world",
        post_filter_type="merlin",
        trajectory_smoothing=True,
        trajectory_smoothing_cutoff=50,
        trajectory_smoothing_cutoff_f0=20,
        vuv_threshold=0.5,
        vibrato_scale=1.0,
        return_states=False,
        force_fix_vuv=False,
        post_filter=None,
        segmented_synthesis=False,
    ):
        """Synthesize waveform from HTS labels with timings.

        Args:
            duration_modified_labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels
                with predicted timings.
            vocoder_type (str): Vocoder type. ``world`` or ``pwg`` is supported.
            post_filter_type (str): Post-filter type. ``merlin``, ``gv`` or ``nnsvs``
                is supported.
            trajectory_smoothing (bool): Whether to smooth acoustic feature trajectory.
            trajectory_smoothing_cutoff (int): Cutoff frequency for trajectory smoothing.
            trajectory_smoothing_cutoff_f0 (int): Cutoff frequency for trajectory
                smoothing of f0.
            vuv_threshold (float): Threshold for VUV.
            vibrato_scale (float): Scale for vibrato. Only valid if the acoustic
                features contain vibrato parameters.
            return_states (bool): Whether to return the internal states (for debugging)
            force_fix_vuv (bool): Whether to correct VUV.
            segmneted_synthesis (bool): Whether to use segmented synthesis.
        """
        if segmented_synthesis:
            segmented_labels = segment_labels(duration_modified_labels)
            from tqdm.auto import tqdm
        else:
            segmented_labels = [duration_modified_labels]

            def tqdm(x, **kwargs):
                return x

        wavs = []
        for seg_labels in tqdm(segmented_labels):
            # Predict acoustic features
            acoustic_features = predict_acoustic(
                self.device,
                seg_labels,
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

            # Apply GV post-filtering
            if post_filter_type in ["nnsvs", "gv"]:
                static_stream_sizes = get_static_stream_sizes(
                    self.acoustic_config.stream_sizes,
                    self.acoustic_config.has_dynamic_features,
                    self.acoustic_config.num_windows,
                )
                mgc_end_dim = static_stream_sizes[0]
                scaler = self.acoustic_out_static_scaler
                acoustic_features[:, :mgc_end_dim] = variance_scaling(
                    scaler.var_.reshape(-1)[:mgc_end_dim],
                    acoustic_features[:, :mgc_end_dim],
                    offset=2,
                )
                # bap
                bap_start_dim = sum(static_stream_sizes[:3])
                bap_end_dim = sum(static_stream_sizes[:4])
                acoustic_features[:, bap_start_dim:bap_end_dim] = variance_scaling(
                    scaler.var_.reshape(-1)[bap_start_dim:bap_end_dim],
                    acoustic_features[:, bap_start_dim:bap_end_dim],
                    offset=0,
                )

            # Learned post-filter using nnsvs
            if post_filter_type == "nnsvs" and self.postfilter_model is not None:
                in_feats = torch.from_numpy(acoustic_features).float().unsqueeze(0)
                in_feats = (
                    self.postfilter_out_scaler.transform(in_feats)
                    .float()
                    .to(self.device)
                )
                out_feats = self.postfilter_model.inference(
                    in_feats, [in_feats.shape[1]]
                )
                acoustic_features = (
                    self.postfilter_out_scaler.inverse_transform(out_feats.cpu())
                    .squeeze(0)
                    .numpy()
                )

            # Generate WORLD parameters
            mgc, lf0, vuv, bap = gen_spsvs_static_features(
                seg_labels,
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

            # Remove high-frequency components of lf0/mgc/bap
            # NOTE: Useful to reduce high-frequency artifacts
            if trajectory_smoothing:
                modfs = int(1 / 0.005)
                lf0[:, 0] = lowpass_filter(
                    lf0[:, 0], modfs, cutoff=trajectory_smoothing_cutoff_f0
                )
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
                    mgc,
                    lf0,
                    vuv,
                    bap,
                    self.config.sample_rate,
                    vuv_threshold=vuv_threshold,
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

            wavs.append(wav)

        # Concatenate segmented wavs
        wav = np.concatenate(wavs, axis=0).reshape(-1)
        wav = self.post_process(wav)

        if return_states:
            assert not segmented_synthesis
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

    @torch.no_grad()
    def svs(
        self,
        labels,
        vocoder_type="world",
        post_filter_type="merlin",
        trajectory_smoothing=True,
        trajectory_smoothing_cutoff=50,
        trajectory_smoothing_cutoff_f0=20,
        vuv_threshold=0.5,
        vibrato_scale=1.0,
        return_states=False,
        force_fix_vuv=False,
        post_filter=None,
        segmented_synthesis=False,
    ):
        """Synthesize waveform from HTS labels.

        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels
            vocoder_type (str): Vocoder type. ``world`` or ``pwg`` is supported.
            post_filter_type (str): Post-filter type. ``merlin``, ``gv`` or ``nnsvs``
                is supported.
            trajectory_smoothing (bool): Whether to smooth acoustic feature trajectory.
            trajectory_smoothing_cutoff (int): Cutoff frequency for trajectory smoothing.
            trajectory_smoothing_cutoff_f0 (int): Cutoff frequency for trajectory
                smoothing of f0.
            vuv_threshold (float): Threshold for VUV.
            vibrato_scale (float): Scale for vibrato. Only valid if the acoustic
                features contain vibrato parameters.
            return_states (bool): Whether to return the internal states (for debugging)
            force_fix_vuv (bool): Whether to correct VUV.
            segmneted_synthesis (bool): Whether to use segmented synthesis.
        """
        vocoder_type = vocoder_type.lower()
        if vocoder_type not in ["world", "pwg"]:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")
        if post_filter_type not in ["merlin", "nnsvs", "gv", "none"]:
            raise ValueError(f"Unknown post-filter type: {post_filter_type}")

        if vocoder_type == "pwg" and self.vocoder is None:
            raise ValueError(
                """Pre-trained vocodr model is not found.
WORLD is only supported for waveform generation"""
            )
        if post_filter is not None:
            warn("post_filter is deprecated. Use post_filter_type instead.")
            post_filter_type = "merlin" if post_filter else "none"

        duration_modified_labels = self.predict_timings(labels)

        return self.synthesis_from_timings(
            duration_modified_labels=duration_modified_labels,
            vocoder_type=vocoder_type,
            post_filter_type=post_filter_type,
            trajectory_smoothing=trajectory_smoothing,
            trajectory_smoothing_cutoff=trajectory_smoothing_cutoff,
            trajectory_smoothing_cutoff_f0=trajectory_smoothing_cutoff_f0,
            vuv_threshold=vuv_threshold,
            vibrato_scale=vibrato_scale,
            return_states=return_states,
            force_fix_vuv=force_fix_vuv,
            segmented_synthesis=segmented_synthesis,
        )

    def post_process(self, wav):
        wav = bandpass_filter(wav, self.config.sample_rate)

        if np.max(wav) > 10:
            # data is likely already in [-32768, 32767]
            wav = wav.astype(np.int16)
        else:
            wav = np.clip(wav, -1.0, 1.0)
            wav = (wav * 32767.0).astype(np.int16)
        return wav
