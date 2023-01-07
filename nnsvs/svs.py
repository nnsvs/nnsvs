import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from nnmnkwii.io import hts
from nnsvs.gen import (
    postprocess_acoustic,
    postprocess_duration,
    postprocess_waveform,
    predict_acoustic,
    predict_duration,
    predict_timelag,
    predict_waveform,
)
from nnsvs.io.hts import get_pitch_index, get_pitch_indices, segment_labels
from nnsvs.logger import getLogger
from nnsvs.usfgan import USFGANWrapper
from nnsvs.util import MinMaxScaler, StandardScaler, extract_static_scaler, load_vocoder
from omegaconf import OmegaConf


class SPSVS(object):
    """Statistical parametric singing voice synthesis (SPSVS)

    This class is meant to be used for inference only. Use the ``svs`` method
    for the simplest inference, or use the separated methods (e.g.,
    ``predict_acoustic`` and ``predict_waveform``) to control each components
    of the SVS system.

    In addition, this class is designed to be language-independent. Therefore,
    frontend functionality such as converting musicXML/UST to HTS labels
    is not included.

    Args:
        model_dir (str): directory of the model
        device (str): cpu or cuda
        verbose (int): verbosity level

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

        # Instantiate the SVS engine
        model_dir = retrieve_pretrained_model("r9y9/yoko_latest")
        engine = SPSVS(model_dir)

        # Extract HTS labels from a MusicXML file
        contexts = pysinsy.extract_fullcontext(example_xml_file(key="get_over"))
        labels = hts.HTSLabelFile.create_from_contexts(contexts)

        # Run inference
        wav, sr = engine.svs(labels)

        # Plot the result
        fig, ax = plt.subplots(figsize=(8,2))
        librosa.display.waveshow(wav.astype(np.float32), sr=sr, ax=ax)


    With WORLD vocoder:

    >>> wav, sr = engine.svs(labels, vocoder_type="world")

    With a uSFGAN or SiFiGAN vocoder:

    >>> wav, sr = engine.svs(labels, vocoder_type="usfgan")
    """

    def __init__(self, model_dir, device="cpu", verbose=0):
        self.device = device

        self.logger = getLogger(verbose=verbose)

        if isinstance(model_dir, str):
            model_dir = Path(model_dir)

        # search for config.yaml
        assert model_dir / "config.yaml"
        self.config = OmegaConf.load(model_dir / "config.yaml")
        self.feature_type = self.config.get("feature_type", "world")

        # qst
        self.binary_dict, self.numeric_dict = hts.load_question_set(
            model_dir / "qst.hed"
        )

        self.pitch_idx = get_pitch_index(self.binary_dict, self.numeric_dict)
        self.pitch_indices = get_pitch_indices(self.binary_dict, self.numeric_dict)

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
            self.postfilter_config = None
            self.postfilter_out_scaler = None

        # Vocoder model
        if (model_dir / "vocoder_model.pth").exists():
            self.vocoder, self.vocoder_in_scaler, self.vocoder_config = load_vocoder(
                model_dir / "vocoder_model.pth", device, self.acoustic_config
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

    def predict_timelag(self, labels):
        """Predict time-ag from HTS labels

        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels.

        Returns:
            ndarray: Predicted time-lag.
        """
        lag = predict_timelag(
            self.device,
            labels,
            timelag_model=self.timelag_model,
            timelag_config=self.timelag_config,
            timelag_in_scaler=self.timelag_in_scaler,
            timelag_out_scaler=self.timelag_out_scaler,
            binary_dict=self.binary_dict,
            numeric_dict=self.numeric_dict,
            pitch_indices=self.pitch_indices,
            log_f0_conditioning=self.config.log_f0_conditioning,
            allowed_range=self.config.timelag.allowed_range,
            allowed_range_rest=self.config.timelag.allowed_range_rest,
            force_clip_input_features=self.config.timelag.force_clip_input_features,
            frame_period=self.config.frame_period,
        )
        return lag

    def predict_duration(self, labels):
        """Predict durations from HTS labels

        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels.

        Returns:
            ndarray: Predicted durations.
        """
        durations = predict_duration(
            self.device,
            labels,
            duration_model=self.duration_model,
            duration_config=self.duration_config,
            duration_in_scaler=self.duration_in_scaler,
            duration_out_scaler=self.duration_out_scaler,
            binary_dict=self.binary_dict,
            numeric_dict=self.numeric_dict,
            pitch_indices=self.pitch_indices,
            log_f0_conditioning=self.config.log_f0_conditioning,
            force_clip_input_features=self.config.duration.force_clip_input_features,
            frame_period=self.config.frame_period,
        )
        return durations

    def postprocess_duration(self, labels, pred_durations, lag):
        """Post-process durations

        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels.
            pred_durations (ndarray): Predicted durations.
            lag (ndarray): Predicted time-lag.

        Returns:
            nnmnkwii.io.hts.HTSLabelFile: duration modified HTS labels.
        """
        duration_modified_labels = postprocess_duration(
            labels, pred_durations, lag, frame_period=self.config.frame_period
        )
        return duration_modified_labels

    def predict_timing(self, labels):
        """Predict timing from HTS labels

        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels.

        Returns:
            nnmnkwii.io.hts.HTSLabelFile: duration modified HTS labels.
        """
        lag = self.predict_timelag(labels)
        durations = self.predict_duration(labels)
        duration_modified_full_labels = self.postprocess_duration(
            labels, durations, lag
        )
        return duration_modified_full_labels

    def predict_acoustic(self, duration_modified_labels, f0_shift_in_cent=0):
        """Predict acoustic features from HTS labels

        Args:
            duration_modified_labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels.
            f0_shift_in_cent (float): F0 shift in cent.

        Returns:
            ndarray: Predicted acoustic features.
        """
        acoustic_features = predict_acoustic(
            device=self.device,
            labels=duration_modified_labels,
            acoustic_model=self.acoustic_model,
            acoustic_config=self.acoustic_config,
            acoustic_in_scaler=self.acoustic_in_scaler,
            acoustic_out_scaler=self.acoustic_out_scaler,
            binary_dict=self.binary_dict,
            numeric_dict=self.numeric_dict,
            subphone_features=self.acoustic_config.get(
                "subphone_features", "coarse_coding"
            ),
            pitch_indices=self.pitch_indices,
            log_f0_conditioning=self.config.log_f0_conditioning,
            force_clip_input_features=self.acoustic_config.get(
                "force_clip_input_features", True
            ),
            frame_period=self.config.frame_period,
            f0_shift_in_cent=f0_shift_in_cent,
        )
        return acoustic_features

    def postprocess_acoustic(
        self,
        duration_modified_labels,
        acoustic_features,
        post_filter_type="gv",
        trajectory_smoothing=True,
        trajectory_smoothing_cutoff=50,
        trajectory_smoothing_cutoff_f0=20,
        vuv_threshold=0.5,
        force_fix_vuv=False,
        fill_silence_to_rest=False,
        f0_shift_in_cent=0,
    ):
        """Post-process acoustic features

        The function converts acoustic features in single ndarray to tuple of
        multi-stream acoustic features.

        e.g., array -> (mgc, lf0, vuv, bap)

        If post_filter_type=``nnsvs`` is specified, learned post-filter is applied.
        However, it is recommended to use ``gv`` in general.

        Args:
            duration_modified_labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels.
            acoustic_features (ndarray): Predicted acoustic features.
            post_filter_type (str): Post-filter type.
                One of ``gv``, ``merlin`` or ``nnsvs``. Recommended to use ``gv``
                for general purpose.
            trajectory_smoothing (bool): Whether to apply trajectory smoothing.
            trajectory_smoothing_cutoff (float): Cutoff frequency for trajectory smoothing
                of spectral features.
            trajectory_smoothing_cutoff_f0 (float): Cutoff frequency for trajectory
                smoothing of f0.
            vuv_threshold (float): V/UV threshold.
            force_fix_vuv (bool): Force fix V/UV.
            fill_silence_to_rest (bool): Fill silence to rest frames.
            f0_shift_in_cent (float): F0 shift in cent.

        Returns:
            tuple: Post-processed multi-stream acoustic features.
        """
        multistream_features = postprocess_acoustic(
            device=self.device,
            duration_modified_labels=duration_modified_labels,
            acoustic_features=acoustic_features,
            binary_dict=self.binary_dict,
            numeric_dict=self.numeric_dict,
            acoustic_config=self.acoustic_config,
            acoustic_out_static_scaler=self.acoustic_out_static_scaler,
            postfilter_model=self.postfilter_model,
            postfilter_config=self.postfilter_config,
            postfilter_out_scaler=self.postfilter_out_scaler,
            sample_rate=self.config.sample_rate,
            frame_period=self.config.frame_period,
            relative_f0=self.config.acoustic.relative_f0,
            feature_type=self.feature_type,
            post_filter_type=post_filter_type,
            trajectory_smoothing=trajectory_smoothing,
            trajectory_smoothing_cutoff=trajectory_smoothing_cutoff,
            trajectory_smoothing_cutoff_f0=trajectory_smoothing_cutoff_f0,
            vuv_threshold=vuv_threshold,
            f0_shift_in_cent=f0_shift_in_cent,
            vibrato_scale=1.0,  # only valid for Sinsy-like models
            force_fix_vuv=force_fix_vuv,
            fill_silence_to_rest=fill_silence_to_rest,
        )
        return multistream_features

    def predict_waveform(
        self,
        multistream_features,
        vocoder_type="world",
        vuv_threshold=0.5,
    ):
        wav = predict_waveform(
            device=self.device,
            multistream_features=multistream_features,
            vocoder=self.vocoder,
            vocoder_config=self.vocoder_config,
            vocoder_in_scaler=self.vocoder_in_scaler,
            sample_rate=self.config.sample_rate,
            frame_period=self.config.frame_period,
            use_world_codec=self.config.get("use_world_codec", False),
            feature_type=self.feature_type,
            vocoder_type=vocoder_type,
            vuv_threshold=vuv_threshold,
        )
        return wav

    def postprocess_waveform(
        self,
        wav,
        dtype=np.int16,
        peak_norm=False,
        loudness_norm=False,
        target_loudness=-20,
    ):
        wav = postprocess_waveform(
            wav=wav,
            sample_rate=self.config.sample_rate,
            dtype=dtype,
            peak_norm=peak_norm,
            loudness_norm=loudness_norm,
            target_loudness=target_loudness,
        )
        return wav

    def svs(
        self,
        labels,
        vocoder_type="world",
        post_filter_type="gv",
        trajectory_smoothing=True,
        trajectory_smoothing_cutoff=50,
        trajectory_smoothing_cutoff_f0=20,
        vuv_threshold=0.5,
        pre_f0_shift_in_cent=0,
        post_f0_shift_in_cent=0,
        force_fix_vuv=False,
        fill_silence_to_rest=False,
        dtype=np.int16,
        peak_norm=False,
        loudness_norm=False,
        target_loudness=-20,
        segmented_synthesis=False,
    ):
        """Synthesize waveform from HTS labels.

        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels
            vocoder_type (str): Vocoder type. ``world``, ``pwg``, ``usfgan``, and ``auto``
                is supported.
            post_filter_type (str): Post-filter type. ``merlin``, ``gv`` or ``nnsvs``
                is supported.
            trajectory_smoothing (bool): Whether to smooth acoustic feature trajectory.
            trajectory_smoothing_cutoff (int): Cutoff frequency for trajectory smoothing.
            trajectory_smoothing_cutoff_f0 (int): Cutoff frequency for trajectory
                smoothing of f0.
            vuv_threshold (float): Threshold for VUV.
            f0_shift_in_cent (float): F0 scaling factor.
            vibrato_scale (float): Scale for vibrato. Only valid if the acoustic
                features contain vibrato parameters.
            return_states (bool): Whether to return the internal states (for debugging)
            force_fix_vuv (bool): Whether to correct VUV.
            segmneted_synthesis (bool): Whether to use segmented synthesis.
        """
        vocoder_type = vocoder_type.lower()
        if vocoder_type not in ["world", "pwg", "usfgan", "auto"]:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")
        if post_filter_type not in ["merlin", "nnsvs", "gv", "none"]:
            raise ValueError(f"Unknown post-filter type: {post_filter_type}")

        if vocoder_type == "pwg" and self.vocoder is None:
            raise ValueError(
                """Pre-trained vocodr model is not found.
WORLD is only supported for waveform generation"""
            )

        if vocoder_type == "auto":
            if self.feature_type == "melf0":
                assert self.vocoder is not None
                vocoder_type = (
                    "usfgan" if isinstance(self.vocoder, USFGANWrapper) else "pwg"
                )
            elif self.feature_type == "world":
                if self.vocoder is None:
                    vocoder_type = "world"
                else:
                    vocoder_type = (
                        "usfgan" if isinstance(self.vocoder, USFGANWrapper) else "pwg"
                    )

        # Predict timinigs
        duration_modified_labels = self.predict_timing(labels)

        # NOTE: segmented synthesis is not well tested. There MUST be better ways
        # to do this.
        if segmented_synthesis:
            self.logger.warning(
                "Segmented synthesis is not well tested. Use it on your own risk."
            )
            duration_modified_labels_segs = segment_labels(
                duration_modified_labels,
                # the following parameters are based on experiments in the NNSVS's paper
                # tuned with Namine Ritsu's database
                silence_threshold=0.1,
                min_duration=5.0,
                force_split_threshold=5.0,
            )
            from tqdm.auto import tqdm
        else:
            duration_modified_labels_segs = [duration_modified_labels]

            def tqdm(x, **kwargs):
                return x

        # Run acoustic model and vocoder
        hts_frame_shift = int(self.config.frame_period * 1e4)
        wavs = []
        for duration_modified_labels_seg in tqdm(
            duration_modified_labels_segs,
            desc="[segment]",
            total=len(duration_modified_labels_segs),
        ):
            duration_modified_labels_seg.frame_shift = hts_frame_shift

            # Predict acoustic features
            # NOTE: if non-zero pre_f0_shift_in_cent is specified, the input pitch
            # will be shifted before running the acoustic model
            acoustic_features = self.predict_acoustic(
                duration_modified_labels_seg,
                f0_shift_in_cent=pre_f0_shift_in_cent,
            )

            # Post-processing for acoustic features
            # NOTE: if non-zero post_f0_shift_in_cent is specified, the output pitch
            # will be shifted as a part of post-processing
            multistream_features = self.postprocess_acoustic(
                acoustic_features=acoustic_features,
                duration_modified_labels=duration_modified_labels_seg,
                trajectory_smoothing=trajectory_smoothing,
                trajectory_smoothing_cutoff=trajectory_smoothing_cutoff,
                trajectory_smoothing_cutoff_f0=trajectory_smoothing_cutoff_f0,
                force_fix_vuv=force_fix_vuv,
                fill_silence_to_rest=fill_silence_to_rest,
                f0_shift_in_cent=post_f0_shift_in_cent,
            )

            # Generate waveform by vocoder
            wav = self.predict_waveform(
                multistream_features=multistream_features,
                vocoder_type=vocoder_type,
                vuv_threshold=vuv_threshold,
            )

            wavs.append(wav)

        # Concatenate segmented waveforms
        wav = np.concatenate(wavs, axis=0).reshape(-1)

        # Post-processing for the output waveform
        wav = self.postprocess_waveform(
            wav,
            dtype=dtype,
            peak_norm=peak_norm,
            loudness_norm=loudness_norm,
            target_loudness=target_loudness,
        )

        return wav, self.config.sample_rate
