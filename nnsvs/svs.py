import json
from pathlib import Path
from warnings import warn

import librosa
import numpy as np
import pysptk
import pyworld
import torch
from hydra.utils import instantiate
from nnmnkwii.frontend import merlin as fe
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
from nnsvs.io.hts import get_pitch_index, get_pitch_indices, segment_labels
from nnsvs.multistream import (
    get_static_features,
    get_static_stream_sizes,
    split_streams,
)
from nnsvs.pitch import lowpass_filter
from nnsvs.postfilters import variance_scaling
from nnsvs.usfgan import USFGANWrapper
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


@torch.no_grad()
def predict_timings(
    device,
    labels,
    binary_dict,
    numeric_dict,
    timelag_model,
    timelag_config,
    timelag_in_scaler,
    timelag_out_scaler,
    duration_model,
    duration_config,
    duration_in_scaler,
    duration_out_scaler,
    log_f0_conditioning=True,
    allowed_range=None,
    allowed_range_rest=None,
    force_clip_input_features=True,
    frame_period=5,
):
    hts_frame_shift = int(frame_period * 1e4)
    labels.frame_shift = hts_frame_shift

    pitch_indices = get_pitch_indices(binary_dict, numeric_dict)

    # Time-lag
    lag = predict_timelag(
        device=device,
        labels=labels,
        timelag_model=timelag_model,
        timelag_config=timelag_config,
        timelag_in_scaler=timelag_in_scaler,
        timelag_out_scaler=timelag_out_scaler,
        binary_dict=binary_dict,
        numeric_dict=numeric_dict,
        pitch_indices=pitch_indices,
        log_f0_conditioning=log_f0_conditioning,
        allowed_range=allowed_range,
        allowed_range_rest=allowed_range_rest,
        force_clip_input_features=force_clip_input_features,
        frame_period=frame_period,
    )

    # Duration predictions
    durations = predict_duration(
        device=device,
        labels=labels,
        duration_model=duration_model,
        duration_config=duration_config,
        duration_in_scaler=duration_in_scaler,
        duration_out_scaler=duration_out_scaler,
        binary_dict=binary_dict,
        numeric_dict=numeric_dict,
        pitch_indices=pitch_indices,
        log_f0_conditioning=log_f0_conditioning,
        force_clip_input_features=force_clip_input_features,
        frame_period=frame_period,
    )

    # Normalize phoneme durations
    duration_modified_labels = postprocess_duration(labels, durations, lag)
    return duration_modified_labels


@torch.no_grad()
def synthesis_from_timings(
    device,
    duration_modified_labels,
    binary_dict,
    numeric_dict,
    acoustic_model,
    acoustic_config,
    acoustic_in_scaler,
    acoustic_out_scaler,
    acoustic_out_static_scaler,
    vocoder=None,
    vocoder_config=None,
    vocoder_in_scaler=None,
    postfilter_model=None,
    postfilter_config=None,
    postfilter_out_scaler=None,
    sample_rate=48000,
    frame_period=5,
    log_f0_conditioning=True,
    subphone_features="coarse_coding",
    use_world_codec=True,
    force_clip_input_features=True,
    relative_f0=False,
    feature_type="world",
    vocoder_type="world",
    post_filter_type="merlin",
    trajectory_smoothing=True,
    trajectory_smoothing_cutoff=50,
    trajectory_smoothing_cutoff_f0=20,
    vuv_threshold=0.5,
    pre_f0_shift_in_cent=0,
    post_f0_shift_in_cent=0,
    vibrato_scale=1.0,
    force_fix_vuv=False,
):
    hts_frame_shift = int(frame_period * 1e4)
    pitch_idx = get_pitch_index(binary_dict, numeric_dict)
    pitch_indices = get_pitch_indices(binary_dict, numeric_dict)

    # Predict acoustic features
    acoustic_features = predict_acoustic(
        device=device,
        labels=duration_modified_labels,
        acoustic_model=acoustic_model,
        acoustic_config=acoustic_config,
        acoustic_in_scaler=acoustic_in_scaler,
        acoustic_out_scaler=acoustic_out_scaler,
        binary_dict=binary_dict,
        numeric_dict=numeric_dict,
        subphone_features=subphone_features,
        pitch_indices=pitch_indices,
        log_f0_conditioning=log_f0_conditioning,
        force_clip_input_features=force_clip_input_features,
        f0_shift_in_cent=pre_f0_shift_in_cent,
    )

    static_stream_sizes = get_static_stream_sizes(
        acoustic_config.stream_sizes,
        acoustic_config.has_dynamic_features,
        acoustic_config.num_windows,
    )

    if post_filter_type == "gv" or (
        post_filter_type == "nnsvs" and feature_type == "world"
    ):
        linguistic_features = fe.linguistic_features(
            duration_modified_labels,
            binary_dict,
            numeric_dict,
            add_frame_features=True,
            subphone_features=subphone_features,
            frame_shift=hts_frame_shift,
        )
        # TODO: remove hardcode
        in_rest_idx = 0
        note_frame_indices = linguistic_features[:, in_rest_idx] <= 0

        if feature_type == "world":
            offset = 2
        elif feature_type == "melf0":
            # NOTE: set offset so that post-filter don't affect F0
            mel_freq = librosa.mel_frequencies(
                n_mels=80, fmin=63, fmax=sample_rate // 2
            )
            offset = np.argmax(mel_freq > 1200)

        mgc_end_dim = static_stream_sizes[0]
        acoustic_features[:, :mgc_end_dim] = variance_scaling(
            acoustic_out_static_scaler.var_.reshape(-1)[:mgc_end_dim],
            acoustic_features[:, :mgc_end_dim],
            offset=offset,
            note_frame_indices=note_frame_indices,
        )

    # Learned post-filter using nnsvs
    if post_filter_type == "nnsvs" and postfilter_model is not None:
        # (1) Raw spectrogram or (2) mgc
        rawsp_output = postfilter_config.stream_sizes[0] >= 128

        # If the post-filter output is raw spectrogrma, convert mgc to log spectrogram
        if rawsp_output:
            outs = split_streams(acoustic_features, static_stream_sizes)
            assert len(outs) == 4
            mgc, lf0, vuv, bap = outs
            fft_size = pyworld.get_cheaptrick_fft_size(sample_rate)
            sp = pyworld.decode_spectral_envelope(
                mgc.astype(np.float64), sample_rate, fft_size
            ).astype(np.float32)
            sp = np.log(sp)
            acoustic_features = np.concatenate([sp, lf0, vuv, bap], axis=-1)

        in_feats = torch.from_numpy(acoustic_features).float().unsqueeze(0)
        in_feats = postfilter_out_scaler.transform(in_feats).float().to(device)
        # Run inference
        out_feats = postfilter_model.inference(in_feats, [in_feats.shape[1]])
        acoustic_features = (
            postfilter_out_scaler.inverse_transform(out_feats.cpu()).squeeze(0).numpy()
        )

        # Convert log spectrogram to mgc
        # NOTE: mgc is used to reduce possible artifacts
        # Ref: https://bit.ly/3AHjstU
        if rawsp_output:
            sp, lf0, vuv, bap = split_streams(
                acoustic_features, postfilter_config.stream_sizes
            )
            sp = np.exp(sp)
            mgc = pyworld.code_spectral_envelope(
                sp.astype(np.float64), sample_rate, 60
            ).astype(np.float32)
            acoustic_features = np.concatenate([mgc, lf0, vuv, bap], axis=-1)

    # Generate WORLD parameters
    if feature_type == "world":
        mgc, lf0, vuv, bap = gen_spsvs_static_features(
            labels=duration_modified_labels,
            acoustic_features=acoustic_features,
            binary_dict=binary_dict,
            numeric_dict=numeric_dict,
            stream_sizes=acoustic_config.stream_sizes,
            has_dynamic_features=acoustic_config.has_dynamic_features,
            subphone_features=subphone_features,
            pitch_idx=pitch_idx,
            num_windows=acoustic_config.num_windows,
            frame_period=frame_period,
            relative_f0=relative_f0,
            vibrato_scale=vibrato_scale,
            vuv_threshold=vuv_threshold,
            force_fix_vuv=force_fix_vuv,
        )
    elif feature_type == "melf0":
        mel, lf0, vuv = split_streams(acoustic_features, [80, 1, 1])
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    if post_f0_shift_in_cent != 0:
        lf0_offset = post_f0_shift_in_cent * np.log(2) / 1200
        lf0 = lf0 + lf0_offset

    # NOTE: spectral enhancement based on the Merlin's post-filter implementation
    if feature_type == "world" and post_filter_type == "merlin":
        alpha = pysptk.util.mcepalpha(sample_rate)
        mgc = merlin_post_filter(mgc, alpha)

    # Remove high-frequency components of lf0/mgc/bap
    # NOTE: Useful to reduce high-frequency artifacts
    if trajectory_smoothing:
        modfs = int(1 / (frame_period * 0.001))
        lf0[:, 0] = lowpass_filter(
            lf0[:, 0], modfs, cutoff=trajectory_smoothing_cutoff_f0
        )
        if feature_type == "world":
            for d in range(mgc.shape[1]):
                mgc[:, d] = lowpass_filter(
                    mgc[:, d], modfs, cutoff=trajectory_smoothing_cutoff
                )
            for d in range(bap.shape[1]):
                bap[:, d] = lowpass_filter(
                    bap[:, d], modfs, cutoff=trajectory_smoothing_cutoff
                )
        elif feature_type == "melf0":
            for d in range(mel.shape[1]):
                mel[:, d] = lowpass_filter(
                    mel[:, d], modfs, cutoff=trajectory_smoothing_cutoff
                )
    if feature_type == "world":
        use_mcep_aperiodicity = bap.shape[-1] > 5
    if feature_type == "world" and not use_mcep_aperiodicity:
        bap = np.clip(bap, a_min=-60, a_max=0)

    # Waveform generation by (1) WORLD or (2) neural vocoder
    if vocoder_type == "world":
        f0, spectrogram, aperiodicity = gen_world_params(
            mgc,
            lf0,
            vuv,
            bap,
            sample_rate,
            vuv_threshold=vuv_threshold,
            use_world_codec=use_world_codec,
        )
        wav = pyworld.synthesize(
            f0,
            spectrogram,
            aperiodicity,
            sample_rate,
            frame_period,
        )
    elif vocoder_type == "pwg":
        # NOTE: So far vocoder models are trained on binary V/UV features
        vuv = (vuv > vuv_threshold).astype(np.float32)
        if feature_type == "world":
            voc_inp = (
                torch.from_numpy(
                    vocoder_in_scaler.transform(
                        np.concatenate([mgc, lf0, vuv, bap], axis=-1)
                    )
                )
                .float()
                .to(device)
            )
        elif feature_type == "melf0":
            voc_inp = (
                torch.from_numpy(
                    vocoder_in_scaler.transform(
                        np.concatenate([mel, lf0, vuv], axis=-1)
                    )
                )
                .float()
                .to(device)
            )
        wav = vocoder.inference(voc_inp).view(-1).to("cpu").numpy()
    elif vocoder_type == "usfgan":
        if feature_type == "world":
            fftlen = pyworld.get_cheaptrick_fft_size(sample_rate)
            if use_mcep_aperiodicity:
                aperiodicity_order = bap.shape[-1] - 1
                alpha = pysptk.util.mcepalpha(sample_rate)
                aperiodicity = pysptk.mc2sp(
                    np.ascontiguousarray(bap).astype(np.float64),
                    fftlen=fftlen,
                    alpha=alpha,
                )
            else:
                aperiodicity = pyworld.decode_aperiodicity(
                    np.ascontiguousarray(bap).astype(np.float64),
                    sample_rate,
                    fftlen,
                )
                # fill aperiodicity with ones for unvoiced regions
            aperiodicity[vuv.reshape(-1) < vuv_threshold, 0] = 1.0
            # WORLD fails catastrophically for out of range aperiodicity
            aperiodicity = np.clip(aperiodicity, 0.0, 1.0)

            if use_mcep_aperiodicity:
                bap = pysptk.sp2mc(
                    aperiodicity,
                    order=aperiodicity_order,
                    alpha=alpha,
                )
            else:
                bap = pyworld.code_aperiodicity(aperiodicity, sample_rate).astype(
                    np.float32
                )
            aux_feats = [mgc, bap]
        elif feature_type == "melf0":
            aux_feats = [mel]

        aux_feats = (
            torch.from_numpy(
                vocoder_in_scaler.transform(np.concatenate(aux_feats, axis=-1))
            )
            .float()
            .to(device)
        )

        contf0 = np.exp(lf0)
        if vocoder_config.data.sine_f0_type in ["contf0", "cf0"]:
            f0_inp = contf0
        elif vocoder_config.data.sine_f0_type == "f0":
            f0_inp = contf0
            f0_inp[vuv < vuv_threshold] = 0
        wav = vocoder.inference(f0_inp, aux_feats).view(-1).to("cpu").numpy()

    if feature_type == "world":
        states = {
            "mgc": mgc,
            "lf0": lf0,
            "vuv": vuv,
            "bap": bap,
        }
    elif feature_type == "melf0":
        states = {
            "mel": mel,
            "lf0": lf0,
            "vuv": vuv,
        }
    if vocoder_type == "world":
        states.update(
            {
                "f0": f0,
                "spectrogram": spectrogram,
                "aperiodicity": aperiodicity,
            }
        )
    return wav, states


def post_process(wav, sample_rate):
    wav = bandpass_filter(wav, sample_rate)

    if np.max(wav) > 10:
        if np.abs(wav).max() > 32767:
            wav = wav / np.abs(wav).max()
        # data is likely already in [-32768, 32767]
        wav = wav.astype(np.int16)
    else:
        if np.abs(wav).max() > 1.0:
            wav = wav / np.abs(wav).max()
        wav = (wav * 32767.0).astype(np.int16)
    return wav


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

        model_dir = retrieve_pretrained_model("r9y9/kiritan_latest")
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
        if (model_dir / "vocoder_model.yaml").exists():
            if not _pwg_available:
                warn("parallel_wavegan is not installed. Vocoder model is disabled.")
                self.vocoder = None
            else:
                self.vocoder_config = OmegaConf.load(model_dir / "vocoder_model.yaml")

                # usfgan
                if (
                    "generator" in self.vocoder_config
                    and "discriminator" in self.vocoder_config
                ):
                    self.vocoder = instantiate(self.vocoder_config.generator).to(device)
                    checkpoint = torch.load(
                        model_dir / "vocoder_model.pth",
                        map_location=device,
                    )
                    self.vocoder.load_state_dict(checkpoint["model"]["generator"])
                    self.vocoder.remove_weight_norm()
                    self.vocoder = USFGANWrapper(self.vocoder_config, self.vocoder)

                    # Extract scaler params for [mgc, bap]
                    mean_ = np.load(model_dir / "in_vocoder_scaler_mean.npy")
                    var_ = np.load(model_dir / "in_vocoder_scaler_var.npy")
                    scale_ = np.load(model_dir / "in_vocoder_scaler_scale.npy")
                    stream_sizes = get_static_stream_sizes(
                        self.acoustic_config.stream_sizes,
                        self.acoustic_config.has_dynamic_features,
                        self.acoustic_config.num_windows,
                    )
                    mgc_end_dim = stream_sizes[0]
                    bap_start_dim = sum(stream_sizes[:3])
                    bap_end_dim = sum(stream_sizes[:4])
                    self.vocoder_in_scaler = StandardScaler(
                        np.concatenate(
                            [mean_[:mgc_end_dim], mean_[bap_start_dim:bap_end_dim]]
                        ),
                        np.concatenate(
                            [var_[:mgc_end_dim], var_[bap_start_dim:bap_end_dim]]
                        ),
                        np.concatenate(
                            [scale_[:mgc_end_dim], scale_[bap_start_dim:bap_end_dim]]
                        ),
                    )
                else:
                    self.vocoder = load_model(
                        model_dir / "vocoder_model.pth", config=self.vocoder_config
                    ).to(device)
                    self.vocoder.remove_weight_norm()
                    self.vocoder_in_scaler = StandardScaler(
                        np.load(model_dir / "in_vocoder_scaler_mean.npy"),
                        np.load(model_dir / "in_vocoder_scaler_var.npy"),
                        np.load(model_dir / "in_vocoder_scaler_scale.npy"),
                    )
                self.vocoder.eval()
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

    def synthesis_from_timings(
        self,
        duration_modified_labels,
        vocoder_type="world",
        post_filter_type="merlin",
        trajectory_smoothing=True,
        trajectory_smoothing_cutoff=50,
        trajectory_smoothing_cutoff_f0=20,
        vuv_threshold=0.5,
        pre_f0_shift_in_cent=0,
        post_f0_shift_in_cent=0,
        vibrato_scale=1.0,
        return_states=False,
        force_fix_vuv=False,
        segmented_synthesis=False,
    ):
        """Synthesize waveform from HTS labels with timings.

        Args:
            duration_modified_labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels
                with predicted timings.
            vocoder_type (str): Vocoder type. ``world``, ``pwg`` and ``usfgan`` is supported.
            post_filter_type (str): Post-filter type. ``merlin``, ``gv`` or ``nnsvs``
                is supported.
            trajectory_smoothing (bool): Whether to smooth acoustic feature trajectory.
            trajectory_smoothing_cutoff (int): Cutoff frequency for trajectory smoothing.
            trajectory_smoothing_cutoff_f0 (int): Cutoff frequency for trajectory
                smoothing of f0.
            vuv_threshold (float): Threshold for VUV.
            f0_scale (float): Scale factor for f0.
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

        hts_frame_shift = int(self.config.frame_period * 1e4)
        wavs = []
        for seg_labels in tqdm(segmented_labels):
            seg_labels.frame_shift = hts_frame_shift
            wav, states = synthesis_from_timings(
                device=self.device,
                duration_modified_labels=duration_modified_labels,
                binary_dict=self.binary_dict,
                numeric_dict=self.numeric_dict,
                acoustic_model=self.acoustic_model,
                acoustic_config=self.acoustic_config,
                acoustic_in_scaler=self.acoustic_in_scaler,
                acoustic_out_scaler=self.acoustic_out_scaler,
                acoustic_out_static_scaler=self.acoustic_out_static_scaler,
                vocoder=self.vocoder,
                vocoder_config=self.vocoder_config,
                vocoder_in_scaler=self.vocoder_in_scaler,
                postfilter_model=self.postfilter_model,
                postfilter_config=self.postfilter_config,
                postfilter_out_scaler=self.postfilter_out_scaler,
                sample_rate=self.config.sample_rate,
                frame_period=self.config.frame_period,
                log_f0_conditioning=self.config.log_f0_conditioning,
                subphone_features=self.config.acoustic.subphone_features,
                use_world_codec=self.config.get("use_world_codec", False),
                force_clip_input_features=self.config.acoustic.force_clip_input_features,
                relative_f0=self.config.acoustic.relative_f0,
                feature_type=self.feature_type,
                vocoder_type=vocoder_type,
                post_filter_type=post_filter_type,
                trajectory_smoothing=trajectory_smoothing,
                trajectory_smoothing_cutoff=trajectory_smoothing_cutoff,
                trajectory_smoothing_cutoff_f0=trajectory_smoothing_cutoff_f0,
                vuv_threshold=vuv_threshold,
                pre_f0_shift_in_cent=pre_f0_shift_in_cent,
                post_f0_shift_in_cent=post_f0_shift_in_cent,
                vibrato_scale=vibrato_scale,
                force_fix_vuv=force_fix_vuv,
            )
            wavs.append(wav)

        # Concatenate segmented wavs
        wav = np.concatenate(wavs, axis=0).reshape(-1)
        wav = post_process(wav, self.config.sample_rate)

        if return_states:
            assert not segmented_synthesis
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
        pre_f0_shift_in_cent=0,
        post_f0_shift_in_cent=0,
        vibrato_scale=1.0,
        return_states=False,
        force_fix_vuv=False,
        post_filter=None,
        segmented_synthesis=False,
    ):
        """Synthesize waveform from HTS labels.

        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels
            vocoder_type (str): Vocoder type. ``world``, ``pwg`` and ``usfgan`` is supported.
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
        if post_filter is not None:
            warn("post_filter is deprecated. Use post_filter_type instead.")
            post_filter_type = "merlin" if post_filter else "none"

        if vocoder_type == "auto":
            if self.vocoder is None:
                vocoder_type = "world"
            else:
                vocoder_type = (
                    "usfgan" if isinstance(self.vocoder, USFGANWrapper) else "pwg"
                )

        duration_modified_labels = predict_timings(
            device=self.device,
            labels=labels,
            binary_dict=self.binary_dict,
            numeric_dict=self.numeric_dict,
            timelag_model=self.timelag_model,
            timelag_config=self.timelag_config,
            timelag_in_scaler=self.timelag_in_scaler,
            timelag_out_scaler=self.timelag_out_scaler,
            duration_model=self.duration_model,
            duration_config=self.duration_config,
            duration_in_scaler=self.duration_in_scaler,
            duration_out_scaler=self.duration_out_scaler,
            log_f0_conditioning=self.config.log_f0_conditioning,
            allowed_range=self.config.timelag.allowed_range,
            allowed_range_rest=self.config.timelag.allowed_range_rest,
            force_clip_input_features=self.config.timelag.force_clip_input_features,
            frame_period=self.config.frame_period,
        )

        return self.synthesis_from_timings(
            duration_modified_labels=duration_modified_labels,
            vocoder_type=vocoder_type,
            post_filter_type=post_filter_type,
            trajectory_smoothing=trajectory_smoothing,
            trajectory_smoothing_cutoff=trajectory_smoothing_cutoff,
            trajectory_smoothing_cutoff_f0=trajectory_smoothing_cutoff_f0,
            vuv_threshold=vuv_threshold,
            pre_f0_shift_in_cent=pre_f0_shift_in_cent,
            post_f0_shift_in_cent=post_f0_shift_in_cent,
            vibrato_scale=vibrato_scale,
            return_states=return_states,
            force_fix_vuv=force_fix_vuv,
            segmented_synthesis=segmented_synthesis,
        )
