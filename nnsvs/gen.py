from warnings import warn

import librosa
import numpy as np
import pyloudnorm as pyln
import pysptk
import pyworld
import scipy
import torch
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from nnmnkwii.postfilters import merlin_post_filter
from nnmnkwii.preprocessing.f0 import interp1d
from nnsvs.base import PredictionType
from nnsvs.dsp import bandpass_filter
from nnsvs.io.hts import (
    get_note_frame_indices,
    get_note_indices,
    get_pitch_index,
    get_pitch_indices,
)
from nnsvs.multistream import (
    get_static_stream_sizes,
    get_windows,
    multi_stream_mlpg,
    split_streams,
)
from nnsvs.pitch import gen_sine_vibrato, lowpass_filter
from nnsvs.postfilters import variance_scaling
from sklearn.preprocessing import MinMaxScaler


def _midi_to_hz(x, idx, log_f0=False):
    z = np.zeros(len(x))
    indices = x[:, idx] > 0
    z[indices] = librosa.midi_to_hz(x[indices, idx])
    if log_f0:
        z[indices] = np.log(z[indices])
    return z


def _is_silence(label):
    is_full_context = "@" in label
    if is_full_context:
        is_silence = "-sil" in label or "-pau" in label
    else:
        is_silence = label == "sil" or label == "pau"
    return is_silence


@torch.no_grad()
def predict_timelag(
    device,
    labels,
    timelag_model,
    timelag_config,
    timelag_in_scaler,
    timelag_out_scaler,
    binary_dict,
    numeric_dict,
    pitch_indices=None,
    log_f0_conditioning=True,
    allowed_range=None,
    allowed_range_rest=None,
    force_clip_input_features=False,
    frame_period=5,
):
    """Predict time-lag from HTS labels

    Args:
        device (torch.device): device
        labels (nnmnkwii.io.hts.HTSLabelFile): HTS-style labels
        timelag_model (nn.Module): time-lag model
        timelag_config (dict): time-lag model config
        timelag_in_scaler (sklearn.preprocessing.MinMaxScaler): input scaler
        timelag_out_scaler (sklearn.preprocessing.MinMaxScaler): output scaler
        binary_dict (dict): binary feature dict
        numeric_dict (dict): numeric feature dict
        pitch_indices (list): indices of pitch features
        log_f0_conditioning (bool): whether to condition on log f0
        allowed_range (list): allowed range of time-lag
        allowed_range_rest (list): allowed range of time-lag for rest
        force_clip_input_features (bool): whether to clip input features

    Returns;
        ndarray: time-lag predictions
    """
    hts_frame_shift = int(frame_period * 1e4)
    # make sure to set frame shift properly before calling round_ method
    labels.frame_shift = hts_frame_shift
    if pitch_indices is None:
        pitch_indices = get_pitch_indices(binary_dict, numeric_dict)

    if allowed_range is None:
        allowed_range = [-20, 20]
    if allowed_range_rest is None:
        allowed_range_rest = [-40, 40]
    # round start/end times just in case.
    labels.round_()

    # Extract note-level labels
    note_indices = get_note_indices(labels)
    note_labels = labels[note_indices]

    # Extract musical/linguistic context
    timelag_linguistic_features = fe.linguistic_features(
        note_labels,
        binary_dict,
        numeric_dict,
        add_frame_features=False,
        subphone_features=None,
        frame_shift=hts_frame_shift,
    ).astype(np.float32)

    # Adjust input features if we use log-f0 conditioning
    if log_f0_conditioning:
        if pitch_indices is None:
            raise ValueError("Pitch feature indices must be specified!")
        for idx in pitch_indices:
            timelag_linguistic_features[:, idx] = interp1d(
                _midi_to_hz(timelag_linguistic_features, idx, log_f0_conditioning),
                kind="slinear",
            )

    # Normalization
    timelag_linguistic_features = timelag_in_scaler.transform(
        timelag_linguistic_features
    )
    if force_clip_input_features and isinstance(timelag_in_scaler, MinMaxScaler):
        # clip to feature range (except for pitch-related features)
        non_pitch_indices = [
            idx
            for idx in range(timelag_linguistic_features.shape[1])
            if idx not in pitch_indices
        ]
        timelag_linguistic_features[:, non_pitch_indices] = np.clip(
            timelag_linguistic_features[:, non_pitch_indices],
            timelag_in_scaler.feature_range[0],
            timelag_in_scaler.feature_range[1],
        )

    # Run model
    x = torch.from_numpy(timelag_linguistic_features).unsqueeze(0).to(device)

    # Run model
    if timelag_model.prediction_type() == PredictionType.PROBABILISTIC:
        # (B, T, D_out)
        max_mu, max_sigma = timelag_model.inference(x, [x.shape[1]])
        if np.any(timelag_config.has_dynamic_features):
            # Apply denormalization
            # (B, T, D_out) -> (T, D_out)
            max_sigma_sq = (
                max_sigma.squeeze(0).cpu().data.numpy() ** 2 * timelag_out_scaler.var_
            )
            max_sigma_sq = np.maximum(max_sigma_sq, 1e-14)
            max_mu = timelag_out_scaler.inverse_transform(
                max_mu.squeeze(0).cpu().data.numpy()
            )
            # (T, D_out) -> (T, static_dim)
            pred_timelag = multi_stream_mlpg(
                max_mu,
                max_sigma_sq,
                get_windows(timelag_config.num_windows),
                timelag_config.stream_sizes,
                timelag_config.has_dynamic_features,
            )
        else:
            # Apply denormalization
            pred_timelag = timelag_out_scaler.inverse_transform(
                max_mu.squeeze(0).cpu().data.numpy()
            )
    else:
        # (T, D_out)
        pred_timelag = (
            timelag_model.inference(x, [x.shape[1]]).squeeze(0).cpu().data.numpy()
        )
        # Apply denormalization
        pred_timelag = timelag_out_scaler.inverse_transform(pred_timelag)
        if np.any(timelag_config.has_dynamic_features):
            # (T, D_out) -> (T, static_dim)
            pred_timelag = multi_stream_mlpg(
                pred_timelag,
                timelag_out_scaler.var_,
                get_windows(timelag_config.num_windows),
                timelag_config.stream_sizes,
                timelag_config.has_dynamic_features,
            )

    # Rounding
    pred_timelag = np.round(pred_timelag)

    # Clip to the allowed range
    for idx in range(len(pred_timelag)):
        if _is_silence(note_labels.contexts[idx]):
            pred_timelag[idx] = np.clip(
                pred_timelag[idx], allowed_range_rest[0], allowed_range_rest[1]
            )
        else:
            pred_timelag[idx] = np.clip(
                pred_timelag[idx], allowed_range[0], allowed_range[1]
            )

    # frames -> 100 ns
    pred_timelag *= hts_frame_shift

    return pred_timelag


@torch.no_grad()
def predict_duration(
    device,
    labels,
    duration_model,
    duration_config,
    duration_in_scaler,
    duration_out_scaler,
    binary_dict,
    numeric_dict,
    pitch_indices=None,
    log_f0_conditioning=True,
    force_clip_input_features=False,
    frame_period=5,
):
    """Predict phoneme durations from HTS labels

    Args:
        device (torch.device): device to run the model on
        labels (nnmnkwii.io.hts.HTSLabelFile): labels
        duration_model (nn.Module): duration model
        duration_config (dict): duration config
        duration_in_scaler (sklearn.preprocessing.MinMaxScaler): duration input scaler
        duration_out_scaler (sklearn.preprocessing.MinMaxScaler): duration output scaler
        binary_dict (dict): binary feature dictionary
        numeric_dict (dict): numeric feature dictionary
        pitch_indices (list): indices of pitch features
        log_f0_conditioning (bool): whether to use log-f0 conditioning
        force_clip_input_features (bool): whether to clip input features

    Returns:
        np.ndarray: predicted durations
    """
    hts_frame_shift = int(frame_period * 1e4)
    if pitch_indices is None:
        pitch_indices = get_pitch_indices(binary_dict, numeric_dict)

    # Extract musical/linguistic features
    duration_linguistic_features = fe.linguistic_features(
        labels,
        binary_dict,
        numeric_dict,
        add_frame_features=False,
        subphone_features=None,
        frame_shift=hts_frame_shift,
    ).astype(np.float32)

    if log_f0_conditioning:
        for idx in pitch_indices:
            duration_linguistic_features[:, idx] = interp1d(
                _midi_to_hz(duration_linguistic_features, idx, log_f0_conditioning),
                kind="slinear",
            )

    # Apply normalization
    duration_linguistic_features = duration_in_scaler.transform(
        duration_linguistic_features
    )
    if force_clip_input_features and isinstance(duration_in_scaler, MinMaxScaler):
        # clip to feature range (except for pitch-related features)
        non_pitch_indices = [
            idx
            for idx in range(duration_linguistic_features.shape[1])
            if idx not in pitch_indices
        ]
        duration_linguistic_features[:, non_pitch_indices] = np.clip(
            duration_linguistic_features[:, non_pitch_indices],
            duration_in_scaler.feature_range[0],
            duration_in_scaler.feature_range[1],
        )

    # Apply model
    x = torch.from_numpy(duration_linguistic_features).float().to(device)
    x = x.view(1, -1, x.size(-1))

    if duration_model.prediction_type() == PredictionType.PROBABILISTIC:
        # (B, T, D_out)
        max_mu, max_sigma = duration_model.inference(x, [x.shape[1]])
        if np.any(duration_config.has_dynamic_features):
            raise RuntimeError(
                "Dynamic features are not supported for duration modeling"
            )
        # Apply denormalization
        max_sigma_sq = (
            max_sigma.squeeze(0).cpu().data.numpy() ** 2 * duration_out_scaler.var_
        )
        max_sigma_sq = np.maximum(max_sigma_sq, 1e-14)
        max_mu = duration_out_scaler.inverse_transform(
            max_mu.squeeze(0).cpu().data.numpy()
        )

        return max_mu, max_sigma_sq
    else:
        # (T, D_out)
        pred_durations = (
            duration_model.inference(x, [x.shape[1]]).squeeze(0).cpu().data.numpy()
        )
        # Apply denormalization
        pred_durations = duration_out_scaler.inverse_transform(pred_durations)
        if np.any(duration_config.has_dynamic_features):
            # (T, D_out) -> (T, static_dim)
            pred_durations = multi_stream_mlpg(
                pred_durations,
                duration_out_scaler.var_,
                get_windows(duration_config.num_windows),
                duration_config.stream_sizes,
                duration_config.has_dynamic_features,
            )

    pred_durations[pred_durations <= 0] = 1
    pred_durations = np.round(pred_durations)

    return pred_durations


def postprocess_duration(labels, pred_durations, lag, frame_period=5):
    """Post-process durations based on predicted time-lag

    Ref : https://arxiv.org/abs/2108.02776

    Args:
        labels (HTSLabelFile): HTS labels
        pred_durations (array or tuple): predicted durations for non-MDN,
            mean and variance for MDN
        lag (array): predicted time-lag

    Returns:
        HTSLabelFile: labels with adjusted durations
    """
    hts_frame_shift = int(frame_period * 1e4)

    note_indices = get_note_indices(labels)
    # append the end of note
    note_indices.append(len(labels))

    is_mdn = isinstance(pred_durations, tuple) and len(pred_durations) == 2

    output_labels = hts.HTSLabelFile()

    for i in range(1, len(note_indices)):
        p = labels[note_indices[i - 1] : note_indices[i]]

        # Compute note duration with time-lag
        # eq (11)
        L = int(fe.duration_features(p)[0])
        if i < len(note_indices) - 1:
            L_hat = L - (lag[i - 1] - lag[i]) / hts_frame_shift
        else:
            L_hat = L - (lag[i - 1]) / hts_frame_shift

        # Prevent negative duration
        L_hat = max(L_hat, 1)

        # adjust the start time of the note
        p.start_times = np.minimum(
            np.asarray(p.start_times) + lag[i - 1].reshape(-1),
            np.asarray(p.end_times) - hts_frame_shift * len(p),
        )
        p.start_times = np.maximum(p.start_times, 0)
        if len(output_labels) > 0:
            p.start_times = np.maximum(
                p.start_times, output_labels.start_times[-1] + hts_frame_shift
            )

        # Compute normalized phoneme durations
        if is_mdn:
            mu = pred_durations[0][note_indices[i - 1] : note_indices[i]]
            sigma_sq = pred_durations[1][note_indices[i - 1] : note_indices[i]]
            # eq (17)
            rho = (L_hat - mu.sum()) / sigma_sq.sum()
            # eq (16)
            d_norm = mu + rho * sigma_sq

            if np.any(d_norm <= 0):
                # eq (12) (using mu as d_hat)
                s = frame_period * 0.001
                print(
                    f"Negative phoneme durations are predicted at {i}-th note. "
                    "The note duration: ",
                    f"{round(float(L)*s,3)} sec -> {round(float(L_hat)*s,3)} sec",
                )
                print(
                    "It's likely that the model couldn't predict correct durations "
                    "for short notes."
                )
                print(
                    f"Variance scaling based durations (in frame):\n{(mu + rho * sigma_sq)}"
                )
                print(
                    f"Fallback to uniform scaling (in frame):\n{(L_hat * mu / mu.sum())}"
                )
                d_norm = L_hat * mu / mu.sum()
        else:
            # eq (12)
            d_hat = pred_durations[note_indices[i - 1] : note_indices[i]]
            d_norm = L_hat * d_hat / d_hat.sum()

        d_norm = np.round(d_norm)
        d_norm[d_norm <= 0] = 1

        p.set_durations(d_norm)

        if len(output_labels) > 0:
            output_labels.end_times[-1] = p.start_times[0]
        for n in p:
            output_labels.append(n)

    return output_labels


@torch.no_grad()
def predict_timing(
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
    """Predict timinigs from HTS labels

    This is equivalent to ``predict_timelag + predict_duration + postprocess_duration``.

    Args:
        device (torch.device): device to run the model on
        labels (nnmnkwii.io.hts.HTSLabelFile): labels
        binary_dict (dict): binary feature dictionary
        numeric_dict (dict): numeric feature dictionary
        timelag_model (nn.Module): timelag model
        timelag_config (dict): timelag config
        timelag_in_scaler (sklearn.preprocessing.MinMaxScaler): timelag input scaler
        timelag_out_scaler (sklearn.preprocessing.MinMaxScaler): timelag output scaler
        duration_model (nn.Module): duration model
        duration_config (dict): duration config
        duration_in_scaler (sklearn.preprocessing.MinMaxScaler): duration input scaler
        duration_out_scaler (sklearn.preprocessing.MinMaxScaler): duration output scaler
        log_f0_conditioning (bool): whether to condition on log f0
        allowed_range (list): allowed range of time-lag
        allowed_range_rest (list): allowed range of time-lag for rest
        force_clip_input_features (bool): whether to clip input features
        frame_period (int): frame period in milliseconds

    Returns:
        nnmnkwii.io.hts.HTSLabelFile: duration modified labels
    """
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
def predict_acoustic(
    device,
    labels,
    acoustic_model,
    acoustic_config,
    acoustic_in_scaler,
    acoustic_out_scaler,
    binary_dict,
    numeric_dict,
    subphone_features="coarse_coding",
    pitch_indices=None,
    log_f0_conditioning=True,
    force_clip_input_features=False,
    frame_period=5,
    f0_shift_in_cent=0,
):
    """Predict acoustic features from HTS labels

    MLPG is applied to the predicted features if the output features have
    dynamic features.

    Args:
        device (torch.device): device to use
        labels (HTSLabelFile): HTS labels
        acoustic_model (nn.Module): acoustic model
        acoustic_config (AcousticConfig): acoustic configuration
        acoustic_in_scaler (sklearn.preprocessing.StandardScaler): input scaler
        acoustic_out_scaler (sklearn.preprocessing.StandardScaler): output scaler
        binary_dict (dict): binary feature dictionary
        numeric_dict (dict): numeric feature dictionary
        subphone_features (str): subphone feature type
        pitch_indices (list): indices of pitch features
        log_f0_conditioning (bool): whether to use log f0 conditioning
        force_clip_input_features (bool): whether to force clip input features
        frame_period (float): frame period in msec
        f0_shift_in_cent (float): F0 shift in cent-scale before the inference

    Returns:
        ndarray: predicted acoustic features
    """
    hts_frame_shift = int(frame_period * 1e4)
    if pitch_indices is None:
        pitch_indices = get_pitch_indices(binary_dict, numeric_dict)

    # Musical/linguistic features
    linguistic_features = fe.linguistic_features(
        labels,
        binary_dict,
        numeric_dict,
        add_frame_features=True,
        subphone_features=subphone_features,
        frame_shift=hts_frame_shift,
    )

    if log_f0_conditioning:
        for idx in pitch_indices:
            linguistic_features[:, idx] = interp1d(
                _midi_to_hz(linguistic_features, idx, log_f0_conditioning),
                kind="slinear",
            )
            if f0_shift_in_cent != 0:
                lf0_offset = f0_shift_in_cent * np.log(2) / 1200
                linguistic_features[:, idx] += lf0_offset

    # Apply normalization
    linguistic_features = acoustic_in_scaler.transform(linguistic_features)
    if force_clip_input_features and isinstance(acoustic_in_scaler, MinMaxScaler):
        # clip to feature range (except for pitch-related features)
        non_pitch_indices = [
            idx
            for idx in range(linguistic_features.shape[1])
            if idx not in pitch_indices
        ]
        linguistic_features[:, non_pitch_indices] = np.clip(
            linguistic_features[:, non_pitch_indices],
            acoustic_in_scaler.feature_range[0],
            acoustic_in_scaler.feature_range[1],
        )

    # Predict acoustic features
    x = torch.from_numpy(linguistic_features).float().to(device)
    x = x.view(1, -1, x.size(-1))

    if acoustic_model.prediction_type() in [
        PredictionType.PROBABILISTIC,
        PredictionType.MULTISTREAM_HYBRID,
    ]:
        # (B, T, D_out)
        max_mu, max_sigma = acoustic_model.inference(x, [x.shape[1]])
        if np.any(acoustic_config.has_dynamic_features):
            # Apply denormalization
            # (B, T, D_out) -> (T, D_out)
            max_sigma_sq = (
                max_sigma.squeeze(0).cpu().data.numpy() ** 2 * acoustic_out_scaler.var_
            )
            max_sigma_sq = np.maximum(max_sigma_sq, 1e-14)
            max_mu = acoustic_out_scaler.inverse_transform(
                max_mu.squeeze(0).cpu().data.numpy()
            )

            # (T, D_out) -> (T, static_dim)
            pred_acoustic = multi_stream_mlpg(
                max_mu,
                max_sigma_sq,
                get_windows(acoustic_config.num_windows),
                acoustic_config.stream_sizes,
                acoustic_config.has_dynamic_features,
            )
        else:
            # Apply denormalization
            pred_acoustic = acoustic_out_scaler.inverse_transform(
                max_mu.squeeze(0).cpu().data.numpy()
            )
    else:
        # (T, D_out)
        pred_acoustic = (
            acoustic_model.inference(x, [x.shape[1]]).squeeze(0).cpu().data.numpy()
        )
        # Apply denormalization
        pred_acoustic = acoustic_out_scaler.inverse_transform(pred_acoustic)
        if np.any(acoustic_config.has_dynamic_features):
            # (T, D_out) -> (T, static_dim)
            pred_acoustic = multi_stream_mlpg(
                pred_acoustic,
                acoustic_out_scaler.var_,
                get_windows(acoustic_config.num_windows),
                acoustic_config.stream_sizes,
                acoustic_config.has_dynamic_features,
            )

    return pred_acoustic


@torch.no_grad()
def postprocess_acoustic(
    device,
    acoustic_features,
    duration_modified_labels,
    binary_dict,
    numeric_dict,
    acoustic_config,
    acoustic_out_static_scaler,
    postfilter_model=None,
    postfilter_config=None,
    postfilter_out_scaler=None,
    sample_rate=48000,
    frame_period=5,
    relative_f0=False,
    feature_type="world",
    post_filter_type="gv",
    trajectory_smoothing=True,
    trajectory_smoothing_cutoff=50,
    trajectory_smoothing_cutoff_f0=20,
    vuv_threshold=0.5,
    f0_shift_in_cent=0,
    vibrato_scale=1.0,
    force_fix_vuv=False,
    fill_silence_to_rest=False,
):
    """Post-process acoustic features

    The function converts acoustic features in single ndarray to tuple of
    multi-stream acoustic features.

    e.g., array -> (mgc, lf0, vuv, bap)

    Args:
        device (torch.device): Device.
        duration_modified_labels (nnmnkwii.io.hts.HTSLabelFile): HTS label file.
        binary_dict (dict): Dictionary of binary features.
        numeric_dict (dict): Dictionary of numeric features.
        acoustic_config (dict): Acoustic model configuration.
        acoustic_features (np.ndarray): Acoustic features.
        acoustic_out_static_scaler (sklearn.preprocessing.StandardScaler): Scaler
            for acoustic features.
        postfilter_model (nn.Module): Post-filter model.
        postfilter_config (dict): Post-filter model configuration.
        postfilter_out_scaler (sklearn.preprocessing.StandardScaler): Scaler for post-filter.
        sample_rate (int): Sampling rate.
        frame_period (float): Frame period in milliseconds.
        relative_f0 (bool): If True, use relative f0.
        feature_type (str): Feature type.
        post_filter_type (str): Post-filter type.
            One of ``gv``, ``merlin`` or ``nnsvs``. Recommended to use ``gv``
            for general purpose.
        trajectory_smoothing (bool): Whether to apply trajectory smoothing.
        trajectory_smoothing_cutoff (float): Cutoff frequency for trajectory smoothing
            of spectral features.
        trajectory_smoothing_cutoff_f0 (float): Cutoff frequency for trajectory smoothing of f0.
        vuv_threshold (float): V/UV threshold.
        f0_shift_in_cent (float): F0 shift in cents.
        vibrato_scale (float): Vibrato scale.
        force_fix_vuv (bool): If True, force to fix V/UV.
        fill_silence_to_rest (bool): Fill silence to rest frames.

    Returns:
        tuple: Post-processed acoustic features.
    """
    hts_frame_shift = int(frame_period * 1e4)
    pitch_idx = get_pitch_index(binary_dict, numeric_dict)

    static_stream_sizes = get_static_stream_sizes(
        acoustic_config.stream_sizes,
        acoustic_config.has_dynamic_features,
        acoustic_config.num_windows,
    )

    linguistic_features = fe.linguistic_features(
        duration_modified_labels,
        binary_dict,
        numeric_dict,
        add_frame_features=True,
        frame_shift=hts_frame_shift,
    )
    # GV post-filter
    if post_filter_type == "gv" or (
        post_filter_type == "nnsvs" and feature_type == "world"
    ):
        note_frame_indices = get_note_frame_indices(
            binary_dict, numeric_dict, linguistic_features
        )
        if feature_type == "world":
            offset = 2
        elif feature_type == "melf0":
            # NOTE: set offset so that post-filter does not affect F0
            mel_freq = librosa.mel_frequencies(
                n_mels=80, fmin=63, fmax=sample_rate // 2
            )
            # NOTE: the threshold could be tuned for better performance
            offset = np.argmax(mel_freq > 1200)

        # NOTE: apply the post-filter for note frames only
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

    if fill_silence_to_rest:
        mask = _get_nonrest_frame_soft_mask(
            binary_dict, numeric_dict, linguistic_features
        )
        if feature_type == "world":
            mgc, lf0, vuv, bap = _fill_silence_to_world_params(mgc, lf0, vuv, bap, mask)
        elif feature_type == "melf0":
            mel, lf0, vuv = _fill_silence_to_mel_params(mel, lf0, vuv, mask)

    if f0_shift_in_cent != 0:
        lf0_offset = f0_shift_in_cent * np.log(2) / 1200
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

    if feature_type == "world":
        return mgc, lf0, vuv, bap
    elif feature_type == "melf0":
        return mel, lf0, vuv


@torch.no_grad()
def predict_waveform(
    device,
    multistream_features,
    vocoder=None,
    vocoder_config=None,
    vocoder_in_scaler=None,
    sample_rate=48000,
    frame_period=5,
    use_world_codec=True,
    feature_type="world",
    vocoder_type="world",
    vuv_threshold=0.5,
):
    """Predict waveform from multi-stream acoustic features

    Vocoders can be 1) WORLD, 2) PWG or 3) uSFGAN.

    Args:
        device (torch.device): Device to run inference
        features (tuple): Acoustic features
        vocoder (nn.Module): Vocoder model
        vocoder_config (dict): Vocoder config
        vocoder_in_scaler (StandardScaler): Vocoder input scaler
        sample_rate (int,): Sampling rate.
        frame_period (float): Frame period in msec.
        use_world_codec (bool): Whether to use WORLD codec for decoding.
        feature_type (str): Feature type.
            ``world`` ``world_org``, ``melf0`` or ``neutrino``.
        vocoder_type (str): Vocoder type. ``world`` or ``pwg`` or ``usfgan``
        vuv_threshold (float): VUV threshold.

    Returns:
        np.ndarray: Predicted waveform
    """
    if feature_type == "world":
        mgc, lf0, vuv, bap = multistream_features
    elif feature_type == "world_org":
        f0, spectrogram, aperiodicity = multistream_features
    elif feature_type == "neutrino":
        mgc, f0, bap = multistream_features
        # prepare (mgc, lf0, vuv, bap) to be compatible with NNSVS
        lf0 = f0.copy()
        lf0[np.nonzero(f0)] = np.log(f0[np.nonzero(f0)])
        vuv = (f0 > 0).astype(np.float32)
    elif feature_type == "melf0":
        mel, lf0, vuv = multistream_features
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    # NOTE: `use_mcep_aperiodicity` was used for experimental purpose but didn't make
    # significant difference. Please just ignore or ping @r9y9 for details.
    if feature_type in ["world", "neutrino"]:
        use_mcep_aperiodicity = bap.shape[-1] > 5

    if feature_type == "neutrino" and not use_world_codec:
        raise ValueError("use_world_codec must be True when feature_type is neutrino")

    # Waveform generation by WORLD
    if vocoder_type == "world":
        if feature_type not in ["world", "world_org", "neutrino"]:
            raise ValueError(f"Invalid feature type for WORLD vocoder: {feature_type}")
        if feature_type == "world_org":
            # NOTE: WORLD-based features are already converted to raw WORLD parameters
            pass
        else:
            f0, spectrogram, aperiodicity = gen_world_params(
                mgc,
                lf0,
                vuv,
                bap,
                sample_rate,
                vuv_threshold=vuv_threshold,
                use_world_codec=use_world_codec,
            )
        # make sure to have float64 typed parameters
        wav = pyworld.synthesize(
            f0.astype(np.float64),
            spectrogram.astype(np.float64),
            aperiodicity.astype(np.float64),
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
        if feature_type in ["world", "neutrino"]:
            fftlen = pyworld.get_cheaptrick_fft_size(sample_rate)
            if use_mcep_aperiodicity:
                # Convert mel-cepstrum-based aperiodicity to WORLD's aperiodicity
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

            # Convert aperiodicity back to BAP
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
        elif feature_type == "world_org":
            # it is possible to implement here but I suppose nobody wants to use
            raise NotImplementedError()

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
        # NOTE: uSFGAN internally performs normalization
        # so we don't need to normalize inputs here
        wav = vocoder.inference(f0_inp, aux_feats).view(-1).to("cpu").numpy()

    return wav


def postprocess_waveform(
    wav,
    sample_rate,
    dtype=np.int16,
    peak_norm=False,
    loudness_norm=False,
    target_loudness=-20.0,
):
    """Perform post-processing for synthesized waveform

    Args:
        wav (ndarray): The input waveform
        sample_rate (int): The sampling rate
        dtype (np.dtype): The dtype of output waveform. Default is np.int16.
        peak_norm (bool): Whether to perform peak normalization
        loudness_norm (bool): Whether to perform loudness normalization
        target_loudness (float): Target loudness in dB

    Returns:
        ndarray: The post-processed waveform
    """
    wav = bandpass_filter(wav, sample_rate)

    # Peak normalize audio to 0 dB
    if peak_norm:
        wav = pyln.normalize.peak(wav, 0.0)

    # Normalize loudnes
    # NOTE: -20 dB is roughly the same as the NEURINO (NSF ver.)
    if loudness_norm:
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, target_loudness)

    # NOTE: use np.int16 to save disk space
    if dtype in [np.int16, "int16"]:
        # NOTE: NNSVS (>=0.0.3) uses waveforms normalized in [-1, 1] for training.
        # so the following code shouldn't be used but in case for models trained
        # with earlier NNSVS
        if np.max(np.abs(wav)) > 10:
            # data is likely already in [-32768, 32767]
            wav = wav.astype(np.int16)
        elif np.max(np.abs(wav)) <= 1:
            wav = (wav * 32767.0).astype(np.int16)
        else:
            # may need to handle int32 data (if any)
            warn("Unexpected waveform range: {} - {}".format(np.min(wav), np.max(wav)))
            warn("Failed to convert to int16. Returning waveform with floating point.")
    elif dtype is None:
        pass
    else:
        wav = wav.astype(dtype)

    return wav


def _get_nonrest_frame_soft_mask(
    binary_dict,
    numeric_dict,
    linguistic_features,
    win_length=200,
    duration_threshold=1.0,
):
    """Get mask for non-rest frames

    Args:
        binary_dict (dict): Dictionary for binary features
        numeric_dict (dict): Dictionary for numeric features
        linguistic_features (ndarray): Linguistic features
        win_length (int): Window length

    Returns:
        ndarray: Soft mask for non-rest frames.
            1 for non-rest frames and 0 for otherwise.
    """
    mask = np.ones(len(linguistic_features))

    in_sil_indices = []
    for k, v in binary_dict.items():
        name, _ = v
        if "C-Phone_sil" in name or "C-Phone_pau" in name:
            in_sil_indices.append(k)
    if len(in_sil_indices) == 0:
        return mask

    in_note_dur_idx = None
    for k, v in numeric_dict.items():
        name, _ = v
        if "e7" in name:
            in_note_dur_idx = k
            break

    dur = linguistic_features[:, len(binary_dict) + in_note_dur_idx]
    dur_in_sec = dur * 0.01
    for in_sil_idx in in_sil_indices:
        # Only mask out sil/pau segments over ${silence_threshold} sec. such as long pause
        mask[
            (linguistic_features[:, in_sil_idx] > 0) & (dur_in_sec > duration_threshold)
        ] = 0

    # make a smoothed mask with ${win_length} * 5ms window length
    mask = scipy.signal.convolve(mask, np.ones(win_length) / win_length, mode="same")

    # make sure that we don't mask out frames where notes are assigned
    pitch_idx = get_pitch_index(binary_dict, numeric_dict)
    score_f0 = linguistic_features[:, pitch_idx]
    mask[score_f0 > 0] = 1.0

    return mask.reshape(-1, 1)


def _fill_silence_to_world_params(mgc, lf0, vuv, bap, mask):
    mgc_sil = np.zeros((1, mgc.shape[1]))

    # NOTE: mgc_sil is a VERY ROUGH estimate of mgc for silence regions
    # the speech signal is assumed to be in [-1, 1].
    #   sr = 48000
    #   noise = np.random.randn(sr * 10) * 1e-5
    #   f0, timeaxis = pyworld.harvest(noise, sr, frame_period=5)
    #   f0[:] = 0
    #   spectrogram = pyworld.cheaptrick(noise, f0, timeaxis, sr)
    #   mgc = pyworld.code_spectral_envelope(spectrogram, sr, 60)
    #   print(mgc.mean(0))
    mgc_sil[0, 0] = -23.3
    mgc_sil[0, 1] = 0.0679
    mgc_sil[0, 2] = 0.00640
    mgc_sil[0, 3:] = 1e-3
    bap_sil = np.zeros_like(bap) + 1e-11

    mgc = mgc * mask + (1 - mask) * mgc_sil
    bap = bap * mask + (1 - mask) * bap_sil

    return mgc, lf0, vuv, bap


def _fill_silence_to_mel_params(mel, lf0, vuv, mask):
    # NOTE: -5.5 is also a very rough estimate of log-melspectrogram
    # for silence regions
    mel_sil = np.zeros((1, mel.shape[1])) - 5.5
    mel = mel * mask + (1 - mask) * mel_sil
    return mel, lf0, vuv


def correct_vuv_by_phone(vuv, binary_dict, linguistic_features):
    """Correct V/UV by phone-related flags in a hed file

    This function allows us to control V/UV explicitly by ``C-VUV_Voiced``
    and ``C-VUV_Unvoied`` flags in a hed file. This is useful when you see
    your trained acoustic model have lots of V/UV errors.
    Note that manually controlling V/UV means we are ignoring the
    acoustic model's prediction. It would have negative impact in some
    cases, but most cases it would help workaround V/UV errors.

    Args:
        vuv (ndarray): V/UV flags
        binary_dict (dict): binary feature dictionary
        linguistic_features (ndarray): linguistic features

    Returns:
        ndarray: corrected V/UV flags
    """
    vuv = vuv.copy()

    # Set V/UV to 1 based on the C-VUV_Voiced flag
    in_voiced_idx = -1
    for k, v in binary_dict.items():
        name, _ = v
        if "C-VUV_Voiced" in name:
            in_voiced_idx = k
            break
    if in_voiced_idx > 0:
        indices = linguistic_features[:, in_voiced_idx : in_voiced_idx + 1] > 0
        vuv[indices] = 1.0

    # Set V/UV to 0 based on the C-VUV_Unvoiced flag
    in_unvoiced_indices = []
    for k, v in binary_dict.items():
        name, _ = v
        if "C-VUV_Unvoiced" in name:
            in_unvoiced_indices.append(k)
    if len(in_unvoiced_indices) > 0:
        for in_unvoiced_idx in in_unvoiced_indices:
            indices = linguistic_features[:, in_unvoiced_idx : in_unvoiced_idx + 1] > 0
            vuv[indices] = 0.0

    # Set V/UV to 0 for sil/pau/br
    in_sil_indices = []
    for k, v in binary_dict.items():
        name, _ = v
        if "C-Phone_sil" in name or "C-Phone_pau" in name or "C-Phone_br" in name:
            in_sil_indices.append(k)
    if len(in_sil_indices) > 0:
        for in_sil_idx in in_sil_indices:
            indices = linguistic_features[:, in_sil_idx : in_sil_idx + 1] > 0
            vuv[indices] = 0.0

    return vuv


def gen_spsvs_static_features(
    labels,
    acoustic_features,
    binary_dict,
    numeric_dict,
    stream_sizes,
    has_dynamic_features,
    pitch_idx=None,
    num_windows=3,
    frame_period=5,
    relative_f0=True,
    vibrato_scale=1.0,
    vuv_threshold=0.3,
    force_fix_vuv=True,
):
    """Generate static features from predicted acoustic features

    Args:
        labels (HTSLabelFile): HTS labels
        acoustic_features (ndarray): predicted acoustic features
        binary_dict (dict): binary feature dictionary
        numeric_dict (dict): numeric feature dictionary
        stream_sizes (list): stream sizes
        has_dynamic_features (list): whether each stream has dynamic features
        pitch_idx (int): index of pitch features
        num_windows (int): number of windows
        frame_period (float): frame period
        relative_f0 (bool): whether to use relative f0
        vibrato_scale (float): vibrato scale
        vuv_threshold (float): vuv threshold
        force_fix_vuv (bool): whether to use post-processing to fix VUV.

    Returns:
        tuple: tuple of mgc, lf0, vuv and bap.
    """
    hts_frame_shift = int(frame_period * 1e4)
    if pitch_idx is None:
        pitch_idx = get_pitch_index(binary_dict, numeric_dict)

    if np.any(has_dynamic_features):
        static_stream_sizes = get_static_stream_sizes(
            stream_sizes, has_dynamic_features, num_windows
        )
    else:
        static_stream_sizes = stream_sizes

    # Copy here to avoid inplace operations on input acoustic features
    acoustic_features = acoustic_features.copy()

    # Split multi-stream features
    streams = split_streams(acoustic_features, static_stream_sizes)

    if len(streams) == 4:
        mgc, target_f0, vuv, bap = streams
        vib, vib_flags = None, None
    elif len(streams) == 5:
        # Assuming diff-based vibrato parameters
        mgc, target_f0, vuv, bap, vib = streams
        vib_flags = None
    elif len(streams) == 6:
        # Assuming sine-based vibrato parameters
        mgc, target_f0, vuv, bap, vib, vib_flags = streams
    else:
        raise RuntimeError("Not supported streams")

    linguistic_features = fe.linguistic_features(
        labels,
        binary_dict,
        numeric_dict,
        add_frame_features=True,
        frame_shift=hts_frame_shift,
    )

    # Correct V/UV based on special phone flags
    if force_fix_vuv:
        vuv = correct_vuv_by_phone(vuv, binary_dict, linguistic_features)

    # F0
    if relative_f0:
        diff_lf0 = target_f0
        f0_score = _midi_to_hz(linguistic_features, pitch_idx, False)[:, None]
        lf0_score = f0_score.copy()
        nonzero_indices = np.nonzero(lf0_score)
        lf0_score[nonzero_indices] = np.log(f0_score[nonzero_indices])
        lf0_score = interp1d(lf0_score, kind="slinear")

        f0 = diff_lf0 + lf0_score
        f0[vuv < vuv_threshold] = 0
        f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])
    else:
        f0 = target_f0
        f0[vuv < vuv_threshold] = 0
        f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])

    if vib is not None:
        if vib_flags is not None:
            # Generate sine-based vibrato
            vib_flags = vib_flags.flatten()
            m_a, m_f = vib[:, 0], vib[:, 1]

            # Fill zeros for non-vibrato frames
            m_a[vib_flags < 0.5] = 0
            m_f[vib_flags < 0.5] = 0

            # Gen vibrato
            sr_f0 = int(1 / (frame_period * 0.001))
            f0 = gen_sine_vibrato(f0.flatten(), sr_f0, m_a, m_f, vibrato_scale)
        else:
            # Generate diff-based vibrato
            f0 = f0.flatten() + vibrato_scale * vib.flatten()

    # NOTE: Back to log-domain for convenience
    lf0 = f0.copy()
    lf0[np.nonzero(lf0)] = np.log(f0[np.nonzero(lf0)])
    # NOTE: interpolation is necessary
    lf0 = interp1d(lf0, kind="slinear")

    lf0 = lf0[:, None] if len(lf0.shape) == 1 else lf0
    vuv = vuv[:, None] if len(vuv.shape) == 1 else vuv

    return mgc, lf0, vuv, bap


def gen_world_params(
    mgc,
    lf0,
    vuv,
    bap,
    sample_rate,
    vuv_threshold=0.3,
    use_world_codec=False,
):
    """Generate WORLD parameters from mgc, lf0, vuv and bap.

    Args:
        mgc (ndarray): mgc
        lf0 (ndarray): lf0
        vuv (ndarray): vuv
        bap (ndarray): bap
        sample_rate (int): sample rate
        vuv_threshold (float): threshold for VUV
        use_world_codec (bool): whether to use WORLD codec for spectral envelope

    Returns:
        tuple: tuple of f0, spectrogram and aperiodicity
    """
    fftlen = pyworld.get_cheaptrick_fft_size(sample_rate)
    alpha = pysptk.util.mcepalpha(sample_rate)
    use_mcep_aperiodicity = bap.shape[-1] > 5

    if use_world_codec:
        spectrogram = pyworld.decode_spectral_envelope(
            np.ascontiguousarray(mgc).astype(np.float64), sample_rate, fftlen
        )
    else:
        spectrogram = pysptk.mc2sp(
            np.ascontiguousarray(mgc), fftlen=fftlen, alpha=alpha
        )

    if use_mcep_aperiodicity:
        aperiodicity = pysptk.mc2sp(
            np.ascontiguousarray(bap), fftlen=fftlen, alpha=alpha
        )
    else:
        aperiodicity = pyworld.decode_aperiodicity(
            np.ascontiguousarray(bap).astype(np.float64), sample_rate, fftlen
        )

    # fill aperiodicity with ones for unvoiced regions
    aperiodicity[vuv.reshape(-1) < vuv_threshold, 0] = 1.0
    # WORLD fails catastrophically for out of range aperiodicity
    aperiodicity = np.clip(aperiodicity, 0.0, 1.0)

    f0 = lf0.copy()
    f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])
    f0[vuv < vuv_threshold] = 0

    f0 = f0.flatten().astype(np.float64)
    spectrogram = spectrogram.astype(np.float64)
    aperiodicity = aperiodicity.astype(np.float64)

    return f0, spectrogram, aperiodicity
