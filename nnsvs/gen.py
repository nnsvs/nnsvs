import librosa
import numpy as np
import pysptk
import pyworld
import torch
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from nnmnkwii.preprocessing.f0 import interp1d
from nnsvs.base import PredictionType
from nnsvs.io.hts import get_note_indices
from nnsvs.multistream import (
    get_static_stream_sizes,
    get_windows,
    multi_stream_mlpg,
    split_streams,
)
from nnsvs.pitch import gen_sine_vibrato
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

    Returns:
        ndarray: predicted acoustic features
    """
    hts_frame_shift = int(frame_period * 1e4)

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
    subphone_features="coarse_coding",
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
        subphone_features (str): subphone feature type
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
        subphone_features=subphone_features,
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
