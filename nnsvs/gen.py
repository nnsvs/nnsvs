import librosa
import numpy as np
import pysptk
import pyworld
import torch
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from nnmnkwii.postfilters import merlin_post_filter
from nnmnkwii.preprocessing.f0 import interp1d
from nnsvs.base import PredictionType
from nnsvs.io.hts import get_note_indices
from nnsvs.multistream import get_static_stream_sizes, multi_stream_mlpg, split_streams
from nnsvs.pitch import gen_sine_vibrato
from sklearn.preprocessing import MinMaxScaler


def get_windows(num_window=1):
    windows = [(0, 0, np.array([1.0]))]
    if num_window >= 2:
        windows.append((1, 1, np.array([-0.5, 0.0, 0.5])))
    if num_window >= 3:
        windows.append((1, 1, np.array([1.0, -2.0, 1.0])))

    if num_window >= 4:
        raise ValueError(f"Not supported num windows: {num_window}")

    return windows


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
):
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
    pred_timelag *= 50000

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
):
    # Extract musical/linguistic features
    duration_linguistic_features = fe.linguistic_features(
        labels,
        binary_dict,
        numeric_dict,
        add_frame_features=False,
        subphone_features=None,
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


def postprocess_duration(labels, pred_durations, lag):
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
            L_hat = L - (lag[i - 1] - lag[i]) / 50000
        else:
            L_hat = L - (lag[i - 1]) / 50000

        # adjust the start time of the note
        p.start_times = np.minimum(
            np.asarray(p.start_times) + lag[i - 1].reshape(-1),
            np.asarray(p.end_times) - 50000 * len(p),
        )
        p.start_times = np.maximum(p.start_times, 0)
        if len(output_labels) > 0:
            p.start_times = np.maximum(
                p.start_times, output_labels.start_times[-1] + 50000
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
                print(
                    f"Negative phoneme durations are predicted at {i}-th note. "
                    "The note duration: ",
                    f"{round(float(L)*0.005,3)} sec -> {round(float(L_hat)*0.005,3)} sec",
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
):

    # Musical/linguistic features
    linguistic_features = fe.linguistic_features(
        labels,
        binary_dict,
        numeric_dict,
        add_frame_features=True,
        subphone_features=subphone_features,
    )

    if log_f0_conditioning:
        for idx in pitch_indices:
            linguistic_features[:, idx] = interp1d(
                _midi_to_hz(linguistic_features, idx, log_f0_conditioning),
                kind="slinear",
            )

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

    if acoustic_model.prediction_type() == PredictionType.PROBABILISTIC:
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


def gen_world_params(
    labels,
    acoustic_features,
    binary_dict,
    numeric_dict,
    stream_sizes,
    has_dynamic_features,
    subphone_features="coarse_coding",
    pitch_idx=None,
    num_windows=3,
    post_filter=True,
    sample_rate=48000,
    frame_period=5,
    relative_f0=True,
    vibrato_scale=1.0,
    vuv_threshold=0.3,
):
    # Apply MLPG if necessary
    if np.any(has_dynamic_features):
        static_stream_sizes = get_static_stream_sizes(
            stream_sizes, has_dynamic_features, num_windows
        )
    else:
        static_stream_sizes = stream_sizes

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

    # Gen waveform by the WORLD vocodoer
    fftlen = pyworld.get_cheaptrick_fft_size(sample_rate)
    alpha = pysptk.util.mcepalpha(sample_rate)

    if post_filter:
        mgc = merlin_post_filter(mgc, alpha)

    spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha)
    aperiodicity = pyworld.decode_aperiodicity(
        bap.astype(np.float64), sample_rate, fftlen
    )

    # fill aperiodicity with ones for unvoiced regions
    aperiodicity[vuv.reshape(-1) < vuv_threshold, :] = 1.0
    # WORLD fails catastrophically for out of range aperiodicity
    aperiodicity = np.clip(aperiodicity, 0.0, 1.0)

    # F0
    if relative_f0:
        diff_lf0 = target_f0
        # need to extract pitch sequence from the musical score
        linguistic_features = fe.linguistic_features(
            labels,
            binary_dict,
            numeric_dict,
            add_frame_features=True,
            subphone_features=subphone_features,
        )
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

    f0 = f0.flatten().astype(np.float64)
    spectrogram = spectrogram.astype(np.float64)
    aperiodicity = aperiodicity.astype(np.float64)

    return f0, spectrogram, aperiodicity
