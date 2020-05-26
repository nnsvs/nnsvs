# coding: utf-8

import numpy as np
import torch
import pysptk
import pyworld
import librosa

from sklearn.preprocessing import MinMaxScaler
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.postfilters import merlin_post_filter
from nnmnkwii.preprocessing.f0 import interp1d
from nnsvs.io.hts import get_note_indices
from nnsvs.multistream import multi_stream_mlpg, get_static_stream_sizes
from nnsvs.multistream import select_streams, split_streams


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

def _is_silence(l):
    is_full_context = "@" in l
    if is_full_context:
        is_silence = ("-sil" in l or "-pau" in l)
    else:
        is_silence = (l == "sil" or l == "pau")
    return is_silence


def predict_timelag(device, labels, timelag_model, timelag_in_scaler, timelag_out_scaler,
        binary_dict, continuous_dict,
        pitch_indices=None, log_f0_conditioning=True,
        allowed_range=[-20, 20], allowed_range_rest=[-40, 40]):
    # round start/end times just in case.
    labels.round_()

    # Extract note-level labels
    note_indices = get_note_indices(labels)
    note_labels = labels[note_indices]

    # Extract musical/linguistic context
    timelag_linguistic_features = fe.linguistic_features(
        note_labels, binary_dict, continuous_dict,
        add_frame_features=False, subphone_features=None).astype(np.float32)

    # Adjust input features if we use log-f0 conditioning
    if log_f0_conditioning:
        if pitch_indices is None:
            raise ValueError("Pitch feature indices must be specified!")
        for idx in pitch_indices:
            timelag_linguistic_features[:, idx] = interp1d(
                _midi_to_hz(timelag_linguistic_features, idx, log_f0_conditioning),
                    kind="slinear")

    # Normalization
    timelag_linguistic_features = timelag_in_scaler.transform(timelag_linguistic_features)
    if isinstance(timelag_in_scaler, MinMaxScaler):
        # clip to feature range
        timelag_linguistic_features = np.clip(
            timelag_linguistic_features, timelag_in_scaler.feature_range[0],
            timelag_in_scaler.feature_range[1])

    # Run model
    x = torch.from_numpy(timelag_linguistic_features).unsqueeze(0).to(device)
    y = timelag_model(x, [x.shape[1]]).squeeze(0).cpu()

    # De-normalization and rounding
    lag = np.round(timelag_out_scaler.inverse_transform(y.data.numpy()))

    # Clip to the allowed range
    for idx in range(len(lag)):
        if _is_silence(note_labels.contexts[idx]):
            lag[idx] = np.clip(lag[idx], allowed_range_rest[0], allowed_range_rest[1])
        else:
            lag[idx] = np.clip(lag[idx], allowed_range[0], allowed_range[1])

    # frames -> 100 ns
    lag *= 50000

    return lag


def postprocess_duration(labels, pred_durations, lag):
    note_indices = get_note_indices(labels)
    # append the end of note
    note_indices.append(len(labels))

    output_labels = hts.HTSLabelFile()

    for i in range(1, len(note_indices)):
        # Apply time lag
        p = labels[note_indices[i-1]:note_indices[i]]
        p.start_times = np.minimum(
            np.asarray(p.start_times) + lag[i-1].reshape(-1),
            np.asarray(p.end_times) - 50000 * len(p))
        p.start_times = np.maximum(p.start_times, 0)
        if len(output_labels) > 0:
            p.start_times = np.maximum(p.start_times, output_labels.start_times[-1] + 50000)

        # Compute normalized phoneme durations
        d = fe.duration_features(p)
        d_hat = pred_durations[note_indices[i-1]:note_indices[i]]
        d_norm = d[0] * d_hat / d_hat.sum()
        d_norm = np.round(d_norm)
        d_norm[d_norm <= 0] = 1

        # TODO: better way to adjust?
        if d_norm.sum() != d[0]:
            d_norm[-1] +=  d[0] - d_norm.sum()
        p.set_durations(d_norm)

        if len(output_labels) > 0:
            output_labels.end_times[-1] = p.start_times[0]
        for n in p:
            output_labels.append(n)

    return output_labels


def predict_duration(device, labels, duration_model, duration_in_scaler, duration_out_scaler,
        lag, binary_dict, continuous_dict, pitch_indices=None, log_f0_conditioning=True):
    # Extract musical/linguistic features
    duration_linguistic_features = fe.linguistic_features(
        labels, binary_dict, continuous_dict,
        add_frame_features=False, subphone_features=None).astype(np.float32)

    if log_f0_conditioning:
        for idx in pitch_indices:
            duration_linguistic_features[:, idx] = interp1d(
                _midi_to_hz(duration_linguistic_features, idx, log_f0_conditioning),
                    kind="slinear")

    # Apply normalization
    duration_linguistic_features = duration_in_scaler.transform(duration_linguistic_features)
    if isinstance(duration_in_scaler, MinMaxScaler):
        # clip to feature range
        duration_linguistic_features = np.clip(
            duration_linguistic_features, duration_in_scaler.feature_range[0],
            duration_in_scaler.feature_range[1])

    # Apply model
    x = torch.from_numpy(duration_linguistic_features).float().to(device)
    x = x.view(1, -1, x.size(-1))
    pred_durations = duration_model(x, [x.shape[1]]).squeeze(0).cpu().data.numpy()

    # Apply denormalization
    pred_durations = duration_out_scaler.inverse_transform(pred_durations)
    pred_durations[pred_durations <= 0] = 1
    pred_durations = np.round(pred_durations)

    return pred_durations


def predict_acoustic(device, labels, acoustic_model, acoustic_in_scaler,
        acoustic_out_scaler, binary_dict, continuous_dict,
        subphone_features="coarse_coding",
        pitch_indices=None, log_f0_conditioning=True):

    # Musical/linguistic features
    linguistic_features = fe.linguistic_features(labels,
                                                  binary_dict, continuous_dict,
                                                  add_frame_features=True,
                                                  subphone_features=subphone_features)

    if log_f0_conditioning:
        for idx in pitch_indices:
            linguistic_features[:, idx] = interp1d(
                _midi_to_hz(linguistic_features, idx, log_f0_conditioning),
                    kind="slinear")

    # Apply normalization
    linguistic_features = acoustic_in_scaler.transform(linguistic_features)
    if isinstance(acoustic_in_scaler, MinMaxScaler):
        # clip to feature range
        linguistic_features = np.clip(
            linguistic_features, acoustic_in_scaler.feature_range[0],
            acoustic_in_scaler.feature_range[1])

    # Predict acoustic features
    x = torch.from_numpy(linguistic_features).float().to(device)
    x = x.view(1, -1, x.size(-1))
    pred_acoustic = acoustic_model(x, [x.shape[1]]).squeeze(0).cpu().data.numpy()

    # Apply denormalization
    pred_acoustic = acoustic_out_scaler.inverse_transform(pred_acoustic)

    return pred_acoustic


def gen_waveform(labels, acoustic_features, acoustic_out_scaler,
        binary_dict, continuous_dict, stream_sizes, has_dynamic_features,
        subphone_features="coarse_coding", log_f0_conditioning=True, pitch_idx=None,
        num_windows=3, post_filter=True, sample_rate=48000, frame_period=5,
        relative_f0=True):

    windows = get_windows(num_windows)

    # Apply MLPG if necessary
    if np.any(has_dynamic_features):
        acoustic_features = multi_stream_mlpg(
            acoustic_features, acoustic_out_scaler.var_, windows, stream_sizes,
            has_dynamic_features)
        static_stream_sizes = get_static_stream_sizes(
            stream_sizes, has_dynamic_features, len(windows))
    else:
        static_stream_sizes = stream_sizes

    # Split multi-stream features
    mgc, target_f0, vuv, bap = split_streams(acoustic_features, static_stream_sizes)

    # Gen waveform by the WORLD vocodoer
    fftlen = pyworld.get_cheaptrick_fft_size(sample_rate)
    alpha = pysptk.util.mcepalpha(sample_rate)

    if post_filter:
        mgc = merlin_post_filter(mgc, alpha)

    spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha)
    aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), sample_rate, fftlen)

    # fill aperiodicity with ones for unvoiced regions
    aperiodicity[vuv.reshape(-1) < 0.5, :] = 1.0
    # WORLD fails catastrophically for out of range aperiodicity
    aperiodicity = np.clip(aperiodicity, 0.0, 1.0)

    ### F0 ###
    if relative_f0:
        diff_lf0 = target_f0
        # need to extract pitch sequence from the musical score
        linguistic_features = fe.linguistic_features(labels,
                                                    binary_dict, continuous_dict,
                                                    add_frame_features=True,
                                                    subphone_features=subphone_features)
        f0_score = _midi_to_hz(linguistic_features, pitch_idx, False)[:, None]
        lf0_score = f0_score.copy()
        nonzero_indices = np.nonzero(lf0_score)
        lf0_score[nonzero_indices] = np.log(f0_score[nonzero_indices])
        lf0_score = interp1d(lf0_score, kind="slinear")

        f0 = diff_lf0 + lf0_score
        f0[vuv < 0.5] = 0
        f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])
    else:
        f0 = target_f0

    generated_waveform = pyworld.synthesize(f0.flatten().astype(np.float64),
                                            spectrogram.astype(np.float64),
                                            aperiodicity.astype(np.float64),
                                            sample_rate, frame_period)

    return generated_waveform
