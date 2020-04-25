# coding: utf-8

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import numpy as np
import joblib
import pysptk
import pyworld
import librosa
import torch
from scipy.io import wavfile
from nnmnkwii import paramgen
from nnmnkwii.io import hts
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.postfilters import merlin_post_filter
from dnnsvs.logger import getLogger
logger = None


# TODO: code is toooo dirty now and need to reorganize

fs = 48000

mgc_dim = 180
lf0_dim = 3
vuv_dim = 1
bap_dim = pyworld.get_num_aperiodicities(fs) * 3
mgc_start_idx = 0
lf0_start_idx = 180
vuv_start_idx = 183
bap_start_idx = 184
frame_period = 5
fftlen = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)
hop_length = int(0.001 * frame_period * fs)
log_f0_conditioning = True
pitch_indices = None
pitch_idx = None

windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]

def get_note_indices(lab):
    note_indices = [0]
    last_start_time = lab.start_times[0]
    for idx in range(1, len(lab)):
        if lab.start_times[idx] != last_start_time:
            note_indices.append(idx)
            last_start_time = lab.start_times[idx]
        else:
            pass
    return note_indices


def midi_to_hz(x, idx, log_f0=False):
    # z = x[:, idx].copy()
    z = np.zeros(len(x))
    indices = x[:, idx] > 0
    z[indices] = librosa.midi_to_hz(x[indices, idx])
    if log_f0:
        z[indices] = np.log(z[indices])
    return z


def gen_parameters(y_predicted, acoustic_out_scaler):
    # Number of time frames
    T = y_predicted.shape[0]

    # Split acoustic features
    mgc = y_predicted[:,:lf0_start_idx]
    lf0 = y_predicted[:,lf0_start_idx:vuv_start_idx]
    vuv = y_predicted[:,vuv_start_idx]
    bap = y_predicted[:,bap_start_idx:]

    # Perform MLPG
    mgc_variances = np.tile(acoustic_out_scaler.var_[:lf0_start_idx], (T, 1))
    mgc = paramgen.mlpg(mgc, mgc_variances, windows)
    lf0_variances = np.tile(acoustic_out_scaler.var_[lf0_start_idx:vuv_start_idx], (T,1))
    lf0 = paramgen.mlpg(lf0, lf0_variances, windows)
    bap_variances = np.tile(acoustic_out_scaler.var_[bap_start_idx:], (T, 1))
    bap = paramgen.mlpg(bap, bap_variances, windows)

    return mgc, lf0, vuv, bap


def gen_waveform(y_predicted, acoustic_out_scaler, linguistic_features, do_postfilter=False, do_clip=False):
    if False:
        y_predicted = trim_zeros_frames(y_predicted)

    # Generate parameters and split streams
    mgc, diff_lf0, vuv, bap = gen_parameters(y_predicted, acoustic_out_scaler)

    if do_clip:
        diff_lf0 = np.clip(diff_lf0, np.log(0.9), np.log(1.1))

    if do_postfilter:
        mgc = merlin_post_filter(mgc, alpha)

    spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha)
    aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), fs, fftlen)

    # Handle lf0 carefully
    if log_f0_conditioning:
        lf0_score = linguistic_features[:, pitch_idx][:, None]
    else:
        f0_score = midi_to_hz(linguistic_features, pitch_idx, False)[:, None]
        lf0_score = f0_score.copy()
        lf0_score[nonzero_indices] = np.log(f0_score[nonzero_indices])
    nonzero_indices = np.nonzero(lf0_score)
    lf0_score = interp1d(lf0_score, kind="slinear")

    f0 = diff_lf0 + lf0_score
    f0[vuv < 0.5] = 0
    f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])

    generated_waveform = pyworld.synthesize(f0.flatten().astype(np.float64),
                                            spectrogram.astype(np.float64),
                                            aperiodicity.astype(np.float64),
                                            fs, frame_period)
    return generated_waveform

def gen_duration(labels, question_path, duration_model, duration_in_scaler, duration_out_scaler,
        lag):
    # Linguistic features for duration
    binary_dict, continuous_dict = hts.load_question_set(
        question_path, append_hat_for_LL=False)
    note_indices = get_note_indices(labels)
    # append the end of note
    note_indices.append(len(labels))

    duration_linguistic_features = fe.linguistic_features(labels,
                                               binary_dict, continuous_dict,
                                               add_frame_features=False,
                                               subphone_features=None).astype(np.float32)

    if log_f0_conditioning:
        for idx in pitch_indices:
            duration_linguistic_features[:, idx] = midi_to_hz(duration_linguistic_features, idx, True)

    # Apply normalization
    duration_linguistic_features = duration_in_scaler.transform(duration_linguistic_features)

    # Apply model
    duration_model = duration_model.cpu()
    duration_model.eval()
    x = torch.from_numpy(duration_linguistic_features).float()
    x = x.view(1, -1, x.size(-1))
    duration_predicted = duration_model(x, [x.shape[1]]).squeeze(0).data.numpy()

    # Apply denormalization
    duration_predicted = duration_out_scaler.inverse_transform(duration_predicted)
    duration_predicted[duration_predicted <= 0] = 1
    duration_predicted = np.round(duration_predicted)

    output_labels = hts.HTSLabelFile()
    for i in range(1, len(note_indices)):
        # Apply time lag
        p = labels[note_indices[i-1]:note_indices[i]]
        p.start_times = np.asarray(p.start_times) + lag[i-1].reshape(-1)
        p.start_times = np.maximum(p.start_times, 0)
        if len(output_labels) > 0:
            p.start_times = np.maximum(p.start_times, output_labels.start_times[-1] + 50000)

        # Compute normalized phoneme durations
        d = fe.duration_features(p)
        d_hat = duration_predicted[note_indices[i-1]:note_indices[i]]
        d_norm = d[0] * d_hat / d_hat.sum()
        d_norm = np.round(d_norm)
        # TODO: better way to adjust?
        if d_norm.sum() != d[0]:
            d_norm[-1] +=  d[0] - d_norm.sum()
        p.set_durations(d_norm)

        if len(output_labels) > 0:
            output_labels.end_times[-1] = p.start_times[0]
        for n in p:
            output_labels.append(n)

    return output_labels


def test_one_utt(config, label_path, question_path,
        timelag_model, timelag_in_scaler, timelag_out_scaler,
        duration_model, duration_in_scaler, duration_out_scaler,
        acoustic_model, acoustic_in_scaler, acoustic_out_scaler,
        post_filter=True, pred_duration=True):
    labels = hts.load(label_path)

    binary_dict, continuous_dict = hts.load_question_set(
        question_path, append_hat_for_LL=False)
    global pitch_indices
    global pitch_idx
    pitch_idx = len(binary_dict) + 1
    pitch_indices = np.arange(len(binary_dict), len(binary_dict)+3)

    # Timelag predictions
    note_indices = get_note_indices(labels)
    note_labels = labels[note_indices]
    timelag_linguistic_features = fe.linguistic_features(
        note_labels, binary_dict, continuous_dict,
        add_frame_features=False, subphone_features=None).astype(np.float32)
    if log_f0_conditioning:
        for idx in pitch_indices:
            timelag_linguistic_features[:, idx] = midi_to_hz(timelag_linguistic_features, idx, True)
    timelag_linguistic_features = timelag_in_scaler.transform(timelag_linguistic_features)

    x = torch.from_numpy(timelag_linguistic_features).unsqueeze(0)
    y = timelag_model(x, [x.shape[1]]).squeeze(0)
    lag = np.round(timelag_out_scaler.inverse_transform(y.data.numpy()) / 50000) * 50000

    # Predict durations
    if not pred_duration:
        duration_modified_hts_labels = labels
    else:
        duration_modified_hts_labels = gen_duration(
            labels, question_path, duration_model, duration_in_scaler, duration_out_scaler, lag)

    # Linguistic features
    linguistic_features = fe.linguistic_features(duration_modified_hts_labels,
                                                  binary_dict, continuous_dict,
                                                  add_frame_features=True,
                                                  subphone_features="coarse_coding")

    if log_f0_conditioning:
        for idx in pitch_indices:
            linguistic_features[:, idx] = midi_to_hz(linguistic_features, idx, True)

    linguistic_features_org = linguistic_features.copy()

    # Apply normalization
    linguistic_features = acoustic_in_scaler.transform(linguistic_features)

    # Predict acoustic features
    acoustic_model = acoustic_model.cpu()
    acoustic_model.eval()
    x = torch.from_numpy(linguistic_features).float()
    x = x.view(1, -1, x.size(-1))
    acoustic_predicted = acoustic_model(x, [x.shape[1]]).squeeze(0).data.numpy()

    # Apply denormalization
    acoustic_predicted = acoustic_out_scaler.inverse_transform(acoustic_predicted)

    return gen_waveform(acoustic_predicted, acoustic_out_scaler, linguistic_features_org, post_filter)


@hydra.main(config_path="conf/inference/config.yaml")
def my_app(config : DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(config.pretty())

    # timelag
    timelag_def = OmegaConf.load(to_absolute_path(config.timelag.model_yaml))
    timelag_model = hydra.utils.instantiate(timelag_def)
    checkpoint = torch.load(to_absolute_path(config.timelag.checkpoint))
    timelag_model.load_state_dict(checkpoint["state_dict"])
    timelag_in_scaler = joblib.load(to_absolute_path(config.timelag.in_scaler_path))
    timelag_out_scaler = joblib.load(to_absolute_path(config.timelag.out_scaler_path))

    # duration
    duration_def = OmegaConf.load(to_absolute_path(config.duration.model_yaml))
    duration_model = hydra.utils.instantiate(duration_def)
    checkpoint = torch.load(to_absolute_path(config.duration.checkpoint))
    duration_model.load_state_dict(checkpoint["state_dict"])
    duration_in_scaler = joblib.load(to_absolute_path(config.duration.in_scaler_path))
    duration_out_scaler = joblib.load(to_absolute_path(config.duration.out_scaler_path))

    # acoustic model
    acoustic_def = OmegaConf.load(to_absolute_path(config.acoustic.model_yaml))
    acoustic_model = hydra.utils.instantiate(acoustic_def)
    checkpoint = torch.load(to_absolute_path(config.acoustic.checkpoint))
    acoustic_model.load_state_dict(checkpoint["state_dict"])
    acoustic_in_scaler = joblib.load(to_absolute_path(config.acoustic.in_scaler_path))
    acoustic_out_scaler = joblib.load(to_absolute_path(config.acoustic.out_scaler_path))

    label_path = to_absolute_path(config.label_path)
    question_path = to_absolute_path(config.question_path)
    out_wave_path = to_absolute_path(config.out_wave_path)
    wav = test_one_utt(config, label_path, question_path,
        timelag_model, timelag_in_scaler, timelag_out_scaler,
        duration_model, duration_in_scaler, duration_out_scaler,
        acoustic_model, acoustic_in_scaler, acoustic_out_scaler)

    wav = wav.astype(np.int16)
    wavfile.write(out_wave_path, rate=fs, data=wav)


def entry():
    my_app()


if __name__ == "__main__":
    my_app()