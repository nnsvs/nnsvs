from os.path import join

import librosa
import numpy as np
import pysptk
import pyworld
from nnmnkwii.datasets import FileDataSource
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.util import apply_delta_windows
from nnsvs.gen import get_windows
from nnsvs.pitch import (
    extract_smoothed_f0,
    extract_vibrato_likelihood,
    extract_vibrato_parameters,
    hz_to_cent_based_c4,
)
from scipy.io import wavfile


def _collect_files(data_root, utt_list, ext):
    with open(utt_list) as f:
        files = f.readlines()
    files = map(lambda utt_id: utt_id.strip(), files)
    files = filter(lambda utt_id: len(utt_id) > 0, files)
    files = list(map(lambda utt_id: join(data_root, f"{utt_id}{ext}"), files))
    return files


def _midi_to_hz(x, idx, log_f0=False):
    z = np.zeros(len(x))
    indices = x[:, idx] > 0
    z[indices] = librosa.midi_to_hz(x[indices, idx])
    if log_f0:
        z[indices] = np.log(z[indices])
    return z


class MusicalLinguisticSource(FileDataSource):
    def __init__(
        self,
        utt_list,
        data_root,
        question_path,
        add_frame_features=False,
        subphone_features=None,
        log_f0_conditioning=True,
    ):
        self.data_root = data_root
        self.utt_list = utt_list
        self.add_frame_features = add_frame_features
        self.subphone_features = subphone_features
        self.binary_dict, self.continuous_dict = hts.load_question_set(
            question_path, append_hat_for_LL=False
        )
        self.log_f0_conditioning = log_f0_conditioning
        self.pitch_idx = np.arange(len(self.binary_dict), len(self.binary_dict) + 3)

    def collect_files(self):
        return _collect_files(self.data_root, self.utt_list, ".lab")

    def collect_features(self, path):
        labels = hts.load(path)
        features = fe.linguistic_features(
            labels,
            self.binary_dict,
            self.continuous_dict,
            add_frame_features=self.add_frame_features,
            subphone_features=self.subphone_features,
        )
        if self.log_f0_conditioning:
            for idx in self.pitch_idx:
                features[:, idx] = interp1d(
                    _midi_to_hz(features, idx, True), kind="slinear"
                )
        return features.astype(np.float32)


class TimeLagFeatureSource(FileDataSource):
    def __init__(self, utt_list, label_phone_score_dir, label_phone_align_dir):
        self.utt_list = utt_list
        self.label_phone_score_dir = label_phone_score_dir
        self.label_phone_align_dir = label_phone_align_dir

    def collect_files(self):
        labels_score = _collect_files(self.label_phone_score_dir, self.utt_list, ".lab")
        labels_align = _collect_files(self.label_phone_align_dir, self.utt_list, ".lab")
        return labels_score, labels_align

    def collect_features(self, label_score_path, label_align_path):
        label_score = hts.load(label_score_path)
        label_align = hts.load(label_align_path)
        timelag = np.asarray(label_align.start_times) - np.asarray(
            label_score.start_times
        )
        # 100ns -> num frames
        timelag = timelag.astype(np.float32) / 50000
        return timelag.reshape(-1, 1)


class DurationFeatureSource(FileDataSource):
    def __init__(self, utt_list, data_root):
        self.utt_list = utt_list
        self.data_root = data_root

    def collect_files(self):
        return _collect_files(self.data_root, self.utt_list, ".lab")

    def collect_features(self, path):
        labels = hts.load(path)
        features = fe.duration_features(labels)
        return features.astype(np.float32)


class WORLDAcousticSource(FileDataSource):
    def __init__(
        self,
        utt_list,
        wav_root,
        label_root,
        question_path,
        use_harvest=True,
        f0_floor=150,
        f0_ceil=700,
        frame_period=5,
        mgc_order=59,
        num_windows=3,
        relative_f0=True,
        interp_unvoiced_aperiodicity=True,
        extract_vibrato=False,
    ):
        self.utt_list = utt_list
        self.wav_root = wav_root
        self.label_root = label_root
        self.binary_dict, self.continuous_dict = hts.load_question_set(
            question_path, append_hat_for_LL=False
        )
        self.pitch_idx = len(self.binary_dict) + 1
        self.use_harvest = use_harvest
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.frame_period = frame_period
        self.mgc_order = mgc_order
        self.relative_f0 = relative_f0
        self.interp_unvoiced_aperiodicity = interp_unvoiced_aperiodicity
        self.extract_vibrato = extract_vibrato
        self.windows = get_windows(num_windows)

    def collect_files(self):
        wav_paths = _collect_files(self.wav_root, self.utt_list, ".wav")
        label_paths = _collect_files(self.label_root, self.utt_list, ".lab")
        return wav_paths, label_paths

    def collect_features(self, wav_path, label_path):
        labels = hts.load(label_path)
        l_features = fe.linguistic_features(
            labels,
            self.binary_dict,
            self.continuous_dict,
            add_frame_features=True,
            subphone_features="coarse_coding",
        )

        f0_score = _midi_to_hz(l_features, self.pitch_idx, False)
        notes = l_features[:, self.pitch_idx]
        notes = notes[notes > 0]

        # allow 200 cent upper/lower to properly handle F0 estimation of
        # preparation, vibrato and overshoot.
        # NOET: set the minimum f0 to 63.5 Hz (125 - 3*20.5)
        # https://acoustics.jp/qanda/answer/50.html
        # NOTE: sinsy allows 30-150 cent frequency range for vibrato (as of 2010)
        # https://staff.aist.go.jp/m.goto/PAPER/SIGMUS201007oura.pdf
        min_f0 = max(63.5, librosa.midi_to_hz(min(notes) - 2))
        max_f0 = librosa.midi_to_hz(max(notes) + 2)
        assert max_f0 > min_f0

        fs, x = wavfile.read(wav_path)
        x = x.astype(np.float64)

        if self.use_harvest:
            f0, timeaxis = pyworld.harvest(
                x, fs, frame_period=self.frame_period, f0_floor=min_f0, f0_ceil=max_f0
            )
        else:
            f0, timeaxis = pyworld.dio(
                x, fs, frame_period=self.frame_period, f0_floor=min_f0, f0_ceil=max_f0
            )
            f0 = pyworld.stonemask(x, f0, timeaxis, fs)

        # Workaround for https://github.com/r9y9/nnsvs/issues/7
        f0 = np.maximum(f0, 0)

        if self.extract_vibrato:
            assert (
                not self.use_harvest
            ), "harvest is not supported for vibrato extraction"
            sr_f0 = int(1 / (self.frame_period * 0.001))
            win_length = 64
            n_fft = 256
            threshold = 0.12

            f0_smooth = extract_smoothed_f0(f0, sr_f0, cutoff=8)
            f0_smooth_cent = hz_to_cent_based_c4(f0_smooth)
            vibrato_likelihood = extract_vibrato_likelihood(
                f0_smooth_cent, sr_f0, win_length=win_length, n_fft=n_fft
            )
            _, m_a, m_f = extract_vibrato_parameters(
                f0_smooth_cent, vibrato_likelihood, sr_f0, threshold=threshold
            )
            vib = np.stack([m_a, m_f], axis=1)
        else:
            vib = None

        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs, f0_floor=min_f0)
        aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)

        mgc = pysptk.sp2mc(
            spectrogram, order=self.mgc_order, alpha=pysptk.util.mcepalpha(fs)
        )
        # F0 of speech
        f0 = f0[:, None]
        lf0 = f0.copy()
        nonzero_indices = np.nonzero(f0)
        lf0[nonzero_indices] = np.log(f0[nonzero_indices])
        if self.use_harvest:
            # https://github.com/mmorise/World/issues/35#issuecomment-306521887
            vuv = (aperiodicity[:, 0] < 0.5).astype(np.float32)[:, None]
        else:
            vuv = (lf0 != 0).astype(np.float32)
        lf0 = interp1d(lf0, kind="slinear")

        # Aperiodicy
        # ref: https://github.com/MTG/WGANSing/blob/mtg/vocoder.py
        if self.interp_unvoiced_aperiodicity:
            is_voiced = (vuv > 0).reshape(-1)
            if not np.any(is_voiced):
                pass  # all unvoiced, do nothing
            else:
                for k in range(aperiodicity.shape[1]):
                    aperiodicity[~is_voiced, k] = np.interp(
                        np.where(~is_voiced)[0],
                        np.where(is_voiced)[0],
                        aperiodicity[is_voiced, k],
                    )
        bap = pyworld.code_aperiodicity(aperiodicity, fs)

        # Adjust lengths
        mgc = mgc[: labels.num_frames()]
        lf0 = lf0[: labels.num_frames()]
        vuv = vuv[: labels.num_frames()]
        bap = bap[: labels.num_frames()]
        vib = vib[: labels.num_frames()] if vib is not None else None

        if self.relative_f0:
            # # F0 derived from the musical score
            f0_score = f0_score[:, None]
            if len(f0_score) > len(f0):
                print(
                    "Warning! likely to have mistakes in alignment in {}".format(
                        label_path
                    )
                )
                print(f0_score.shape, f0.shape)
                f0_score = f0_score[: len(f0)]

            lf0_score = f0_score.copy()
            nonzero_indices = np.nonzero(f0_score)
            lf0_score[nonzero_indices] = np.log(f0_score[nonzero_indices])
            lf0_score = interp1d(lf0_score, kind="slinear")
            # relative f0
            diff_lf0 = lf0 - lf0_score
            diff_lf0 = np.clip(diff_lf0, np.log(0.5), np.log(2.0))

            f0_target = diff_lf0
        else:
            f0_target = lf0

        mgc = apply_delta_windows(mgc, self.windows)
        f0_target = apply_delta_windows(f0_target, self.windows)
        bap = apply_delta_windows(bap, self.windows)
        vib = apply_delta_windows(vib, self.windows) if vib is not None else None

        if vib is None:
            features = np.hstack((mgc, f0_target, vuv, bap)).astype(np.float32)
        else:
            features = np.hstack((mgc, f0_target, vuv, bap, vib)).astype(np.float32)

        # Align waveform and features
        wave = x.astype(np.float32) / 2 ** 15
        T = int(features.shape[0] * (fs * self.frame_period / 1000))
        if len(wave) < T:
            if T - len(wave) > int(fs * 0.005):
                print("Warn!!", T, len(wave), T - len(wave))
                print("you have unepxcted input. Please debug though ipdb")
                import ipdb

                ipdb.set_trace()
            else:
                pass
            wave = np.pad(wave, (0, T - len(wave)))
        assert wave.shape[0] >= T
        wave = wave[:T]

        return features, wave
