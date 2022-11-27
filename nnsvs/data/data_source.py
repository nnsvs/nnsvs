from os.path import join

import librosa
import numpy as np
import pysptk
import pyworld
import soundfile as sf
from nnmnkwii.datasets import FileDataSource
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.util import apply_delta_windows
from nnsvs.io.hts import get_pitch_index, get_pitch_indices
from nnsvs.multistream import get_windows
from nnsvs.pitch import (
    compute_f0_correction_ratio,
    extract_smoothed_continuous_f0,
    extract_smoothed_f0,
    extract_vibrato_likelihood,
    extract_vibrato_parameters,
    hz_to_cent_based_c4,
    lowpass_filter,
)
from nnsvs.util import init_seed
from parallel_wavegan.bin.preprocess import logmelfilterbank
from scipy.signal import firwin, lfilter


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


def low_cut_filter(x, fs, cutoff=70):
    """Low cut filter

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence

    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


class MusicalLinguisticSource(FileDataSource):
    def __init__(
        self,
        utt_list,
        data_root,
        question_path,
        add_frame_features=False,
        subphone_features=None,
        log_f0_conditioning=True,
        frame_period=5,
    ):
        self.data_root = data_root
        self.utt_list = utt_list
        self.add_frame_features = add_frame_features
        self.subphone_features = subphone_features
        self.binary_dict, self.numeric_dict = hts.load_question_set(
            question_path, append_hat_for_LL=False
        )
        self.log_f0_conditioning = log_f0_conditioning
        self.frame_period = frame_period
        self.pitch_indices = get_pitch_indices(self.binary_dict, self.numeric_dict)

    def collect_files(self):
        return _collect_files(self.data_root, self.utt_list, ".lab")

    def collect_features(self, path):
        labels = hts.load(path)
        hts_frame_shift = int(self.frame_period * 1e4)
        labels.frame_shift = hts_frame_shift
        features = fe.linguistic_features(
            labels,
            self.binary_dict,
            self.numeric_dict,
            add_frame_features=self.add_frame_features,
            subphone_features=self.subphone_features,
            frame_shift=hts_frame_shift,
        )
        if self.log_f0_conditioning:
            for idx in self.pitch_indices:
                lf0_score = _midi_to_hz(features, idx, True)
                features[:, idx] = interp1d(lf0_score, kind="slinear")

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
        f0_extractor="harvest",
        f0_floor=150,
        f0_ceil=700,
        frame_period=5,
        mgc_order=59,
        num_windows=3,
        relative_f0=True,
        interp_unvoiced_aperiodicity=True,
        vibrato_mode="none",  # diff, sine
        sample_rate=48000,
        d4c_threshold=0.85,
        trajectory_smoothing=False,
        trajectory_smoothing_cutoff=50,
        trajectory_smoothing_f0=True,
        trajectory_smoothing_cutoff_f0=20,
        correct_vuv=False,
        correct_f0=False,
        dynamic_features_flags=None,
        use_world_codec=False,
        use_mcep_aperiodicity=False,
        mcep_aperiodicity_order=24,
        res_type="scipy",
    ):
        self.utt_list = utt_list
        self.wav_root = wav_root
        self.label_root = label_root
        self.binary_dict, self.numeric_dict = hts.load_question_set(
            question_path, append_hat_for_LL=False
        )
        self.pitch_idx = get_pitch_index(self.binary_dict, self.numeric_dict)
        self.f0_extractor = f0_extractor
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.frame_period = frame_period
        self.mgc_order = mgc_order
        self.relative_f0 = relative_f0
        self.interp_unvoiced_aperiodicity = interp_unvoiced_aperiodicity
        self.vibrato_mode = vibrato_mode
        self.windows = get_windows(num_windows)
        self.sample_rate = sample_rate
        self.d4c_threshold = d4c_threshold
        self.trajectory_smoothing = trajectory_smoothing
        self.trajectory_smoothing_cutoff = trajectory_smoothing_cutoff
        self.trajectory_smoothing_f0 = trajectory_smoothing_f0
        self.trajectory_smoothing_cutoff_f0 = trajectory_smoothing_cutoff_f0
        self.correct_vuv = correct_vuv
        self.correct_f0 = correct_f0
        self.use_world_codec = use_world_codec
        self.use_mcep_aperiodicity = use_mcep_aperiodicity
        self.mcep_aperiodicity_order = mcep_aperiodicity_order
        if dynamic_features_flags is None:
            # NOTE: we have up to 6 streams: (mgc, lf0, vuv, bap, vib, vib_flags)
            dynamic_features_flags = [True, True, False, True, True, False]
        self.dynamic_features_flags = dynamic_features_flags
        self.res_type = res_type

    def collect_files(self):
        wav_paths = _collect_files(self.wav_root, self.utt_list, ".wav")
        label_paths = _collect_files(self.label_root, self.utt_list, ".lab")
        return wav_paths, label_paths

    def collect_features(self, wav_path, label_path):
        labels = hts.load(label_path)
        hts_frame_shift = int(self.frame_period * 1e4)
        labels.frame_shift = hts_frame_shift
        num_frames = int(labels.num_frames(frame_shift=hts_frame_shift))

        l_features = fe.linguistic_features(
            labels,
            self.binary_dict,
            self.numeric_dict,
            add_frame_features=True,
            subphone_features="coarse_coding",
            frame_shift=hts_frame_shift,
        )

        f0_score = _midi_to_hz(l_features, self.pitch_idx, False)
        notes = l_features[:, self.pitch_idx]
        notes = notes[notes > 0]

        # allow 200 cent upper and 600 cent lower to properly handle F0 estimation of
        # preparation, vibrato and overshoot.
        # NOET: set the minimum f0 to 63.5 Hz (125 - 3*20.5)
        # https://acoustics.jp/qanda/answer/50.html
        # NOTE: sinsy allows 30-150 cent frequency range for vibrato (as of 2010)
        # https://staff.aist.go.jp/m.goto/PAPER/SIGMUS201007oura.pdf
        min_f0 = max(63.5, librosa.midi_to_hz(min(notes) - 6))
        max_f0 = librosa.midi_to_hz(max(notes) + 2)
        assert max_f0 > min_f0

        # Use fixed f0 range for HARVEST if specified
        if self.f0_floor is not None:
            min_f0 = self.f0_floor
        if self.f0_ceil is not None:
            max_f0 = self.f0_ceil

        # Workaround segfault issues of WORLD's CheapTrick
        min_f0 = min(min_f0, 500)

        x, fs = sf.read(wav_path)
        assert np.max(x) <= 1.0
        assert x.dtype == np.float64

        if fs != self.sample_rate:
            x = librosa.resample(
                x, orig_sr=fs, target_sr=self.sample_rate, res_type=self.res_type
            )
            fs = self.sample_rate

        if self.f0_extractor == "parselmouth":
            import parselmouth

            assert (
                self.f0_floor is not None and self.f0_ceil is not None
            ), "must be set manually"
            harvest_num_frames = int(int(1000 * len(x) / fs) / self.frame_period) + 1
            f0 = (
                parselmouth.Sound(x.astype(np.float64), fs)
                .to_pitch_ac(
                    time_step=self.frame_period * 0.001,
                    voicing_threshold=0.6,
                    very_accurate=False,
                    pitch_floor=min_f0,
                    pitch_ceiling=max_f0,
                )
                .selected_array["frequency"]
            )
            pad = int(np.round((3 / min_f0) / (self.frame_period * 0.001)))
            f0 = np.pad(f0, [[0, pad]], mode="constant")
            if len(f0) > harvest_num_frames:
                f0 = f0[:harvest_num_frames]
            elif len(f0) < harvest_num_frames:
                f0 = np.pad(f0, (0, harvest_num_frames - len(f0)), mode="constant")

            timeaxis = np.arange(harvest_num_frames) * self.frame_period * 0.001
        elif self.f0_extractor == "harvest":
            f0, timeaxis = pyworld.harvest(
                x, fs, frame_period=self.frame_period, f0_floor=min_f0, f0_ceil=max_f0
            )
        elif self.f0_extractor == "dio":
            f0, timeaxis = pyworld.dio(
                x, fs, frame_period=self.frame_period, f0_floor=min_f0, f0_ceil=max_f0
            )
            f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        else:
            raise ValueError(f"unknown f0 extractor: {self.f0_extractor}")

        # Workaround for https://github.com/r9y9/nnsvs/issues/7
        f0 = np.maximum(f0, 0)

        # Correct V/UV (and F0) based on the musical score information
        # treat frames where musical notes are not assigned as unvoiced
        # Use smoothed mask so that we don't mask out overshoot or something
        # that could happen at the start/end of notes
        # 0.5 sec. window (could be tuned for better results)
        if self.correct_vuv:
            win_length = int(0.5 / (self.frame_period * 0.001))
            mask = np.convolve(f0_score, np.ones(win_length) / win_length, "same")
            if len(f0) > len(mask):
                mask = np.pad(mask, (0, len(f0) - len(mask)), "constant")
            elif len(f0) < len(mask):
                mask = mask[: len(f0)]
            f0 = f0 * np.sign(mask)

        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
        aperiodicity = pyworld.d4c(x, f0, timeaxis, fs, threshold=self.d4c_threshold)

        if np.isnan(aperiodicity).any():
            print(wav_path)
            print(min_f0, max_f0, aperiodicity.shape, fs)
            print(np.isnan(aperiodicity).sum())
            print(aperiodicity)
            raise RuntimeError("Aperiodicity has NaN")

        # Apply pitch correction
        # NOTE: we should better to apply pitch corrections
        # or manually adjust UST/musicxml in advance.
        sr_f0 = int(1 / (self.frame_period * 0.001))
        if self.correct_f0:
            f0_smooth = extract_smoothed_f0(f0, sr_f0, cutoff=20)
            ratio = compute_f0_correction_ratio(f0_smooth, f0_score)
            f0 *= ratio

        lf0 = f0[:, np.newaxis].copy()
        nonzero_indices = np.nonzero(lf0)
        lf0[nonzero_indices] = np.log(f0[:, np.newaxis][nonzero_indices])
        if self.f0_extractor == "harvest":
            # https://github.com/mmorise/World/issues/35#issuecomment-306521887
            vuv = (aperiodicity[:, 0] < 0.5).astype(np.float32)[:, None]
        else:
            vuv = (lf0 != 0).astype(np.float32)

        # F0 -> continuous F0
        lf0 = interp1d(lf0, kind="slinear")

        # Smooth continuous F0 to avoid discontinuities
        if self.trajectory_smoothing_f0:
            lf0 = extract_smoothed_continuous_f0(
                lf0, sr_f0, cutoff=self.trajectory_smoothing_cutoff_f0
            )

        # Fill continuous F0s for segments where no notes are assigned & no F0s are detected.
        lf0_score = _midi_to_hz(l_features, self.pitch_idx, True)
        clf0_score = interp1d(lf0_score, kind="slinear")
        mask = np.convolve(lf0_score, np.ones(1), "same")
        if len(f0) > len(mask):
            mask = np.pad(mask, (0, len(f0) - len(mask)), "constant")
            clf0_score = np.pad(clf0_score, (0, len(f0) - len(clf0_score)), "constant")
        elif len(f0) < len(mask):
            mask = mask[: len(f0)]
            clf0_score = clf0_score[: len(f0)]
        ind = (mask + f0.reshape(-1)) <= 0
        lf0[ind, 0] = clf0_score[ind]

        # Vibrato parameter extraction
        if self.vibrato_mode == "sine":
            win_length = 64
            n_fft = 256
            threshold = 0.12

            if self.f0_extractor == "harvest":
                # NOTE: harvest is not supported here since the current implemented algorithm
                # relies on v/uv flags to find vibrato sections.
                # We use DIO since it provides more accurate v/uv detection in my experience.
                _f0, _timeaxis = pyworld.dio(
                    x,
                    fs,
                    frame_period=self.frame_period,
                    f0_floor=min_f0,
                    f0_ceil=max_f0,
                )
                _f0 = pyworld.stonemask(x, _f0, _timeaxis, fs)
                f0_smooth = extract_smoothed_f0(_f0, sr_f0, cutoff=8)
            else:
                f0_smooth = extract_smoothed_f0(f0, sr_f0, cutoff=8)

            f0_smooth_cent = hz_to_cent_based_c4(f0_smooth)
            vibrato_likelihood = extract_vibrato_likelihood(
                f0_smooth_cent, sr_f0, win_length=win_length, n_fft=n_fft
            )
            vib_flags, m_a, m_f = extract_vibrato_parameters(
                f0_smooth_cent, vibrato_likelihood, sr_f0, threshold=threshold
            )
            m_a = interp1d(m_a, kind="linear")
            m_f = interp1d(m_f, kind="linear")
            vib = np.stack([m_a, m_f], axis=1)
            vib_flags = vib_flags[:, np.newaxis]
        elif self.vibrato_mode == "diff":
            # NOTE: vibrato is known to have 3 ~ 8 Hz range (in general)
            # remove higher frequency than 3 to separate vibrato from the original F0
            f0_smooth = extract_smoothed_f0(f0, sr_f0, cutoff=3)
            assert len(f0.shape) == 1 and len(f0_smooth.shape) == 1
            vib = (f0 - f0_smooth)[:, np.newaxis]
            vib_flags = None
        elif self.vibrato_mode == "none":
            vib, vib_flags = None, None
        else:
            raise RuntimeError("Unknown vibrato mode: {}".format(self.vibrato_mode))

        if self.use_world_codec:
            mgc = pyworld.code_spectral_envelope(spectrogram, fs, self.mgc_order + 1)
        else:
            mgc = pysptk.sp2mc(
                spectrogram, order=self.mgc_order, alpha=pysptk.util.mcepalpha(fs)
            )
        # NOTE: used as the target for post-filters
        spectrogram = np.log(spectrogram)

        # Post-processing for aperiodicy
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

        if self.use_mcep_aperiodicity:
            bap = pysptk.sp2mc(
                aperiodicity,
                order=self.mcep_aperiodicity_order,
                alpha=pysptk.util.mcepalpha(fs),
            )
        else:
            bap = pyworld.code_aperiodicity(aperiodicity, fs)

        # Parameter trajectory smoothing
        if self.trajectory_smoothing:
            modfs = int(1 / (self.frame_period * 0.001))
            for d in range(mgc.shape[1]):
                mgc[:, d] = lowpass_filter(
                    mgc[:, d], modfs, cutoff=self.trajectory_smoothing_cutoff
                )
            for d in range(bap.shape[1]):
                bap[:, d] = lowpass_filter(
                    bap[:, d], modfs, cutoff=self.trajectory_smoothing_cutoff
                )

        # Adjust lengths
        sp = spectrogram[:num_frames]
        mgc = mgc[:num_frames]
        lf0 = lf0[:num_frames]
        vuv = vuv[:num_frames]
        bap = bap[:num_frames]
        vib = vib[:num_frames] if vib is not None else None
        vib_flags = vib_flags[:num_frames] if vib_flags is not None else None

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

        # Compute delta features if necessary
        if self.dynamic_features_flags[0]:
            mgc = apply_delta_windows(mgc, self.windows)
        if self.dynamic_features_flags[1]:
            f0_target = apply_delta_windows(f0_target, self.windows)
        if self.dynamic_features_flags[3]:
            bap = apply_delta_windows(bap, self.windows)
        if vib is not None and self.dynamic_features_flags[4]:
            vib = apply_delta_windows(vib, self.windows)

        # Concat features
        if vib is None and vib_flags is None:
            features = np.hstack((mgc, f0_target, vuv, bap)).astype(np.float32)
            pf_features = np.hstack((sp, f0_target, vuv, bap)).astype(np.float32)
        elif vib is not None and vib_flags is None:
            features = np.hstack((mgc, f0_target, vuv, bap, vib)).astype(np.float32)
            pf_features = np.hstack((sp, f0_target, vuv, bap, vib)).astype(np.float32)
        elif vib is not None and vib_flags is not None:
            features = np.hstack((mgc, f0_target, vuv, bap, vib, vib_flags)).astype(
                np.float32
            )
            pf_features = np.hstack((sp, f0_target, vuv, bap, vib, vib_flags)).astype(
                np.float32
            )
        else:
            raise RuntimeError("Unknown combination of features")

        if len(features) < num_frames:
            print(f"WARN: length mismatch for {wav_path}")
            print(len(features), num_frames)
            return None, None, None

        # Align waveform and features
        wave = x.astype(np.float32)

        # NOTE: since neural vocoders need to perform integer-valued up-sampling
        # (e.g., 120x upsampling with 5ms and 24kHz sampling), we must ensure
        # that the length of the waveform is a multiple of the integer-valued
        # up-sampling factor.
        frame_shift_int = int(fs * self.frame_period / 1000)
        T = int(features.shape[0] * frame_shift_int)
        if len(wave) < T:
            if T - len(wave) > int(fs * (self.frame_period * 0.001)):
                print("Length mismatch", T, len(wave), T - len(wave))
                print(f"Unaligned data: {wav_path} and {label_path}")
                raise RuntimeError("Unaligned data")
            else:
                pass
            wave = np.pad(wave, (0, T - len(wave)))
        assert wave.shape[0] >= T
        wave = wave[:T]

        assert np.isfinite(features).all()
        assert np.isfinite(wave).all()
        assert np.isfinite(pf_features).all()

        return features, wave, pf_features


class MelF0AcousticSource(FileDataSource):
    def __init__(
        self,
        utt_list,
        wav_root,
        label_root,
        question_path,
        f0_extractor="harvest",
        f0_floor=150,
        f0_ceil=700,
        frame_period=5,
        vibrato_mode="none",  # diff, sine
        sample_rate=48000,
        d4c_threshold=0.85,
        trajectory_smoothing_f0=True,
        trajectory_smoothing_cutoff_f0=20,
        correct_vuv=False,
        correct_f0=False,
        res_type="scipy",
        fft_size=512,
        win_length=480,
        hop_size=120,
        fmin=30,
        fmax=None,
        eps=1e-10,
        num_mels=80,
    ):
        self.utt_list = utt_list
        self.wav_root = wav_root
        self.label_root = label_root
        self.binary_dict, self.numeric_dict = hts.load_question_set(
            question_path, append_hat_for_LL=False
        )
        self.pitch_idx = get_pitch_index(self.binary_dict, self.numeric_dict)
        self.f0_extractor = f0_extractor
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.frame_period = frame_period
        self.vibrato_mode = vibrato_mode
        self.sample_rate = sample_rate
        self.d4c_threshold = d4c_threshold
        self.trajectory_smoothing_f0 = trajectory_smoothing_f0
        self.trajectory_smoothing_cutoff_f0 = trajectory_smoothing_cutoff_f0
        self.correct_vuv = correct_vuv
        self.correct_f0 = correct_f0
        self.res_type = res_type
        self.fft_size = fft_size
        self.win_length = win_length
        self.hop_size = hop_size
        self.fmin = fmin
        if fmax is None:
            fmax = sample_rate // 2
        self.fmax = fmax
        self.eps = eps
        self.num_mels = num_mels

    def collect_files(self):
        wav_paths = _collect_files(self.wav_root, self.utt_list, ".wav")
        label_paths = _collect_files(self.label_root, self.utt_list, ".lab")
        return wav_paths, label_paths

    def collect_features(self, wav_path, label_path):
        labels = hts.load(label_path)
        hts_frame_shift = int(self.frame_period * 1e4)
        labels.frame_shift = hts_frame_shift
        num_frames = int(labels.num_frames(frame_shift=hts_frame_shift))

        l_features = fe.linguistic_features(
            labels,
            self.binary_dict,
            self.numeric_dict,
            add_frame_features=True,
            subphone_features="coarse_coding",
            frame_shift=hts_frame_shift,
        )

        f0_score = _midi_to_hz(l_features, self.pitch_idx, False)
        notes = l_features[:, self.pitch_idx]
        notes = notes[notes > 0]

        # allow 200 cent upper and 600 cent lower to properly handle F0 estimation of
        # preparation, vibrato and overshoot.
        # NOET: set the minimum f0 to 63.5 Hz (125 - 3*20.5)
        # https://acoustics.jp/qanda/answer/50.html
        # NOTE: sinsy allows 30-150 cent frequency range for vibrato (as of 2010)
        # https://staff.aist.go.jp/m.goto/PAPER/SIGMUS201007oura.pdf
        min_f0 = max(63.5, librosa.midi_to_hz(min(notes) - 6))
        max_f0 = librosa.midi_to_hz(max(notes) + 2)
        assert max_f0 > min_f0

        # Use fixed f0 range for HARVEST if specified
        if self.f0_floor is not None:
            min_f0 = self.f0_floor
        if self.f0_ceil is not None:
            max_f0 = self.f0_ceil

        # Workaround segfault issues of WORLD's CheapTrick
        min_f0 = min(min_f0, 500)

        x, fs = sf.read(wav_path)
        assert np.max(x) <= 1.0
        assert x.dtype == np.float64

        if fs != self.sample_rate:
            x = librosa.resample(
                x, orig_sr=fs, target_sr=self.sample_rate, res_type=self.res_type
            )
            if self.sample_rate > fs:
                # NOTE: add a tiny white noise to compensate missing frequency bands
                init_seed(len(x))
                x = x + np.random.randn(len(x)) * 1e-7
            fs = self.sample_rate

        if self.f0_extractor == "parselmouth":
            import parselmouth

            assert (
                self.f0_floor is not None and self.f0_ceil is not None
            ), "must be set manually"
            harvest_num_frames = int(int(1000 * len(x) / fs) / self.frame_period) + 1
            f0 = (
                parselmouth.Sound(x.astype(np.float64), fs)
                .to_pitch_ac(
                    time_step=self.frame_period * 0.001,
                    voicing_threshold=0.6,
                    very_accurate=False,
                    pitch_floor=min_f0,
                    pitch_ceiling=max_f0,
                )
                .selected_array["frequency"]
            )
            pad = int(np.round((3 / min_f0) / (self.frame_period * 0.001)))
            f0 = np.pad(f0, [[0, pad]], mode="constant")
            if len(f0) > harvest_num_frames:
                f0 = f0[:harvest_num_frames]
            elif len(f0) < harvest_num_frames:
                f0 = np.pad(f0, (0, harvest_num_frames - len(f0)), mode="constant")

            timeaxis = np.arange(harvest_num_frames) * self.frame_period * 0.001
        elif self.f0_extractor == "harvest":
            f0, timeaxis = pyworld.harvest(
                x, fs, frame_period=self.frame_period, f0_floor=min_f0, f0_ceil=max_f0
            )
        elif self.f0_extractor == "dio":
            f0, timeaxis = pyworld.dio(
                x, fs, frame_period=self.frame_period, f0_floor=min_f0, f0_ceil=max_f0
            )
            f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        else:
            raise ValueError(f"unknown f0 extractor: {self.f0_extractor}")

        # Workaround for https://github.com/r9y9/nnsvs/issues/7
        f0 = np.maximum(f0, 0)

        # Correct V/UV (and F0) based on the musical score information
        # treat frames where musical notes are not assigned as unvoiced
        # Use smoothed mask so that we don't mask out overshoot or something
        # that could happen at the start/end of notes
        # 0.5 sec. window (could be tuned for better results)
        if self.correct_vuv:
            win_length = int(0.5 / (self.frame_period * 0.001))
            mask = np.convolve(f0_score, np.ones(win_length) / win_length, "same")
            if len(f0) > len(mask):
                mask = np.pad(mask, (0, len(f0) - len(mask)), "constant")
            elif len(f0) < len(mask):
                mask = mask[: len(f0)]
            f0 = f0 * np.sign(mask)

        aperiodicity = pyworld.d4c(x, f0, timeaxis, fs, threshold=self.d4c_threshold)

        if np.isnan(aperiodicity).any():
            print(wav_path)
            print(min_f0, max_f0, aperiodicity.shape, fs)
            print(np.isnan(aperiodicity).sum())
            print(aperiodicity)
            raise RuntimeError("Aperiodicity has NaN")

        # Apply pitch correction
        # NOTE: we should better to apply pitch corrections
        # or manually adjust UST/musicxml in advance.
        sr_f0 = int(1 / (self.frame_period * 0.001))
        if self.correct_f0:
            f0_smooth = extract_smoothed_f0(f0, sr_f0, cutoff=20)
            ratio = compute_f0_correction_ratio(f0_smooth, f0_score)
            f0 *= ratio

        lf0 = f0[:, np.newaxis].copy()
        nonzero_indices = np.nonzero(lf0)
        lf0[nonzero_indices] = np.log(f0[:, np.newaxis][nonzero_indices])
        if self.f0_extractor == "harvest":
            # https://github.com/mmorise/World/issues/35#issuecomment-306521887
            vuv = (aperiodicity[:, 0] < 0.5).astype(np.float32)[:, None]
        else:
            vuv = (lf0 != 0).astype(np.float32)

        # F0 -> continuous F0
        lf0 = interp1d(lf0, kind="slinear")

        # Smooth continuous F0 to avoid discontinuities
        if self.trajectory_smoothing_f0:
            lf0 = extract_smoothed_continuous_f0(
                lf0, sr_f0, cutoff=self.trajectory_smoothing_cutoff_f0
            )

        # Fill continuous F0s for segments where no notes are assigned & no F0s are detected.
        lf0_score = _midi_to_hz(l_features, self.pitch_idx, True)
        clf0_score = interp1d(lf0_score, kind="slinear")
        mask = np.convolve(lf0_score, np.ones(1), "same")
        if len(f0) > len(mask):
            mask = np.pad(mask, (0, len(f0) - len(mask)), "constant")
            clf0_score = np.pad(clf0_score, (0, len(f0) - len(clf0_score)), "constant")
        elif len(f0) < len(mask):
            mask = mask[: len(f0)]
            clf0_score = clf0_score[: len(f0)]
        ind = (mask + f0.reshape(-1)) <= 0
        lf0[ind, 0] = clf0_score[ind]

        # Mel-spectrogram
        logmel = logmelfilterbank(
            x,
            fs,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            win_length=self.win_length,
            window="hann",
            fmin=self.fmin,
            fmax=self.fmax,
            eps=self.eps,
            num_mels=self.num_mels,
        )

        # Adjust lengths
        logmel = logmel[:num_frames]
        lf0 = lf0[:num_frames]
        vuv = vuv[:num_frames]

        # Concat features
        features = np.hstack((logmel, lf0, vuv)).astype(np.float32)
        pf_features = features

        if len(features) < num_frames:
            print(f"WARN: length mismatch for {wav_path}")
            print(len(features), num_frames)
            return None, None, None

        # Align waveform and features
        wave = x.astype(np.float32)

        # NOTE: since neural vocoders need to perform integer-valued up-sampling
        # (e.g., 120x upsampling with 5ms and 24kHz sampling), we must ensure
        # that the length of the waveform is a multiple of the integer-valued
        # up-sampling factor.
        frame_shift_int = int(fs * self.frame_period / 1000)
        T = int(features.shape[0] * frame_shift_int)
        if len(wave) < T:
            if T - len(wave) > int(fs * (self.frame_period * 0.001)):
                print("Length mismatch", T, len(wave), T - len(wave))
                print(f"Unaligned data: {wav_path} and {label_path}")
                raise RuntimeError("Unaligned data")
            else:
                pass
            wave = np.pad(wave, (0, T - len(wave)))
        assert wave.shape[0] >= T
        wave = wave[:T]

        assert np.isfinite(features).all()
        assert np.isfinite(wave).all()
        assert np.isfinite(pf_features).all()

        return features, wave, pf_features
