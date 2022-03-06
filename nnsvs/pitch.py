import librosa
import numpy as np
from scipy import signal
from scipy.signal import argrelmax, argrelmin

_c4_hz = 440 * 2 ** (3 / 12 - 1)
_c4_cent = 4800


def hz_to_cent_based_c4(hz):
    out = hz.copy()
    nonzero_indices = np.where(hz > 0)[0]
    out[nonzero_indices] = (
        1200 * np.log(hz[nonzero_indices] / _c4_hz) / np.log(2) + _c4_cent
    )
    return out


def cent_to_hz_based_c4(cent):
    out = cent.copy()
    nonzero_indices = np.where(cent > 0)[0]
    out[nonzero_indices] = (
        np.exp((cent[nonzero_indices] - _c4_cent) * np.log(2) / 1200) * _c4_hz
    )
    return out


def lowpass_filter(x, fs, cutoff=5):
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    Wn = [norm_cutoff]

    b, a = signal.butter(5, Wn, "lowpass")
    if len(x) <= len(b) * 3:
        # NOTE: input signal is too short
        return x

    # NOTE: use zero-phase filter
    y = signal.filtfilt(b, a, x)

    return y


def get_voiced_segments(f0):
    vuv = f0 > 0
    started = False
    s, e = 0, 0
    segments = []
    for idx in range(len(f0)):
        if vuv[idx] and not started:
            started = True
            s = idx
        elif started and not vuv[idx]:
            e = idx
            started = False
            segments.append((s, e))
        else:
            pass
    return segments


def extract_vibrato_parameters_impl(pitch_seg, sr):
    peak_high_pos = argrelmax(pitch_seg)[0]
    peak_low_pos = argrelmin(pitch_seg)[0]

    m_a = np.zeros(len(pitch_seg))
    m_f = np.zeros(len(pitch_seg))

    if len(peak_high_pos) != len(peak_low_pos) + 1:
        print("warning!")
        print(peak_high_pos, peak_low_pos)
        return None, None, None, None

    peak_high_pos_diff = np.diff(peak_high_pos)
    peak_low_pos_diff = np.diff(peak_low_pos)

    R = np.zeros(len(peak_high_pos_diff) + len(peak_low_pos_diff))
    R[0::2] = peak_high_pos_diff
    R[1::2] = peak_low_pos_diff

    m_f_ind = np.zeros(len(R), dtype=int)
    m_f_ind[0::2] = peak_high_pos[:-1]
    m_f_ind[1::2] = peak_low_pos[:-1]
    m_f[m_f_ind] = (1 / R) * sr

    peak_high_pitch = pitch_seg[peak_high_pos]
    peak_low_pitch = pitch_seg[peak_low_pos]

    E = np.zeros(len(R))
    E[0::2] = (peak_high_pitch[1:] + peak_high_pitch[:-1]) / 2 - peak_low_pitch
    E[1::2] = peak_high_pitch[1:-1] - (peak_low_pitch[1:] + peak_low_pitch[:-1]) / 2

    m_a_ind = np.zeros(len(R), dtype=int)
    m_a_ind[0::2] = peak_low_pos
    m_a_ind[1::2] = peak_high_pos[1:-1]
    m_a[m_a_ind] = 0.5 * E

    rate = 1 / R.mean() * sr
    extent = 0.5 * E.mean()
    print(f"Rate: {rate}, Extent: {extent}")

    return R, E, m_a, m_f


def compute_extent(pitch_seg):
    peak_high_pos = argrelmax(pitch_seg)[0]
    peak_low_pos = argrelmin(pitch_seg)[0]

    if len(peak_high_pos) < len(peak_low_pos):
        peak_low_pos = peak_low_pos[:-2]
    elif len(peak_high_pos) == len(peak_low_pos):
        peak_low_pos = peak_low_pos[:-1]

    peak_high_pitch = pitch_seg[peak_high_pos]
    peak_low_pitch = pitch_seg[peak_low_pos]

    # NOTE: don't have enough peaks to compute parameters
    if len(peak_low_pitch) <= 2:
        return np.array([-1])

    peak_high_pos_diff = np.diff(peak_high_pos)
    peak_low_pos_diff = np.diff(peak_low_pos)

    E = np.zeros(len(peak_high_pos_diff) + len(peak_low_pos_diff))
    E[0::2] = (peak_high_pitch[1:] + peak_high_pitch[:-1]) / 2 - peak_low_pitch
    E[1::2] = peak_high_pitch[1:-1] - (peak_low_pitch[1:] + peak_low_pitch[:-1]) / 2

    return E


def extract_smoothed_f0(f0, sr, cutoff=8):
    segments = get_voiced_segments(f0)

    f0_smooth = f0.copy()
    for s, e in segments:
        f0_smooth[s:e] = lowpass_filter(f0[s:e], sr, cutoff=cutoff)

    return f0_smooth


def extract_vibrato_likelihood(
    f0_smooth, sr, win_length=32, n_fft=128, min_freq=3, max_freq=8
):
    # STFT on 1st order diffference of F0
    X = np.abs(
        librosa.stft(
            np.diff(f0_smooth),
            hop_length=1,
            win_length=win_length,
            n_fft=n_fft,
            window="hanning",
        )
    )
    X_norm = X / (X.sum(0) + 1e-7)

    freq_per_bin = sr / n_fft
    min_freq_bin = int(min_freq / freq_per_bin)
    max_freq_bin = int(max_freq / freq_per_bin)

    # Compute vibrato likelhiood
    St = np.abs(np.diff(X_norm, axis=0)).sum(0)
    Ft = X_norm[min_freq_bin:max_freq_bin, :].sum(0)
    vibrato_likelihood = St * Ft

    return vibrato_likelihood


def interp_vibrato(m_f):
    nonzero_indices = np.where(m_f > 0)[0]
    nonzero_indices = [0] + list(nonzero_indices) + [len(m_f) - 1]
    out = np.interp(np.arange(len(m_f)), nonzero_indices, m_f[nonzero_indices])
    return out


def extract_vibrato_parameters(
    pitch,
    vibrato_likelihood,
    sr=100,
    threshold=0.12,
    min_cross_count=5,
    min_extent=30,
    max_extent=150,
    interp_params=True,
    smooth_params=True,
    smooth_width=15,
    clip_extent=True,
):
    T = len(vibrato_likelihood)

    results = np.zeros(T, dtype=int)
    m_a = np.zeros(T)
    m_f = np.zeros(T)

    peak_high_pos = argrelmax(pitch)[0]
    peak_low_pos = argrelmin(pitch)[0]

    num_vibrato = 0
    # iterate over every peak position
    peak_high_idx = 0
    while peak_high_idx < len(peak_high_pos):
        peak_frame_idx = peak_high_pos[peak_high_idx]

        found = False
        if vibrato_likelihood[peak_frame_idx] > threshold:
            # Initial positions for vibrato section
            start_index = peak_frame_idx
            end_index = peak_low_pos[peak_low_pos > peak_frame_idx][0]
            next_start_peak_high_idx = -1

            # Find a peak position that is close to the next non-speech segment
            # assuming that there's a non-speech segment right after vibrato
            # NOTE: we may want to remove this constraint
            peak_high_pos_rest = peak_high_pos[peak_high_pos > peak_frame_idx]
            for frame_idx in range(end_index, T):
                if pitch[frame_idx] <= 0:
                    peaks = peak_high_pos_rest[peak_high_pos_rest < frame_idx]
                    if len(peaks) > 0:
                        end_index = peaks[-1]
                        next_start_peak_high_idx = (
                            len(peak_high_pos[peak_high_pos < end_index]) + 1
                        )
                    break

            # Set the search width (backward)
            search_width_backward = 0
            for frame_idx in range(start_index, 0, -1):
                if pitch[frame_idx] <= 0:
                    peaks_backward = peak_high_pos[
                        (peak_high_pos < peak_frame_idx) & (peak_high_pos > frame_idx)
                    ]
                    if len(peaks_backward) > 0:
                        backward = peaks_backward[0]
                        search_width_backward = len(
                            peak_high_pos[
                                (peak_high_pos > backward)
                                & (peak_high_pos <= peak_frame_idx)
                            ]
                        )
                    break

            # Find a peak position that satisfies the following vibrato constraints
            # 1) more than 5 times crossing
            # 2) 30 ~ 150 cent oscillation
            estimate_start_index = start_index
            rate = 0
            for peak_idx in range(
                max(peak_high_idx - search_width_backward, 0), peak_high_idx
            ):
                if peak_high_pos[peak_idx] >= T:
                    break
                f0_seg = pitch[peak_high_pos[peak_idx] : end_index]

                # Check if the segment satisfies vibrato constraints
                m = f0_seg.mean()
                cross_count = len(np.where(np.diff(np.sign(f0_seg - m)))[0])

                # Find the start_index so that the vibrato section has more than 5 crossing
                E = compute_extent(f0_seg)
                extent = 0.5 * E.mean()
                having_large_deviation = ((0.5 * E) > max_extent * 2).any()
                if (
                    cross_count >= min_cross_count
                    and cross_count >= rate
                    and extent >= min_extent
                    and extent <= max_extent
                    and not having_large_deviation
                    and (E > 0).all()
                ):
                    rate = cross_count
                    estimate_start_index = peak_high_pos[peak_idx]

            start_index = estimate_start_index

            if rate >= min_cross_count:
                R, E, m_a_seg, m_f_seg = extract_vibrato_parameters_impl(
                    pitch[start_index - 1 : end_index + 2], sr
                )
                # TODO
                if m_a_seg is None:
                    found = False
                    break
                found = True
                results[start_index:end_index] = 1
                num_vibrato += 1

                if interp_params:
                    m_a_seg = interp_vibrato(m_a_seg)
                    m_f_seg = np.clip(interp_vibrato(m_f_seg), 3, 8)
                if smooth_params:
                    m_a_seg = np.convolve(
                        m_a_seg, np.ones(smooth_width) / smooth_width, mode="same"
                    )
                    m_f_seg = np.convolve(
                        m_f_seg, np.ones(smooth_width) / smooth_width, mode="same"
                    )

                if clip_extent:
                    m_a_seg = np.clip(m_a_seg, min_extent, max_extent)
                m_a[start_index:end_index] = m_a_seg[1:-2]
                m_f[start_index:end_index] = m_f_seg[1:-2]

                assert next_start_peak_high_idx > 0
                peak_high_idx = next_start_peak_high_idx

        if not found:
            peak_high_idx += 1

    return results, m_a, m_f


def gen_sine_vibrato(f0, sr, m_a, m_f):
    f0_gen = f0.copy()

    for s, e in get_voiced_segments(m_a):
        cent = m_a[s:e] * np.sin(2 * np.pi / sr * m_f[s:e] * np.arange(0, e - s))
        new_f0 = f0[s:e] * np.exp(cent * np.log(2) / 1200)
        f0_gen[s:e] = new_f0

    return f0_gen
