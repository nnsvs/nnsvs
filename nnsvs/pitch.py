"""This module provides functionality for pitch analysis.

References:

Nakano et al, "An Automatic Singing Skill Evaluation Method
for Unknown Melodies Using Pitch Interval Accuracy and Vibrato Features"
Proc. Interspeech 2006.

山田 et al, "HMM に基づく歌声合成のためのビブラートモデル化"
IPSJ SIG Tech. Report 2009.

Note that vibrato extraction method in this module is exerimental.
Because details of the vibrato extraction method are not described
in the above papers and not trivial to implement (in my opinion),
my implementation may not work well compared to the original author's one.
Also note that there are a lot of tunable parameters (threshold,
window size, min/max extent, cut-off frequency, etc.).
If you want to get maximum performance, you might want to tune these
parameters with your dataset.
I tested this code with kiritan_singing and nit-song070 database.
"""
import librosa
import numpy as np
import torch
from nnsvs.dsp import lowpass_filter
from scipy.signal import argrelmax, argrelmin

_c4_hz = 440 * 2 ** (3 / 12 - 1)
_c4_cent = 4800


def hz_to_cent_based_c4(hz):
    """Convert Hz to cent based on C4

    Args:
        hz (np.ndarray): array of Hz

    Returns:
        np.ndarray: array of cent
    """
    out = hz.copy()
    nonzero_indices = np.where(hz > 0)[0]
    out[nonzero_indices] = (
        1200 * np.log(hz[nonzero_indices] / _c4_hz) / np.log(2) + _c4_cent
    )
    return out


def cent_to_hz_based_c4(cent):
    """Convert cent to Hz based on C4

    Args:
        cent (np.ndarray): array of cent

    Returns:
        np.ndarray: array of Hz
    """
    out = cent.copy()
    nonzero_indices = np.where(cent > 0)[0]
    out[nonzero_indices] = (
        np.exp((cent[nonzero_indices] - _c4_cent) * np.log(2) / 1200) * _c4_hz
    )
    return out


def nonzero_segments(f0):
    """Find nonzero segments

    Args:
        f0 (np.ndarray): array of f0

    Returns:
        list: list of (start, end)
    """
    vuv = f0 > 0
    started = False
    s, e = 0, 0
    segments = []
    for idx in range(len(f0)):
        if vuv[idx] > 0 and not started:
            started = True
            s = idx
        elif started and (vuv[idx] <= 0):
            e = idx
            started = False
            segments.append((s, e))
        else:
            pass

    if started and vuv[-1] > 0:
        segments.append((s, len(vuv) - 1))

    return segments


def note_segments(lf0_score_denorm):
    """Compute note segments (start and end indices) from log-F0

    Note that unvoiced frames must be set to 0 in advance.

    Args:
        lf0_score_denorm (Tensor): (B, T)

    Returns:
        list: list of note (start, end) indices
    """
    segments = []
    for s, e in nonzero_segments(lf0_score_denorm):
        out = torch.sign(torch.abs(torch.diff(lf0_score_denorm[s : e + 1])))
        transitions = torch.where(out > 0)[0]
        note_start, note_end = s, -1
        for pos in transitions:
            note_end = int(s + pos)
            segments.append((note_start, note_end))
            note_start = note_end + 1

        # Handle last note
        while (
            note_start < len(lf0_score_denorm) - 1 and lf0_score_denorm[note_start] <= 0
        ):
            note_start += 1
        note_end = note_start + 1
        while note_end < len(lf0_score_denorm) - 1 and lf0_score_denorm[note_end] > 0:
            note_end += 1

        if note_end != note_start + 1:
            segments.append((note_start, note_end))

    return segments


def compute_f0_correction_ratio(
    f0,
    f0_score,
    edges_to_be_excluded=50,
    out_of_tune_threshold=200,
    correction_threshold=100,
):
    """Compute f0 correction ratio

    Args:
        f0 (np.ndarray): array of f0
        f0_score (np.ndarray): array of f0 score

    Returns:
        float: correction ratio to multiplied to F0 (i.e. f0 * ratio)
    """
    segments = note_segments(torch.from_numpy(f0_score))

    center_f0s = []
    center_score_f0s = []
    # edges_to_be_excluded = 50  # 0.25 sec for excluding overshoot/preparation
    for s, e in segments:
        L = e - s
        if L > edges_to_be_excluded * 2:
            center_f0s.append(f0[s + edges_to_be_excluded : e - edges_to_be_excluded])
            center_score_f0s.append(
                f0_score[s + edges_to_be_excluded : e - edges_to_be_excluded]
            )
    center_f0s = np.concatenate(center_f0s)
    center_score_f0s = np.concatenate(center_score_f0s)

    # Compute pitch ratio to be multiplied
    nonzero_indices = (center_f0s > 0) & (center_score_f0s > 0)
    ratio = center_score_f0s[nonzero_indices] / center_f0s[nonzero_indices]

    # Exclude too out-of-tune frames (over 2 semitone)
    up_threshold = np.exp(out_of_tune_threshold * np.log(2) / 1200)
    low_threshold = np.exp(-out_of_tune_threshold * np.log(2) / 1200)

    ratio = ratio[(ratio < up_threshold) & (ratio > low_threshold)]

    global_offset = ratio.mean()

    # Avoid corrections over semi-tone
    # If more than semi-tone pitch correction is needed, it is better to correct
    # data by hand or fix musicxml or UST instead.
    up_threshold = np.exp(correction_threshold * np.log(2) / 1200)
    low_threshold = np.exp(-correction_threshold * np.log(2) / 1200)

    if global_offset > up_threshold or global_offset < low_threshold:
        print(
            f"""warn: more than 1 semitone pitch correction is needed.
global_offset: {global_offset} cent.
It is likely that manual pitch corrections are preferable."""
        )
    global_offset = np.clip(global_offset, low_threshold, up_threshold)

    return global_offset


def extract_vibrato_parameters_impl(pitch_seg, sr):
    """Extract vibrato parameters for a single pitch segment

    Nakano et al, "An Automatic Singing Skill Evaluation Method
    for Unknown Melodies Using Pitch Interval Accuracy and Vibrato Features"
    Proc. Interspeech 2006.

    山田 et al, "HMM に基づく歌声合成のためのビブラートモデル化"
    IPSJ SIG Tech. Report 2009.

    Args:
        pitch_seg (np.ndarray): array of pitch
        sr (int): sampling rate

    Returns:
        tuple: (R, E, m_a, m_f)
    """
    peak_high_pos = argrelmax(pitch_seg)[0]
    peak_low_pos = argrelmin(pitch_seg)[0]

    m_a = np.zeros(len(pitch_seg))
    m_f = np.zeros(len(pitch_seg))

    if len(peak_high_pos) != len(peak_low_pos) + 1:
        print("Warning! Probably a bug...T.T")
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
    """Compute extent of a pitch segment

    Args:
        pitch_seg (np.ndarray): array of pitch

    Returns:
        np.ndarray: array of extent
    """
    peak_high_pos = argrelmax(pitch_seg)[0]
    peak_low_pos = argrelmin(pitch_seg)[0]

    if len(peak_high_pos) == 1 or len(peak_low_pos) == 1:
        return np.array([-1])

    if len(peak_high_pos) < len(peak_low_pos):
        peak_low_pos = peak_low_pos[:-2]
    elif len(peak_high_pos) == len(peak_low_pos):
        peak_low_pos = peak_low_pos[:-1]

    peak_high_pitch = pitch_seg[peak_high_pos]
    peak_low_pitch = pitch_seg[peak_low_pos]

    peak_high_pos_diff = np.diff(peak_high_pos)
    peak_low_pos_diff = np.diff(peak_low_pos)

    # TODO: would probably be a bug...
    if len(peak_high_pitch) != len(peak_low_pitch) + 1:
        return np.array([-1])

    E = np.zeros(len(peak_high_pos_diff) + len(peak_low_pos_diff))
    E[0::2] = (peak_high_pitch[1:] + peak_high_pitch[:-1]) / 2 - peak_low_pitch
    E[1::2] = peak_high_pitch[1:-1] - (peak_low_pitch[1:] + peak_low_pitch[:-1]) / 2

    return E


def extract_smoothed_f0(f0, sr, cutoff=8):
    """Extract smoothed f0 by low-pass filtering

    Note that the low-pass filter is only applied to voiced segments.

    Args:
        f0 (np.ndarray): array of f0
        sr (int): sampling rate
        cutoff (float): cutoff frequency

    Returns:
        np.ndarray: array of smoothed f0
    """
    segments = nonzero_segments(f0)

    f0_smooth = f0.copy()
    for s, e in segments:
        f0_smooth[s:e] = lowpass_filter(f0[s:e], sr, cutoff=cutoff)

    return f0_smooth


def extract_smoothed_continuous_f0(f0, sr, cutoff=20):
    """Extract smoothed continuous f0 by low-pass filtering

    Note that the input must be continuous F0 or log-F0.

    Args:
        f0 (np.ndarray): array of continuous f0
        sr (int): sampling rate
        cutoff (float): initial cutoff frequency

    Returns:
        np.ndarray: array of smoothed continuous f0
    """
    is_2d = len(f0.shape) == 2
    f0 = f0.reshape(-1) if is_2d else f0

    # Ref: https://bit.ly/3SOePFw
    f0_smooth = lowpass_filter(f0, sr, cutoff=cutoff)

    # Fallback case: shound't happen I believe
    # NOTE: hard-coded for now
    next_cutoff = 50
    while (f0_smooth < 0).any():
        f0_smooth = lowpass_filter(f0, sr, cutoff=next_cutoff)
        next_cutoff *= 2

    f0_smooth = f0_smooth.reshape(len(f0), 1) if is_2d else f0_smooth
    return f0_smooth


def extract_vibrato_likelihood(
    f0_smooth, sr, win_length=32, n_fft=128, min_freq=3, max_freq=8
):
    """Extract vibrato likelihood

    Args:
        f0_smooth (np.ndarray): array of smoothed f0
        sr (int): sampling rate
        win_length (int): window length
        n_fft (int): FFT size
        min_freq (float): minimum frequency of the vibrato
        max_freq (float): maximum frequency of the vibrato

    Returns:
        np.ndarray: array of vibrato likelihood
    """
    # STFT on 1st order diffference of F0
    X = np.abs(
        librosa.stft(
            np.diff(f0_smooth),
            hop_length=1,
            win_length=win_length,
            n_fft=n_fft,
            window="hann",
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
    """Interpolate a sequence of vibrato parameter by linear interpolation

    Args:
        m_f (np.ndarray): array of vibrato parameter

    Returns:
        np.ndarray: array of vibrato parameter
    """
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
    smooth_params=False,
    smooth_width=15,
    clip_extent=True,
):
    """Extract vibrato parameters

    Args:
        pitch (np.ndarray): array of pitch (smoothed f0)
        vibrato_likelihood (np.ndarray): array of vibrato likelihood
        sr (int): sampling rate
        threshold (float): threshold of vibrato likelihood
        min_cross_count (int): minimum number of cross points
        min_extent (int): minimum extent of vibrato (cent)
        max_extent (int): maximum extent of vibrato (cent)
        interp_params (bool): whether to interpolate vibrato parameters
        smooth_params (bool): whether to smooth vibrato parameters
        smooth_width (int): width of smoothing window
        clip_extent (bool): whether to clip extent

    Returns:
        tuple: tuple of vibrato parameters
    """
    T = len(vibrato_likelihood)

    vibrato_flags = np.zeros(T, dtype=int)
    m_a = np.zeros(T)
    m_f = np.zeros(T)

    peak_high_pos = argrelmax(pitch)[0]
    peak_low_pos = argrelmin(pitch)[0]

    # iterate over every peak position
    peak_high_idx = 0
    while peak_high_idx < len(peak_high_pos):
        peak_frame_idx = peak_high_pos[peak_high_idx]

        found = False
        if vibrato_likelihood[peak_frame_idx] > threshold:
            # Initial positions for vibrato section
            start_index = peak_frame_idx
            peaks = peak_low_pos[peak_low_pos > peak_frame_idx]
            if len(peaks) > 0:
                end_index = peaks[0]
            else:
                peak_high_idx += 1
                continue
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
                if m_a_seg is None:
                    found = False
                    break
                found = True
                vibrato_flags[start_index:end_index] = 1

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

                assert next_start_peak_high_idx > peak_high_idx
                peak_high_idx = next_start_peak_high_idx

        if not found:
            peak_high_idx += 1

    return vibrato_flags, m_a, m_f


def gen_sine_vibrato(f0, sr, m_a, m_f, scale=1.0):
    """Generate F0 with sine-based vibrato

    Args:
        f0 (ndarray): fundamental frequency
        sr (int): sampling rate
        m_a (ndarray): amplitude of vibrato
        m_f (ndarray): frequency of vibrato
        scale (float): scale factor

    Returns:
        ndarray: F0 with sine-based vibrato
    """
    f0_gen = f0.copy()

    voiced_end_indices = np.asarray([e for _, e in nonzero_segments(f0)])

    for s, e in nonzero_segments(m_a):
        # limit vibrato rate to [3, 8] Hz
        m_f_seg = np.clip(m_f[s:e], 3, 8)
        # limit vibrato extent to [30, 150] cent
        m_a_seg = np.clip(m_a[s:e], 30, 150)

        cent = scale * m_a_seg * np.sin(2 * np.pi / sr * m_f_seg * np.arange(0, e - s))
        new_f0 = f0[s:e] * np.exp(cent * np.log(2) / 1200)
        f0_gen[s:e] = new_f0

        # NOTE: this is a hack to avoid discontinuity at the end of vibrato
        voiced_ends_next_to_vibrato = voiced_end_indices[voiced_end_indices > e]
        if len(voiced_ends_next_to_vibrato) > 0:
            voiced_end = voiced_ends_next_to_vibrato[0]
            f0_gen[s:voiced_end] = lowpass_filter(f0_gen[s:voiced_end], sr, cutoff=12)

    return f0_gen
