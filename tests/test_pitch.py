from os.path import dirname, join

import numpy as np
import pytest
import pyworld
from nnsvs.pitch import (
    extract_smoothed_f0,
    extract_vibrato_likelihood,
    extract_vibrato_parameters,
    gen_sine_vibrato,
    hz_to_cent_based_c4,
)
from scipy.io import wavfile

__test_wav_file = join(dirname(__file__), "data", "nitech_jp_song070_f001_004.wav")


@pytest.mark.parametrize("frame_period", [5, 10])
def test_sine_vibrato(frame_period):
    sr, x = wavfile.read(__test_wav_file)

    if frame_period == 10:
        win_length = 32
        n_fft = 128
        threshold = 0.12
    elif frame_period == 5:
        win_length = 64
        n_fft = 256
        threshold = 0.12

    frame_shift = int(frame_period * 0.001 * sr)
    sr_f0 = int(sr / frame_shift)

    f0, timeaxis = pyworld.dio(x.astype(np.float64), sr, frame_period=frame_period)
    f0 = pyworld.stonemask(x.astype(np.float64), f0, timeaxis, sr)

    f0_smooth = extract_smoothed_f0(f0, sr_f0, cutoff=8)
    f0_smooth_cent = hz_to_cent_based_c4(f0_smooth)
    vibrato_likelihood = extract_vibrato_likelihood(
        f0_smooth_cent, sr_f0, win_length=win_length, n_fft=n_fft
    )
    vib_flags, m_a, m_f = extract_vibrato_parameters(
        f0_smooth_cent, vibrato_likelihood, sr_f0, threshold=threshold
    )
    # at least one vibrato section should be detected
    assert vib_flags.sum() > 0

    assert m_a.shape == (len(f0),)
    assert m_f.shape == (len(f0),)
    assert vib_flags.shape == (len(f0),)

    # Generate F0 with the vibrato parameters (m_a, m_f)
    f0_no_vib = extract_smoothed_f0(f0, sr_f0, cutoff=3)
    f0_with_vib = gen_sine_vibrato(f0_no_vib, sr_f0, m_a, m_f)
    assert f0_no_vib.shape == f0_with_vib.shape


def test_diff_vibrato():
    sr, x = wavfile.read(__test_wav_file)

    frame_period = 5
    frame_shift = int(frame_period * 0.001 * sr)
    sr_f0 = int(sr / frame_shift)

    f0, timeaxis = pyworld.dio(x.astype(np.float64), sr, frame_period=frame_period)
    f0 = pyworld.stonemask(x.astype(np.float64), f0, timeaxis, sr)

    f0_smooth = extract_smoothed_f0(f0, sr_f0, cutoff=8)
    vib = f0 - f0_smooth

    assert vib.shape == (len(f0),)
