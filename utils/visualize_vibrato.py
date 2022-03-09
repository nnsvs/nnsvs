import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyworld
from nnmnkwii.preprocessing.f0 import interp1d
from nnsvs.pitch import (
    extract_smoothed_f0,
    extract_vibrato_likelihood,
    extract_vibrato_parameters,
    gen_sine_vibrato,
    hz_to_cent_based_c4,
    lowpass_filter,
    nonzero_segments,
)
from scipy.io import wavfile


def get_parser():
    parser = argparse.ArgumentParser(
        description="Visualize vibrato",
    )
    parser.add_argument("input_file", type=str, help="Input wav file")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    sr, x = wavfile.read(args.input_file)

    print(sr, x.dtype)
    frame_perioid = 5

    if frame_perioid == 10:
        win_length = 32
        n_fft = 128
        threshold = 0.12
    elif frame_perioid == 5:
        win_length = 64
        n_fft = 256
        threshold = 0.12

    frame_shift = int(frame_perioid * 0.001 * sr)
    sr_f0 = int(sr / frame_shift)

    f0, timeaxis = pyworld.dio(x.astype(np.float64), sr, frame_period=frame_perioid)
    f0 = pyworld.stonemask(x.astype(np.float64), f0, timeaxis, sr)

    f0_smooth = extract_smoothed_f0(f0, sr_f0, cutoff=8)
    f0_smooth_cent = hz_to_cent_based_c4(f0_smooth)
    vibrato_likelihood = extract_vibrato_likelihood(
        f0_smooth_cent, sr_f0, win_length=win_length, n_fft=n_fft
    )
    results, m_a, m_f = extract_vibrato_parameters(
        f0_smooth_cent, vibrato_likelihood, sr_f0, threshold=threshold
    )

    fig, ax = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    ax[0].plot(timeaxis, f0, label="Original F0")
    ax[0].plot(timeaxis, f0_smooth, label="Smoothed F0")
    ax[0].plot(timeaxis, results * 15, "*", label="Vibrato sections")
    ax[0].set_ylim(12)
    ax[0].set_ylabel("Frequency [cent]")
    ax[0].legend()
    ax[0].set_title("F0")
    ax[1].plot(timeaxis, interp1d(m_a))
    ax[1].set_title("m_a(t)")
    ax[1].set_ylabel("Frequency [cent]")
    ax[2].plot(timeaxis, interp1d(m_f))
    ax[2].set_title("m_f(t)")
    ax[2].set_ylabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()

    # Let's reconstruct vibrato
    f0_no_vib = f0.copy()
    segments = nonzero_segments(f0)
    for s, e in segments:
        f0_no_vib[s:e] = lowpass_filter(f0[s:e], sr_f0, cutoff=1)
    f0_gen = gen_sine_vibrato(f0_no_vib, sr_f0, m_a, m_f)

    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    ax[0].plot(timeaxis, f0, label="Original F0")
    ax[0].plot(timeaxis, f0_smooth, label="Smoothed F0")
    ax[0].plot(timeaxis, results * 15, "*", label="Vibrato sections")
    ax[0].set_ylim(12)
    ax[0].set_ylabel("Frequency [cent]")
    ax[0].legend()
    ax[0].set_title("F0")

    ax[1].plot(timeaxis, f0_no_vib, label="Pseudo smoothed F0")
    ax[1].plot(timeaxis, f0_gen, label="Generated F0")
    ax[1].legend()
    plt.tight_layout()
    plt.show()
