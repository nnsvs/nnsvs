"""Predict waveform by neural vocoders with NEUTRINO-compatible file IO

NOTE: options are not yet fully implemented

NSF - Neural Source Filter (v1.2.0-Stable)
        Copyright (c) 2020-2022 STUDIO NEUTRINO All rights reserved.

usage:
    NSF input.f0 input.mgc input.bap model_name output_wav [option]
    options : description [default]
    -s i            : sampling rate (kHz)             [   48]
    -n i            : number of parallel              [  MAX]
    -p i            : number of parallel in session   [    1]
    -l file name    : multi phrase prediction         [ none]
    -g              : use gpu                         [  off]
    -i i            : gpu id                          [    0]
    -t              : view information                [  off]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyworld
import torch
from nnsvs.svs import NEUTRINO
from scipy.io import wavfile


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pretend as if the script is NSF of NEUTRINO",
    )
    parser.add_argument("input_f0", type=str, help="Path of input F0")
    parser.add_argument("input_mgc", type=str, help="Path of input MGC")
    parser.add_argument("input_bap", type=str, help="Path of input BAP")
    parser.add_argument("model_dir", type=str, help="model_dir")
    parser.add_argument("output_wav", type=str, help="Path of output wav")
    return parser


def main():
    args = get_parser().parse_args(sys.argv[1:])
    model_dir = Path(args.model_dir)
    engine = NEUTRINO(model_dir, device="cuda" if torch.cuda.is_available() else "cpu")

    f0 = np.fromfile(args.input_f0, dtype=np.float64).reshape(-1, 1)
    mgc = np.fromfile(args.input_mgc, dtype=np.float64).reshape(-1, 60)
    bap = np.fromfile(args.input_bap, dtype=np.float64).reshape(
        -1, pyworld.get_num_aperiodicities(engine.sample_rate)
    )

    # NOTE: `auto` will run uSFGAN or PWG if a trained one is in the model_dir
    # and fallback to WORLD it doesn't exist.
    wav = engine.predict_waveform(f0, mgc, bap, vocoder_type="auto", dtype=np.int16)

    wavfile.write(args.output_wav, engine.sample_rate, wav)


if __name__ == "__main__":
    main()
