"""Predict acoustic features by NNSVS with NEUTRINO-compatible file IO

NOTE: options are not yet fully implemented

NEUTRINO - NEURAL SINGING SYNTHESIZER (Electron v1.2.0-Stable)
            Copyright (c) 2020-2022 STUDIO NEUTRINO All rights reserved.

usage:
    NEUTRINO full.lab timing.lab output.f0 output.mgc output.bap model_directory [option]
    options : description [default]
    -n i        : number of threads (CPU)                 [MAX]
    -k i        : style shift                             [  0]
    -s          : skip timing prediction                  [off]
    -a          : skip acoustic features prediction       [off]
    -p i        : single phrase prediction                [ -1]
    -i filename : trace phrase information                [off]
    -t          : view information                        [off]

"""
import argparse
import sys
from pathlib import Path

import torch
from nnmnkwii.io import hts
from nnsvs.io.hts import full_to_mono
from nnsvs.svs import NEUTRINO


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pretend as if the script is NEUTRINO",
    )
    parser.add_argument("full_lab", type=str, help="Full context labels")
    parser.add_argument("timing_lab", type=str, help="Path of timing labels")
    parser.add_argument("output_f0", type=str, help="Path of output F0")
    parser.add_argument("output_mgc", type=str, help="Path of output MGC")
    parser.add_argument("output_bap", type=str, help="Path of output BAP")
    parser.add_argument("model_dir", type=str, help="model_dir")
    parser.add_argument(
        "-i", "--phraselist", type=str, default=None, help="Path of phraselist"
    ),
    parser.add_argument(
        "-p", "--phrase_num", type=int, default=-1, help="Phrase number"
    ),
    return parser


def main():
    args = get_parser().parse_args(sys.argv[1:])
    model_dir = Path(args.model_dir)
    engine = NEUTRINO(model_dir, device="cuda" if torch.cuda.is_available() else "cpu")

    full_lab = Path(args.full_lab)
    assert full_lab.exists()
    full_labels = hts.load(full_lab)

    timing_lab = Path(args.timing_lab)
    if not timing_lab.exists():
        timing_labels = full_to_mono(engine.predict_timing(full_labels))
        with open(timing_lab, "w") as f:
            f.write(str(timing_labels))
    else:
        timing_labels = hts.load(timing_lab)

    if args.phraselist is not None:
        phraselist = Path(args.phraselist)
        if not phraselist.exists():
            phraselist_str = engine.get_phraselist(full_labels, timing_labels)
            with open(phraselist, "w") as f:
                f.write(phraselist_str)
        num_phrases = engine.get_num_phrases(full_labels)
        if args.phrase_num < 0 or args.phrase_num >= num_phrases:
            raise ValueError(f"phrase_num must be in [0, {num_phrases - 1}]")

    f0, mgc, bap = engine.predict_acoustic(
        full_labels, timing_labels, phrase_num=args.phrase_num
    )

    # Save to file
    f0.tofile(args.output_f0)
    mgc.tofile(args.output_mgc)
    bap.tofile(args.output_bap)


if __name__ == "__main__":
    main()
