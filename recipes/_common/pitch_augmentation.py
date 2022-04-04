"""Pitch augmentation for lab and wav files
"""
import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from os.path import basename, join

import librosa
import numpy as np
import pysptk
import soundfile as sf
from nnmnkwii.frontend import NOTE_MAPPING
from nnmnkwii.io import hts
from pysptk.synthesis import AllPoleDF, AllZeroDF, Synthesizer
from scipy.io import wavfile
from tqdm.auto import tqdm

MIDI_MAPPING = {v: k for k, v in NOTE_MAPPING.items()}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pitch data augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("in_dir", type=str, help="Input directory")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument(
        "shift_in_cent", default=100, type=int, help="Pitch shift in cent"
    )
    parser.add_argument("--n_jobs", help="n_jobs")
    parser.add_argument(
        "--filter_augmented_files",
        action="store_true",
        help="filter out already augmented files",
    )
    return parser


def pitch_shift_on_lpc_residual(
    wav,
    sr,
    shift_in_cent,
    frame_length=4096,
    hop_length=240,
    mgc_order=59,
):
    assert wav.dtype == np.int16
    frames = (
        librosa.util.frame(wav, frame_length=frame_length, hop_length=hop_length)
        .astype(np.float64)
        .T
    )
    frames *= pysptk.blackman(frame_length)
    alpha = pysptk.util.mcepalpha(sr)

    mgc = pysptk.mcep(frames, mgc_order, alpha, eps=1e-5, etype=1)
    c = pysptk.freqt(mgc, mgc_order, -alpha)

    lpc = pysptk.levdur(pysptk.c2acr(c, mgc_order, frame_length))
    # remove gain
    lpc[:, 0] = 0

    # Compute LPC residual
    synth = Synthesizer(AllZeroDF(mgc_order), hop_length)
    wav_lpc = synth.synthesis(wav.astype(np.float64), -lpc)
    residual = wav - wav_lpc

    # Pitch-shift on LPC residual
    residual_shifted = librosa.effects.pitch_shift(
        residual, sr, shift_in_cent, bins_per_octave=1200
    )

    # Filtering by LPC
    synth = Synthesizer(AllPoleDF(mgc_order), hop_length)
    wav_shifted = synth.synthesis(residual_shifted, lpc)

    return wav_shifted.astype(np.int16)


def process_wav(wav_file, out_dir, shift_in_cent):
    sr, wav = wavfile.read(wav_file)
    assert wav.dtype == np.int16

    y = pitch_shift_on_lpc_residual(wav, sr, shift_in_cent)

    postfix = str(shift_in_cent).replace("-", "minus") + "cent_aug"

    out_file = join(out_dir, basename(wav_file).replace(".wav", f"_{postfix}.wav"))
    sf.write(out_file, y, sr)


def process_lab(lab_files, out_dir, shift_in_cent):
    shift_in_note = args.shift_in_cent // 100

    for lab_file in tqdm(lab_files):
        labels = hts.load(lab_file)
        name = basename(lab_file)
        new_contexts = []
        for label in labels:
            context = label[-1]

            for pre, post in [("/D:", "!"), ("/E:", "]"), ("/F:", "#")]:
                match = re.search(f"{pre}([A-Z][b]?[0-9]+){post}", context)
                # if not "xx"
                if match is not None:
                    assert len(match.groups()) == 1
                    note = match.group(0)[3:-1]
                    note_index = NOTE_MAPPING[note]
                    note_shifted = MIDI_MAPPING[note_index + shift_in_note]
                    context = context.replace(
                        match.group(0), f"{pre}{note_shifted}{post}", 1
                    )
            new_contexts.append(context)

        labels.contexts = new_contexts
        postfix = str(shift_in_cent).replace("-", "minus") + "cent_aug"
        dst_lab_file = join(out_dir, name.replace(".lab", f"_{postfix}.lab"))
        with open(dst_lab_file, "w") as of:
            of.write(str(labels))


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    assert args.shift_in_cent % 100 == 0

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    wav_files = sorted(glob(join(args.in_dir, "*.wav")))
    if args.filter_augmented_files:
        wav_files = list(filter(lambda x: not x.endswith("aug.wav"), wav_files))
    if len(wav_files) == 0:
        lab_files = sorted(glob(join(args.in_dir, "*.lab")))
        if args.filter_augmented_files:
            lab_files = list(filter(lambda x: not x.endswith("aug.lab"), lab_files))
        assert len(lab_files) > 0
        process_lab(lab_files, out_dir, args.shift_in_cent)
    else:
        with ProcessPoolExecutor(args.n_jobs) as executor:
            futures = [
                executor.submit(
                    process_wav,
                    wav_file,
                    out_dir,
                    args.shift_in_cent,
                )
                for wav_file in wav_files
            ]
            for future in tqdm(futures):
                future.result()
