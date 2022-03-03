"""Pitch augmentation for lab and wav files
"""
import argparse
import os
import re
import sys
from glob import glob
from os.path import basename, join

import librosa
import soundfile as sf
import torch
from nnmnkwii.frontend import NOTE_MAPPING
from nnmnkwii.io import hts
from torchaudio.sox_effects import apply_effects_tensor
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
    parser.add_argument(
        "--filter_augmented_files",
        action="store_true",
        help="filter out already augmented files",
    )
    return parser


def process_wav(wav_files, out_dir, shift_in_cent):
    for wav_file in tqdm(wav_files):
        wav, sr = librosa.load(wav_file, sr=None)
        x = torch.from_numpy(wav).view(1, -1)

        # pitch shift by sox
        effects = [["pitch", f"{shift_in_cent}"], ["rate", f"{sr}"]]
        y, y_sr = apply_effects_tensor(x, sr, effects)
        y = y.view(-1)

        assert y_sr == sr
        if len(y) != len(wav):
            print(f"{wav_file}: there's small difference between the length of wavs")
            print(y.shape, wav.shape)
            if len(y) > len(wav):
                y = y[: len(wav)]
            else:
                y = torch.cat([y, torch.zeros(len(wav) - len(y))])
        assert len(y) == len(wav)

        postfix = str(shift_in_cent).replace("-", "minus") + "cent_aug"

        out_file = join(out_dir, basename(wav_file).replace(".wav", f"_{postfix}.wav"))
        sf.write(out_file, y.numpy(), sr)


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
        process_wav(wav_files, out_dir, args.shift_in_cent)
