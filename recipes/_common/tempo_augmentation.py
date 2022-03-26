"""Tempo data augmentation for lab and wav files
"""
import argparse
import os
import random
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
        description="Tempo data augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("in_dir", type=str, help="Input directory")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("tempo", default=1.0, type=float, help="Tempo")
    parser.add_argument(
        "--random_tempo", action="store_true", help="Random tempo augmentation"
    )
    parser.add_argument(
        "--filter_augmented_files",
        action="store_true",
        help="filter out already augmented files",
    )
    return parser


def process_wav(wav_files, out_dir, tempo, random_tempo=False):
    # NOTE: must be careful about random seed
    if random_tempo:
        random.seed(int(tempo * 100))

    for wav_file in tqdm(wav_files):
        if random_tempo:
            tempo = round(random.uniform(0.9, 1.1), 2)

        wav, sr = librosa.load(wav_file, sr=None)
        x = torch.from_numpy(wav).view(1, -1)

        # pitch shift by sox
        effects = [["tempo", f"{tempo}"], ["rate", f"{sr}"]]
        y, y_sr = apply_effects_tensor(x, sr, effects)
        y = y.view(-1)

        assert y_sr == sr

        postfix = str(tempo).replace("-", "minus") + "tempo_aug"

        out_file = join(out_dir, basename(wav_file).replace(".wav", f"_{postfix}.wav"))
        sf.write(out_file, y.numpy(), sr)


def process_lab(lab_files, out_dir, tempo, random_tempo=False):
    # NOTE: must be careful about random seed
    if random_tempo:
        random.seed(int(tempo * 100))

    for lab_file in tqdm(lab_files):
        if random_tempo:
            tempo = round(random.uniform(0.9, 1.1), 2)

        labels = hts.load(lab_file)
        name = basename(lab_file)
        new_s = []
        new_e = []
        new_contexts = []
        for s, e, context in labels:
            new_s.append(int(round(s / tempo / 50000) * 50000))
            new_e.append(int(round(e / tempo / 50000) * 50000))

            # Tempo: d5, e5, f5
            for pre, post in [("%", "\\|"), ("~", "!"), ("\\$", "\\$")]:
                match = re.search(f"{pre}([0-9]+){post}", context)
                # if not "xx"
                if match is not None:
                    assert len(match.groups()) == 1
                    num = match.group(0)[1:-1]
                    if len(num) > 0:
                        pre = pre.replace("\\", "")
                        post = post.replace("\\", "")
                        new_num = int(round(float(num) * tempo))
                        context = context.replace(
                            match.group(0), f"{pre}{new_num}{post}", 1
                        )

            # Length in sec or msec: d7, e7, f7
            # e12/13, e20/21, e31/32, e37/e38, e43/44, e51/52
            for pre, post in [
                ("&", ";"),
                ("@", "#"),
                ("\\+", "%"),
                ("\\|", "\\["),
                ("\\[", "&"),
                ("_", ";"),
                (";", "\\$"),
                ("~", "="),
                ("=", "@"),
                ("#", "\\|"),
                ("\\|", "\\|"),
                ("\\+", "\\["),
                ("\\[", ";"),
                ("\\^", "@"),
                ("@", "\\["),
            ]:
                match = re.search(f"{pre}([0-9]+){post}", context)
                # if not "xx"
                if match is not None:
                    assert len(match.groups()) == 1
                    num = match.group(0)[1:-1]
                    if len(num) > 0:
                        pre = pre.replace("\\", "")
                        post = post.replace("\\", "")
                        # NOTE: ensure > 0
                        new_num = max(int(round(float(num) / tempo)), 1)
                        context = context.replace(
                            match.group(0), f"{pre}{new_num}{post}", 1
                        )

            new_contexts.append(context)

        labels.start_times = new_s
        labels.end_times = new_e
        labels.contexts = new_contexts

        postfix = str(tempo).replace("-", "minus") + "tempo_aug"
        dst_lab_file = join(out_dir, name.replace(".lab", f"_{postfix}.lab"))
        with open(dst_lab_file, "w") as of:
            of.write(str(labels))


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

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
        process_lab(lab_files, out_dir, args.tempo, args.random_tempo)
    else:
        process_wav(wav_files, out_dir, args.tempo, args.random_tempo)
