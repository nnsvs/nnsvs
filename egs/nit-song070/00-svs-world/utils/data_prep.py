# coding: utf-8
import os

import argparse
from glob import glob
from os.path import join, basename, splitext, exists, expanduser
from nnmnkwii.io import hts
from scipy.io import wavfile
import librosa
import soundfile as sf
import sys
import numpy as np

from nnsvs.io.hts import get_note_indices

def get_parser():
    parser = argparse.ArgumentParser(
        description="Data preparation for NIT-SONG070",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("hts_demo_root", type=str, help="HTS demo root")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument(
        "--max-timelag", type=int, default=50, help="Max allowed time-lag (in frames)"
    )
    parser.add_argument("--gain-normalize", action='store_true')
    return parser

args = get_parser().parse_args(sys.argv[1:])

hts_demo_root = args.hts_demo_root
out_dir = args.out_dir
max_timelag = args.max_timelag
hts_label_root = join(hts_demo_root, "data/labels")
gain_normalize = args.gain_normalize

mono_dir = join(hts_label_root, "mono")
full_dir = join(hts_label_root, "full")

### Make aligned full context labels

# Note: this will be saved under hts_label_root directory
full_align_dir = join(hts_label_root, "full_align")
os.makedirs(full_align_dir, exist_ok=True)

mono_lab_files = sorted(glob(join(mono_dir, "*.lab")))
full_lab_files = sorted(glob(join(full_dir, "*.lab")))
for m, f in zip(mono_lab_files, full_lab_files):
    mono_lab = hts.load(m)
    full_lab = hts.load(f)
    assert len(mono_lab) == len(full_lab)
    full_lab.start_times = mono_lab.start_times
    full_lab.end_times = mono_lab.end_times
    name = basename(m)
    dst_path = join(full_align_dir, name)
    with open(dst_path, "w") as of:
        of.write(str(full_lab))

### Prepare data for time-lag models

dst_dir = join(out_dir, "timelag")
lab_align_dst_dir  = join(dst_dir, "label_phone_align")
lab_score_dst_dir  = join(dst_dir, "label_phone_score")

for d in [lab_align_dst_dir, lab_score_dst_dir]:
    os.makedirs(d, exist_ok=True)

print("Prepare data for time-lag models")
full_lab_align_files = sorted(glob(join(full_align_dir, "*.lab")))
for lab_align_path in full_lab_align_files:
    lab_score_path = join(full_dir, basename(lab_align_path))
    assert exists(lab_score_path)
    name = basename(lab_align_path)

    lab_align = hts.load(lab_align_path)
    lab_score = hts.load(lab_score_path)

    # Extract note onsets and let's compute a offset
    note_indices = get_note_indices(lab_score)

    onset_align = np.asarray(lab_align[note_indices].start_times)
    onset_score = np.asarray(lab_score[note_indices].start_times)
    # Exclude large diff parts (probably a bug of musicxml though)
    diff = np.abs(onset_align - onset_score) / 50000
    if diff.max() > max_timelag:
        print(f"{name}: {np.sum(diff > max_timelag)}/{len(diff)} of time-lags are excluded. max/max: {diff.min()}/{diff.max()}")
        note_indices = list(np.asarray(note_indices)[diff <= max_timelag])

    # Note onsets as labels
    lab_align = lab_align[note_indices]
    lab_score = lab_score[note_indices]

    # Save lab files
    lab_align_dst_path = join(lab_align_dst_dir, name)
    with open(lab_align_dst_path, "w") as of:
        of.write(str(lab_align))

    lab_score_dst_path = join(lab_score_dst_dir, name)
    with open(lab_score_dst_path, "w") as of:
        of.write(str(lab_score))

### Prepare data for duration models

dst_dir = join(out_dir, "duration")
lab_align_dst_dir  = join(dst_dir, "label_phone_align")

for d in [lab_align_dst_dir]:
    os.makedirs(d, exist_ok=True)

print("Prepare data for duration models")
full_lab_align_files = sorted(glob(join(full_align_dir, "*.lab")))
for lab_align_path in full_lab_align_files:
    lab_score_path = join(full_dir, basename(lab_align_path))
    name = basename(lab_align_path)
    assert exists(lab_score_path)

    lab_align = hts.load(lab_align_path)

    # Save lab file
    lab_align_dst_path = join(lab_align_dst_dir, name)
    with open(lab_align_dst_path, "w") as of:
        of.write(str(lab_align))


### Prepare data for acoustic models

dst_dir = join(out_dir, "acoustic")
wav_dst_dir  = join(dst_dir, "wav")
lab_align_dst_dir  = join(dst_dir, "label_phone_align")
lab_score_dst_dir  = join(dst_dir, "label_phone_score")

for d in [wav_dst_dir, lab_align_dst_dir, lab_score_dst_dir]:
    os.makedirs(d, exist_ok=True)

print("Prepare data for acoustic models")
full_lab_align_files = sorted(glob(join(full_align_dir, "*.lab")))
for lab_align_path in full_lab_align_files:
    name = splitext(basename(lab_align_path))[0]
    lab_score_path = join(full_dir, name + ".lab")
    assert exists(lab_score_path)
    wav_path = join(hts_demo_root, "data", "wav", name + ".wav")
    raw_path = join(hts_demo_root, "data", "raw", name + ".raw")

    # We can load and manupulate audio (e.g., normalizing gain), but for now just copy it as is
    if exists(wav_path):
        # sr, wave = wavfile.read(wav_path)
        wav, sr = librosa.load(wav_path, sr=48000)
    else:
        assert raw_path
        wav = np.fromfile(raw_path, dtype=np.int16)
        wav = wav.astype(np.float32) / 2**15
        sr = 48000

    if gain_normalize:
        wav = wav / wav.max() * 0.99

    lab_align = hts.load(lab_align_path)
    lab_score = hts.load(lab_score_path)

    # Save caudio
    wav_dst_path = join(wav_dst_dir, name + ".wav")
    # TODO: consider explicit subtype
    sf.write(wav_dst_path, wav, sr)

    # Save label
    lab_align_dst_path = join(lab_align_dst_dir, name + ".lab")
    with open(lab_align_dst_path, "w") as of:
        of.write(str(lab_align))

    lab_score_dst_path = join(lab_score_dst_dir, name + ".lab")
    with open(lab_score_dst_path, "w") as of:
        of.write(str(lab_score))

sys.exit(0)