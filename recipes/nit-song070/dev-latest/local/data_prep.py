import argparse
import os
import sys
from glob import glob
from os.path import basename, exists, join, splitext

import librosa
import numpy as np
import soundfile as sf
from nnmnkwii.io import hts
from nnsvs.io.hts import get_note_indices


def _is_silence(label):
    is_full_context = "@" in label
    if is_full_context:
        is_silence = "-sil" in label or "-pau" in label
    else:
        is_silence = label == "sil" or label == "pau"
    return is_silence


def remove_sil_and_pau(lab):
    newlab = hts.HTSLabelFile()
    for label in lab:
        if "-sil" not in label[-1] and "-pau" not in label[-1]:
            newlab.append(label, strict=False)

    return newlab


def get_parser():
    parser = argparse.ArgumentParser(
        description="Data preparation for NIT-SONG070",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("hts_demo_root", type=str, help="HTS demo root")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("--gain-normalize", action="store_true")
    return parser


args = get_parser().parse_args(sys.argv[1:])

hts_demo_root = args.hts_demo_root
out_dir = args.out_dir
hts_label_root = join(hts_demo_root, "data/labels")
gain_normalize = args.gain_normalize

# Time-lag constraints to filter outliers
timelag_allowed_range = (-20, 19)
timelag_allowed_range_rest = (-40, 39)

offset_correction_threshold = 0.005

mono_dir = join(hts_label_root, "mono")
full_dir = join(hts_label_root, "full")

# Make aligned full context labels

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

# Prepare data for time-lag models

dst_dir = join(out_dir, "timelag")
lab_align_dst_dir = join(dst_dir, "label_phone_align")
lab_score_dst_dir = join(dst_dir, "label_phone_score")

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

    global_offset = (onset_align - onset_score).mean()
    global_offset = int(round(global_offset / 50000) * 50000)

    # Apply offset correction only when there is a big gap
    apply_offset_correction = np.abs(global_offset * 1e-7) > offset_correction_threshold
    if apply_offset_correction:
        print(f"{name}: Global offset (in sec): {global_offset * 1e-7}")
        lab_score.start_times = list(np.asarray(lab_score.start_times) + global_offset)
        lab_score.end_times = list(np.asarray(lab_score.end_times) + global_offset)
        onset_score += global_offset

    # Exclude large diff parts (probably a bug of musicxml or alignment though)
    valid_note_indices = []
    for idx, (a, b) in enumerate(zip(onset_align, onset_score)):
        note_idx = note_indices[idx]
        lag = np.abs(a - b) / 50000
        if _is_silence(lab_score.contexts[note_idx]):
            if (
                lag >= timelag_allowed_range_rest[0]
                and lag <= timelag_allowed_range_rest[1]
            ):
                valid_note_indices.append(note_idx)
        else:
            if lag >= timelag_allowed_range[0] and lag <= timelag_allowed_range[1]:
                valid_note_indices.append(note_idx)

    if len(valid_note_indices) < len(note_indices):
        D = len(note_indices) - len(valid_note_indices)
        print(f"{name}: {D}/{len(note_indices)} time-lags are excluded.")

    # Note onsets as labels
    lab_align = lab_align[valid_note_indices]
    lab_score = lab_score[valid_note_indices]

    # Save lab files
    lab_align_dst_path = join(lab_align_dst_dir, name)
    with open(lab_align_dst_path, "w") as of:
        of.write(str(lab_align))

    lab_score_dst_path = join(lab_score_dst_dir, name)
    with open(lab_score_dst_path, "w") as of:
        of.write(str(lab_score))

# Prepare data for duration models

dst_dir = join(out_dir, "duration")
lab_align_dst_dir = join(dst_dir, "label_phone_align")

for d in [lab_align_dst_dir]:
    os.makedirs(d, exist_ok=True)

print("Prepare data for duration models")
full_lab_align_files = sorted(glob(join(full_align_dir, "*.lab")))
for lab_align_path in full_lab_align_files:
    name = basename(lab_align_path)

    lab_align = hts.load(lab_align_path)

    # Save lab file
    lab_align_dst_path = join(lab_align_dst_dir, name)
    with open(lab_align_dst_path, "w") as of:
        of.write(str(lab_align))


# Prepare data for acoustic models

dst_dir = join(out_dir, "acoustic")
wav_dst_dir = join(dst_dir, "wav")
lab_align_dst_dir = join(dst_dir, "label_phone_align")
lab_score_dst_dir = join(dst_dir, "label_phone_score")

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

    # We can load and manipulate audio (e.g., normalizing gain), but for now just copy it as is
    if exists(wav_path):
        # sr, wave = wavfile.read(wav_path)
        wav, sr = librosa.load(wav_path, sr=48000)
    else:
        assert raw_path
        wav = np.fromfile(raw_path, dtype=np.int16)
        wav = wav.astype(np.float32) / 2 ** 15
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
