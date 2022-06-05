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
    parser.add_argument("jsut_song_root", type=str, help="JSUT song dir")
    parser.add_argument("jsut_lab_root", type=str, help="JSUT lab dir")
    parser.add_argument("hts_demo_root", type=str, help="HTS demo root")
    parser.add_argument("out_dir", type=str, help="Output directory")
    return parser


args = get_parser().parse_args(sys.argv[1:])

out_dir = args.out_dir
nit_song080_label_root = join(args.hts_demo_root, "data/labels/full")
jsut_song_root = join(args.jsut_song_root, "child_song")

# Time-lag constraints to filter outliers
timelag_allowed_range = (-20, 19)
timelag_allowed_range_rest = (-40, 39)

offset_correction_threshold = 0.005

prefix = "jsut"

# Make aligned full context labels

full_align_dir = join(args.jsut_lab_root)
# Note: this will be saved under jsut_song_root directory
full_align_new_dir = join(jsut_song_root, "label_phone_align")

os.makedirs(full_align_new_dir, exist_ok=True)

# replace contexts with HTS demo's one
# this is needed because nnmnkwii assumes that pitch is represented by midi note (e.g. E4),
# while jsut-song's label uses numbers (e.g. 87) to represent pitch.
full_lab_files = sorted(glob(join(full_align_dir, "*.lab")))
names = list(map(lambda s: basename(s), full_lab_files))
nit_lab_files = list(
    map(lambda s: join(nit_song080_label_root, f"nitech_jp_song070_f001_{s}"), names)
)
for jsut, nit in zip(full_lab_files, nit_lab_files):
    assert exists(jsut) and exists(nit)
    jsut_lab = hts.load(jsut)
    nit_lab = hts.load(nit)
    assert len(jsut_lab) == len(nit_lab)

    # make sure that each label represents the same
    for a, b in zip(jsut_lab, nit_lab):
        assert a[-1].split("/A")[0] == b[-1].split("/A")[0]

    jsut_lab.contexts = nit_lab.contexts
    name = basename(jsut)
    dst_path = join(full_align_new_dir, name)
    with open(dst_path, "w") as of:
        of.write(str(jsut_lab))


# Prepare data for time-lag models

dst_dir = join(out_dir, "timelag")
lab_align_dst_dir = join(dst_dir, "label_phone_align")
lab_score_dst_dir = join(dst_dir, "label_phone_score")

for d in [lab_align_dst_dir, lab_score_dst_dir]:
    os.makedirs(d, exist_ok=True)

print("Prepare data for time-lag models")
full_lab_align_files = sorted(glob(join(full_align_new_dir, "*.lab")))
for lab_align_path in full_lab_align_files:
    name = basename(lab_align_path)
    lab_score_path = join(nit_song080_label_root, f"nitech_jp_song070_f001_{name}")
    assert exists(lab_score_path)

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

    # NOTE: before saving file, let's add a prefix
    # 01.lab -> ${prefix}_01.lab
    name = prefix + "_" + name

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
full_lab_align_files = sorted(glob(join(full_align_new_dir, "*.lab")))
for lab_align_path in full_lab_align_files:
    name = basename(lab_align_path)

    lab_align = hts.load(lab_align_path)

    # NOTE: before saving file, let's add a prefix
    # 01.lab -> ${prefix}_01.lab
    name = prefix + "_" + name

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
full_lab_align_files = sorted(glob(join(full_align_new_dir, "*.lab")))
for lab_align_path in full_lab_align_files:
    name = splitext(basename(lab_align_path))[0]
    lab_score_path = join(nit_song080_label_root, f"nitech_jp_song070_f001_{name}.lab")
    assert exists(lab_score_path)
    wav_path = join(jsut_song_root, "wav", name + ".wav")

    assert wav_path
    # sr, wave = wavfile.read(wav_path)
    wav, sr = librosa.load(wav_path, sr=48000)

    lab_align = hts.load(lab_align_path)
    lab_score = hts.load(lab_score_path)

    # NOTE: before saving file, let's add a prefix
    # 01.lab -> ${prefix}_01.lab
    name = prefix + "_" + name

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
