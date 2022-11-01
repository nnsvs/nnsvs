import argparse
import os
import sys
from glob import glob
from os.path import basename, join, splitext

import librosa
import numpy as np
import pysinsy
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
        description="Data preparation for PJS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pjs_root", type=str, help="PJS song dir")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("--gain-normalize", action="store_true")
    return parser


args = get_parser().parse_args(sys.argv[1:])

out_dir = args.out_dir
gain_normalize = args.gain_normalize

# Time-lag constraints to filter outliers
timelag_allowed_range = (-20, 19)
timelag_allowed_range_rest = (-40, 39)

offset_correction_threshold = 0.01


# Make aligned full context labels

full_align_dir = join(out_dir, "label_phone_align")
full_score_dir = join(out_dir, "label_phone_score")
for d in [full_align_dir, full_score_dir]:
    os.makedirs(d, exist_ok=True)

sinsy = pysinsy.sinsy.Sinsy()
assert sinsy.setLanguages("j", pysinsy.get_default_dic_dir())

mono_lab_files = sorted(glob(join(args.pjs_root, "**/*.lab")))
muxicxml_files = sorted(glob(join(args.pjs_root, "**/*.musicxml")))
assert len(mono_lab_files) == len(muxicxml_files)
for mono_path, xml_path in zip(mono_lab_files, muxicxml_files):
    align_mono_lab = hts.load(mono_path)
    name = basename(mono_path)

    assert sinsy.loadScoreFromMusicXML(xml_path)
    # check if sinsy's phoneme output is same as the provided alignment format
    sinsy_labels = sinsy.createLabelData(True, 1, 1).getData()
    sinsy_mono_lab = hts.HTSLabelFile()
    for label in sinsy_labels:
        sinsy_mono_lab.append(label.split(), strict=False)

    assert len(align_mono_lab) == len(sinsy_mono_lab)
    assert (
        np.asarray(align_mono_lab.contexts) == np.asarray(sinsy_mono_lab.contexts)
    ).all()

    # rounding
    has_too_short_ph = False
    for idx in range(len(align_mono_lab)):
        b, e = align_mono_lab.start_times[idx], align_mono_lab.end_times[idx]
        bb, ee = round(b / 50000) * 50000, round(e / 50000) * 50000
        # TODO: better way
        if bb == ee:
            # ensure minimum frame length 1
            align_mono_lab.end_times[idx] = align_mono_lab.start_times[idx] + 50000
            align_mono_lab.start_times[idx + 1] = align_mono_lab.end_times[idx]
            print(align_mono_lab[idx - 1 : idx + 2])
            has_too_short_ph = True

    if has_too_short_ph:
        sinsy.clearScore()
    else:
        # gen full-context
        sinsy_labels = sinsy.createLabelData(False, 1, 1).getData()
        align_full_lab = hts.HTSLabelFile()
        score_full_lab = hts.HTSLabelFile()
        for idx, label in enumerate(sinsy_labels):
            b, e = align_mono_lab.start_times[idx], align_mono_lab.end_times[idx]
            try:
                align_full_lab.append((b, e, label.split()[-1]), strict=True)
            except BaseException:
                # TODO
                import ipdb

                ipdb.set_trace()
            b, e, c = label.split()
            b, e = round(int(b) / 50000) * 50000, round(int(e) / 50000) * 50000
            assert b != e
            score_full_lab.append((b, e, c), strict=False)

        with open(join(full_score_dir, name), "w") as of:
            of.write(str(score_full_lab))
        with open(join(full_align_dir, name), "w") as of:
            of.write(str(align_full_lab))
        sinsy.clearScore()


# Prepare data for time-lag models

dst_dir = join(out_dir, "timelag")
lab_align_dst_dir = join(dst_dir, "label_phone_align")
lab_score_dst_dir = join(dst_dir, "label_phone_score")

for d in [lab_align_dst_dir, lab_score_dst_dir]:
    os.makedirs(d, exist_ok=True)

print("Prepare data for time-lag models")
full_lab_align_files = sorted(glob(join(full_align_dir, "*.lab")))
full_lab_score_files = sorted(glob(join(full_score_dir, "*.lab")))
for lab_align_path, lab_score_path in zip(full_lab_align_files, full_lab_score_files):
    name = basename(lab_align_path)

    lab_align = hts.load(lab_align_path)
    lab_score = hts.load(lab_score_path)

    # this may harm for computing offset
    lab_align = remove_sil_and_pau(lab_align)
    lab_score = remove_sil_and_pau(lab_score)

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
full_lab_score_files = sorted(glob(join(full_score_dir, "*.lab")))
for lab_align_path, lab_score_path in zip(full_lab_align_files, full_lab_score_files):
    name = splitext(basename(lab_align_path))[0]
    wav_path = join(args.pjs_root, name, f"{name}_song.wav")
    assert wav_path
    # sr, wave = wavfile.read(wav_path)
    wav, sr = librosa.load(wav_path, sr=48000)

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
