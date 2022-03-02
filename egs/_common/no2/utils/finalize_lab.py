import os
import sys
from glob import glob
from os.path import basename, exists, expanduser, join, splitext

import librosa
import numpy as np
import soundfile as sf
import yaml
from nnmnkwii.io import hts
from tqdm import tqdm
from util import _is_silence, fix_offset, get_note_indices, trim_sil_and_pau

if len(sys.argv) != 2:
    print(f"USAGE: {sys.argv[0]} config_path")
    sys.exit(-1)

config = None
with open(sys.argv[1], "r") as yml:
    config = yaml.load(yml, Loader=yaml.FullLoader)
if config is None:
    print(f"Cannot read config file: {sys.argv[1]}.")
    sys.exit(-1)

full_align_dir = join(config["out_dir"], "full_dtw_seg")
full_score_dir = join(config["out_dir"], "sinsy_full_round_seg")


def sanity_check_lab(lab):
    for b, e, _ in lab:
        assert e - b > 0


def remove_sil_and_pau(lab):
    newlab = hts.HTSLabelFile()
    for l_inner in lab:
        if "-sil" not in l_inner[-1] and "-pau" not in l_inner[-1]:
            newlab.append(l_inner, strict=False)

    return newlab


# Prepare data for time-lag models

dst_dir = join(config["out_dir"], "timelag")
lab_align_dst_dir = join(dst_dir, "label_phone_align")
lab_score_dst_dir = join(dst_dir, "label_phone_score")

for d in [lab_align_dst_dir, lab_score_dst_dir]:
    os.makedirs(d, exist_ok=True)


base_files = sorted(glob(join(config["out_dir"], "full_dtw", "*.lab")))

print("Prepare data for time-lag models")
for base in tqdm(base_files):
    utt_id = splitext(basename(base))[0]
    seg_idx = 0

    # Compute offset for the entire song
    lab_align_path = join(config["out_dir"], "full_dtw", f"{utt_id}.lab")
    lab_score_path = join(config["out_dir"], "sinsy_full_round", f"{utt_id}.lab")
    lab_align = trim_sil_and_pau(hts.load(lab_align_path))
    lab_score = trim_sil_and_pau(hts.load(lab_score_path))

    # this may harm for computing offset
    lab_align = remove_sil_and_pau(lab_align)
    lab_score = remove_sil_and_pau(lab_score)

    # Extract note onsets and let's compute a offset
    note_indices = get_note_indices(lab_score)

    # offset = argmin_{b} \sum_{t=1}^{T}|x-(y+b)|^2
    # assuming there's a constant offset; tempo is same through the song
    onset_align = np.asarray(lab_align[note_indices].start_times)
    onset_score = np.asarray(lab_score[note_indices].start_times)
    global_offset = (onset_align - onset_score).mean()
    global_offset = int(round(global_offset / 50000) * 50000)

    # Apply offset correction only when there is a big gap
    apply_offset_correction = np.abs(global_offset * 1e-7) > float(
        config["offset_correction_threshold"]
    )
    if apply_offset_correction:
        print(f"{utt_id}: Global offset (in sec): {global_offset * 1e-7}")

    while True:
        lab_align_path = join(full_align_dir, f"{utt_id}_seg{seg_idx}.lab")
        lab_score_path = join(full_score_dir, f"{utt_id}_seg{seg_idx}.lab")
        name = basename(lab_align_path)
        assert seg_idx > 0 or exists(lab_align_path)
        if not exists(lab_align_path):
            break
        assert exists(lab_score_path)

        lab_align = hts.load(lab_align_path)
        lab_score = hts.load(lab_score_path)
        sanity_check_lab(lab_align)

        # Pau/sil lengths may differ in score and alignment, so remove it in case.
        lab_align = trim_sil_and_pau(lab_align)
        lab_score = trim_sil_and_pau(lab_score)

        # Extract note onsets and let's compute a offset
        note_indices = get_note_indices(lab_score)

        # offset = argmin_{b} \sum_{t=1}^{T}|x-(y+b)|^2
        # assuming there's a constant offset; tempo is same through the song
        onset_align = np.asarray(lab_align[note_indices].start_times)
        onset_score = np.asarray(lab_score[note_indices].start_times)

        # Offset adjustment
        segment_offset = (onset_align - onset_score).mean()
        segment_offset = int(round(segment_offset / 50000) * 50000)
        if apply_offset_correction:
            if config["global_offset_correction"]:
                offset_ = global_offset
            else:
                offset_ = segment_offset
            print(f"{name} offset (in sec): {offset_ * 1e-7}")
        else:
            offset_ = 0
        # apply
        lab_score.start_times = list(np.asarray(lab_score.start_times) + offset_)
        lab_score.end_times = list(np.asarray(lab_score.end_times) + offset_)
        onset_score += offset_

        # Exclude large diff parts (probably a bug of musicxml or alignment though)
        valid_note_indices = []
        for idx, (a, b) in enumerate(zip(onset_align, onset_score)):
            note_idx = note_indices[idx]
            lag = np.abs(a - b) / 50000
            if _is_silence(lab_score.contexts[note_idx]):
                if lag >= float(
                    config["timelag_allowed_range_rest"][0]
                ) and lag <= float(config["timelag_allowed_range_rest"][1]):
                    valid_note_indices.append(note_idx)
            else:
                if lag >= float(config["timelag_allowed_range"][0]) and lag <= float(
                    config["timelag_allowed_range"][1]
                ):
                    valid_note_indices.append(note_idx)

        if len(valid_note_indices) < len(note_indices):
            D = len(note_indices) - len(valid_note_indices)
            print(f"{utt_id}.lab: {D}/{len(note_indices)} time-lags are excluded.")

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

        seg_idx += 1


# Prepare data for duration models

dst_dir = join(config["out_dir"], "duration")
lab_align_dst_dir = join(dst_dir, "label_phone_align")

for d in [lab_align_dst_dir, lab_score_dst_dir]:
    os.makedirs(d, exist_ok=True)

print("Prepare data for duration models")
for base in tqdm(base_files):
    utt_id = splitext(basename(base))[0]
    seg_idx = 0

    while True:
        lab_align_path = join(full_align_dir, f"{utt_id}_seg{seg_idx}.lab")
        name = basename(lab_align_path)
        assert seg_idx > 0 or exists(lab_align_path)
        if not exists(lab_align_path):
            break

        lab_align = hts.load(lab_align_path)
        sanity_check_lab(lab_align)
        lab_align = fix_offset(lab_align)

        # Save lab file
        lab_align_dst_path = join(lab_align_dst_dir, name)
        with open(lab_align_dst_path, "w") as of:
            of.write(str(lab_align))

        seg_idx += 1

# Prepare data for acoustic models

dst_dir = join(config["out_dir"], "acoustic")
wav_dst_dir = join(dst_dir, "wav")
lab_align_dst_dir = join(dst_dir, "label_phone_align")
lab_score_dst_dir = join(dst_dir, "label_phone_score")

for d in [wav_dst_dir, lab_align_dst_dir, lab_score_dst_dir]:
    os.makedirs(d, exist_ok=True)

print("Prepare data for acoustic models")
for base in tqdm(base_files):
    utt_id = splitext(basename(base))[0]
    if config["spk"] == "natsumeyuuri":
        # natsume_singing
        wav_path = join(expanduser(config["db_root"]), f"wav/{utt_id}.wav")
    else:
        # ofuton_p_utagoe_db, oniku_kurumi_utagoe_db
        wav_path = join(expanduser(config["db_root"]), f"{utt_id}/{utt_id}.wav")

    assert exists(wav_path)
    # sr, wave = wavfile.read(wav_path)
    wav, sr = librosa.load(wav_path, sr=config["sample_rate"])
    assert sr == config["sample_rate"]

    # gain normalize
    wav = wav / wav.max() * 0.99

    seg_idx = 0
    while True:
        lab_align_path = join(full_align_dir, f"{utt_id}_seg{seg_idx}.lab")
        lab_score_path = join(full_score_dir, f"{utt_id}_seg{seg_idx}.lab")
        name = basename(lab_align_path)
        assert seg_idx > 0 or exists(lab_align_path)
        if not exists(lab_align_path):
            break
        lab_align = hts.load(lab_align_path)
        lab_score = hts.load(lab_score_path)

        # Make a slice of audio and then save it
        b, e = int(lab_align[0][0] * 1e-7 * sr), int(lab_align[-1][1] * 1e-7 * sr)
        wav_silce = wav[b:e]
        wav_slice_path = join(wav_dst_dir, name.replace(".lab", ".wav"))
        # TODO: consider explicit subtype
        sf.write(wav_slice_path, wav_silce, sr)

        # Set the beginning time to be zero for convenience
        lab_align = fix_offset(lab_align)
        sanity_check_lab(lab_align)
        lab_score = fix_offset(lab_score)

        # Save label
        lab_align_dst_path = join(lab_align_dst_dir, name)
        with open(lab_align_dst_path, "w") as of:
            of.write(str(lab_align))

        lab_score_dst_path = join(lab_score_dst_dir, name)
        with open(lab_score_dst_path, "w") as of:
            of.write(str(lab_score))

        seg_idx += 1

sys.exit(0)
