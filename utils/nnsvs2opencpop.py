"""Convert NNSVS's segmented data to Opencpop's structure

so that the code for DiffSinger can be used.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import librosa
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from tqdm.auto import tqdm


def note_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return 0
    return librosa.note_to_midi(match.group(1))


def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return 0
    return int(match.group(1))


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert NNSVS's segmented data to Opencpop's structure",
    )
    parser.add_argument("in_dir", type=str, help="Path to input dir")
    parser.add_argument("out_dir", type=str, help="Output directory")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    label_score_dir = in_dir / "label_phone_score"
    label_align_dir = in_dir / "label_phone_align"
    in_wav_dir = in_dir / "wav"

    out_wav_dir = out_dir / "wavs"
    out_wav_dir.mkdir(exist_ok=True, parents=True)

    label_score_files = sorted(label_score_dir.glob("*.lab"))
    utt_ids = [f.stem for f in label_score_files]

    rows = []
    for utt_id in tqdm(utt_ids):
        if utt_id in ["namine_ritsu_hana_seg12"]:
            continue
        label_score = hts.load(label_score_dir / f"{utt_id}.lab")
        label_align = hts.load(label_align_dir / f"{utt_id}.lab")

        ph = [
            re.search(r"\-(.*?)\+", context).group(1)
            for context in label_score.contexts
        ]
        note = [
            note_by_regex(r"/E:([A-Z][b]?[0-9]+)]", context)
            for context in label_score.contexts
        ]
        note_dur = [
            numeric_feature_by_regex(r"@(\d+)#", context) / 100.0
            for context in label_score.contexts
        ]
        ph_dur = fe.duration_features(label_align).reshape(-1) * 0.005
        is_slur = [0] * len(ph_dur)
        assert len(ph) == len(note) == len(note_dur) == len(ph_dur) == len(is_slur)
        cols = [
            utt_id,
            " ".join(ph),
            " ".join(ph),
            " ".join(str(n) for n in note),
            " ".join(str(n) for n in note_dur),
            " ".join(str(round(n, 3)) for n in ph_dur),
            " ".join(str(n) for n in is_slur),
        ]
        rows.append("|".join(cols))
        shutil.copyfile(in_wav_dir / f"{utt_id}.wav", out_wav_dir / f"{utt_id}.wav")

    with open(out_dir / "transcriptions.txt", "w") as f:
        for row in rows:
            f.write(row + "\n")
