"""Pitch augmentation for extracted features
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from nnmnkwii.io import hts
from nnsvs.util import load_utt_list
from tqdm.auto import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pitch data augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("utt_list", type=str, help="utt list")
    parser.add_argument("in_dir", type=str, help="Input directory")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("qst_path", type=str, help="qst path")
    parser.add_argument("input_type", choices=["in", "out"], help="Input type")
    parser.add_argument(
        "shift_in_cent", default=100, type=int, help="Pitch shift in cent"
    )
    parser.add_argument("--n_jobs", help="n_jobs")
    return parser


def process_in_feats(utt_id, in_dir, out_dir, note_indices, shift_in_cent):
    in_feats = np.load(in_dir / f"{utt_id}-feats.npy")

    lf0_offset = shift_in_cent * np.log(2) / 1200

    for idx in note_indices:
        in_feats[:, idx] += lf0_offset

    postfix = str(shift_in_cent).replace("-", "minus") + "cent_aug"

    out_file = out_dir / f"{utt_id}_{postfix}-feats.npy"
    np.save(out_file, in_feats, allow_pickle=False)


def process_out_feats(utt_id, in_dir, out_dir, lf0_idx, shift_in_cent):
    out_feats = np.load(in_dir / f"{utt_id}-feats.npy")

    lf0_offset = shift_in_cent * np.log(2) / 1200
    out_feats[:, lf0_idx] += lf0_offset

    postfix = str(shift_in_cent).replace("-", "minus") + "cent_aug"

    out_file = out_dir / f"{utt_id}_{postfix}-feats.npy"
    np.save(out_file, out_feats, allow_pickle=False)


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    assert args.shift_in_cent % 100 == 0

    utt_ids = load_utt_list(args.utt_list)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    lf0_idx = 60
    binary_dict, numeric_dict = hts.load_question_set(args.qst_path)
    note_indices = []
    for idx, (_, v) in enumerate(numeric_dict.items()):
        name, regex = v
        if name in ["d1", "e1", "f1"]:
            note_indices.append(len(binary_dict) + idx)

    with ProcessPoolExecutor(args.n_jobs) as executor:
        if args.input_type == "in":
            futures = [
                executor.submit(
                    process_in_feats,
                    utt_id,
                    in_dir,
                    out_dir,
                    note_indices,
                    args.shift_in_cent,
                )
                for utt_id in utt_ids
            ]
        else:
            futures = [
                executor.submit(
                    process_out_feats,
                    utt_id,
                    in_dir,
                    out_dir,
                    lf0_idx,
                    args.shift_in_cent,
                )
                for utt_id in utt_ids
            ]

        for future in tqdm(futures):
            future.result()

    sys.exit(0)
