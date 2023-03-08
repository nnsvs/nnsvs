"""Overwrite phoneme flags for HTS full-context labels
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from nnmnkwii.io import hts
from nnsvs.io.hts import overwrite_phoneme_flags_
from nnsvs.util import load_utt_list
from tqdm.auto import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Overwrite phoneme flags for HTS full-context labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("utt_list", type=str, help="utt list")
    parser.add_argument("in_dir", type=str, help="Input directory")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("flag", type=str, help="flag to overwrite")
    parser.add_argument("--n_jobs", help="n_jobs")
    return parser


def process_labels(utt_id, in_dir, out_dir, flag):
    labels = hts.load(in_dir / f"{utt_id}.lab")
    overwrite_phoneme_flags_(labels, flag)
    out_file = out_dir / f"{utt_id}.lab"
    with open(out_file, "w") as f:
        f.write(str(labels))


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    utt_ids = load_utt_list(args.utt_list)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    with ProcessPoolExecutor(args.n_jobs) as executor:
        futures = [
            executor.submit(
                process_labels,
                utt_id,
                in_dir,
                out_dir,
                args.flag,
            )
            for utt_id in utt_ids
        ]

        for future in tqdm(futures):
            future.result()

    sys.exit(0)
