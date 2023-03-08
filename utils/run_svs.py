"""Run NNSVS"s SVS inference
"""
import argparse
import sys
from pathlib import Path

import pysinsy
from nnmnkwii.io import hts
from nnsvs.io.hts import merge_sil, overwrite_phoneme_flags_
from nnsvs.svs import SPSVS
from nnsvs.util import init_seed
from scipy.io import wavfile
from utaupy.utils import ust2hts


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run SVS",
    )
    parser.add_argument("model_dir", type=str, help="Model dir")
    parser.add_argument("input_file", type=str, help="Input file")
    parser.add_argument("out_dir", type=str, help="Output dir")
    parser.add_argument(
        "--vocoder_type", type=str, default="usfgan", help="Vocoder type"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--phoneme-flag", type=str, help="phoneme flag (p9)")
    parser.add_argument("--verbose", default=100, type=int, help="Verbose level")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    model_dir = Path(args.model_dir)
    input_file = Path(args.input_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    vocoder_type = args.vocoder_type

    model_name = model_dir.name

    engine = SPSVS(model_dir, device=args.device, verbose=args.verbose)

    if input_file.suffix == ".ust":
        table_path = model_dir / "kana2phonemes.table"
        assert table_path.exists()
        full_lab_path = out_dir / f"{input_file.stem}_full.lab"
        ust2hts(
            str(input_file),
            full_lab_path,
            table_path,
            strict_sinsy_style=False,
            as_mono=False,
        )
        labels = hts.HTSLabelFile()
        with open(full_lab_path) as f:
            for label in f.readlines():
                labels.append(label.split(), strict=False)
        labels = merge_sil(labels)
    elif input_file.suffix in [".xml", ".musicxml"]:
        contexts = pysinsy.extract_fullcontext(str(input_file))
        labels = hts.HTSLabelFile.create_from_contexts(contexts)
    elif input_file.suffix == ".lab":
        labels = hts.load(input_file)
    else:
        raise ValueError(f"Unknown input file type: {input_file.suffix}")

    if args.phoneme_flag is not None:
        overwrite_phoneme_flags_(labels, args.phoneme_flag)

    init_seed(1234)
    wav, sr = engine.svs(
        labels,
        vocoder_type=vocoder_type,
        post_filter_type="gv",
        force_fix_vuv=True,
        segmented_synthesis=False,
    )

    if args.phoneme_flag is not None:
        filename = f"{input_file.stem}_{model_name}_{args.phoneme_flag}.wav"
    else:
        filename = f"{input_file.stem}_{model_name}.wav"

    wavfile.write(out_dir / filename, sr, wav)
