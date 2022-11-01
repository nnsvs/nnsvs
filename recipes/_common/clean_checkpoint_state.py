import argparse
import os
import sys

import torch


def get_parser():
    parser = argparse.ArgumentParser(
        description="Clean checkpoint state and make a new checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", type=str, help="input file")
    parser.add_argument("output_file", type=str, help="output file")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    checkpoint = torch.load(args.input_file, map_location=torch.device("cpu"))
    size = os.path.getsize(args.input_file)
    print("Processisng:", args.input_file)
    print(f"File size (before): {size / 1024/1024:.3f} MB")
    for k in ["optimizer_state", "lr_scheduler_state"]:
        if k in checkpoint.keys():
            del checkpoint[k]

    # For https://github.com/kan-bayashi/ParallelWaveGAN
    for k in ["optimizer", "lr_scheduler", "scheduler"]:
        if k in checkpoint.keys():
            del checkpoint[k]
    if "model" in checkpoint and "discriminator" in checkpoint["model"]:
        del checkpoint["model"]["discriminator"]

    torch.save(checkpoint, args.output_file)
    size = os.path.getsize(args.output_file)
    print(f"File size (after): {size / 1024/1024:.3f} MB")
