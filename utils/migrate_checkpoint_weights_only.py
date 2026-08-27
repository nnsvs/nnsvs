import argparse
import os
import sys

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert OmegaConf containers (e.g. optimizer betas) embedded in an "
            "old checkpoint into plain Python types, so the checkpoint becomes "
            "loadable with torch.load's default weights_only=True."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", type=str, help="input checkpoint")
    parser.add_argument("output_file", type=str, help="output checkpoint")

    return parser


def _sanitize(obj, count):
    if isinstance(obj, (ListConfig, DictConfig)):
        count[0] += 1
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: _sanitize(v, count) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v, count) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize(v, count) for v in obj)
    return obj


def migrate_checkpoint(input_file, output_file, verbose=True):
    """Convert OmegaConf containers embedded in ``input_file`` to plain Python
    types and save the result to ``output_file``.

    Returns the number of OmegaConf containers that were converted.
    """
    # NOTE: weights_only=False is required here since old checkpoints may embed
    # OmegaConf containers (e.g. optimizer betas) that this script converts away.
    checkpoint = torch.load(input_file, map_location="cpu", weights_only=False)
    if verbose:
        size = os.path.getsize(input_file)
        print("Processing:", input_file)
        print(f"File size (before): {size / 1024 / 1024:.3f} MB")

    count = [0]
    checkpoint = _sanitize(checkpoint, count)
    if verbose:
        print(f"Converted {count[0]} OmegaConf container(s) to plain Python types")

    torch.save(checkpoint, output_file)
    if verbose:
        size = os.path.getsize(output_file)
        print(f"File size (after): {size / 1024 / 1024:.3f} MB")

    # Verify the migrated checkpoint is loadable under the strict default.
    torch.load(output_file, map_location="cpu", weights_only=True)
    if verbose:
        print("Verified: migrated checkpoint loads with weights_only=True")

    return count[0]


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    migrate_checkpoint(args.input_file, args.output_file)
