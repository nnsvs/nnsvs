import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from migrate_checkpoint_weights_only import migrate_checkpoint  # noqa: E402


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Migrate all resume checkpoints (latest.pth / latest_D.pth) under a "
            "pretrained_expdir-style directory tree so they become loadable with "
            "torch.load's default weights_only=True. Only these files are read by "
            "the resume path (see train_*.sh's pretrained_expdir handling); other "
            "checkpoints (best_loss.pth, epochNNNN.pth, ...) are left untouched."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=str, help="input pretrained_expdir")
    parser.add_argument("output_dir", type=str, help="output pretrained_expdir")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    checkpoints = sorted(input_dir.rglob("latest*.pth"))
    if len(checkpoints) == 0:
        print(f"No latest*.pth found under {input_dir}")
        sys.exit(1)

    migrated, already_safe = 0, 0
    for input_file in checkpoints:
        rel_path = input_file.relative_to(input_dir)
        output_file = output_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.load(input_file, map_location="cpu", weights_only=True)
        except Exception:  # pylint: disable=broad-except
            print(f"[migrate] {rel_path}")
            migrate_checkpoint(str(input_file), str(output_file))
            migrated += 1
        else:
            print(f"[already safe, copied as-is] {rel_path}")
            shutil.copy2(input_file, output_file)
            already_safe += 1
        print()

    print(
        f"Done: {migrated} checkpoint(s) migrated, "
        f"{already_safe} already weights_only=True-safe (copied as-is)."
    )
    print(f"New pretrained_expdir: {output_dir}")
