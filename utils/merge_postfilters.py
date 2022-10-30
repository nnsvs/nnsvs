import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def get_parser():
    parser = argparse.ArgumentParser(
        description="Merge post-filters",
    )
    parser.add_argument("mgc_checkpoint", type=str, help="mgc checkpoint")
    parser.add_argument("bap_checkpoint", type=str, help="bap checkpoint")
    parser.add_argument("output_dir", type=str, help="out_dir")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    mgc_checkpoint = torch.load(args.mgc_checkpoint, map_location="cpu")
    bap_checkpoint = torch.load(args.bap_checkpoint, map_location="cpu")

    for path in [args.mgc_checkpoint, args.bap_checkpoint]:
        size = os.path.getsize(path)
        print("Processisng:", path)
        print(f"File size: {size / 1024/1024:.3f} MB")

    mgc_model = OmegaConf.load(Path(args.mgc_checkpoint).parent / "model.yaml")
    bap_model = OmegaConf.load(Path(args.bap_checkpoint).parent / "model.yaml")

    if "postfilters.MultistreamPostFilter" not in mgc_model.netG._target_:
        raise ValueError("Only MultistreamPostFilter is supported for now")

    checkpoint = mgc_checkpoint
    checkpoint["state_dict"].update(bap_checkpoint["state_dict"])

    for k in ["optimizer_state", "lr_scheduler_state"]:
        if k in checkpoint.keys():
            del checkpoint[k]

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    # Model definition
    yaml_path = Path(args.output_dir) / "model.yaml"
    mgc_model.netG.bap_postfilter = bap_model.netG.bap_postfilter
    OmegaConf.save(mgc_model, yaml_path)

    # Checkpoint
    checkpoint_path = Path(args.output_dir) / "latest.pth"
    torch.save(checkpoint, checkpoint_path)
    size = os.path.getsize(checkpoint_path)
    print(f"File size (after): {size / 1024/1024:.3f} MB")
