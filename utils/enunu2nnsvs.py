"""Convert ENUNU's packed model to NNSVS's style
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from nnsvs.logger import getLogger
from nnsvs.util import StandardScaler as NNSVSStandardScaler
from omegaconf import OmegaConf
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert ENUNU's packed model to NNSVS's style",
    )
    parser.add_argument("enunu_dir", type=str, help="ENUNU's model dir")
    parser.add_argument("out_dir", type=str, help="Output dir")
    parser.add_argument("--verbose", type=int, default=100, help="Verbose level")
    return parser


def _scaler2numpy(input_file, out_dir, logger):
    scaler = joblib.load(input_file)
    if isinstance(scaler, StandardScaler) or isinstance(scaler, NNSVSStandardScaler):
        logger.info(f"Converting {input_file} mean/scale npy files")
        mean_path = out_dir / (input_file.stem + "_mean.npy")
        scale_path = out_dir / (input_file.stem + "_scale.npy")
        var_path = out_dir / (input_file.stem + "_var.npy")

        np.save(mean_path, scaler.mean_, allow_pickle=False)
        np.save(scale_path, scaler.scale_, allow_pickle=False)
        np.save(var_path, scaler.var_, allow_pickle=False)
    elif isinstance(scaler, MinMaxScaler):
        logger.info(f"Converting {input_file} min/max npy files")
        min_path = out_dir / (input_file.stem + "_min.npy")
        scale_path = out_dir / (input_file.stem + "_scale.npy")

        np.save(min_path, scaler.min_, allow_pickle=False)
        np.save(scale_path, scaler.scale_, allow_pickle=False)
    else:
        raise ValueError(f"Unknown scaler type: {type(scaler)}")


def _save_checkpoint(input_file, output_file, logger):
    checkpoint = torch.load(
        input_file, map_location=torch.device("cpu")  # pylint: disable='no-member'
    )
    size = os.path.getsize(input_file)
    logger.info(f"Processisng: {input_file}")
    logger.info(f"File size (before): {size / 1024/1024:.3f} MB")
    for k in ["optimizer_state", "lr_scheduler_state"]:
        if k in checkpoint.keys():
            del checkpoint[k]

    # For https://github.com/kan-bayashi/ParallelWaveGAN
    for k in ["optimizer", "lr_scheduler"]:
        if k in checkpoint.keys():
            del checkpoint[k]
    if "model" in checkpoint and "discriminator" in checkpoint["model"]:
        del checkpoint["model"]["discriminator"]

    torch.save(checkpoint, output_file)
    size = os.path.getsize(output_file)
    logger.info(f"File size (after): {size / 1024/1024:.3f} MB")


def main(enunu_dir, out_dir, verbose=100):
    """Run the main function

    NOTE: This function is used by https://github.com/oatsu-gh/SimpleEnunu.
    So we need to be careful about the changes.
    It would be probably better to move this functionality to under the nnsvs
    directory.
    """
    logger = getLogger(verbose=verbose)
    enunu_dir = Path(enunu_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    enuconfig = OmegaConf.load(enunu_dir / "enuconfig.yaml")

    # Hed
    qst_path = enunu_dir / enuconfig.question_path
    shutil.copyfile(qst_path, out_dir / "qst.hed")
    # Table
    table_path = enunu_dir / enuconfig.table_path
    shutil.copyfile(table_path, out_dir / "kana2phonemes.table")

    # Models
    model_dir = enunu_dir / enuconfig.model_dir
    assert model_dir.exists()
    for typ in ["timelag", "duration", "acoustic"]:
        model_config = model_dir / typ / "model.yaml"
        assert model_config.exists()
        checkpoint = model_dir / typ / enuconfig[typ]["checkpoint"]
        assert checkpoint.exists()

        shutil.copyfile(model_config, out_dir / f"{typ}_model.yaml")
        _save_checkpoint(checkpoint, out_dir / f"{typ}_model.pth", logger)

        for inout in ["in", "out"]:
            scaler_path = (
                enunu_dir / enuconfig.stats_dir / f"{inout}_{typ}_scaler.joblib"
            )
            _scaler2numpy(scaler_path, out_dir, logger)

    # Config
    s = f"""# Global configs
sample_rate: {enuconfig.sample_rate}
frame_period: 5
log_f0_conditioning: {enuconfig.log_f0_conditioning}
use_world_codec: false

# Model-specific synthesis configs
timelag:
    allowed_range: {enuconfig.timelag.allowed_range}
    allowed_range_rest: {enuconfig.timelag.allowed_range_rest}
    force_clip_input_features: true
duration:
    force_clip_input_features: true
acoustic:
    subphone_features: "coarse_coding"
    force_clip_input_features: true
    relative_f0: {enuconfig.acoustic.relative_f0}
"""
    with open(out_dir / "config.yaml", "w") as f:
        f.write(s)

    logger.info(f"Contents of config.yaml: \n{s}")
    logger.warning(
        """Assuming `use_world_codec: false` since the most of released ENUNU models
were trained with `use_world_codec: false`.
If you use the default feature extarction settings in newer NNSVS (> 0.0.3),
please set `use_world_codec: true` in the config.yaml
`use_world_codec` must be the same during the feature extraction and synthesis time.
"""
    )


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    main(args.enunu_dir, args.out_dir, args.verbose)
