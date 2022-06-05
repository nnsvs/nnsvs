import argparse
import sys

import joblib
import numpy as np
from nnsvs.multistream import get_static_features
from nnsvs.util import StandardScaler
from omegaconf import OmegaConf


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract static scaler from static+dynamic scaler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", type=str, help="input file")
    parser.add_argument("model_config", type=str, help="model config")
    parser.add_argument("output_file", type=str, help="output file")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    model_config = OmegaConf.load(args.model_config)

    out_scaler = joblib.load(args.input_file)

    mean_ = get_static_features(
        out_scaler.mean_.reshape(1, 1, out_scaler.mean_.shape[-1]),
        model_config.num_windows,
        model_config.stream_sizes,
        model_config.has_dynamic_features,
    )
    mean_ = np.concatenate(mean_, -1).reshape(1, -1)
    var_ = get_static_features(
        out_scaler.var_.reshape(1, 1, out_scaler.var_.shape[-1]),
        model_config.num_windows,
        model_config.stream_sizes,
        model_config.has_dynamic_features,
    )
    var_ = np.concatenate(var_, -1).reshape(1, -1)
    scale_ = get_static_features(
        out_scaler.scale_.reshape(1, 1, out_scaler.scale_.shape[-1]),
        model_config.num_windows,
        model_config.stream_sizes,
        model_config.has_dynamic_features,
    )
    scale_ = np.concatenate(scale_, -1).reshape(1, -1)
    static_scaler = StandardScaler(mean_, var_, scale_)

    joblib.dump(static_scaler, args.output_file)
