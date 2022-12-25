"""Save npy-format scalers for vocoders

NOTE: The input must be out_acoutic_scaler.joblib that is used for normalizing
the vocoder's input features
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from nnsvs.multistream import get_static_features
from nnsvs.util import get_world_stream_info
from sklearn.preprocessing import StandardScaler


def get_parser():
    parser = argparse.ArgumentParser(description="joblib scaler to npy files")
    parser.add_argument("input_file", type=str, help="input file")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--sample_rate", type=int, default=48000, help="sampling rate")
    parser.add_argument(
        "--feature_type", type=str, default="world", help="world or melf0"
    )
    parser.add_argument("--mgc_order", type=int, default=59, help="mgc order")
    parser.add_argument("--num_windows", type=int, default=3, help="number of windows")
    parser.add_argument("--vibrato_mode", type=str, default="none", help="vibrato mode")
    parser.add_argument(
        "--use_mcep_aperiodicity",
        action="store_true",
        help="use mcep-based aperiodicity",
    )
    parser.add_argument(
        "--mcep_aperiodicity_order",
        type=int,
        default=24,
        help="order of mcep-based aperiodicity",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    input_file = Path(args.input_file)
    scaler = joblib.load(input_file)
    assert isinstance(scaler, StandardScaler)
    feature_type = args.feature_type

    if input_file.stem != "out_acoustic_scaler":
        raise ValueError(
            f"Expected input_file.stem to be 'out_acoustic_scaler', got {input_file.stem}"
        )

    # NOTE: The output files are supposed to be used for normalizing
    # the vocoder's input features.
    out_file_name = "in_vocoder_scaler"

    mean = scaler.mean_
    scale = scaler.scale_
    var = scaler.var_

    if feature_type == "melf0":
        stream_sizes = [len(mean) - 2, 1, 1]
    else:
        stream_sizes = get_world_stream_info(
            args.sample_rate,
            args.mgc_order,
            args.num_windows,
            args.vibrato_mode,
            use_mcep_aperiodicity=args.use_mcep_aperiodicity,
            mcep_aperiodicity_order=args.mcep_aperiodicity_order,
        )
        assert len(mean) == sum(stream_sizes)

    has_dynamic_features = [False] * len(stream_sizes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {input_file} mean/scale npy files")
    lf0_params = {}
    for name, in_feats in [("mean", mean), ("scale", scale), ("var", var)]:
        streams = get_static_features(
            in_feats.reshape(1, -1, in_feats.shape[-1]),
            args.num_windows,
            stream_sizes,
            has_dynamic_features,
        )

        # NOTE: use up to 4 streams
        # [mgc, lf0, bap, vuv]
        # or
        # (mel, lf0, vuv)
        streams = list(map(lambda x: x.reshape(-1), streams))[:4]
        lf0_params[name] = float(streams[1])
        out_feats = np.concatenate(streams)

        print(f"[{name}] dim: {in_feats.shape} -> {out_feats.shape}")
        out_path = out_dir / (out_file_name + f"_{name}.npy")
        np.save(out_path, out_feats, allow_pickle=False)

    print(
        f"""
If you are going to train NSF-based vocoders, please set the following parameters:

out_lf0_mean: {lf0_params["mean"]}
out_lf0_scale: {lf0_params["scale"]}

NOTE: If you are using the same data for training acoustic/vocoder models, the F0 statistics
for those models should be the same. If you are using different data for training
acoustic/vocoder models (e.g., training a vocoder model on a multiple DBs),
you will likely need to set different F0 statistics for acoustic/vocoder models.

If you use uSFGAN/SiFI-GAN, it is not necessary to manually set the F0 statistics."""
    )
