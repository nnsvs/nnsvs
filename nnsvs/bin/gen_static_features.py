import os
from os.path import exists, join

import hydra
import joblib
import numpy as np
import pyworld
import torch
from hydra.utils import to_absolute_path
from nnsvs.acoustic_models.util import pad_inference
from nnsvs.base import PredictionType
from nnsvs.gen import get_windows
from nnsvs.logger import getLogger
from nnsvs.mdn import mdn_get_most_probable_sigma_and_mu
from nnsvs.multistream import (
    get_static_features,
    get_static_stream_sizes,
    multi_stream_mlpg,
    split_streams,
)
from nnsvs.postfilters import variance_scaling
from nnsvs.util import StandardScaler, load_utt_list
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logger = None

use_cuda = torch.cuda.is_available()


@torch.no_grad()
def _gen_static_features(
    model, model_config, in_feats, gt_feats, out_scaler, gta=False
):
    if model.prediction_type() == PredictionType.PROBABILISTIC:
        if gta:
            raise ValueError("GTA not supported for probabilistic models for now")
        else:
            max_mu, max_sigma = model.inference(in_feats, [in_feats.shape[1]])

        if np.any(model_config.has_dynamic_features):
            # Apply denormalization
            # (B, T, D_out) -> (T, D_out)
            max_sigma_sq = (
                max_sigma.squeeze(0).cpu().data.numpy() ** 2 * out_scaler.var_
            )
            max_mu = out_scaler.inverse_transform(max_mu.squeeze(0).cpu().data.numpy())
            # Apply MLPG
            # (T, D_out) -> (T, static_dim)
            out_feats = multi_stream_mlpg(
                max_mu,
                max_sigma_sq,
                get_windows(model_config.num_windows),
                model_config.stream_sizes,
                model_config.has_dynamic_features,
            )
        else:
            # (T, D_out)
            out_feats = max_mu.squeeze(0).cpu().data.numpy()
            out_feats = out_scaler.inverse_transform(out_feats)
    else:
        if gta:
            if hasattr(model, "reduction_factor"):
                out_feats = pad_inference(
                    model,
                    in_feats,
                    [in_feats.shape[1]],
                    model.reduction_factor,
                    mode="replicate",
                    y=gt_feats,
                    mdn=model.prediction_type() in [PredictionType.PROBABILISTIC, 3],
                )
                if model.prediction_type() == PredictionType.PROBABILISTIC:
                    # (sigma, mu) -> (mu)
                    out_feats = mdn_get_most_probable_sigma_and_mu(out_feats)[1]
                elif model.prediction_type() == 3:
                    # (mu, sigma) -> (mu)
                    out_feats = out_feats[0]
            else:
                out_feats = model(in_feats, [in_feats.shape[1]], gt_feats)
                # out + residual case
                out_feats = out_feats[0] if isinstance(out_feats, tuple) else out_feats
                # multiple outputs case
                out_feats = out_feats[-1] if isinstance(out_feats, list) else out_feats
            out_feats = out_feats.squeeze(0).cpu().data.numpy()
        else:
            out_feats = (
                model.inference(in_feats, [in_feats.shape[1]])
                .squeeze(0)
                .cpu()
                .data.numpy()
            )
        out_feats = out_scaler.inverse_transform(out_feats)

        # Apply MLPG if necessary
        if np.any(model_config.has_dynamic_features):
            out_feats = multi_stream_mlpg(
                out_feats,
                out_scaler.var_,
                get_windows(model_config.num_windows),
                model_config.stream_sizes,
                model_config.has_dynamic_features,
            )

    return out_feats.astype(np.float32)


@hydra.main(config_path="conf/gen_static_features", config_name="config")
def my_app(config: DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    device = torch.device("cuda" if use_cuda else "cpu")
    utt_list = to_absolute_path(config.utt_list)
    in_dir = to_absolute_path(config.in_dir)
    gt_dir = to_absolute_path(config.gt_dir)
    out_dir = to_absolute_path(config.out_dir)

    utt_ids = load_utt_list(utt_list)

    os.makedirs(out_dir, exist_ok=True)

    model_config = OmegaConf.load(to_absolute_path(config.model.model_yaml))
    model = hydra.utils.instantiate(model_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.model.checkpoint),
        map_location=lambda storage, loc: storage,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    out_scaler = joblib.load(to_absolute_path(config.out_scaler_path))

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

    static_stream_sizes = get_static_stream_sizes(
        model_config.stream_sizes,
        model_config.has_dynamic_features,
        model_config.num_windows,
    )

    for utt_id in tqdm(utt_ids):
        if not exists(join(gt_dir, utt_id + "-feats.npy")):
            print(f"Skip {utt_id}")
            continue
        in_feats = (
            torch.from_numpy(np.load(join(in_dir, utt_id + "-feats.npy")))
            .unsqueeze(0)
            .to(device)
        )
        gt_feats = (
            torch.from_numpy(np.load(join(gt_dir, utt_id + "-feats.npy")))
            .unsqueeze(0)
            .to(device)
        )
        static_feats = _gen_static_features(
            model, model_config, in_feats, gt_feats, out_scaler, config.gta
        )
        outs = split_streams(static_feats, static_stream_sizes)

        if len(outs) == 4:
            mgc, lf0, vuv, bap = outs

            mgc_end_dim = static_stream_sizes[0]
            bap_start_dim = sum(static_stream_sizes[:3])
            bap_end_dim = sum(static_stream_sizes[:4])

            if config.gv_postfilter:
                # mgc
                mgc = variance_scaling(
                    static_scaler.var_.reshape(-1)[:mgc_end_dim],
                    mgc,
                    offset=config.mgc_offset,
                )
                # bap
                bap = variance_scaling(
                    static_scaler.var_.reshape(-1)[bap_start_dim:bap_end_dim],
                    bap,
                    offset=config.bap_offset,
                )

            if config.mgc2sp:
                # 24k: 1024
                # 48: 2048
                fft_size = pyworld.get_cheaptrick_fft_size(config.sample_rate)
                sp = np.log(
                    pyworld.decode_spectral_envelope(
                        mgc.astype(np.float64), config.sample_rate, fft_size
                    ).astype(np.float32)
                )
                static_feats = np.concatenate([sp, lf0, vuv, bap], axis=-1)
            else:
                static_feats = np.concatenate([mgc, lf0, vuv, bap], axis=-1)
        elif len(outs) == 3:
            mel, lf0, vuv = outs
            # NOTE: may add postfilter here
            static_feats = np.concatenate([mel, lf0, vuv], axis=-1)
        else:
            raise ValueError("Invalid number of streams")

        gt_feats = gt_feats.squeeze(0)
        if len(static_feats) != len(gt_feats):
            print(static_feats.shape, gt_feats.shape)
            raise RuntimeError(f"Length mismatch in {utt_id}")

        if config.normalize:
            assert not config.mgc2sp, "need to compute normalization stats"
            static_feats = static_scaler.transform(static_feats)
        out_path = join(out_dir, f"{utt_id}-feats.npy")
        np.save(out_path, static_feats.astype(np.float32), allow_pickle=False)


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
