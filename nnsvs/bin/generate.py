# coding: utf-8

import os
from os.path import basename, join

import hydra
import joblib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from nnmnkwii.datasets import FileSourceDataset
from nnsvs.base import PredictionType
from nnsvs.logger import getLogger
from nnsvs.multistream import get_windows, multi_stream_mlpg
from nnsvs.train_util import NpyFileSource
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logger = None

use_cuda = torch.cuda.is_available()


@hydra.main(config_path="conf/generate", config_name="config")
def my_app(config: DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    device = torch.device("cuda" if use_cuda else "cpu")
    in_dir = to_absolute_path(config.in_dir)
    out_dir = to_absolute_path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model_config = OmegaConf.load(to_absolute_path(config.model.model_yaml))
    model = hydra.utils.instantiate(model_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.model.checkpoint),
        map_location=lambda storage, loc: storage,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scaler = joblib.load(to_absolute_path(config.out_scaler_path))

    in_feats = FileSourceDataset(NpyFileSource(in_dir, logger))

    with torch.no_grad():
        for idx in tqdm(range(len(in_feats))):
            feats = torch.from_numpy(in_feats[idx]).unsqueeze(0).to(device)

            if model.prediction_type() == PredictionType.PROBABILISTIC:

                max_mu, max_sigma = model.inference(feats, [feats.shape[1]])

                if np.any(model_config.has_dynamic_features):
                    # Apply denormalization
                    # (B, T, D_out) -> (T, D_out)
                    max_sigma_sq = (
                        max_sigma.squeeze(0).cpu().data.numpy() ** 2 * scaler.var_
                    )
                    max_mu = scaler.inverse_transform(
                        max_mu.squeeze(0).cpu().data.numpy()
                    )
                    # Apply MLPG
                    # (T, D_out) -> (T, static_dim)
                    out = multi_stream_mlpg(
                        max_mu,
                        max_sigma_sq,
                        get_windows(model_config.num_windows),
                        model_config.stream_sizes,
                        model_config.has_dynamic_features,
                    )

                else:
                    # (T, D_out)
                    out = max_mu.squeeze(0).cpu().data.numpy()
                    out = scaler.inverse_transform(out)
            else:
                out = (
                    model.inference(feats, [feats.shape[1]])
                    .squeeze(0)
                    .cpu()
                    .data.numpy()
                )
                out = scaler.inverse_transform(out)

                # Apply MLPG if necessary
                if np.any(model_config.has_dynamic_features):
                    out = multi_stream_mlpg(
                        out,
                        scaler.var_,
                        get_windows(model_config.num_windows),
                        model_config.stream_sizes,
                        model_config.has_dynamic_features,
                    )

            name = basename(in_feats.collected_files[idx][0])
            out_path = join(out_dir, name)
            np.save(out_path, out, allow_pickle=False)


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
