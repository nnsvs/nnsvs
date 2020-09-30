# coding: utf-8

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import numpy as np
import joblib
from tqdm import tqdm
from os.path import basename, splitext, exists, join
import os
import torch
from torch import nn
from torch.nn import functional as F
from nnmnkwii.datasets import FileSourceDataset

from nnsvs.gen import get_windows
from nnsvs.multistream import multi_stream_mlpg
from nnsvs.bin.train import NpyFileSource
from nnsvs.logger import getLogger
from nnsvs.mdn import mdn_sample

logger = None

use_cuda = torch.cuda.is_available()


@hydra.main(config_path="conf/generate/config.yaml")
def my_app(config : DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(config.pretty())

    device = torch.device("cuda" if use_cuda else "cpu")
    in_dir = to_absolute_path(config.in_dir)
    out_dir = to_absolute_path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model_config = OmegaConf.load(to_absolute_path(config.model.model_yaml))
    model = hydra.utils.instantiate(model_config.netG).to(device)
    checkpoint = torch.load(to_absolute_path(config.model.checkpoint),
        map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    scaler = joblib.load(to_absolute_path(config.out_scaler_path))

    in_feats = FileSourceDataset(NpyFileSource(in_dir))

    with torch.no_grad():
        for idx in tqdm(range(len(in_feats))):
            feats = torch.from_numpy(in_feats[idx]).unsqueeze(0).to(device)
            if model.prediction_type == "probabilistic":
                pi, sigma, mu = model(feats, [feats.shape[1]])
                out = mdn_sample(pi, sigma, mu).squeeze(0).cpu().data.numpy()
            else:
                out = model(feats, [feats.shape[1]]).squeeze(0).cpu().data.numpy()

            out = scaler.inverse_transform(out)

            # Apply MLPG if necessary
            if np.any(model_config.has_dynamic_features):
                windows = get_windows(3)
                out = multi_stream_mlpg(
                    out, scaler.var_, windows, model_config.stream_sizes,
                    model_config.has_dynamic_features)

            name = basename(in_feats.collected_files[idx][0])
            out_path = join(out_dir, name)
            np.save(out_path, out, allow_pickle=False)


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
