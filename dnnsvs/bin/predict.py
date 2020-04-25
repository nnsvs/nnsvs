# coding: utf-8

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import numpy as np
from glob import glob
from tqdm import tqdm
from os.path import basename, splitext, exists, join
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from dnnsvs.logger import getLogger
from dnnsvs.bin.train import NpyFileSource

logger = None

use_cuda = torch.cuda.is_available()


@hydra.main(config_path="conf/predict/config.yaml")
def my_app(config : DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(config.pretty())

    device = torch.device("cuda" if use_cuda else "cpu")
    in_dir = to_absolute_path(config.in_dir)
    out_dir = to_absolute_path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model_def = OmegaConf.load(to_absolute_path(config.model.model_yaml))
    model = hydra.utils.instantiate(model_def).to(device)

    in_feats = FileSourceDataset(NpyFileSource(in_dir))

    with torch.no_grad():
        for idx in tqdm(range(len(in_feats))):
            feats = torch.from_numpy(in_feats[idx]).unsqueeze(0)
            out = model(feats, [feats.shape[1]]).squeeze(0)
            name = basename(in_feats.collected_files[idx][0])
            out_path = join(out_dir, name)
            np.save(out_path, out.data.numpy(), allow_pickle=False)


def entry():
    my_app()


if __name__ == "__main__":
    my_app()