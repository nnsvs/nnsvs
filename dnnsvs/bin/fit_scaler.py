# coding: utf-8

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dnnsvs.logger import getLogger
logger = None

@hydra.main(config_path="conf/fit_scaler/config.yaml")
def my_app(config : DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(config.pretty())

    list_path = to_absolute_path(config.list_path)
    out_path = to_absolute_path(config.out_path)

    scaler = hydra.utils.instantiate(config.scaler)
    with open(list_path) as f:
        for path in f:
            c = np.load(to_absolute_path(path.strip()))
            scaler.partial_fit(c)
        joblib.dump(scaler, out_path)

    if config.verbose > 0:
        if isinstance(scaler, StandardScaler):
            logger.info("mean:\n{}".format(scaler.mean_))
            logger.info("std:\n{}".format(np.sqrt(scaler.var_)))
        if isinstance(scaler, MinMaxScaler):
            logger.info("data min:\n{}".format(scaler.data_min_))
            logger.info("data max:\n{}".format(np.sqrt(scaler.data_max_)))


def entry():
    my_app()


if __name__ == "__main__":
    my_app()