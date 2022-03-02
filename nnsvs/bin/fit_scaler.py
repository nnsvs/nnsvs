# coding: utf-8

import hydra
import joblib
import numpy as np
from hydra.utils import to_absolute_path
from nnsvs.logger import getLogger
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = None


@hydra.main(config_path="conf/fit_scaler", config_name="config")
def my_app(config: DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

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
            logger.info("mean:\n%s", scaler.mean_)
            logger.info("std:\n%s", np.sqrt(scaler.var_))
        if isinstance(scaler, MinMaxScaler):
            logger.info("data min:\n%s", scaler.data_min_)
            logger.info("data max:\n%s", scaler.data_max_)


def entry():
    my_app()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    my_app()  # pylint: disable=no-value-for-parameter
