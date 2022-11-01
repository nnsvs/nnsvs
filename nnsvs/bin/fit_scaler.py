from os.path import exists

import hydra
import joblib
import numpy as np
from hydra.utils import to_absolute_path
from nnsvs.logger import getLogger
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@hydra.main(config_path="conf/fit_scaler", config_name="config")
def my_app(config: DictConfig) -> None:
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    list_path = to_absolute_path(config.list_path)
    out_path = to_absolute_path(config.out_path)

    if config.external_scaler is not None:
        scaler = joblib.load(config.external_scaler)
    else:
        scaler = hydra.utils.instantiate(config.scaler)
    with open(list_path) as f:
        for path in f:
            path = to_absolute_path(path.strip())
            c = np.load(path)
            if "out_acoustic/" in path:
                assert "org" in path
                in_path = path.replace("out_acoustic/", "in_acoustic/")
                in_feats = np.load(in_path)
                note_frame_indices = in_feats[:, config.in_rest_idx] <= 0
                assert exists(in_path)
                # Removed non-voice segments to compute statistics
                c = c[note_frame_indices]
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
