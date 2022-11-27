# coding: utf-8

import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob
from itertools import zip_longest
from os.path import basename, exists, join, splitext

import hydra
import joblib
import numpy as np
from hydra.utils import to_absolute_path
from nnsvs.logger import getLogger
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logger = None


def get_paths_by_glob(in_dir, filt):
    return glob(join(in_dir, filt))


def _process_utterance(
    out_dir, audio_path, feat_path, scaler, inverse, clipping_threshold
):
    # [Optional] copy audio with the same name if exists
    if audio_path is not None and exists(audio_path):
        name = splitext(basename(audio_path))[0]
        np.save(join(out_dir, name), np.load(audio_path), allow_pickle=False)

    # [Required] apply normalization for features
    assert exists(feat_path)
    x = np.load(feat_path)
    if inverse:
        y = scaler.inverse_transform(x)
    else:
        y = scaler.transform(x)

    # NOTE: workaround for outliers
    # may need to adjust the clipping value
    if isinstance(scaler, StandardScaler):
        y = np.clip(y, -clipping_threshold, clipping_threshold)

    assert x.dtype == y.dtype
    name = splitext(basename(feat_path))[0]
    np.save(join(out_dir, name), y, allow_pickle=False)


def apply_normalization_dir2dir(
    in_dir, out_dir, scaler, inverse, num_workers, clipping_threshold
):
    # NOTE: at this point, audio_paths can be empty
    audio_paths = get_paths_by_glob(in_dir, "*-wave.npy")
    feature_paths = get_paths_by_glob(in_dir, "*-feats.npy")
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for audio_path, feature_path in zip_longest(audio_paths, feature_paths):
        futures.append(
            executor.submit(
                partial(
                    _process_utterance,
                    out_dir,
                    audio_path,
                    feature_path,
                    scaler,
                    inverse,
                    clipping_threshold,
                )
            )
        )
    for future in tqdm(futures):
        future.result()


@hydra.main(config_path="conf/preprocess_normalize", config_name="config")
def my_app(config: DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    in_dir = to_absolute_path(config.in_dir)
    out_dir = to_absolute_path(config.out_dir)
    scaler_path = to_absolute_path(config.scaler_path)
    scaler = joblib.load(scaler_path)
    inverse = config.inverse
    num_workers = config.num_workers
    clipping_threshold = config.clipping_threshold

    os.makedirs(out_dir, exist_ok=True)
    apply_normalization_dir2dir(
        in_dir, out_dir, scaler, inverse, num_workers, clipping_threshold
    )


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
