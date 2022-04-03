import os
from concurrent.futures import ProcessPoolExecutor
from os.path import exists, join

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from nnsvs.logger import getLogger
from nnsvs.multistream import get_static_features
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def load_utt_list(utt_list):
    with open(utt_list) as f:
        utt_ids = f.readlines()
    utt_ids = map(lambda utt_id: utt_id.strip(), utt_ids)
    utt_ids = filter(lambda utt_id: len(utt_id) > 0, utt_ids)
    return list(utt_ids)


def _gen_static_features(
    in_dir,
    out_dir,
    utt_id,
    config,
) -> None:
    feats = np.load(join(in_dir, utt_id + "-feats.npy"))
    in_wave_path = join(in_dir, utt_id + "-wave.npy")
    assert exists(in_wave_path)

    assert np.any(config.has_dynamic_features)
    streams = get_static_features(
        feats.reshape(1, -1, feats.shape[-1]),
        config.num_windows,
        config.stream_sizes,
        config.has_dynamic_features,
    )

    # remove batch-axis
    streams = list(map(lambda x: x.squeeze(0), streams))

    assert len(streams) >= 4
    mgc, lf0, vuv, bap = streams[0], streams[1], streams[2], streams[3]

    if config.relative_f0:
        raise ValueError("Relative lf0 is not supported")

    static_feats = np.hstack((mgc, lf0, vuv, bap)).astype(np.float32)

    static_path = join(out_dir, utt_id + "-feats.npy")
    np.save(static_path, static_feats, allow_pickle=False)
    save_wave_path = join(out_dir, utt_id + "-wave.npy")
    if not exists(save_wave_path):
        os.symlink(join(in_dir, utt_id + "-wave.npy"), save_wave_path)


@hydra.main(config_path="conf/gen_static_features", config_name="config")
def my_app(config: DictConfig) -> None:
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    utt_list = to_absolute_path(config.utt_list)
    in_dir = to_absolute_path(config.in_dir)
    out_dir = to_absolute_path(config.out_dir)

    utt_ids = load_utt_list(utt_list)

    os.makedirs(out_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(
                _gen_static_features,
                in_dir,
                out_dir,
                utt_id,
                config.acoustic,
            )
            for utt_id in utt_ids
        ]
        for future in tqdm(futures):
            future.result()


def entry():
    my_app()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    my_app()  # pylint: disable=no-value-for-parameter
