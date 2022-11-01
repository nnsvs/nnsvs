"""Prepare static features from static + dynamic features
"""
import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join

import hydra
import numpy as np
import pyworld
from hydra.utils import to_absolute_path
from nnsvs.logger import getLogger
from nnsvs.multistream import get_static_features
from nnsvs.util import get_world_stream_info, load_utt_list
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def _extract_static_features(
    in_dir,
    out_dir,
    utt_id,
    num_windows,
    stream_sizes,
    has_dynamic_features,
    mgc2sp=False,
    sample_rate=48000,
) -> None:
    feats = np.load(join(in_dir, utt_id + "-feats.npy"))

    assert np.any(has_dynamic_features)
    streams = get_static_features(
        feats.reshape(1, -1, feats.shape[-1]),
        num_windows,
        stream_sizes,
        has_dynamic_features,
    )

    # remove batch-axis
    streams = list(map(lambda x: x.squeeze(0), streams))

    # Convert mgc2sp
    if mgc2sp:
        mgc = streams[0]
        fft_size = pyworld.get_cheaptrick_fft_size(sample_rate)
        sp = np.log(
            pyworld.decode_spectral_envelope(
                mgc.astype(np.float64), sample_rate, fft_size
            ).astype(np.float32)
        )
        streams[0] = sp

    static_feats = np.concatenate(streams, axis=-1).astype(np.float32)

    static_path = join(out_dir, utt_id + "-feats.npy")
    np.save(static_path, static_feats, allow_pickle=False)


@hydra.main(config_path="conf/prepare_static_features", config_name="config")
def my_app(config: DictConfig) -> None:
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    utt_list = to_absolute_path(config.utt_list)
    in_dir = to_absolute_path(config.in_dir)
    out_dir = to_absolute_path(config.out_dir)

    utt_ids = load_utt_list(utt_list)

    stream_sizes, has_dynamic_features = get_world_stream_info(
        config.acoustic.sample_rate,
        config.acoustic.mgc_order,
        config.acoustic.num_windows,
        config.acoustic.vibrato_mode,
    )

    os.makedirs(out_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(
                _extract_static_features,
                in_dir,
                out_dir,
                utt_id,
                config.acoustic.num_windows,
                stream_sizes,
                has_dynamic_features,
                mgc2sp=config.mgc2sp,
                sample_rate=config.sample_rate,
            )
            for utt_id in utt_ids
        ]
        for future in tqdm(futures):
            future.result()


def entry():
    my_app()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    my_app()  # pylint: disable=no-value-for-parameter
