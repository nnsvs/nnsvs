"""Prepare input features for training neural vocoders
"""
import os
from concurrent.futures import ProcessPoolExecutor
from os.path import exists, islink, join

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from nnsvs.logger import getLogger
from nnsvs.multistream import get_static_features
from nnsvs.util import get_world_stream_info, load_utt_list
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def _prepare_voc_features(
    in_dir,
    out_dir,
    utt_id,
    num_windows,
    stream_sizes,
    has_dynamic_features,
) -> None:
    feats = np.load(join(in_dir, utt_id + "-feats.npy"))
    in_wave_path = join(in_dir, utt_id + "-wave.npy")
    assert exists(in_wave_path)

    streams = get_static_features(
        feats.reshape(1, -1, feats.shape[-1]),
        num_windows,
        stream_sizes,
        has_dynamic_features,
    )

    # remove batch-axis
    streams = list(map(lambda x: x.squeeze(0), streams))

    # NOTE: even if the number of streams are larger than 4, we only use the first 4 streams
    # for training neural vocoders
    if len(streams) >= 4:
        mgc, lf0, vuv, bap = streams[0], streams[1], streams[2], streams[3]
        voc_feats = np.hstack((mgc, lf0, vuv, bap)).astype(np.float32)
    elif len(streams) == 3:
        mel, lf0, vuv = streams[0], streams[1], streams[2]
        voc_feats = np.hstack((mel, lf0, vuv)).astype(np.float32)

    voc_feats_path = join(out_dir, utt_id + "-feats.npy")
    np.save(voc_feats_path, voc_feats, allow_pickle=False)

    # NOTE: To train vocoders with https://github.com/kan-bayashi/ParallelWaveGAN
    # target waveform needs to be created in the same directory as the vocoder input features.
    save_wave_path = join(out_dir, utt_id + "-wave.npy")
    if (not exists(save_wave_path)) and (not islink(save_wave_path)):
        os.symlink(join(in_dir, utt_id + "-wave.npy"), save_wave_path)


@hydra.main(config_path="conf/prepare_static_features", config_name="config")
def my_app(config: DictConfig) -> None:
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    utt_list = to_absolute_path(config.utt_list)
    in_dir = to_absolute_path(config.in_dir)
    out_dir = to_absolute_path(config.out_dir)

    utt_ids = load_utt_list(utt_list)

    if config.acoustic.relative_f0:
        raise ValueError("Relative F0 is not supported")

    if config.acoustic.feature_type == "world":
        stream_sizes = get_world_stream_info(
            config.acoustic.sample_rate,
            config.acoustic.mgc_order,
            config.acoustic.num_windows,
            config.acoustic.vibrato_mode,
            use_mcep_aperiodicity=config.acoustic.use_mcep_aperiodicity,
            mcep_aperiodicity_order=config.acoustic.mcep_aperiodicity_order,
        )
    elif config.acoustic.feature_type == "melf0":
        stream_sizes = [config.acoustic.num_mels, 1, 1]
    else:
        raise ValueError(f"Unknown feature type: {config.acoustic.feature_type}")

    os.makedirs(out_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(
                _prepare_voc_features,
                in_dir,
                out_dir,
                utt_id,
                config.acoustic.num_windows,
                stream_sizes,
                config.acoustic.dynamic_features_flags,
            )
            for utt_id in utt_ids
        ]
        for future in tqdm(futures):
            future.result()


def entry():
    my_app()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    my_app()  # pylint: disable=no-value-for-parameter
