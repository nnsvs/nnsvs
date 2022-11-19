# coding: utf-8
import os
from concurrent.futures import ProcessPoolExecutor
from os.path import basename, join, splitext

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from nnmnkwii.datasets import FileSourceDataset
from nnsvs.data import (
    DurationFeatureSource,
    MelF0AcousticSource,
    MusicalLinguisticSource,
    TimeLagFeatureSource,
    WORLDAcousticSource,
)
from nnsvs.logger import getLogger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logger = None


def _prepare_timelag_feature(
    in_timelag_root,
    out_timelag_root,
    in_timelag: FileSourceDataset,
    out_timelag: FileSourceDataset,
    idx: int,
) -> None:
    """prepare timelag feature for one item of in_duration"""
    x, y = in_timelag[idx], out_timelag[idx]
    name = splitext(basename(in_timelag.collected_files[idx][0]))[0]
    xpath = join(in_timelag_root, name + "-feats.npy")
    ypath = join(out_timelag_root, name + "-feats.npy")
    np.save(xpath, x, allow_pickle=False)
    np.save(ypath, y, allow_pickle=False)


def _prepare_duration_feature(
    in_duration_root,
    out_duration_root,
    in_duration: FileSourceDataset,
    out_duration: FileSourceDataset,
    idx: int,
) -> None:
    """prepare duration feature for one item of in_duration"""
    x, y = in_duration[idx], out_duration[idx]
    name = splitext(basename(in_duration.collected_files[idx][0]))[0]
    xpath = join(in_duration_root, name + "-feats.npy")
    ypath = join(out_duration_root, name + "-feats.npy")
    np.save(xpath, x, allow_pickle=False)
    np.save(ypath, y, allow_pickle=False)


def _prepare_acoustic_feature(
    in_acoustic_root,
    out_acoustic_root,
    out_postfilter_root,
    in_acoustic: FileSourceDataset,
    out_acoustic: FileSourceDataset,
    idx: int,
) -> None:
    """prepare acoustic feature for one item of in_acoustic"""
    x, (y, wave, y_pf) = in_acoustic[idx], out_acoustic[idx]
    name = splitext(basename(in_acoustic.collected_files[idx][0]))[0]
    if y is None:
        print(f"{name} is skipped")
        return
    xpath = join(in_acoustic_root, name + "-feats.npy")
    ypath = join(out_acoustic_root, name + "-feats.npy")
    wpath = join(out_acoustic_root, name + "-wave.npy")
    pfpath = join(out_postfilter_root, name + "-feats.npy")
    np.save(xpath, x, allow_pickle=False)
    np.save(ypath, y, allow_pickle=False)
    np.save(wpath, wave, allow_pickle=False)
    np.save(pfpath, y_pf, allow_pickle=False)


@hydra.main(config_path="conf/prepare_features", config_name="config")
def my_app(config: DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    utt_list = to_absolute_path(config.utt_list)
    out_dir = to_absolute_path(config.out_dir)
    question_path_general = to_absolute_path(config.question_path)

    # Time-lag model
    # in: musical/linguistic context
    # out: time-lag (i.e. onset time deviation)
    if config.timelag.question_path is not None:
        question_path = config.timelag.question_path
    else:
        question_path = question_path_general

    in_timelag_source = MusicalLinguisticSource(
        utt_list,
        to_absolute_path(config.timelag.label_phone_score_dir),
        add_frame_features=False,
        subphone_features=None,
        question_path=question_path,
        log_f0_conditioning=config.log_f0_conditioning,
    )
    out_timelag_source = TimeLagFeatureSource(
        utt_list,
        to_absolute_path(config.timelag.label_phone_score_dir),
        to_absolute_path(config.timelag.label_phone_align_dir),
    )

    in_timelag = FileSourceDataset(in_timelag_source)
    out_timelag = FileSourceDataset(out_timelag_source)

    # Duration model
    # in: musical/linguistic context
    # out: duration
    if config.duration.question_path is not None:
        question_path = config.duration.question_path
    else:
        question_path = question_path_general

    in_duration_source = MusicalLinguisticSource(
        utt_list,
        to_absolute_path(config.duration.label_dir),
        add_frame_features=False,
        subphone_features=None,
        question_path=question_path,
        log_f0_conditioning=config.log_f0_conditioning,
    )
    out_duration_source = DurationFeatureSource(
        utt_list, to_absolute_path(config.duration.label_dir)
    )

    in_duration = FileSourceDataset(in_duration_source)
    out_duration = FileSourceDataset(out_duration_source)

    # Acoustic model
    # in: musical/linguistic context
    # out: acoustic features
    if config.acoustic.question_path is not None:
        question_path = config.acoustic.question_path
    else:
        question_path = question_path_general
    in_acoustic_source = MusicalLinguisticSource(
        utt_list,
        to_absolute_path(config.acoustic.label_dir),
        question_path,
        add_frame_features=True,
        subphone_features=config.acoustic.subphone_features,
        log_f0_conditioning=config.log_f0_conditioning,
        frame_period=config.acoustic.frame_period,
    )
    if config.acoustic.feature_type == "world":
        out_acoustic_source = WORLDAcousticSource(
            utt_list,
            to_absolute_path(config.acoustic.wav_dir),
            to_absolute_path(config.acoustic.label_dir),
            question_path,
            f0_extractor=config.acoustic.f0_extractor,
            f0_ceil=config.acoustic.f0_ceil,
            f0_floor=config.acoustic.f0_floor,
            frame_period=config.acoustic.frame_period,
            mgc_order=config.acoustic.mgc_order,
            num_windows=config.acoustic.num_windows,
            relative_f0=config.acoustic.relative_f0,
            vibrato_mode=config.acoustic.vibrato_mode,
            sample_rate=config.acoustic.sample_rate,
            d4c_threshold=config.acoustic.d4c_threshold,
            trajectory_smoothing=config.acoustic.trajectory_smoothing,
            trajectory_smoothing_cutoff=config.acoustic.trajectory_smoothing_cutoff,
            trajectory_smoothing_f0=config.acoustic.trajectory_smoothing_f0,
            trajectory_smoothing_cutoff_f0=config.acoustic.trajectory_smoothing_cutoff_f0,
            correct_vuv=config.acoustic.correct_vuv,
            correct_f0=config.acoustic.correct_f0,
            dynamic_features_flags=config.acoustic.dynamic_features_flags,
            use_world_codec=config.acoustic.use_world_codec,
            res_type=config.acoustic.res_type,
            use_mcep_aperiodicity=config.acoustic.use_mcep_aperiodicity,
            mcep_aperiodicity_order=config.acoustic.mcep_aperiodicity_order,
        )
    elif config.acoustic.feature_type == "melf0":
        out_acoustic_source = MelF0AcousticSource(
            utt_list,
            to_absolute_path(config.acoustic.wav_dir),
            to_absolute_path(config.acoustic.label_dir),
            question_path,
            f0_extractor=config.acoustic.f0_extractor,
            f0_ceil=config.acoustic.f0_ceil,
            f0_floor=config.acoustic.f0_floor,
            frame_period=config.acoustic.frame_period,
            sample_rate=config.acoustic.sample_rate,
            d4c_threshold=config.acoustic.d4c_threshold,
            trajectory_smoothing_f0=config.acoustic.trajectory_smoothing_f0,
            trajectory_smoothing_cutoff_f0=config.acoustic.trajectory_smoothing_cutoff_f0,
            correct_vuv=config.acoustic.correct_vuv,
            correct_f0=config.acoustic.correct_f0,
            res_type=config.acoustic.res_type,
            fft_size=config.acoustic.fft_size,
            win_length=config.acoustic.win_length,
            hop_size=config.acoustic.hop_size,
            fmin=config.acoustic.fmin,
            fmax=config.acoustic.fmax,
            eps=config.acoustic.eps,
            num_mels=config.acoustic.num_mels,
        )
    else:
        raise ValueError(
            "Unknown feature type: {}".format(config.acoustic.feature_type)
        )

    in_acoustic = FileSourceDataset(in_acoustic_source)
    out_acoustic = FileSourceDataset(out_acoustic_source)

    # Save as files
    in_timelag_root = join(out_dir, "in_timelag")
    out_timelag_root = join(out_dir, "out_timelag")
    in_duration_root = join(out_dir, "in_duration")
    out_duration_root = join(out_dir, "out_duration")
    in_acoustic_root = join(out_dir, "in_acoustic")
    out_acoustic_root = join(out_dir, "out_acoustic")
    out_postfilter_root = join(out_dir, "out_postfilter")

    for d in [
        in_timelag_root,
        out_timelag_root,
        in_duration_root,
        out_duration_root,
        in_acoustic_root,
        out_acoustic_root,
        out_postfilter_root,
    ]:
        if not os.path.exists(d):
            logger.info("mkdirs: %s", format(d))
            os.makedirs(d)

    # Save features for timelag model
    if config.timelag.enabled:
        logger.info("Timelag linguistic feature dim: %s", str(in_timelag[0].shape[1]))
        logger.info("Timelag feature dim: %s", str(out_timelag[0].shape[1]))
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = [
                executor.submit(
                    _prepare_timelag_feature,
                    in_timelag_root,
                    out_timelag_root,
                    in_timelag,
                    out_timelag,
                    idx,
                )
                for idx in range(len(in_timelag))
            ]
            for future in tqdm(futures):
                future.result()

    # Save features for duration model
    if config.duration.enabled:
        logger.info("Duration linguistic feature dim: %s", str(in_duration[0].shape[1]))
        logger.info("Duration feature dim: %s", str(out_duration[0].shape[1]))
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = [
                executor.submit(
                    _prepare_duration_feature,
                    in_duration_root,
                    out_duration_root,
                    in_duration,
                    out_duration,
                    idx,
                )
                for idx in range(len(in_duration))
            ]
            for future in tqdm(futures):
                future.result()

    # Save features for acoustic model
    if config.acoustic.enabled:
        logger.info("Acoustic linguistic feature dim: %s", str(in_acoustic[0].shape[1]))
        logger.info("Acoustic feature dim: %s", str(out_acoustic[0][0].shape[1]))
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = [
                executor.submit(
                    _prepare_acoustic_feature,
                    in_acoustic_root,
                    out_acoustic_root,
                    out_postfilter_root,
                    in_acoustic,
                    out_acoustic,
                    idx,
                )
                for idx in range(len(in_acoustic))
            ]
            for future in tqdm(futures):
                future.result()


def entry():
    my_app()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    my_app()  # pylint: disable=no-value-for-parameter
