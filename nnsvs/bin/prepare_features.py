# coding: utf-8

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import numpy as np
from os.path import join
from tqdm import tqdm
from os.path import basename, splitext, exists
import os
import sys

from nnmnkwii.datasets import FileSourceDataset

from nnsvs.data import (
    MusicalLinguisticSource, TimeLagFeatureSource,
    DurationFeatureSource, WORLDAcousticSource)
from nnsvs.logger import getLogger
logger = None


@hydra.main(config_path="conf/prepare_features/config.yaml")
def my_app(config : DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(config.pretty())

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
    in_timelag_source = MusicalLinguisticSource(utt_list,
        to_absolute_path(config.timelag.label_phone_score_dir),
        add_frame_features=False, subphone_features=None,
        question_path=question_path,
        log_f0_conditioning=config.log_f0_conditioning)
    out_timelag_source = TimeLagFeatureSource(utt_list,
        to_absolute_path(config.timelag.label_phone_score_dir),
        to_absolute_path(config.timelag.label_phone_align_dir))

    in_timelag = FileSourceDataset(in_timelag_source)
    out_timelag = FileSourceDataset(out_timelag_source)

    # Duration model
    # in: musical/linguistic context
    # out: duration
    if config.duration.question_path is not None:
        question_path = config.duration.question_path
    else:
        question_path = question_path_general
    in_duration_source = MusicalLinguisticSource(utt_list,
        to_absolute_path(config.duration.label_dir),
        add_frame_features=False, subphone_features=None,
        question_path=question_path,
        log_f0_conditioning=config.log_f0_conditioning)
    out_duration_source = DurationFeatureSource(
        utt_list, to_absolute_path(config.duration.label_dir))

    in_duration = FileSourceDataset(in_duration_source)
    out_duration = FileSourceDataset(out_duration_source)

    # Acoustic model
    # in: musical/linguistic context
    # out: acoustic features
    if config.acoustic.question_path is not None:
        question_path = config.acoustic.question_path
    else:
        question_path = question_path_general
    in_acoustic_source = MusicalLinguisticSource(utt_list,
        to_absolute_path(config.acoustic.label_dir), question_path,
        add_frame_features=True, subphone_features=config.acoustic.subphone_features,
        log_f0_conditioning=config.log_f0_conditioning)
    out_acoustic_source = WORLDAcousticSource(utt_list,
        to_absolute_path(config.acoustic.wav_dir), to_absolute_path(config.acoustic.label_dir),
        question_path, use_harvest=config.acoustic.use_harvest,
        f0_ceil=config.acoustic.f0_ceil, f0_floor=config.acoustic.f0_floor,
        frame_period=config.acoustic.frame_period, mgc_order=config.acoustic.mgc_order,
        num_windows=config.acoustic.num_windows,
        relative_f0=config.acoustic.relative_f0)
    in_acoustic = FileSourceDataset(in_acoustic_source)
    out_acoustic = FileSourceDataset(out_acoustic_source)

    # Save as files
    in_timelag_root = join(out_dir, "in_timelag")
    out_timelag_root = join(out_dir, "out_timelag")
    in_duration_root = join(out_dir, "in_duration")
    out_duration_root = join(out_dir, "out_duration")
    in_acoustic_root = join(out_dir, "in_acoustic")
    out_acoustic_root = join(out_dir, "out_acoustic")

    for d in [in_timelag_root, out_timelag_root, in_duration_root, out_duration_root,
            in_acoustic_root, out_acoustic_root]:
        if not os.path.exists(d):
            logger.info("mkdirs: {}".format(d))
            os.makedirs(d)

    # Save features for timelag model
    logger.info("Timelag linguistic feature dim: {}".format(in_timelag[0].shape))
    logger.info("Timelag feature dim: {}".format(out_timelag[0].shape))
    for idx, (x, y) in tqdm(enumerate(zip(in_timelag, out_timelag))):
        name = splitext(basename(in_timelag.collected_files[idx][0]))[0]
        xpath = join(in_timelag_root, name + "-feats.npy")
        ypath = join(out_timelag_root, name + "-feats.npy")
        np.save(xpath, x, allow_pickle=False)
        np.save(ypath, y, allow_pickle=False)

    # Save features for duration model
    logger.info("Duration linguistic feature dim: {}".format(in_duration[0].shape))
    logger.info("Duration feature dim: {}".format(out_duration[0].shape))
    for idx, (x, y) in tqdm(enumerate(zip(in_duration, out_duration))):
        name = splitext(basename(in_duration.collected_files[idx][0]))[0]
        xpath = join(in_duration_root, name + "-feats.npy")
        ypath = join(out_duration_root, name + "-feats.npy")
        np.save(xpath, x, allow_pickle=False)
        np.save(ypath, y, allow_pickle=False)

    # Save features for acoustic model
    logger.info("Acoustic linguistic feature dim: {}".format(in_acoustic[0].shape))
    logger.info("Acoustic feature dim: {}".format(out_acoustic[0][0].shape))
    for idx, (x, yw) in tqdm(enumerate(zip(in_acoustic, out_acoustic))):
        y, wave = yw
        name = splitext(basename(in_acoustic.collected_files[idx][0]))[0]
        xpath = join(in_acoustic_root, name + "-feats.npy")
        ypath = join(out_acoustic_root, name + "-feats.npy")
        wpath = join(out_acoustic_root, name + "-wave.npy")
        np.save(xpath, x, allow_pickle=False)
        np.save(ypath, y, allow_pickle=False)
        np.save(wpath, wave, allow_pickle=False)


def entry():
    my_app()


if __name__ == "__main__":
    my_app()