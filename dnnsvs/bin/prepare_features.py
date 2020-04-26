# coding: utf-8

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import numpy as np
from os.path import join
from glob import glob
import pysptk
import pyworld
import librosa
from scipy.io import wavfile
from tqdm import tqdm
from os.path import basename, splitext, exists
import os
import sys

from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.util import apply_delta_windows
from dnnsvs.logger import getLogger
logger = None


def midi_to_hz(x, idx, log_f0=False):
    z = np.zeros(len(x))
    indices = x[:, idx] > 0
    z[indices] = librosa.midi_to_hz(x[indices, idx])
    if log_f0:
        z[indices] = np.log(z[indices])
    return z

def _collect_files(data_root, utt_list, ext):
    with open(utt_list) as f:
        files = f.readlines()
    files = map(lambda utt_id: utt_id.strip(), files)
    files = filter(lambda utt_id: len(utt_id) > 0, files)
    files = list(map(lambda utt_id: join(data_root, f"{utt_id}{ext}"), files))
    return files


class LinguisticSource(FileDataSource):
    def __init__(self, utt_list, data_root, question_path, add_frame_features=False,
                subphone_features=None, log_f0_conditioning=True):
        self.data_root = data_root
        self.utt_list = utt_list
        self.add_frame_features = add_frame_features
        self.subphone_features = subphone_features
        self.binary_dict, self.continuous_dict = hts.load_question_set(
            question_path, append_hat_for_LL=False)
        self.log_f0_conditioning = log_f0_conditioning
        self.pitch_idx = np.arange(len(self.binary_dict), len(self.binary_dict)+3)

    def collect_files(self):
        return _collect_files(self.data_root, self.utt_list, ".lab")

    def collect_features(self, path):
        labels = hts.load(path)
        features = fe.linguistic_features(
            labels, self.binary_dict, self.continuous_dict,
            add_frame_features=self.add_frame_features,
            subphone_features=self.subphone_features)
        if self.log_f0_conditioning:
            for idx in self.pitch_idx:
                features[:, idx] = interp1d(midi_to_hz(features, idx, True), kind="slinear")
        return features.astype(np.float32)


class TimeLagFeatureSource(FileDataSource):
    def __init__(self, utt_list, label_phone_score_dir, label_phone_align_dir):
        self.utt_list = utt_list
        self.label_phone_score_dir = label_phone_score_dir
        self.label_phone_align_dir = label_phone_align_dir

    def collect_files(self):
        labels_score = _collect_files(self.label_phone_score_dir, self.utt_list, ".lab")
        labels_align = _collect_files(self.label_phone_align_dir, self.utt_list, ".lab")
        return labels_score, labels_align

    def collect_features(self, label_score_path, label_align_path):
        label_score = hts.load(label_score_path)
        label_align = hts.load(label_align_path)
        timelag = np.asarray(label_align.start_times) - np.asarray(label_score.start_times)
        return timelag.astype(np.float32).reshape(-1, 1)


class DurationFeatureSource(FileDataSource):
    def __init__(self, utt_list, data_root):
        self.utt_list = utt_list
        self.data_root = data_root

    def collect_files(self):
        return _collect_files(self.data_root, self.utt_list, ".lab")

    def collect_features(self, path):
        labels = hts.load(path)
        features = fe.duration_features(labels)
        return features.astype(np.float32)


class AcousticSource(FileDataSource):
    def __init__(self, utt_list, wav_root, label_root, question_path,
            use_harvest=True, f0_floor=150, f0_ceil=700, frame_period=5,
            mgc_order=59):
        self.utt_list = utt_list
        self.wav_root = wav_root
        self.label_root = label_root
        self.binary_dict, self.continuous_dict = hts.load_question_set(
            question_path, append_hat_for_LL=False)
        self.pitch_idx = len(self.binary_dict) + 1
        self.use_harvest = use_harvest
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.frame_period = frame_period
        self.mgc_order = mgc_order

        self.windows = [
            (0, 0, np.array([1.0])),
            (1, 1, np.array([-0.5, 0.0, 0.5])),
            (1, 1, np.array([1.0, -2.0, 1.0])),
        ]

    def collect_files(self):
        wav_paths = _collect_files(self.wav_root, self.utt_list, ".wav")
        label_paths = _collect_files(self.label_root, self.utt_list, ".lab")
        return wav_paths, label_paths

    def collect_features(self, wav_path, label_path):
        labels = hts.load(label_path)
        l_features = fe.linguistic_features(
            labels, self.binary_dict, self.continuous_dict,
            add_frame_features=True,
            subphone_features="coarse_coding")

        f0_score = midi_to_hz(l_features, self.pitch_idx, False)
        # TODO: better to set the margin carefully
        max_f0 = int(max(f0_score)) + 100
        min_f0 = int(max(self.f0_floor, min(f0_score[f0_score > 0]) - 20))
        assert max_f0 > min_f0

        fs, x = wavfile.read(wav_path)
        x = x.astype(np.float64)

        if self.use_harvest:
            f0, timeaxis = pyworld.harvest(x, fs, frame_period=self.frame_period,
            f0_floor=min_f0, f0_ceil=max_f0)
        else:
            f0, timeaxis = pyworld.dio(x, fs, frame_period=frame_period,
                f0_floor=min_f0, f0_ceil=max_f0)
            f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs, f0_floor=self.f0_floor)
        aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)

        bap = pyworld.code_aperiodicity(aperiodicity, fs)
        mgc = pysptk.sp2mc(spectrogram, order=self.mgc_order,
                           alpha=pysptk.util.mcepalpha(fs))
        # F0 of speech
        f0 = f0[:, None]
        lf0 = f0.copy()
        nonzero_indices = np.nonzero(f0)
        lf0[nonzero_indices] = np.log(f0[nonzero_indices])
        if self.use_harvest:
            # https://github.com/mmorise/World/issues/35#issuecomment-306521887
            vuv = (aperiodicity[:, 0] < 0.5).astype(np.float32)[:, None]
        else:
            vuv = (lf0 != 0).astype(np.float32)
        lf0 = interp1d(lf0, kind="slinear")

        # # F0 derived from the musical score
        f0_score = f0_score[:, None]
        lf0_score = f0_score.copy()
        nonzero_indices = np.nonzero(f0_score)
        lf0_score[nonzero_indices] = np.log(f0_score[nonzero_indices])
        lf0_score = interp1d(lf0_score, kind="slinear")


        # Adjust lengths
        mgc = mgc[:labels.num_frames()]
        lf0 = lf0[:labels.num_frames()]
        vuv = vuv[:labels.num_frames()]
        bap = bap[:labels.num_frames()]

        diff_lf0 = lf0 - lf0_score
        diff_lf0 = np.clip(diff_lf0, np.log(0.5), np.log(2.0))

        mgc = apply_delta_windows(mgc, self.windows)
        diff_lf0 = apply_delta_windows(diff_lf0, self.windows)
        bap = apply_delta_windows(bap, self.windows)

        features = np.hstack((mgc, diff_lf0, vuv, bap))

        return features.astype(np.float32)


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
    in_timelag_source = LinguisticSource(utt_list,
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
    in_duration_source = LinguisticSource(utt_list,
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
    in_acoustic_source = LinguisticSource(utt_list,
        to_absolute_path(config.acoustic.label_dir), question_path,
        add_frame_features=True, subphone_features="coarse_coding",
        log_f0_conditioning=config.log_f0_conditioning)
    out_acoustic_source = AcousticSource(utt_list,
        to_absolute_path(config.acoustic.wav_dir), to_absolute_path(config.acoustic.label_dir),
        question_path, use_harvest=config.acoustic.use_harvest,
        f0_ceil=config.acoustic.f0_ceil, f0_floor=config.acoustic.f0_floor,
        frame_period=config.acoustic.frame_period, mgc_order=config.acoustic.mgc_order)
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
    logger.info("Acoustic feature dim: {}".format(out_acoustic[0].shape))
    for idx, (x, y) in tqdm(enumerate(zip(in_acoustic, out_acoustic))):
        name = splitext(basename(in_acoustic.collected_files[idx][0]))[0]
        xpath = join(in_acoustic_root, name + "-feats.npy")
        ypath = join(out_acoustic_root, name + "-feats.npy")
        np.save(xpath, x, allow_pickle=False)
        np.save(ypath, y, allow_pickle=False)


def entry():
    my_app()


if __name__ == "__main__":
    my_app()