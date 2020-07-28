# coding: utf-8
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

import sys
from glob import glob
import os
from os.path import join, basename, splitext, exists
import numpy as np
import librosa
import soundfile as sf
import torch

from tqdm import tqdm

from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.preprocessing.f0 import interp1d

from nnsvs.multistream import get_static_features, get_static_stream_sizes, split_streams
from nnsvs.logger import getLogger
logger = None

def _midi_to_hz(x, idx, log_f0=False):
    z = np.zeros(len(x))
    indices = x[:, idx] > 0
    z[indices] = librosa.midi_to_hz(x[indices, idx])
    if log_f0:
        z[indices] = np.log(z[indices])
    return z

@hydra.main(config_path="conf/prepare_nsf_data/config.yaml")
def my_app(config : DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(config.pretty())

    in_dir = to_absolute_path(config.in_dir)
    out_dir = to_absolute_path(config.out_dir)

    label_dir = to_absolute_path(config.label_dir)
    question_path = to_absolute_path(config.question_path)

    relative_f0=config.relative_f0
    stream_sizes=config.stream_sizes
    has_dynamic_features=config.has_dynamic_features
    num_windows=config.num_windows
    sample_rate=config.sample_rate
    test_set=config.test_set

    binary_dict, continuous_dict = hts.load_question_set(question_path, append_hat_for_LL=False)
    pitch_idx=len(binary_dict)+1

    feats_files = sorted(glob(join(in_dir, "*-feats.npy")))
    logger.info(f"Process {len(feats_files)} feature files for NSF.")
    for idx in tqdm(range(len(feats_files))):
        feat_file=feats_files[idx]
        utt_id = splitext(basename(feat_file))[0].replace("-feats", "")
        label_path= join(label_dir, utt_id + ".lab")
        labels = hts.load(label_path)

        acoustic_features = np.load(feat_file)

        if np.any(has_dynamic_features):
            # get_static_features requires torch.Tensor, so it is necessary to wrap acoustic_features
            _acoustic_features = torch.from_numpy(acoustic_features.astype(np.float32)).unsqueeze(0)
            acoustic_features = get_static_features(_acoustic_features, num_windows, stream_sizes, has_dynamic_features).squeeze(0).cpu().data.numpy()
            static_stream_sizes = get_static_stream_sizes(
                stream_sizes, has_dynamic_features, num_windows)
        else:
            static_stream_sizes = stream_sizes
        
        mgc, target_f0, vuv, bap = split_streams(acoustic_features, static_stream_sizes)

        if relative_f0:
            diff_lf0 = target_f0
            # need to extract pitch sequence from the musical score
            linguistic_features = fe.linguistic_features(labels, binary_dict, continuous_dict,
                                                         add_frame_features=True,
                                                         subphone_features="coarse_coding")
            f0_score = _midi_to_hz(linguistic_features, pitch_idx, False)[:, None]

            if len(f0_score) > len(diff_lf0):
                print("Warning! likely to have mistakes in alignment in {}".format(label_path))
                print(f0_score.shape, diff_lf0.shape)

            f0_score = f0_score[:len(diff_lf0)]        
            lf0_score = f0_score.copy()
            nonzero_indices = np.nonzero(lf0_score)
            lf0_score[nonzero_indices] = np.log(f0_score[nonzero_indices])
            lf0_score = interp1d(lf0_score, kind="slinear")
            f0 = diff_lf0 + lf0_score
            f0[vuv < 0.5] = 0
            f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])
        else:
            f0 = target_f0
                
        if test_set:
            feats_out_dir = join(out_dir , "test_input_dirs")
        else:
            feats_out_dir = join(out_dir , "input_dirs")

        if exists(feats_out_dir) != True:
            os.makedirs(feats_out_dir)

        # NSF binary data format is required to be read by numpy.fromfile.
        # "npy" format may not be adequate.
        with open(join(feats_out_dir, utt_id + ".f0"), "wb") as f:
            f0.astype(np.float32).tofile(f)
        with open(join(feats_out_dir, utt_id + ".mgc"), "wb") as f:
            mgc.astype(np.float32).tofile(f)
        with open(join(feats_out_dir, utt_id + ".bap"), "wb") as f:
            bap.astype(np.float32).tofile(f)
    
    if test_set != True:
        wave_files = sorted(glob(join(in_dir, "*-wave.npy")))
        logger.info(f"Process {len(wave_files)} wave files for NSF.")
        for idx in tqdm(range(len(wave_files))):
            wave_file=wave_files[idx]
            utt_id = splitext(basename(wave_file))[0].replace("-wave", "")
                        
            data = np.load(wave_file)
            wave_out_dir = join(out_dir, "output_dirs")
            if exists(wave_out_dir) != True:
                os.makedirs(wave_out_dir)
        
            wav_output_path = join(wave_out_dir, utt_id + ".wav")
            sf.write(wav_output_path, data, sample_rate)

def entry():
    my_app()

if __name__ == "__main__":
    my_app()
                            
