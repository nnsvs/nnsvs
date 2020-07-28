# coding: utf-8
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import argparse

import numpy as np
import joblib
import torch
import sys
import os
from os.path import exists, join, splitext

import pysptk
import librosa

from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.postfilters import merlin_post_filter
from nnmnkwii.preprocessing.f0 import interp1d
from nnsvs.multistream import multi_stream_mlpg, get_static_stream_sizes, split_streams

from nnsvs.gen import (
    predict_timelag, predict_duration, predict_acoustic, postprocess_duration)

from nnsvs.logger import getLogger
logger = None

# This function is originated from nnsvs/gen.py
def get_windows(num_window=1):
    windows = [(0, 0, np.array([1.0]))]
    if num_window >= 2:
        windows.append((1, 1, np.array([-0.5, 0.0, 0.5])))
    if num_window >= 3:
        windows.append((1, 1, np.array([1.0, -2.0, 1.0])))

    if num_window >= 4:
        raise ValueError(f"Not supported num windows: {num_window}")

    return windows

# This function is originated from nsvs/gen.py
def _midi_to_hz(x, idx, log_f0=False):
    z = np.zeros(len(x))
    indices = x[:, idx] > 0
    z[indices] = librosa.midi_to_hz(x[indices, idx])
    if log_f0:
        z[indices] = np.log(z[indices])
    return z

# This function is derived from gen_wavform in nnsvs/gen.py
def get_static_acoustic_features(labels, acoustic_features, acoustic_out_scaler,
                                 binary_dict, continuous_dict, stream_sizes, has_dynamic_features,
                                 subphone_features="coarse_coding", log_f0_conditioning=True, pitch_idx=None,
                                 num_windows=3, post_filter=True, sample_rate=48000, frame_period=5,
                                 relative_f0=True):
    windows = get_windows(num_windows)

    # Apply MLPG if necessary
    if np.any(has_dynamic_features):
        acoustic_features = multi_stream_mlpg(
            acoustic_features, acoustic_out_scaler.var_, windows, stream_sizes,
            has_dynamic_features)
        static_stream_sizes = get_static_stream_sizes(
            stream_sizes, has_dynamic_features, len(windows))
    else:
        static_stream_sizes = stream_sizes

    # Split multi-stream features
    mgc, target_f0, vuv, bap = split_streams(acoustic_features, static_stream_sizes)

    alpha = pysptk.util.mcepalpha(sample_rate)

    if post_filter:
        mgc = merlin_post_filter(mgc, alpha)

    ### F0 ###
    if relative_f0:
        diff_lf0 = target_f0
        # need to extract pitch sequence from the musical score
        linguistic_features = fe.linguistic_features(labels,
                                                    binary_dict, continuous_dict,
                                                    add_frame_features=True,
                                                    subphone_features=subphone_features)
        f0_score = _midi_to_hz(linguistic_features, pitch_idx, False)[:, None]
        lf0_score = f0_score.copy()
        nonzero_indices = np.nonzero(lf0_score)
        lf0_score[nonzero_indices] = np.log(f0_score[nonzero_indices])
        lf0_score = interp1d(lf0_score, kind="slinear")

        f0 = diff_lf0 + lf0_score
        f0[vuv < 0.5] = 0
        f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])
    else:
        f0 = target_f0
    return mgc, f0, bap
    
def dump_acoustic_features(config, device, label_path, question_path,
                           timelag_model, timelag_in_scaler, timelag_out_scaler,
                           duration_model, duration_in_scaler, duration_out_scaler,
                           acoustic_model, acoustic_in_scaler, acoustic_out_scaler,
                           out_dir, utt_id):
    # load labels and question
    labels = hts.load(label_path).round_()
    binary_dict, continuous_dict = hts.load_question_set(
        question_path, append_hat_for_LL=False)

    # pitch indices in the input features
    # TODO: configuarable
    pitch_idx = len(binary_dict) + 1
    pitch_indices = np.arange(len(binary_dict), len(binary_dict)+3)

    log_f0_conditioning = config.log_f0_conditioning

    if config.ground_truth_duration:
        # Use provided alignment
        duration_modified_labels = labels
    else:
        # Time-lag
        lag = predict_timelag(device, labels, timelag_model, timelag_in_scaler,
            timelag_out_scaler, binary_dict, continuous_dict, pitch_indices,
            log_f0_conditioning, config.timelag.allowed_range)

        # Timelag predictions
        durations = predict_duration(device, labels, duration_model,
            duration_in_scaler, duration_out_scaler, lag, binary_dict, continuous_dict,
            pitch_indices, log_f0_conditioning)

        # Normalize phoneme durations
        duration_modified_labels = postprocess_duration(labels, durations, lag)

    # Predict acoustic features
    acoustic_features = predict_acoustic(device, duration_modified_labels, acoustic_model,
        acoustic_in_scaler, acoustic_out_scaler, binary_dict, continuous_dict,
        config.acoustic.subphone_features, pitch_indices, log_f0_conditioning)

    # Get mgc/f0/bap from acoustic features
    mgc, f0, bap = get_static_acoustic_features(duration_modified_labels, acoustic_features,
                                                acoustic_out_scaler, binary_dict,
                                                continuous_dict, config.acoustic.stream_sizes,
                                                config.acoustic.has_dynamic_features,
                                                config.acoustic.subphone_features,
                                                log_f0_conditioning,
                                                pitch_idx, config.acoustic.num_windows,
                                                config.acoustic.post_filter, config.sample_rate,
                                                config.frame_period,
                                                config.acoustic.relative_f0)

    # Save mgc/f0/bap
    with open(join(out_dir, utt_id + ".f0"), "wb") as f:
        f0.astype(np.float32).tofile(f)
    with open(join(out_dir, utt_id + ".mgc"), "wb") as f:
        mgc.astype(np.float32).tofile(f)
    with open(join(out_dir, utt_id + ".bap"), "wb") as f:
        bap.astype(np.float32).tofile(f)

def synthesis_nsf(config, utt_list, input_dir, output_dir):
    # load NSF modules
    assert config.nsf_root_dir
    nsf_root_dir = to_absolute_path(config.nsf_root_dir)
    sys.path.append(nsf_root_dir)
    import core_scripts.data_io.default_data_io as nii_dset
    import core_scripts.other_tools.list_tools as nii_list_tool
    import core_scripts.nn_manager.nn_manager as nii_nn_wrapper

    # load NSF model
    if config.nsf_type == "hn-sinc-nsf":
        sys.path.append(to_absolute_path(join(config.nsf_root_dir, "project/hn-sinc-nsf-9")))
    elif config.nsf_type == "hn-nsf":
        sys.path.append(to_absolute_path(join(config.nsf_root_dir, "project/hn-nsf")))
    elif config.nsf_type == "cyc-noise-nsf":
        sys.path.append(to_absolute_path(join(config.nsf_root_dir, "project/cyc-noise-nsf-4")))
    else:
        raise Exception(f"Unknown NSF type: {config.nsf_type}")

    import model as nsf_model

    # Initialization
    # All NSF related settings are copied to argparse.Namespace object, because NSF core scripts are written
    # to work with argparse, not hydra.
    # Setting of file paths are converted to absolute one(save_model_dir, trained_model, output_dir)
    args = argparse.Namespace()
    args.batch_size = config.nsf.args.batch_size
    args.epochs = config.nsf.args.epochs
    args.no_best_epochs = config.nsf.args.no_best_epochs
    args.lr = config.nsf.args.lr
    args.no_cuda = config.nsf.args.no_cuda
    args.seed = config.nsf.args.seed
    args.eval_mode_for_validation = config.nsf.args.eval_mode_for_validation
    args.model_forward_with_target = config.nsf.args.model_forward_with_target
    args.model_forward_with_file_name = config.nsf.args.model_forward_with_file_name
    args.shuffle = config.nsf.args.shuffle
    args.num_workers = config.nsf.args.num_workers
    args.multi_gpu_data_parallel = config.nsf.args.multi_gpu_data_parallel
    if config.nsf.args.save_model_dir != None:
        args.save_model_dir = to_absolute_path(config.nsf.args.save_model_dir)
    else:
        args.save_model_dir = None 
    args.not_save_each_epoch = config.nsf.args.not_save_each_epoch
    args.save_epoch_name = config.nsf.args.save_epoch_name
    args.save_trained_name = config.nsf.args.save_trained_name
    args.save_model_ext = config.nsf.args.save_model_ext
    if config.nsf.args.trained_model != None:
        args.trained_model = to_absolute_path(config.nsf.args.trained_model)
    else:
        args.trained_model = None
    args.ignore_training_history_in_trained_model = config.nsf.args.ignore_training_history_in_trained_model
    args.inference = config.nsf.args.inference
    args.output_dir = to_absolute_path(output_dir)
    args.optimizer = config.nsf.args.optimizer
    args.verbose = config.nsf.args.verbose
    
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Prepare data io    
    params = {'batch_size':  args.batch_size,
              'shuffle':  args.shuffle,
              'num_workers': args.num_workers}

    test_set = nii_dset.NIIDataSetLoader("eval",
                                         utt_list, 
                                         [input_dir] * 3,
                                         config.nsf.model.input_exts, 
                                         config.nsf.model.input_dims, 
                                         config.nsf.model.input_reso, 
                                         config.nsf.model.input_norm, 
                                         [],
                                         config.nsf.model.output_exts, 
                                         config.nsf.model.output_dims, 
                                         config.nsf.model.output_reso, 
                                         config.nsf.model.output_norm, 
                                         args.save_model_dir,
                                         params = params,
                                         truncate_seq = None,
                                         min_seq_len = None,
                                         save_mean_std = False,
                                         wav_samp_rate = config.nsf.model.wav_samp_rate)


    # Initialize the model and loss function
    model = nsf_model.Model(test_set.get_in_dim(),
                            test_set.get_out_dim(), 
                            args)

    if not args.trained_model:
        print("trained_model is not set, so try to load default trained model")
        default_trained_model_path = join(args.save_model_dir,
                                          "{}{}".format(args.save_trained_name,
                                                        args.save_model_ext))
        if not exists(default_trained_model_path):
            raise Exception("No trained model found")
        checkpoint = torch.load(default_trained_model_path)
    else:
        checkpoint = torch.load(args.trained_model)

    # do inference and output data
    nii_nn_wrapper.f_inference_wrapper(args, model, device,
                                       test_set, checkpoint)
    
@hydra.main(config_path="conf/synthesis_nsf/config.yaml")
def my_app(config : DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(config.pretty())

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    # timelag
    timelag_config = OmegaConf.load(to_absolute_path(config.timelag.model_yaml))
    timelag_model = hydra.utils.instantiate(timelag_config.netG).to(device)
    checkpoint = torch.load(to_absolute_path(config.timelag.checkpoint),
        map_location=lambda storage, loc: storage)
    timelag_model.load_state_dict(checkpoint["state_dict"])
    timelag_in_scaler = joblib.load(to_absolute_path(config.timelag.in_scaler_path))
    timelag_out_scaler = joblib.load(to_absolute_path(config.timelag.out_scaler_path))
    timelag_model.eval()

    # duration
    duration_config = OmegaConf.load(to_absolute_path(config.duration.model_yaml))
    duration_model = hydra.utils.instantiate(duration_config.netG).to(device)
    checkpoint = torch.load(to_absolute_path(config.duration.checkpoint),
        map_location=lambda storage, loc: storage)
    duration_model.load_state_dict(checkpoint["state_dict"])
    duration_in_scaler = joblib.load(to_absolute_path(config.duration.in_scaler_path))
    duration_out_scaler = joblib.load(to_absolute_path(config.duration.out_scaler_path))
    duration_model.eval()

    # acoustic model
    acoustic_config = OmegaConf.load(to_absolute_path(config.acoustic.model_yaml))
    acoustic_model = hydra.utils.instantiate(acoustic_config.netG).to(device)
    checkpoint = torch.load(to_absolute_path(config.acoustic.checkpoint),
        map_location=lambda storage, loc: storage)
    acoustic_model.load_state_dict(checkpoint["state_dict"])
    acoustic_in_scaler = joblib.load(to_absolute_path(config.acoustic.in_scaler_path))
    acoustic_out_scaler = joblib.load(to_absolute_path(config.acoustic.out_scaler_path))
    acoustic_model.eval()

    # Run synthesis for each utt.
    question_path = to_absolute_path(config.question_path)

    assert config.utt_list
    in_dir = to_absolute_path(config.in_dir)
    out_dir = to_absolute_path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    utt_list = []
    with open(to_absolute_path(config.utt_list)) as f:
        lines = f.readlines()
        for l in lines:
            if len(l.strip()) > 0:
                utt_list.append(l.strip())
                
    logger.info(f"Processes {len(utt_list)} utterances...")
    for utt_id in utt_list:
        label_path = join(in_dir, f"{utt_id}.lab")
        if not exists(label_path):
            raise RuntimeError(f"Label file does not exist: {label_path}")
        dump_acoustic_features(config, device, label_path, question_path,
                               timelag_model, timelag_in_scaler, timelag_out_scaler,
                               duration_model, duration_in_scaler, duration_out_scaler,
                               acoustic_model, acoustic_in_scaler, acoustic_out_scaler,
                               out_dir, utt_id)
    synthesis_nsf(config, utt_list, out_dir, out_dir)

def entry():
    my_app()

if __name__ == "__main__":
    my_app()
                            
