# coding: utf-8

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import numpy as np
import joblib
import torch
from scipy.io import wavfile
from os.path import join, basename, exists
import os
from tqdm import tqdm
from nnmnkwii.io import hts

from nnsvs.gen import (
    predict_timelag, predict_duration, predict_acoustic, postprocess_duration,
    gen_waveform)
from nnsvs.logger import getLogger
logger = None


def synthesis(config, device, label_path, question_path,
              timelag_model, timelag_config, timelag_in_scaler, timelag_out_scaler,
              duration_model, duration_config, duration_in_scaler, duration_out_scaler,
              acoustic_model, acoustic_config, acoustic_in_scaler, acoustic_out_scaler):
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
        lag = predict_timelag(device, labels, timelag_model, timelag_config, timelag_in_scaler,
            timelag_out_scaler, binary_dict, continuous_dict, pitch_indices,
            log_f0_conditioning, config.timelag.allowed_range)

        # Timelag predictions
        durations = predict_duration(device, labels, duration_model, duration_config, 
            duration_in_scaler, duration_out_scaler, lag, binary_dict, continuous_dict,
            pitch_indices, log_f0_conditioning)

        # Normalize phoneme durations
        duration_modified_labels = postprocess_duration(labels, durations, lag)

    # Predict acoustic features
    acoustic_features = predict_acoustic(device, duration_modified_labels, acoustic_model, acoustic_config,
        acoustic_in_scaler, acoustic_out_scaler, binary_dict, continuous_dict,
        config.acoustic.subphone_features, pitch_indices, log_f0_conditioning)

    # Waveform generation
    generated_waveform = gen_waveform(
        duration_modified_labels, acoustic_features,
        binary_dict, continuous_dict, acoustic_config.stream_sizes,
        acoustic_config.has_dynamic_features,
        config.acoustic.subphone_features, log_f0_conditioning,
        pitch_idx, acoustic_config.num_windows,
        config.acoustic.post_filter, config.sample_rate, config.frame_period,
        config.acoustic.relative_f0)

    return generated_waveform


@hydra.main(config_path="conf/synthesis/config.yaml")
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

    if config.utt_list is not None:
        in_dir = to_absolute_path(config.in_dir)
        out_dir = to_absolute_path(config.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        with open(to_absolute_path(config.utt_list)) as f:
            lines = list(filter(lambda s : len(s.strip()) > 0, f.readlines()))
            logger.info(f"Processes {len(lines)} utterances...")
            for idx in tqdm(range(len(lines))):
                utt_id = lines[idx].strip()
                label_path = join(in_dir, f"{utt_id}.lab")
                if not exists(label_path):
                    raise RuntimeError(f"Label file does not exist: {label_path}")

                wav = synthesis(config, device, label_path, question_path,
                                timelag_model, timelag_config, timelag_in_scaler, timelag_out_scaler,
                                duration_model, duration_config, duration_in_scaler, duration_out_scaler,
                                acoustic_model, acoustic_config, acoustic_in_scaler, acoustic_out_scaler)
                wav = wav / np.max(np.abs(wav)) * (2**15 - 1)

                out_wav_path = join(out_dir, f"{utt_id}.wav")
                wavfile.write(out_wav_path, rate=config.sample_rate, data=wav.astype(np.int16))
    else:
        assert config.label_path is not None
        logger.info(f"Process the label file: {config.label_path}")
        label_path = to_absolute_path(config.label_path)
        out_wav_path = to_absolute_path(config.out_wav_path)

        wav = synthesis(config, device, label_path, question_path,
            timelag_model, timelag_in_scaler, timelag_out_scaler,
            duration_model, duration_in_scaler, duration_out_scaler,
            acoustic_model, acoustic_in_scaler, acoustic_out_scaler)
        wav = wav / np.max(np.abs(wav)) * (2**15 - 1)
        wavfile.write(out_wav_path, rate=config.sample_rate, data=wav.astype(np.int16))


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
