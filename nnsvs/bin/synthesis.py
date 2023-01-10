import os
from os.path import join

import hydra
import joblib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from nnmnkwii.io import hts
from nnsvs.gen import (
    postprocess_acoustic,
    postprocess_waveform,
    predict_acoustic,
    predict_timing,
    predict_waveform,
)
from nnsvs.logger import getLogger
from nnsvs.util import extract_static_scaler, init_seed, load_utt_list, load_vocoder
from omegaconf import DictConfig, OmegaConf
from scipy.io import wavfile
from tqdm.auto import tqdm


@hydra.main(config_path="conf/synthesis", config_name="config")
def my_app(config: DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    # timelag
    timelag_config = OmegaConf.load(to_absolute_path(config.timelag.model_yaml))
    timelag_model = hydra.utils.instantiate(timelag_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.timelag.checkpoint),
        map_location=lambda storage, loc: storage,
    )
    timelag_model.load_state_dict(checkpoint["state_dict"])
    timelag_in_scaler = joblib.load(to_absolute_path(config.timelag.in_scaler_path))
    timelag_out_scaler = joblib.load(to_absolute_path(config.timelag.out_scaler_path))
    timelag_model.eval()

    # duration
    duration_config = OmegaConf.load(to_absolute_path(config.duration.model_yaml))
    duration_model = hydra.utils.instantiate(duration_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.duration.checkpoint),
        map_location=lambda storage, loc: storage,
    )
    duration_model.load_state_dict(checkpoint["state_dict"])
    duration_in_scaler = joblib.load(to_absolute_path(config.duration.in_scaler_path))
    duration_out_scaler = joblib.load(to_absolute_path(config.duration.out_scaler_path))
    duration_model.eval()

    # acoustic model
    acoustic_config = OmegaConf.load(to_absolute_path(config.acoustic.model_yaml))
    acoustic_model = hydra.utils.instantiate(acoustic_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.acoustic.checkpoint),
        map_location=lambda storage, loc: storage,
    )
    acoustic_model.load_state_dict(checkpoint["state_dict"])
    acoustic_in_scaler = joblib.load(to_absolute_path(config.acoustic.in_scaler_path))
    acoustic_out_scaler = joblib.load(to_absolute_path(config.acoustic.out_scaler_path))
    acoustic_model.eval()

    # NOTE: this is used for GV post-filtering
    acoustic_out_static_scaler = extract_static_scaler(
        acoustic_out_scaler, acoustic_config
    )

    # Vocoder
    if config.vocoder.checkpoint is not None and len(config.vocoder.checkpoint) > 0:
        vocoder, vocoder_in_scaler, vocoder_config = load_vocoder(
            to_absolute_path(config.vocoder.checkpoint),
            device,
            acoustic_config,
        )
    else:
        vocoder, vocoder_in_scaler, vocoder_config = None, None, None
        if config.synthesis.vocoder_type != "world":
            logger.warning("Vocoder checkpoint is not specified")
            logger.info(f"Use world instead of {config.synthesis.vocoder_type}.")
        config.synthesis.vocoder_type = "world"

    # Run synthesis for each utt.
    binary_dict, numeric_dict = hts.load_question_set(
        to_absolute_path(config.synthesis.qst)
    )

    in_dir = to_absolute_path(config.in_dir)
    out_dir = to_absolute_path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    utt_ids = load_utt_list(to_absolute_path(config.utt_list))
    logger.info("Processes %s utterances...", len(utt_ids))
    for utt_id in tqdm(utt_ids):
        labels = hts.load(join(in_dir, f"{utt_id}.lab"))
        hts_frame_shift = int(config.synthesis.frame_period * 1e4)
        labels.frame_shift = hts_frame_shift
        init_seed(1234)

        if config.synthesis.ground_truth_duration:
            duration_modified_labels = labels
        else:
            duration_modified_labels = predict_timing(
                device=device,
                labels=labels,
                binary_dict=binary_dict,
                numeric_dict=numeric_dict,
                timelag_model=timelag_model,
                timelag_config=timelag_config,
                timelag_in_scaler=timelag_in_scaler,
                timelag_out_scaler=timelag_out_scaler,
                duration_model=duration_model,
                duration_config=duration_config,
                duration_in_scaler=duration_in_scaler,
                duration_out_scaler=duration_out_scaler,
                log_f0_conditioning=config.synthesis.log_f0_conditioning,
                allowed_range=config.timelag.allowed_range,
                allowed_range_rest=config.timelag.allowed_range_rest,
                force_clip_input_features=config.timelag.force_clip_input_features,
                frame_period=config.synthesis.frame_period,
            )

        # Predict acoustic features
        acoustic_features = predict_acoustic(
            device=device,
            labels=duration_modified_labels,
            acoustic_model=acoustic_model,
            acoustic_config=acoustic_config,
            acoustic_in_scaler=acoustic_in_scaler,
            acoustic_out_scaler=acoustic_out_scaler,
            binary_dict=binary_dict,
            numeric_dict=numeric_dict,
            subphone_features=config.synthesis.subphone_features,
            log_f0_conditioning=config.synthesis.log_f0_conditioning,
            force_clip_input_features=config.acoustic.force_clip_input_features,
            f0_shift_in_cent=config.synthesis.pre_f0_shift_in_cent,
        )

        # NOTE: the output of this function is tuple of features
        # e.g., (mgc, lf0, vuv, bap)
        multistream_features = postprocess_acoustic(
            device=device,
            acoustic_features=acoustic_features,
            duration_modified_labels=duration_modified_labels,
            binary_dict=binary_dict,
            numeric_dict=numeric_dict,
            acoustic_config=acoustic_config,
            acoustic_out_static_scaler=acoustic_out_static_scaler,
            postfilter_model=None,  # NOTE: learned post-filter is not supported
            postfilter_config=None,
            postfilter_out_scaler=None,
            sample_rate=config.synthesis.sample_rate,
            frame_period=config.synthesis.frame_period,
            relative_f0=config.synthesis.relative_f0,
            feature_type=config.synthesis.feature_type,
            post_filter_type=config.synthesis.post_filter_type,
            trajectory_smoothing=config.synthesis.trajectory_smoothing,
            trajectory_smoothing_cutoff=config.synthesis.trajectory_smoothing_cutoff,
            trajectory_smoothing_cutoff_f0=config.synthesis.trajectory_smoothing_cutoff_f0,
            vuv_threshold=config.synthesis.vuv_threshold,
            f0_shift_in_cent=config.synthesis.post_f0_shift_in_cent,
            vibrato_scale=1.0,
            force_fix_vuv=config.synthesis.force_fix_vuv,
        )

        # Generate waveform by vocoder
        wav = predict_waveform(
            device=device,
            multistream_features=multistream_features,
            vocoder=vocoder,
            vocoder_config=vocoder_config,
            vocoder_in_scaler=vocoder_in_scaler,
            sample_rate=config.synthesis.sample_rate,
            frame_period=config.synthesis.frame_period,
            use_world_codec=config.synthesis.use_world_codec,
            feature_type=config.synthesis.feature_type,
            vocoder_type=config.synthesis.vocoder_type,
            vuv_threshold=config.synthesis.vuv_threshold,
        )

        wav = postprocess_waveform(
            wav=wav,
            sample_rate=config.synthesis.sample_rate,
            dtype=np.int16,
            peak_norm=False,
            loudness_norm=False,
        )

        out_wav_path = join(out_dir, f"{utt_id}.wav")
        wavfile.write(
            out_wav_path, rate=config.synthesis.sample_rate, data=wav.astype(np.int16)
        )


def entry():
    my_app()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    my_app()  # pylint: disable=no-value-for-parameter
