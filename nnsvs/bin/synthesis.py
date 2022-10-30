import os
from os.path import join
from pathlib import Path

import hydra
import joblib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from nnmnkwii.io import hts
from nnsvs.logger import getLogger
from nnsvs.multistream import get_static_features, get_static_stream_sizes
from nnsvs.svs import post_process, predict_timings, synthesis_from_timings
from nnsvs.usfgan import USFGANWrapper
from nnsvs.util import StandardScaler, init_seed, load_utt_list
from omegaconf import DictConfig, OmegaConf
from parallel_wavegan.utils import load_model
from scipy.io import wavfile
from tqdm.auto import tqdm


def extract_static_scaler(out_scaler, model_config):
    mean_ = get_static_features(
        out_scaler.mean_.reshape(1, 1, out_scaler.mean_.shape[-1]),
        model_config.num_windows,
        model_config.stream_sizes,
        model_config.has_dynamic_features,
    )
    mean_ = np.concatenate(mean_, -1).reshape(1, -1)
    var_ = get_static_features(
        out_scaler.var_.reshape(1, 1, out_scaler.var_.shape[-1]),
        model_config.num_windows,
        model_config.stream_sizes,
        model_config.has_dynamic_features,
    )
    var_ = np.concatenate(var_, -1).reshape(1, -1)
    scale_ = get_static_features(
        out_scaler.scale_.reshape(1, 1, out_scaler.scale_.shape[-1]),
        model_config.num_windows,
        model_config.stream_sizes,
        model_config.has_dynamic_features,
    )
    scale_ = np.concatenate(scale_, -1).reshape(1, -1)
    static_scaler = StandardScaler(mean_, var_, scale_)
    return static_scaler


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
        path = Path(to_absolute_path(config.vocoder.checkpoint))
        vocoder_dir = path.parent
        if (vocoder_dir / "vocoder_model.yaml").exists():
            # packed model
            vocoder_config = OmegaConf.load(vocoder_dir / "vocoder_model.yaml")
        elif (vocoder_dir / "config.yml").exists():
            # PWG checkpoint
            vocoder_config = OmegaConf.load(vocoder_dir / "config.yml")
        else:
            # usfgan
            vocoder_config = OmegaConf.load(vocoder_dir / "config.yaml")

        if "generator" in vocoder_config and "discriminator" in vocoder_config:
            # usfgan
            checkpoint = torch.load(
                path,
                map_location=lambda storage, loc: storage,
            )
            vocoder = hydra.utils.instantiate(vocoder_config.generator).to(device)
            vocoder.load_state_dict(checkpoint["model"]["generator"])
            vocoder.remove_weight_norm()
            vocoder = USFGANWrapper(vocoder_config, vocoder)

            # Extract scaler params for [mgc, bap]
            if vocoder_config.data.aux_feats == ["mcep", "codeap"]:
                mean_ = np.load(vocoder_dir / "in_vocoder_scaler_mean.npy")
                var_ = np.load(vocoder_dir / "in_vocoder_scaler_var.npy")
                scale_ = np.load(vocoder_dir / "in_vocoder_scaler_scale.npy")
                stream_sizes = get_static_stream_sizes(
                    acoustic_config.stream_sizes,
                    acoustic_config.has_dynamic_features,
                    acoustic_config.num_windows,
                )
                mgc_end_dim = stream_sizes[0]
                bap_start_dim = sum(stream_sizes[:3])
                bap_end_dim = sum(stream_sizes[:4])
                vocoder_in_scaler = StandardScaler(
                    np.concatenate(
                        [mean_[:mgc_end_dim], mean_[bap_start_dim:bap_end_dim]]
                    ),
                    np.concatenate(
                        [var_[:mgc_end_dim], var_[bap_start_dim:bap_end_dim]]
                    ),
                    np.concatenate(
                        [scale_[:mgc_end_dim], scale_[bap_start_dim:bap_end_dim]]
                    ),
                )
            else:
                vocoder_in_scaler = StandardScaler(
                    np.load(vocoder_dir / "in_vocoder_scaler_mean.npy")[:80],
                    np.load(vocoder_dir / "in_vocoder_scaler_var.npy")[:80],
                    np.load(vocoder_dir / "in_vocoder_scaler_scale.npy")[:80],
                )
        else:
            # Normal pwg
            vocoder = load_model(path, config=vocoder_config).to(device)
            vocoder.remove_weight_norm()
            vocoder_in_scaler = StandardScaler(
                np.load(vocoder_dir / "in_vocoder_scaler_mean.npy"),
                np.load(vocoder_dir / "in_vocoder_scaler_var.npy"),
                np.load(vocoder_dir / "in_vocoder_scaler_scale.npy"),
            )

        vocoder.eval()
    else:
        vocoder = None
        vocoder_config = None
        vocoder_in_scaler = None
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
            duration_modified_labels = predict_timings(
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

        wav, _ = synthesis_from_timings(
            device=device,
            duration_modified_labels=duration_modified_labels,
            binary_dict=binary_dict,
            numeric_dict=numeric_dict,
            acoustic_model=acoustic_model,
            acoustic_config=acoustic_config,
            acoustic_in_scaler=acoustic_in_scaler,
            acoustic_out_scaler=acoustic_out_scaler,
            acoustic_out_static_scaler=acoustic_out_static_scaler,
            vocoder=vocoder,
            vocoder_config=vocoder_config,
            vocoder_in_scaler=vocoder_in_scaler,
            sample_rate=config.synthesis.sample_rate,
            frame_period=config.synthesis.frame_period,
            log_f0_conditioning=config.synthesis.log_f0_conditioning,
            subphone_features=config.synthesis.subphone_features,
            use_world_codec=config.synthesis.use_world_codec,
            force_clip_input_features=config.acoustic.force_clip_input_features,
            relative_f0=config.synthesis.relative_f0,
            feature_type=config.synthesis.feature_type,
            vocoder_type=config.synthesis.vocoder_type,
            post_filter_type=config.synthesis.post_filter_type,
            trajectory_smoothing=config.synthesis.trajectory_smoothing,
            trajectory_smoothing_cutoff=config.synthesis.trajectory_smoothing_cutoff,
            trajectory_smoothing_cutoff_f0=config.synthesis.trajectory_smoothing_cutoff_f0,
            vuv_threshold=config.synthesis.vuv_threshold,
            pre_f0_shift_in_cent=config.synthesis.pre_f0_shift_in_cent,
            post_f0_shift_in_cent=config.synthesis.post_f0_shift_in_cent,
            vibrato_scale=config.synthesis.vibrato_scale,
            force_fix_vuv=config.synthesis.force_fix_vuv,
        )
        wav = post_process(wav, config.synthesis.sample_rate)
        out_wav_path = join(out_dir, f"{utt_id}.wav")
        wavfile.write(
            out_wav_path, rate=config.synthesis.sample_rate, data=wav.astype(np.int16)
        )


def entry():
    my_app()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    my_app()  # pylint: disable=no-value-for-parameter
