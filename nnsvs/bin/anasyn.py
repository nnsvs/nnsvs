import os
from os.path import join

import hydra
import numpy as np
import pysptk
import pyworld
import torch
from hydra.utils import to_absolute_path
from nnsvs.dsp import bandpass_filter
from nnsvs.gen import gen_world_params
from nnsvs.logger import getLogger
from nnsvs.multistream import get_static_stream_sizes, split_streams
from nnsvs.svs import load_vocoder
from nnsvs.util import init_seed, load_utt_list
from omegaconf import DictConfig, OmegaConf
from scipy.io import wavfile
from tqdm.auto import tqdm


@torch.no_grad()
def anasyn(
    device,
    acoustic_features,
    acoustic_config,
    vocoder=None,
    vocoder_config=None,
    vocoder_in_scaler=None,
    sample_rate=48000,
    frame_period=5,
    vuv_threshold=0.5,
    use_world_codec=True,
    feature_type="world",
    vocoder_type="world",
):
    static_stream_sizes = get_static_stream_sizes(
        acoustic_config.stream_sizes,
        acoustic_config.has_dynamic_features,
        acoustic_config.num_windows,
    )

    # Split multi-stream features
    streams = split_streams(acoustic_features, static_stream_sizes)

    # Generate WORLD parameters
    if feature_type == "world":
        assert len(streams) == 4
        mgc, lf0, vuv, bap = streams
    elif feature_type == "melf0":
        mel, lf0, vuv = split_streams(acoustic_features, [80, 1, 1])
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    # Waveform generation by (1) WORLD or (2) neural vocoder
    if vocoder_type == "world":
        f0, spectrogram, aperiodicity = gen_world_params(
            mgc,
            lf0,
            vuv,
            bap,
            sample_rate,
            vuv_threshold=vuv_threshold,
            use_world_codec=use_world_codec,
        )
        wav = pyworld.synthesize(
            f0,
            spectrogram,
            aperiodicity,
            sample_rate,
            frame_period,
        )
    elif vocoder_type == "pwg":
        # NOTE: So far vocoder models are trained on binary V/UV features
        vuv = (vuv > vuv_threshold).astype(np.float32)
        if feature_type == "world":
            voc_inp = (
                torch.from_numpy(
                    vocoder_in_scaler.transform(
                        np.concatenate([mgc, lf0, vuv, bap], axis=-1)
                    )
                )
                .float()
                .to(device)
            )
        elif feature_type == "melf0":
            voc_inp = (
                torch.from_numpy(
                    vocoder_in_scaler.transform(
                        np.concatenate([mel, lf0, vuv], axis=-1)
                    )
                )
                .float()
                .to(device)
            )
        wav = vocoder.inference(voc_inp).view(-1).to("cpu").numpy()
    elif vocoder_type == "usfgan":
        if feature_type == "world":
            fftlen = pyworld.get_cheaptrick_fft_size(sample_rate)
            use_mcep_aperiodicity = bap.shape[-1] > 5
            if use_mcep_aperiodicity:
                mcep_aperiodicity_order = bap.shape[-1] - 1
                alpha = pysptk.util.mcepalpha(sample_rate)
                aperiodicity = pysptk.mc2sp(
                    np.ascontiguousarray(bap).astype(np.float64),
                    fftlen=fftlen,
                    alpha=alpha,
                )
            else:
                aperiodicity = pyworld.decode_aperiodicity(
                    np.ascontiguousarray(bap).astype(np.float64), sample_rate, fftlen
                )
            # fill aperiodicity with ones for unvoiced regions
            aperiodicity[vuv.reshape(-1) < vuv_threshold, 0] = 1.0
            # WORLD fails catastrophically for out of range aperiodicity
            aperiodicity = np.clip(aperiodicity, 0.0, 1.0)
            # back to bap
            if use_mcep_aperiodicity:
                bap = pysptk.sp2mc(
                    aperiodicity,
                    order=mcep_aperiodicity_order,
                    alpha=alpha,
                )
            else:
                bap = pyworld.code_aperiodicity(aperiodicity, sample_rate).astype(
                    np.float32
                )

            aux_feats = (
                torch.from_numpy(
                    vocoder_in_scaler.transform(np.concatenate([mgc, bap], axis=-1))
                )
                .float()
                .to(device)
            )
        elif feature_type == "melf0":
            # NOTE: So far vocoder models are trained on binary V/UV features
            vuv = (vuv > vuv_threshold).astype(np.float32)
            aux_feats = (
                torch.from_numpy(vocoder_in_scaler.transform(mel)).float().to(device)
            )
        contf0 = np.exp(lf0)
        if vocoder_config.data.sine_f0_type in ["contf0", "cf0"]:
            f0_inp = contf0
        elif vocoder_config.data.sine_f0_type == "f0":
            f0_inp = contf0
            f0_inp[vuv < vuv_threshold] = 0
        wav = vocoder.inference(f0_inp, aux_feats).view(-1).to("cpu").numpy()

    return wav


def post_process(wav, sample_rate):
    wav = bandpass_filter(wav, sample_rate)

    if np.max(wav) > 10:
        if np.abs(wav).max() > 32767:
            wav = wav / np.abs(wav).max()
        # data is likely already in [-32768, 32767]
        wav = wav.astype(np.int16)
    else:
        if np.abs(wav).max() > 1.0:
            wav = wav / np.abs(wav).max()
        wav = (wav * 32767.0).astype(np.int16)
    return wav


@hydra.main(config_path="conf/synthesis", config_name="config")
def my_app(config: DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(OmegaConf.to_yaml(config))

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    acoustic_config = OmegaConf.load(to_absolute_path(config.acoustic.model_yaml))

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

    in_dir = to_absolute_path(config.in_dir)
    out_dir = to_absolute_path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    utt_ids = load_utt_list(to_absolute_path(config.utt_list))
    logger.info("Processes %s utterances...", len(utt_ids))
    for utt_id in tqdm(utt_ids):
        acoustic_features = np.load(join(in_dir, f"{utt_id}-feats.npy"))
        init_seed(1234)

        wav = anasyn(
            device=device,
            acoustic_features=acoustic_features,
            acoustic_config=acoustic_config,
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
        wav = post_process(wav, config.synthesis.sample_rate)
        out_wav_path = join(out_dir, f"{utt_id}.wav")
        wavfile.write(
            out_wav_path, rate=config.synthesis.sample_rate, data=wav.astype(np.int16)
        )


def entry():
    my_app()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    my_app()  # pylint: disable=no-value-for-parameter
