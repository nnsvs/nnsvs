"""Predict waveform by neural vocoders with NEUTRINO-compatible file IO

NOTE: options are not yet fully implemented

NSF - Neural Source Filter (v1.2.0-Stable)
        Copyright (c) 2020-2022 STUDIO NEUTRINO All rights reserved.

usage:
    NSF input.f0 input.mgc input.bap model_name output_wav [option]
    options : description [default]
    -s i            : sampling rate (kHz)             [   48]
    -n i            : number of parallel              [  MAX]
    -p i            : number of parallel in session   [    1]
    -l file name    : multi phrase prediction         [ none]
    -g              : use gpu                         [  off]
    -i i            : gpu id                          [    0]
    -t              : view information                [  off]
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pretend as if the script is NSF of NEUTRINO",
    )
    parser.add_argument("input_f0", type=str, help="Path of input F0")
    parser.add_argument("input_mgc", type=str, help="Path of input MGC")
    parser.add_argument("input_bap", type=str, help="Path of input BAP")
    parser.add_argument("model_dir", type=str, help="model_dir")
    parser.add_argument("output_wav", type=str, help="Path of output wav")
    parser.add_argument("--use_api", action="store_true", help="Use web API")
    parser.add_argument(
        "--url", type=str, default="http://127.0.0.1:8001", help="URL of the server"
    )
    return parser


def run_local(args, _):
    import pyworld
    import torch
    from nnsvs.svs import NEUTRINO

    model_dir = Path(args.model_dir)
    # NOTE: this is needed to be compatible with NEUTRINO's Run.bat
    if not model_dir.exists():
        model_dir = "model" / model_dir
    engine = NEUTRINO(model_dir, device="cuda" if torch.cuda.is_available() else "cpu")

    f0 = np.fromfile(args.input_f0, dtype=np.float64).reshape(-1, 1)
    mgc = np.fromfile(args.input_mgc, dtype=np.float64).reshape(-1, 60)
    bap = np.fromfile(args.input_bap, dtype=np.float64).reshape(
        -1, pyworld.get_num_aperiodicities(engine.sample_rate)
    )

    # NOTE: `auto` will run uSFGAN or PWG if a trained one is in the model_dir
    # and fallback to WORLD it doesn't exist.
    wav = engine.predict_waveform(f0, mgc, bap, vocoder_type="auto", dtype=np.int16)

    return wav, engine.sample_rate


def run_api(args, logger):
    url = args.url[:-1] if args.url[-1] == "/" else args.url

    # NOTE: for webAPI, these local files are not used for inference
    # files on the remote server are used instead.
    input_f0 = Path(args.input_f0)
    input_mgc = Path(args.input_mgc)
    input_bap = Path(args.input_bap)
    assert input_f0.exists()
    assert input_mgc.exists()
    assert input_bap.exists()

    name = input_f0.stem
    # TODO: better way to handle phrase-based synthesis
    # At the moment, we need to tell the server which phrase we are synthesizing
    if "-" in name:
        # NOTE: this code does not work for complicated filenames
        phrase_num = int(name.split("-")[-1])
        name = name.split("-")[0]
    else:
        phrase_num = -1

    # pretend as if the model_dir is model_id
    model_id = args.model_dir

    # Get sampling rate
    res = requests.get(
        url + f"/models/{model_id}",
    )
    if res.status_code != 200:
        raise RuntimeError(f"Failed to fetch model info: {res.status_code}")
    res = res.json()
    sample_rate = res["config"]["sample_rate"]

    # Run vocoder
    logger.info("Predicting waveform")
    dtype = "int16"
    res = requests.get(
        url + "/run/vocoder",
        params={
            "name": name,
            "model_id": model_id,
            "phrase_num": phrase_num,
            "vocoder_type": "auto",
            "dtype": dtype,
            "loudness_norm": False,
        },
    )
    if res.status_code != 200:
        raise RuntimeError(f"Failed to generate waveform: {res.status_code}")

    wav = np.frombuffer(res.content, dtype=dtype).reshape(-1)

    return wav, sample_rate


def main():
    args = get_parser().parse_args(sys.argv[1:])

    start_time = time.time()
    format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logger = logging.getLogger("NSF")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(stream_handler)

    if args.use_api:
        logger.info(f"Using webapi: {args.url} for inference")
        wav, sr = run_api(args, logger)
    else:
        logger.info("Using local machine for inference")
        wav, sr = run_local(args, logger)

    sf.write(args.output_wav, wav, sr)
    logger.info(f"Elapsed time: {time.time() - start_time:.2f} sec")


if __name__ == "__main__":
    main()
