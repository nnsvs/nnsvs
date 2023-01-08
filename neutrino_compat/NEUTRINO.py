"""Predict acoustic features by NNSVS with NEUTRINO-compatible file IO

NOTE: options are not yet fully implemented

NEUTRINO - NEURAL SINGING SYNTHESIZER (Electron v1.2.0-Stable)
            Copyright (c) 2020-2022 STUDIO NEUTRINO All rights reserved.

usage:
    NEUTRINO full.lab timing.lab output.f0 output.mgc output.bap model_directory [option]
    options : description [default]
    -n i        : number of threads (CPU)                 [MAX]
    -k i        : style shift                             [  0]
    -s          : skip timing prediction                  [off]
    -a          : skip acoustic features prediction       [off]
    -p i        : single phrase prediction                [ -1]
    -i filename : trace phrase information                [off]
    -t          : view information                        [off]

"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from nnmnkwii.io import hts
from nnsvs.io.hts import full_to_mono
from nnsvs.logger import getLogger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pretend as if the script is NEUTRINO",
    )
    parser.add_argument("full_lab", type=str, help="Full context labels")
    parser.add_argument("timing_lab", type=str, help="Path of timing labels")
    parser.add_argument("output_f0", type=str, help="Path of output F0")
    parser.add_argument("output_mgc", type=str, help="Path of output MGC")
    parser.add_argument("output_bap", type=str, help="Path of output BAP")
    parser.add_argument("model_dir", type=str, help="model_dir")
    parser.add_argument(
        "-i", "--phraselist", type=str, default=None, help="Path of phraselist"
    ),
    parser.add_argument(
        "-p", "--phrase_num", type=int, default=-1, help="Phrase number"
    )
    parser.add_argument("--use_api", action="store_true", help="Use web API")
    parser.add_argument(
        "--url", type=str, default="http://127.0.0.1:8001", help="URL of the server"
    )
    return parser


def run_local(args, _):
    from nnsvs.svs import NEUTRINO

    model_dir = Path(args.model_dir)
    engine = NEUTRINO(
        model_dir, device="cuda" if torch.cuda.is_available() else "cpu", verbose=100
    )

    full_lab = Path(args.full_lab)
    assert full_lab.exists()
    full_labels = hts.load(full_lab)

    timing_lab = Path(args.timing_lab)
    if not timing_lab.exists():
        timing_labels = full_to_mono(engine.predict_timing(full_labels))
        with open(timing_lab, "w") as f:
            f.write(str(timing_labels))
    else:
        timing_labels = hts.load(timing_lab)

    if args.phraselist is not None:
        phraselist = Path(args.phraselist)
        if not phraselist.exists():
            phraselist_str = engine.get_phraselist(full_labels, timing_labels)
            with open(phraselist, "w") as f:
                f.write(phraselist_str)
        num_phrases = engine.get_num_phrases(full_labels)
        if args.phrase_num < 0 or args.phrase_num >= num_phrases:
            raise ValueError(f"phrase_num must be in [0, {num_phrases - 1}]")

    f0, mgc, bap = engine.predict_acoustic(
        full_labels, timing_labels, phrase_num=args.phrase_num
    )
    return f0, mgc, bap


def run_api(args, logger):
    full_lab = Path(args.full_lab)
    url = args.url[:-1] if args.url[-1] == "/" else args.url
    assert full_lab.exists()

    name = full_lab.stem
    # pretend as if the model_dir is model_id
    model_id = args.model_dir

    # Upload full-context labels
    logger.info(f"Uploading full_lab: {full_lab}")
    res = requests.post(
        url + "/score/full/upload",
        files={
            "full_lab": open(full_lab, "rb"),
        },
    )
    if res.status_code != 200:
        raise RuntimeError(f"Failed to upload full_lab: {res.status_code}")

    # Predict timing
    logger.info("Predicting timing")
    res = requests.get(
        url + "/run/timing",
        params={
            "name": name,
            "model_id": model_id,
        },
    )
    if res.status_code != 200:
        raise RuntimeError(f"Failed to predict timing: {res.status_code}")
    timing_str = res.json()["timing"]
    logger.info(timing_str)
    with open(args.timing_lab, "w") as f:
        f.write(timing_str)

    # Phraselist
    if args.phraselist is not None:
        logger.info("Predicting phraselist")
        res = requests.get(
            url + "/run/phrases",
            params={
                "name": name,
                "model_id": model_id,
            },
        )
        if res.status_code != 200:
            raise RuntimeError(f"Failed to predict phraselist: {res.status_code}")
        phraselist_str = res.json()["phraselist"]
        logger.info(phraselist_str)
        with open(args.phraselist, "w") as f:
            f.write(phraselist_str)

    # Predict acoustic features
    logger.info("Predicting acoustic features")
    res = requests.get(
        url + "/run/acoustic",
        params={
            "name": name,
            "model_id": model_id,
            "phrase_num": args.phrase_num,
        },
    )
    if res.status_code != 200:
        raise RuntimeError(f"Failed to predict acoustic features: {res.status_code}")

    feats = np.frombuffer(res.content, dtype=np.float64).reshape(-1, 66)
    f0 = feats[:, :1]
    mgc = feats[:, 1:61]
    bap = feats[:, 61:]
    logger.info(f"f0: {f0.shape}")
    logger.info(f"mgc: {mgc.shape}")
    logger.info(f"bap: {bap.shape}")

    return f0, mgc, bap


def main():
    args = get_parser().parse_args(sys.argv[1:])

    start_time = time.time()
    logger = getLogger(verbose=100, name="neutrino")
    if args.use_api:
        logger.info(f"Using webapi: {args.url}")
        f0, mgc, bap = run_api(args, logger)
    else:
        logger.info("Using local machine")
        f0, mgc, bap = run_local(args, logger)

    # Save to file
    f0.tofile(args.output_f0)
    mgc.tofile(args.output_mgc)
    bap.tofile(args.output_bap)

    logger.info(f"Elapsed time: {time.time() - start_time:.2f} sec")


if __name__ == "__main__":
    main()
