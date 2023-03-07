"""Web server implementation for singing voice synthesis

NOTE: validation is not implemented. Expect 500 errors for unexpected inputs.
"""

import tarfile
from os import listdir, rmdir
from pathlib import Path
from shutil import move

import numpy as np
import pyworld
import torch
from fastapi import FastAPI, UploadFile
from nnmnkwii.io import hts
from nnsvs.io.hts import full_to_mono
from nnsvs.svs import NEUTRINO
from omegaconf import OmegaConf
from scipy.io import wavfile
from starlette.responses import RedirectResponse, StreamingResponse
from utaupy.utils import ust2hts

SCORE_DIR = Path("./score")
MUSICXML_DIR = SCORE_DIR / "musicxml"
FULL_LAB_DIR = SCORE_DIR / "label" / "full"
MONO_LAB_DIR = SCORE_DIR / "label" / "mono"
UST_DIR = SCORE_DIR / "ust"

TIMING_LAB_DIR = SCORE_DIR / "label" / "timing"
OUTPUT_DIR = Path("./output")
MODEL_DIR = Path("./model")

for d in [
    SCORE_DIR,
    MUSICXML_DIR,
    FULL_LAB_DIR,
    MONO_LAB_DIR,
    UST_DIR,
    TIMING_LAB_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
]:
    d.mkdir(exist_ok=True, parents=True)

app = FastAPI()

_models = {}


def _instantiate_model(model_id):
    global _models
    if model_id in _models:
        return _models[model_id]
    model = NEUTRINO(
        MODEL_DIR / model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=100,
    )
    _models[model_id] = model
    return model


def _finalize():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@app.get("/healthcheck")
def perform_healthcheck():
    return {"healthcheck": "OK"}


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")


@app.get("/models/list")
async def model_list():
    model_ids = listdir(MODEL_DIR)
    return {"model_ids": model_ids}


@app.get("/models/{model_id}")
async def model_info(model_id: str):
    model = NEUTRINO(
        MODEL_DIR / model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=100,
    )

    return {
        "config": OmegaConf.to_container(model.config),
        "repr": repr(model),
    }


@app.post("/models/create")
def create_model(model: UploadFile, model_id: str):
    filename = Path(model.filename)

    if str(filename).endswith(".tar.gz"):
        model_dir = MODEL_DIR / model_id
        model_dir.mkdir(exist_ok=True)
        with tarfile.open(fileobj=model.file, mode="r|gz") as f:
            f.extractall(path=model_dir)

        tar_dir_path = model_dir / filename.name.replace(".tar.gz", "")
        # Move all contents to model_dir / model_id
        if (tar_dir_path).exists():
            for name in listdir(tar_dir_path):
                move(tar_dir_path / name, model_dir / name)
            rmdir(tar_dir_path)
    else:
        raise NotImplementedError()

    return {"filename": model.filename}


@app.post("/score/full/upload")
async def upload_full_lab(full_lab: UploadFile):
    with open(f"{FULL_LAB_DIR}/{full_lab.filename}", "wb") as f:
        f.write(full_lab.file.read())
    return {"filename": full_lab.filename}


@app.post("/score/timing/upload")
async def upload_timing_lab(timing_lab: UploadFile):
    with open(f"{TIMING_LAB_DIR}/{timing_lab.filename}", "wb") as f:
        f.write(timing_lab.file.read())
    return {"filename": timing_lab.filename}


@app.post("/score/musicxml/upload")
async def upload_musicxml(musicxml: UploadFile):
    filename = musicxml.filename
    musicxml_path = MUSICXML_DIR / filename
    with open(musicxml_path, "wb") as f:
        f.write(musicxml.file.read())

    full_labels, mono_labels = NEUTRINO.musicxml2label(str(musicxml_path))
    full_lab_path = FULL_LAB_DIR / (
        filename.replace(".musicxml", "").replace(".xml", "") + ".lab"
    )
    mono_lab_path = MONO_LAB_DIR / (
        filename.replace(".musicxml", "").replace(".xml", "") + ".lab"
    )
    with open(full_lab_path, "w") as f:
        f.write(str(full_labels))
    with open(mono_lab_path, "w") as f:
        f.write(str(mono_labels))

    return {"filename": filename}


@app.post("/score/ust/upload")
async def upload_ust(ust: UploadFile, model_id: str):
    ust_path = UST_DIR / ust.filename
    with open(ust_path, "wb") as f:
        f.write(ust.file.read())

    model_dir = MODEL_DIR / model_id
    table_path = model_dir / "kana2phonemes.table"
    assert table_path.exists()

    full_lab = FULL_LAB_DIR / ust.filename.replace(".ust", ".lab")
    ust2hts(
        str(ust_path), full_lab, table_path, strict_sinsy_style=False, as_mono=False
    )

    return {"filename": ust.filename}


@app.get("/run/timing")
async def run_timing(name: str, model_id: str):
    model = _instantiate_model(model_id)
    model.set_device("cuda" if torch.cuda.is_available() else "cpu")

    full_lab = FULL_LAB_DIR / (name + ".lab")
    assert full_lab.exists(), f"{full_lab} does not exist"
    full_labels = hts.load(full_lab)
    timing_labels = full_to_mono(model.predict_timing(full_labels))

    # TODO: Do we want to save timing for each model?
    # timing_lab = TIMING_LAB_DIR / model_id / (name + ".lab")
    timing_lab = TIMING_LAB_DIR / (name + ".lab")
    with open(timing_lab, "w") as f:
        f.write(str(timing_labels))

    _finalize()

    return {"timing": str(timing_labels)}


@app.get("/run/phrases")
async def run_phrases(name: str, model_id: str):
    model = _instantiate_model(model_id)
    model.set_device("cuda" if torch.cuda.is_available() else "cpu")

    full_lab = FULL_LAB_DIR / (name + ".lab")
    assert full_lab.exists(), f"{full_lab} does not exist"
    full_labels = hts.load(full_lab)

    timing_lab = TIMING_LAB_DIR / (name + ".lab")
    assert timing_lab.exists(), "Timing labels not found. "
    timing_labels = hts.load(timing_lab)

    model_output_dir = OUTPUT_DIR / model_id
    model_output_dir.mkdir(exist_ok=True)

    # Dump phraselist
    phraselist_path = model_output_dir / (name + "-phraselist.txt")
    phraselist = model.get_phraselist(full_labels, timing_labels)
    with open(phraselist_path, "w") as f:
        f.write(str(phraselist))

    # Dump num_phrases for convenience
    num_phrases = model.get_num_phrases(full_labels)

    return {"phraselist": phraselist, "num_phrases": num_phrases}


@app.get("/run/acoustic")
async def run_acoustic(name: str, model_id: str, phrase_num: int = -1):
    model = _instantiate_model(model_id)
    model.set_device("cuda" if torch.cuda.is_available() else "cpu")

    full_lab = FULL_LAB_DIR / (name + ".lab")
    assert full_lab.exists(), f"{full_lab} does not exist"
    full_labels = hts.load(full_lab)
    timing_lab = TIMING_LAB_DIR / (name + ".lab")
    assert timing_lab.exists(), "Timing labels not found. "
    timing_labels = hts.load(timing_lab)

    model_output_dir = OUTPUT_DIR / model_id
    model_output_dir.mkdir(exist_ok=True)

    f0, mgc, bap = model.predict_acoustic(
        full_labels,
        timing_labels,
        phrase_num=phrase_num,
    )
    _finalize()

    if phrase_num > 0:
        name = f"{name}-{phrase_num}"

    f0.tofile(model_output_dir / (name + ".f0"))
    mgc.tofile(model_output_dir / (name + ".mgc"))
    bap.tofile(model_output_dir / (name + ".bap"))

    # NOTE: pack into a single file for convenience
    feats = np.concatenate([f0, mgc, bap], axis=1).astype(np.float64)
    path = model_output_dir / (name + ".bin")
    feats.tofile(path)

    def iterfile():
        with open(path, mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile())


@app.get("/run/vocoder")
async def run_vocoder(
    name: str,
    model_id: str,
    vocoder_type: str = "world",
    phrase_num: int = -1,
    loudness_norm: bool = False,
    dtype: str = "int16",
):
    model = _instantiate_model(model_id)
    model.set_device("cuda" if torch.cuda.is_available() else "cpu")

    if phrase_num > 0:
        name = f"{name}-{phrase_num}"

    f0_path = OUTPUT_DIR / model_id / (name + ".f0")
    mgc_path = OUTPUT_DIR / model_id / (name + ".mgc")
    bap_path = OUTPUT_DIR / model_id / (name + ".bap")

    f0 = np.fromfile(f0_path, dtype=np.float64).reshape(-1, 1)
    mgc = np.fromfile(mgc_path, dtype=np.float64).reshape(-1, 60)
    bap = np.fromfile(bap_path, dtype=np.float64).reshape(
        -1, pyworld.get_num_aperiodicities(model.sample_rate)
    )

    wav = model.predict_waveform(
        f0,
        mgc,
        bap,
        vocoder_type=vocoder_type,
        loudness_norm=loudness_norm,
        dtype=dtype,
    )
    _finalize()

    if vocoder_type == "world":
        suffix = "_syn.wav"
    else:
        suffix = "_nsf.wav"

    wav_path = OUTPUT_DIR / model_id / (name + suffix)
    wavfile.write(wav_path, model.sample_rate, wav)

    path = OUTPUT_DIR / model_id / (name + suffix.replace(".wav", ".raw"))
    wav.tofile(path)

    def iterfile():
        with open(path, mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile())
