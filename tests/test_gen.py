from pathlib import Path

import numpy as np
import pysptk
import pyworld
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from nnmnkwii.preprocessing.f0 import interp1d
from nnsvs.gen import correct_vuv_by_phone, gen_spsvs_static_features
from scipy.io import wavfile


def _extract_static_feats(wav, sr):
    f0, timeaxis = pyworld.dio(wav, sr, frame_period=5)
    spectrogram = pyworld.cheaptrick(wav, f0, timeaxis, sr)
    aperiodicity = pyworld.d4c(wav, f0, timeaxis, sr)

    mgc = pysptk.sp2mc(spectrogram, order=59, alpha=pysptk.util.mcepalpha(sr))
    f0 = f0[:, None]
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    vuv = (lf0 != 0).astype(np.float32)
    lf0 = interp1d(lf0, kind="slinear")
    bap = pyworld.code_aperiodicity(aperiodicity, sr)

    feats = np.hstack((mgc, lf0, vuv, bap)).astype(np.float32)
    stream_sizes = [mgc.shape[1], lf0.shape[1], vuv.shape[1], bap.shape[1]]

    return feats, stream_sizes


def test_gen_spsvs_static_features():
    wav_path = Path(__file__).parent / "data" / "nitech_jp_song070_f001_004.wav"
    lab_path = Path(__file__).parent / "data" / "nitech_jp_song070_f001_004.lab"

    binary_dict, numeric_dict = hts.load_question_set(
        Path(__file__).parent / "data" / "jp_test.hed"
    )

    labels = hts.load(lab_path)
    sr, wav = wavfile.read(wav_path)
    wav = wav.astype(np.float64)
    assert sr == 48000

    out_feats, stream_sizes = _extract_static_feats(wav, sr)
    has_dynamic_features = [False] * len(stream_sizes)
    pitch_idx = len(binary_dict) + 1

    params = {
        "labels": labels,
        "acoustic_features": out_feats,
        "binary_dict": binary_dict,
        "numeric_dict": numeric_dict,
        "stream_sizes": stream_sizes,
        "has_dynamic_features": has_dynamic_features,
        "pitch_idx": pitch_idx,
        "relative_f0": False,
        "frame_period": 5,
        "force_fix_vuv": False,
    }

    mgc, lf0, vuv, bap = gen_spsvs_static_features(**params)
    assert mgc.shape[1] == 60
    assert lf0.shape[1] == 1
    assert vuv.shape[1] == 1
    assert bap.shape[1] == 5

    # w/o V/UV correction, vuv should't change
    out_vuv_idx = 61
    N = np.abs(vuv - out_feats[:, out_vuv_idx : out_vuv_idx + 1]).sum()
    assert int(N) == 0


def test_correct_vuv_by_phone():
    wav_path = Path(__file__).parent / "data" / "nitech_jp_song070_f001_004.wav"
    lab_path = Path(__file__).parent / "data" / "nitech_jp_song070_f001_004.lab"

    binary_dict, numeric_dict = hts.load_question_set(
        Path(__file__).parent / "data" / "jp_test.hed"
    )

    labels = hts.load(lab_path)
    sr, wav = wavfile.read(wav_path)
    wav = wav.astype(np.float64)
    assert sr == 48000

    out_feats, stream_sizes = _extract_static_feats(wav, sr)
    has_dynamic_features = [False] * len(stream_sizes)
    pitch_idx = len(binary_dict) + 1

    linguistic_features = fe.linguistic_features(
        labels,
        binary_dict,
        numeric_dict,
        add_frame_features=True,
        subphone_features="coarse_coding",
    )

    params = {
        "labels": labels,
        "acoustic_features": out_feats,
        "binary_dict": binary_dict,
        "numeric_dict": numeric_dict,
        "stream_sizes": stream_sizes,
        "has_dynamic_features": has_dynamic_features,
        "pitch_idx": pitch_idx,
        "relative_f0": False,
        "frame_period": 5,
    }

    out_vuv_idx = 61
    vuv = out_feats[:, out_vuv_idx : out_vuv_idx + 1]

    vuv_corrected = correct_vuv_by_phone(vuv, binary_dict, linguistic_features)
    # by correcting VUV should make a difference
    _, _, vuv_fixed, _ = gen_spsvs_static_features(**{**params, "force_fix_vuv": True})
    assert np.any(vuv_corrected != vuv)

    # 0: Rest 1: Voiced 2: Unvoiced
    rest_idx = 0
    voiced_idx = 1
    unvoiced_idx = 2
    assert np.all(vuv_corrected[linguistic_features[:, rest_idx] > 0] < 0.5)
    assert np.all(vuv_corrected[linguistic_features[:, voiced_idx] > 0] > 0.5)
    assert np.all(vuv_corrected[linguistic_features[:, unvoiced_idx] > 0] < 0.5)
