"""Normalize audio gain with sv56

About sv56:
International Telecommunication Union, Recommendation G.191:
Software Tools and Audio Coding Standardization, Nov 11 2005.
Sv56 has been used for gain normalization in several speech research including:
- Using Cyclic Noise as the Source Signal for Neural Source-Filter-based Speech Waveform Model
    - https://arxiv.org/abs/2004.02191
- Zero-Shot Multi-Speaker Text-To-Speech with State-of-the-art Neural Speaker Embeddings
    - https://arxiv.org/abs/1910.10838

sv56demo: https://github.com/foss-for-synopsys-dwc-arc-processors/G722
"""
import argparse
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from os.path import join
from subprocess import PIPE, Popen

import librosa
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Normalize audio gain with sv56",
    )
    parser.add_argument("in_dir", type=str, help="Input wav directory")
    parser.add_argument("--ndb", type=int, default=-26, help="Gain level in dB")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
    return parser


def sv56(x, sr, ndb):
    """Run sv56 gain normalization

    Args:
        x (array): waveform
        sr (int): Sampling rate
        ndb (int): Gain level in dB

    Return
        array: gain-normalized waveform
    """
    assert x.dtype == np.int16
    with tempfile.TemporaryDirectory() as f:
        tmp_dir = f
        name = "gen0001"
        pcm_path = join(tmp_dir, "{}.pcm".format(name))
        out_path = join(tmp_dir, "{}_norm.pcm".format(name))
        x.tofile(pcm_path)
        cmd = "sv56demo -qq -lev {} -sf {} {} {}".format(ndb, sr, pcm_path, out_path)
        p = Popen(cmd, shell=True, stdout=PIPE)
        r = p.wait()
        if r != 0:
            raise RuntimeError("sv56 failed to execute.")
        x_norm = np.fromfile(out_path, dtype=np.int16)
        assert len(x) == len(x_norm)
        return x_norm


def __process(in_file, out_file, ndb):
    x, sr = sf.read(in_file)

    if sr != 24000:
        y = librosa.resample(x.astype(np.float64), sr, 24000)
        sr = 24000
    else:
        y = x

    print(x.dtype, x.min(), x.max())
    if x.dtype == np.float32 or x.dtype == np.float64:
        assert np.abs(x).max() < 1.0
        x = (x * 32767).astype(np.int16)
    assert x.dtype == np.int16
    y = sv56(x, sr, ndb)
    assert y.dtype == np.int16
    sf.write(out_file, y, sr)


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    in_files = glob(join(args.in_dir, "**/*.wav"))

    with ProcessPoolExecutor(args.n_jobs) as executor:
        futures = [
            executor.submit(__process, in_file, in_file, args.ndb)
            for in_file in in_files
        ]
        for future in tqdm(futures):
            future.result()
