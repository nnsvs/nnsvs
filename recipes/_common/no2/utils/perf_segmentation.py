import os
import sys
from glob import glob
from os.path import basename, join, splitext

import numpy as np
import yaml
from nnmnkwii.io import hts
from tqdm import tqdm
from util import segment_labels

if len(sys.argv) != 2:
    print(f"USAGE: {sys.argv[0]} config_path")
    sys.exit(-1)

config = None
with open(sys.argv[1], "r") as yml:
    config = yaml.load(yml, Loader=yaml.FullLoader)
if config is None:
    print(f"Cannot read config file: {sys.argv[1]}.")
    sys.exit(-1)

# copy mono alignments to full
mono_files = sorted(glob(join(config["out_dir"], "mono_dtw", "*.lab")))
full_files = sorted(glob(join(config["out_dir"], "generated_full_round", "*.lab")))
dst_dir = join(config["out_dir"], "full_dtw")
os.makedirs(dst_dir, exist_ok=True)

for mono, full in tqdm(zip(mono_files, full_files)):
    m, f = hts.load(mono), hts.load(full)
    assert len(m) == len(f)
    f.start_times = m.start_times
    f.end_times = m.end_times
    name = basename(mono)
    with open(join(dst_dir, name), "w") as of:
        of.write(str(f))


# segmentation
base_files = sorted(glob(join(config["out_dir"], "mono_dtw", "*.lab")))

lengths = {}

for name in ["full_dtw", "generated_full_round", "generated_mono_round"]:
    files = sorted(glob(join(config["out_dir"], name, "*.lab")))
    for idx, base in tqdm(enumerate(base_files)):
        utt_id = splitext(basename(base))[0]
        base_lab = hts.load(base)
        base_segments, start_indices, end_indices = segment_labels(
            base_lab,
            True,
            config["segmentation_threshold"],
            min_duration=config["segment_min_duration"],
            force_split_threshold=config["force_split_threshold"],
        )
        if name == "full_dtw":
            d = []
            for seg in base_segments:
                d.append((seg.end_times[-1] - seg.start_times[0]) * 1e-7)
            lengths[utt_id] = d

        lab = hts.load(files[idx])
        #        print("{}: len:{}".format(files[idx], len(lab)))
        #        print("{}: len:{}".format(base, len(base_lab)))
        assert len(lab) == len(base_lab)
        segments = []
        for s, e in zip(start_indices, end_indices):
            segments.append(lab[s : e + 1])

        dst_dir = join(config["out_dir"], f"{name}_seg")
        os.makedirs(dst_dir, exist_ok=True)
        for idx, seg in enumerate(segments):
            with open(join(dst_dir, f"{utt_id}_seg{idx}.lab"), "w") as of:
                of.write(str(seg))

        base_dst_dir = join(config["out_dir"], "mono_label_round_seg")
        os.makedirs(base_dst_dir, exist_ok=True)
        for idx, seg in enumerate(base_segments):
            with open(join(base_dst_dir, f"{utt_id}_seg{idx}.lab"), "w") as of:
                of.write(str(seg))

for ls in [lengths]:
    for k, v in ls.items():
        print(
            "{}.lab: segment duration min {:.02f}, max {:.02f}, mean {:.02f}".format(
                k, np.min(v), np.max(v), np.mean(v)
            )
        )

    flatten_lengths = []
    for k, v in ls.items():
        sys.stdout.write(f"{k}.lab: segment lengths: ")
        for d in v:
            sys.stdout.write("{:.02f}, ".format(d))
            flatten_lengths.append(d)
        sys.stdout.write("\n")

    print(
        "Segmentation stats: min {:.02f}, max {:.02f}, mean {:.02f}".format(
            np.min(flatten_lengths), np.max(flatten_lengths), np.mean(flatten_lengths)
        )
    )

    print("Total number of segments: {}".format(len(flatten_lengths)))

sys.exit(0)
