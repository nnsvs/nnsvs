import os
import sys
from glob import glob
from os.path import basename, expanduser, join, splitext

import yaml
from nnmnkwii.io import hts
from tqdm import tqdm
from util import fix_mono_lab_before_align

if len(sys.argv) != 2:
    print(f"USAGE: {sys.argv[0]} config_path")
    sys.exit(-1)

config = None
with open(sys.argv[1], "r") as yml:
    config = yaml.load(yml, Loader=yaml.FullLoader)
if config is None:
    print(f"Cannot read config file: {sys.argv[1]}.")
    sys.exit(-1)

print("Copy original label files.")
files = sorted(glob(join(expanduser(config["db_root"]), "**/*.lab"), recursive=True))
dst_dir = join(config["out_dir"], "mono_label")
os.makedirs(dst_dir, exist_ok=True)
for m in tqdm(files):
    if config["spk"] == "natsumeyuuri":
        # natsume_singing
        name = splitext(basename(m))[0]
        if name in config["exclude_songs"]:
            continue
        h = hts.HTSLabelFile()
        with open(m) as f:
            for label in f:
                s, e, lab = label.strip().split()
                if config["label_time_unit"] == "sec":
                    s, e = int(float(s) * 1e7), int(float(e) * 1e7)
                h.append((s, e, lab))
            with open(join(dst_dir, basename(m)), "w") as of:
                of.write(str(fix_mono_lab_before_align(h, config["spk"])))
    else:
        # ofuton_p_utagoe_db, oniku_kurumi_utagoe_db
        name = splitext(basename(m))[0]
        if name in config["exclude_songs"]:
            continue
        f = hts.load(m)
        with open(join(dst_dir, basename(m)), "w") as of:
            of.write(str(fix_mono_lab_before_align(f, config["spk"])))

# Rounding
print("Round label files.")
frame_shift = 50000
for lab_type in ["generated_mono", "generated_full", "mono_label"]:
    files = sorted(glob(join(config["out_dir"], lab_type, "*.lab")))
    dst_dir = join(config["out_dir"], lab_type + "_round")
    os.makedirs(dst_dir, exist_ok=True)

    for path in tqdm(files):
        lab = hts.load(path)
        name = basename(path)

        for x in range(len(lab)):
            lab.start_times[x] = round(lab.start_times[x] / 50000) * 50000
            lab.end_times[x] = round(lab.end_times[x] / 50000) * 50000

        # Check if rounding is done property
        if lab_type == "mono_label":
            for i in range(len(lab) - 1):
                # Corner case: rounded to zero duration
                if lab.end_times[i] == lab.start_times[i]:
                    print(
                        "Detected zero frames. Assign one frame from the next phoneme"
                    )
                    print(name, lab[i])
                    # let's consume one frame from the next (presumably) vowel
                    d = (lab.end_times[i + 1] - lab.start_times[i + 1]) // frame_shift
                    assert d >= 2
                    lab.end_times[i] += frame_shift
                    lab.start_times[i + 1] += frame_shift

                if lab.end_times[i] != lab.start_times[i + 1]:
                    print(path)
                    print(i, lab[i])
                    print(i + 1, lab[i + 1])

        with open(join(dst_dir, name), "w") as of:
            of.write(str(lab))
