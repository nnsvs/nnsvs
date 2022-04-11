import os
import sys
from glob import glob
from os.path import basename, expanduser, join, splitext

import pysinsy
import yaml
from nnmnkwii.io import hts
from tqdm import tqdm
from util import fix_mono_lab_before_align, merge_sil

if len(sys.argv) != 2:
    print(f"USAGE: {sys.argv[0]} config_path")
    sys.exit(-1)

config = None
with open(sys.argv[1], "r") as yml:
    config = yaml.load(yml, Loader=yaml.FullLoader)
if config is None:
    print(f"Cannot read config file: {sys.argv[1]}.")
    sys.exit(-1)

sinsy = pysinsy.sinsy.Sinsy()

assert sinsy.setLanguages("j", config["sinsy_dic"])

# generate full/mono labels by sinsy
print("Convert musicxml to label files.")
files = sorted(glob(join(expanduser(config["db_root"]), "**/*.*xml"), recursive=True))
for path in tqdm(files):
    name = splitext(basename(path))[0]
    if name in config["exclude_songs"]:
        continue

    assert sinsy.loadScoreFromMusicXML(path)
    for is_mono in [True, False]:
        n = "generated_mono" if is_mono else "generated_full"
        labels = sinsy.createLabelData(is_mono, 1, 1).getData()
        lab = hts.HTSLabelFile()
        for label in labels:
            lab.append(label.split(), strict=False)
        lab = merge_sil(lab)
        dst_dir = join(config["out_dir"], f"{n}")
        os.makedirs(dst_dir, exist_ok=True)
        with open(join(dst_dir, name + ".lab"), "w") as f:
            f.write(str(lab))
    sinsy.clearScore()

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
for name in ["generated_mono", "generated_full", "mono_label"]:
    files = sorted(glob(join(config["out_dir"], name, "*.lab")))
    dst_dir = join(config["out_dir"], name + "_round")
    os.makedirs(dst_dir, exist_ok=True)

    for path in tqdm(files):
        lab = hts.load(path)
        name = basename(path)

        for x in range(len(lab)):
            lab.start_times[x] = round(lab.start_times[x] / 50000) * 50000
            lab.end_times[x] = round(lab.end_times[x] / 50000) * 50000

        # Check if rounding is done property
        if name == "mono_label":
            for i in range(len(lab) - 1):
                if lab.end_times[i] != lab.start_times[i + 1]:
                    print(path)
                    print(i, lab[i])
                    print(i + 1, lab[i + 1])
                    import ipdb

                    ipdb.set_trace()

        with open(join(dst_dir, name), "w") as of:
            of.write(str(lab))
