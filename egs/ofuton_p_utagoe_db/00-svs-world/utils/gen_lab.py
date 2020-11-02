import pysinsy
import os
from glob import glob
from os.path import join, basename, expanduser, splitext
from nnmnkwii.io import hts
from util import merge_sil, fix_mono_lab_before_align
from tqdm import tqdm

import yaml
with open('config.yaml', 'r') as yml:
    config = yaml.load(yml, Loader=yaml.FullLoader)

sinsy = pysinsy.sinsy.Sinsy()

assert sinsy.setLanguages("j", config["sinsy_dic"])

# generate full/mono labels by sinsy
print("Convert musicxml to label files.")
files = sorted(glob(join(expanduser(config["db_root"]), "**/*.musicxml"), recursive=True))
for path in tqdm(files):
    assert sinsy.loadScoreFromMusicXML(path)
    for is_mono in [True, False]:
        n = "sinsy_mono" if is_mono else "sinsy_full"
        labels = sinsy.createLabelData(is_mono, 1, 1).getData()
        lab = hts.HTSLabelFile()
        for l in labels:
            lab.append(l.split(), strict=False)
        lab = merge_sil(lab)
        dst_dir = join(config["out_dir"], f"{n}")
        os.makedirs(dst_dir, exist_ok=True)
        name = splitext(basename(path))[0]
        with open(join(dst_dir, name + ".lab"), "w") as f:
            f.write(str(lab))
    sinsy.clearScore()

print("Copy original label files.")
files = sorted(glob(join(expanduser(config["db_root"]), "**/*.lab"), recursive=True))
dst_dir = join(config["out_dir"], "mono_label")
os.makedirs(dst_dir, exist_ok=True)
for m in tqdm(files):
    f = hts.load(m)
    with open(join(dst_dir, basename(m)), "w") as of:
        of.write(str(f))

# Rounding
print("Round label files.")
for name in ["sinsy_mono", "sinsy_full", "mono_label"]:
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
            for i in range(len(lab)-1):
                if lab.end_times[i] != lab.start_times[i+1]:
                    print(path)
                    print(i, lab[i])
                    print(i+1, lab[i+1])
                    import ipdb; ipdb.set_trace()

        with open(join(dst_dir, name), "w") as of:
            of.write(str(lab))
