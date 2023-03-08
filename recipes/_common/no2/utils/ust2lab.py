import os
import sys
from glob import glob
from os.path import basename, expanduser, join, splitext

import yaml
from nnmnkwii.io import hts
from tqdm import tqdm
from utaupy.utils import ust2hts
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

table_path = join(config["utaupy_table_path"])

# generate full/mono labels by utaupy
print("Convert ust to label files.")
files = sorted(glob(join(expanduser(config["db_root"]), "**/*.ust"), recursive=True))
for ust_path in tqdm(files):
    name = splitext(basename(ust_path))[0]
    if name in config["exclude_songs"]:
        continue

    for as_mono in [True, False]:
        n = "generated_mono" if as_mono else "generated_full"
        dst_dir = join(config["out_dir"], f"{n}")
        os.makedirs(dst_dir, exist_ok=True)

        lab_path = join(dst_dir, name + ".lab")

        ust2hts(
            ust_path, lab_path, table_path, strict_sinsy_style=False, as_mono=as_mono
        )

        lab = hts.HTSLabelFile()

        with open(lab_path, "r") as f:
            for label in f.readlines():
                lab.append(label.split(), strict=False)

        lab = merge_sil(lab)

        with open(lab_path, "w") as f:
            f.write(str(fix_mono_lab_before_align(lab, config["spk"])))
