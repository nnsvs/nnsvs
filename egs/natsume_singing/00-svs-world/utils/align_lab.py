import os

from glob import glob
from os.path import join, basename, splitext
from nnmnkwii.io import hts
from fastdtw import fastdtw
import sys
from util import prep_ph2num, ph2numeric, fix_mono_lab_after_align
from tqdm import tqdm

import yaml
with open('config.yaml', 'r') as yml:
    config = yaml.load(yml, Loader=yaml.FullLoader)

# Get rough alignment between
# 1) mono-phone labels of singing voice database and
# 2) generated labels by sinsy

ph2num = prep_ph2num()

sinsy_files = sorted(glob(join(config["out_dir"], "sinsy_mono_round/*.lab")))
mono_label_files = sorted(glob(join(config["out_dir"], "mono_label_round/*.lab")))

dst_dir = join(config["out_dir"], "mono_dtw")
os.makedirs(dst_dir, exist_ok=True)

excludes = []
for (path1, path2) in tqdm(zip(sinsy_files, mono_label_files)):
    lab_sinsy = hts.load(path1)
    lab_mono_label = hts.load(path2)
    name = basename(path1)
    if name in excludes:
        print("Skip!", name)
        continue

    # align two labels roughly based on the phoneme labels
    d, path = fastdtw(ph2numeric(lab_sinsy.contexts,ph2num),
        ph2numeric(lab_mono_label.contexts, ph2num), radius=len(lab_mono_label))

    # Edit sinsy labels with hand-annontated aligments
    for x, y in path:
        lab_sinsy.start_times[x] = lab_mono_label.start_times[y]
        lab_sinsy.end_times[x] = lab_mono_label.end_times[y]

    lab_sinsy = fix_mono_lab_after_align(lab_sinsy)
    with open(join(dst_dir, name), "w") as of:
        of.write(str(lab_sinsy))
    print(name, d)

sys.exit(0)
