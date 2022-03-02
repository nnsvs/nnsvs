# /usr/bin/python

import argparse
import glob
import os
import re
import sys
from os.path import basename, expanduser, join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def process_log(f):
    log_re = re.compile(
        ".* \\- \\[(?P<dataset>[\\w_]+)\\] \\[Epoch (?P<epoch>\\d+)\\]: loss (?P<loss>[\\-\\d.]+)"  # noqa
    )
    model_re = re.compile("\\s+.*_target_:\\s+nnsvs\\.model\\.(?P<model_name>\\w+)$")
    out_dir_re = re.compile("\\s{2}out_dir:\\s+(?P<out_dir>[\\w./]+)$")
    data = []
    model_name = None
    training_type = None

    for line in f:
        # model name
        match = model_re.match(line)
        if match:
            model_name = match["model_name"]

        # Which model is trained? (timelag or  duration or acoustic)
        match = out_dir_re.match(line)
        if match:
            training_type = basename(match["out_dir"])

        # train information
        match = log_re.match(line)
        if match:
            data.append([match["dataset"], int(match["epoch"]), float(match["loss"])])

    ret = {
        "model_name": model_name,
        "training_type": training_type,
        "log": pd.DataFrame(data, columns=["dataset", "epoch", "loss"]),
    }

    return ret


def get_parser():
    parser = argparse.ArgumentParser(
        description="Make graphs of learning curves from NNSVS outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("recipe_root", type=str, help="Recipe Root directory")
    parser.add_argument("--output_dir", type=str, help="Directory of output files")
    return parser


args = get_parser().parse_args(sys.argv[1:])
recipe_root = expanduser(args.recipe_root)

if args.output_dir is not None:
    output_dir = expanduser(args.output_dir)
else:
    output_dir = recipe_root

log_path_re = join(recipe_root, "outputs", "**", "train.log")

for log_path in glob.iglob(log_path_re, recursive=True):
    datetime_re = re.compile(
        ".*(?P<date>\\d{4}-\\d{2}-\\d{2})/(?P<time>\\d{2}-\\d{2}-\\d{2})/.*"
    )
    match = datetime_re.match(log_path)
    with open(log_path, "r", encoding="utf-8") as f:
        ret = process_log(f)
        df = ret["log"]
        if df.empty:
            continue
        sns.set_style("whitegrid")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        title = f"{ret['training_type'].capitalize()} ({ret['model_name']})"
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        for dataset in ["train_no_dev", "dev"]:
            df[df["dataset"] == dataset].plot(x="epoch", y="loss", label=dataset, ax=ax)

        os.makedirs(output_dir, exist_ok=True)
        output_filename = join(
            output_dir, f"{match['date']}_{match['time']}_{ret['training_type']}.png"
        )
        plt.savefig(output_filename)
