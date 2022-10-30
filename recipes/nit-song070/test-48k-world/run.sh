#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

function xrun () {
    set -x
    $@
    set +x
}

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
NNSVS_ROOT=$script_dir/../../..
NNSVS_COMMON_ROOT=$NNSVS_ROOT/recipes/_common/spsvs
. $NNSVS_ROOT/utils/yaml_parser.sh || exit 1;

eval $(parse_yaml "./config.yaml" "")

train_set="train_no_dev"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)
testsets=($eval_set)

dumpdir=dump

dump_org_dir=$dumpdir/$spk/org
dump_norm_dir=$dumpdir/$spk/norm

stage=0
stop_stage=0

. $NNSVS_ROOT/utils/parse_options.sh || exit 1;

# exp name
if [ -z ${tag:=} ]; then
    expname=${spk}
else
    expname=${spk}_${tag}
fi
expdir=exp/$expname

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    if [ ! -e downloads/HTS-demo_NIT-SONG070-F001 ]; then
        echo "stage -1: Downloading data"
        mkdir -p downloads
        cd downloads
        curl -LO http://hts.sp.nitech.ac.jp/archives/2.3/HTS-demo_NIT-SONG070-F001.tar.bz2
        tar jxvf HTS-demo_NIT-SONG070-F001.tar.bz2
        cd $script_dir
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    # the following three directories will be created
    # 1) data/timelag 2) data/duration 3) data/acoustic
    python $NNSVS_ROOT/recipes/_common/db/nit-song070/data_prep.py $db_root $out_dir

    echo "train/dev/eval split"
    mkdir -p data/list
    find data/acoustic/ -type f -name "*.wav" -exec basename {} .wav \; \
        | sort > data/list/utt_list.txt
    grep _003 data/list/utt_list.txt > data/list/$eval_set.list
    grep _004 data/list/utt_list.txt > data/list/$dev_set.list
    grep -e _007 -e _010 data/list/utt_list.txt > data/list/$train_set.list
fi

# Run the rest of the steps
# Please check the script file for more details
. $NNSVS_COMMON_ROOT/run_common_steps_dev.sh
