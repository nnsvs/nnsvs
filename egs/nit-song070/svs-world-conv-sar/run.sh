#!/bin/bash

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
NNSVS_ROOT=$script_dir/../../..
NNSVS_COMMON_ROOT=$NNSVS_ROOT/egs/_common/spsvs

spk="yoko"

dumpdir=dump

# HTS-style question used for extracting musical/linguistic context from musicxml files
question_path=$NNSVS_ROOT/egs/_common/hed/jp_qst003_nnsvs.hed

# speficy if you have it locally, otherwise it will be downloaded at stage -1
hts_demo_root=downloads/HTS-demo_NIT-SONG070-F001

# Models
# To customize, put your config in conf/train/model/ and
# specify the config name below
timelag_model=timelag_ffn
duraiton_model=duration_lstm
acoustic_model=acoustic_conv

# Pretrained model dir
# leave empty to disable
pretrained_expdir=

batch_size=4

stage=0
stop_stage=0

# exp tag
tag="" # tag for managing experiments.

. $NNSVS_ROOT/utils/parse_options.sh || exit 1;

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

train_set="train_no_dev"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)
testsets=($eval_set)

dump_org_dir=$dumpdir/$spk/org
dump_norm_dir=$dumpdir/$spk/norm

# exp name
if [ -z ${tag} ]; then
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
    python local/data_prep.py $hts_demo_root ./data --gain-normalize

    echo "train/dev/eval split"
    mkdir -p data/list
    find data/acoustic/ -type f -name "*.wav" -exec basename {} .wav \; \
        | sort > data/list/utt_list.txt
    grep _003 data/list/utt_list.txt > data/list/$eval_set.list
    grep _004 data/list/utt_list.txt > data/list/$dev_set.list
    grep -v _003 data/list/utt_list.txt | grep -v _004 > data/list/$train_set.list
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature generation"
    . $NNSVS_COMMON_ROOT/feature_generation.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Training time-lag model"
    . $NNSVS_COMMON_ROOT/train_timelag.sh
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Training duration model"
    . $NNSVS_COMMON_ROOT/train_duration.sh
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Training acoustic model"
    . $NNSVS_COMMON_ROOT/train_acoustic.sh
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Generate features from timelag/duration/acoustic models"
    . $NNSVS_COMMON_ROOT/generate.sh
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Synthesis waveforms"
    . $NNSVS_COMMON_ROOT/synthesis.sh
fi
