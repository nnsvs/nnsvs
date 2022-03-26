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
NNSVS_ROOT=$script_dir/../../../
NNSVS_COMMON_ROOT=$NNSVS_ROOT/egs/_common/spsvs
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
    python local/data_prep.py $db_root $out_dir --gain-normalize

    # Pitch data augmentation (in cent)
    for cent in -100 100
    do
        # timelag
        for typ in label_phone_align label_phone_score
        do
            python $NNSVS_COMMON_ROOT/../pitch_augmentation.py $out_dir/timelag/$typ $out_dir/timelag/$typ \
                $cent --filter_augmented_files
        done
        # duration
        for typ in label_phone_align
        do
            python $NNSVS_COMMON_ROOT/../pitch_augmentation.py $out_dir/duration/$typ $out_dir/duration/$typ \
                $cent --filter_augmented_files
        done
        # acoustic
        for typ in wav label_phone_align label_phone_score
        do
            python $NNSVS_COMMON_ROOT/../pitch_augmentation.py $out_dir/acoustic/$typ $out_dir/acoustic/$typ \
                $cent --filter_augmented_files
        done
    done

    # Tempo data augmentation
    for tempo in 0.9 1.1
    do
        # timelag
        for typ in label_phone_align label_phone_score
        do
            python $NNSVS_COMMON_ROOT/../tempo_augmentation.py $out_dir/timelag/$typ $out_dir/timelag/$typ \
                $tempo --filter_augmented_files
        done
        # duration
        for typ in label_phone_align
        do
            python $NNSVS_COMMON_ROOT/../tempo_augmentation.py $out_dir/duration/$typ $out_dir/duration/$typ \
                $tempo --filter_augmented_files
        done
        # acoustic
        for typ in wav label_phone_align label_phone_score
        do
            python $NNSVS_COMMON_ROOT/../tempo_augmentation.py $out_dir/acoustic/$typ $out_dir/acoustic/$typ \
                $tempo --filter_augmented_files
        done
    done

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
