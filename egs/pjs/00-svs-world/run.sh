#!/bin/bash

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
NNSVS_ROOT=$script_dir/../../../

spk="pjs"

dumpdir=dump

# HTS-style question used for extracting musical/linguistic context from musicxml files
question_path=./conf/jp_qst001_nnsvs.hed

# Pretrained model dir
# leave empty to disable
pretrained_expdir=../../kiritan_singing/00-svs-world/exp/kiritan

batch_size=8

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
testsets=($dev_set $eval_set)

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
    if [ ! -d downloads/PJS_corpus_ver1.1 ]; then
        echo "stage -1: Downloading PJS"
        echo "run `pip install gdown` if you don't have it locally"
        mkdir -p downloads && cd downloads
        gdown "https://drive.google.com/uc?id=1hPHwOkSe2Vnq6hXrhVtzNskJjVMQmvN_"
        unzip PJS_corpus_ver1.1.zip
        cd -
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    python utils/data_prep.py downloads/PJS_corpus_ver1.1 data --gain-normalize
    echo "train/dev/eval split"
    mkdir -p data/list
    # exclude utts that are not strictly aligned
    find data/acoustic/ -type f -name "*.wav" -exec basename {} .wav \; \
        | grep -v 030 | sort > data/list/utt_list.txt
    grep 056 data/list/utt_list.txt > data/list/$eval_set.list
    grep 055 data/list/utt_list.txt > data/list/$dev_set.list
    grep -v 056 data/list/utt_list.txt | grep -v 056 > data/list/$train_set.list
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature generation"

    for s in ${datasets[@]};
    do
      nnsvs-prepare-features utt_list=data/list/$s.list out_dir=$dump_org_dir/$s/  \
        question_path=$question_path
    done

    # Compute normalization stats for each input/output
    mkdir -p $dump_norm_dir
    for inout in "in" "out"; do
        if [ $inout = "in" ]; then
            scaler_class="sklearn.preprocessing.MinMaxScaler"
        else
            scaler_class="sklearn.preprocessing.StandardScaler"
        fi
        for typ in timelag duration acoustic;
        do
            find $dump_org_dir/$train_set/${inout}_${typ} -name "*feats.npy" > train_list.txt
            scaler_path=$dump_org_dir/${inout}_${typ}_scaler.joblib
            nnsvs-fit-scaler list_path=train_list.txt scaler.class=$scaler_class \
                out_path=$scaler_path
            rm -f train_list.txt
            cp -v $scaler_path $dump_norm_dir/${inout}_${typ}_scaler.joblib
        done
    done

    # apply normalization
    for s in ${datasets[@]}; do
        for inout in "in" "out"; do
            for typ in timelag duration acoustic;
            do
                nnsvs-preprocess-normalize in_dir=$dump_org_dir/$s/${inout}_${typ}/ \
                    scaler_path=$dump_org_dir/${inout}_${typ}_scaler.joblib \
                    out_dir=$dump_norm_dir/$s/${inout}_${typ}/
            done
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Training time-lag model"
    if [ ! -z "${pretrained_expdir}" ]; then
        resume_checkpoint=$pretrained_expdir/timelag/latest.pth
    else
        resume_checkpoint=
    fi
   xrun nnsvs-train data.train_no_dev.in_dir=$dump_norm_dir/$train_set/in_timelag/ \
        data.train_no_dev.out_dir=$dump_norm_dir/$train_set/out_timelag/ \
        data.dev.in_dir=$dump_norm_dir/$dev_set/in_timelag/ \
        data.dev.out_dir=$dump_norm_dir/$dev_set/out_timelag/ \
        model=timelag train.out_dir=$expdir/timelag \
        data.batch_size=$batch_size \
        resume.checkpoint=$resume_checkpoint
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Training phoneme duration model"
    if [ ! -z "${pretrained_expdir}" ]; then
        resume_checkpoint=$pretrained_expdir/duration/latest.pth
    else
        resume_checkpoint=
    fi
    xrun nnsvs-train data.train_no_dev.in_dir=$dump_norm_dir/$train_set/in_duration/ \
        data.train_no_dev.out_dir=$dump_norm_dir/$train_set/out_duration/ \
        data.dev.in_dir=$dump_norm_dir/$dev_set/in_duration/ \
        data.dev.out_dir=$dump_norm_dir/$dev_set/out_duration/ \
        model=duration train.out_dir=$expdir/duration \
        data.batch_size=$batch_size \
        resume.checkpoint=$resume_checkpoint
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Training acoustic model"
    if [ ! -z "${pretrained_expdir}" ]; then
        resume_checkpoint=$pretrained_expdir/acoustic/latest.pth
    else
        resume_checkpoint=
    fi
    xrun nnsvs-train data.train_no_dev.in_dir=$dump_norm_dir/$train_set/in_acoustic/ \
        data.train_no_dev.out_dir=$dump_norm_dir/$train_set/out_acoustic/ \
        data.dev.in_dir=$dump_norm_dir/$dev_set/in_acoustic/ \
        data.dev.out_dir=$dump_norm_dir/$dev_set/out_acoustic/ \
        model=acoustic train.out_dir=$expdir/acoustic \
        data.batch_size=$batch_size \
        resume.checkpoint=$resume_checkpoint
fi


# NOTE: step 5 does not generate waveform. It just saves neural net's outputs.
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Generation features from timelag/duration/acoustic models"
    for s in ${testsets[@]}; do
        for typ in timelag duration acoustic; do
            checkpoint=$expdir/$typ/latest.pth
            name=$(basename $checkpoint)
            xrun nnsvs-generate model.checkpoint=$checkpoint \
                model.model_yaml=$expdir/$typ/model.yaml \
                out_scaler_path=$dump_norm_dir/out_${typ}_scaler.joblib \
                in_dir=$dump_norm_dir/$s/in_${typ}/ \
                out_dir=$expdir/$typ/predicted/$s/${name%.*}/
        done
    done
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Synthesis waveforms"
    for s in ${testsets[@]}; do
        for input in label_phone_score label_phone_align; do
            if [ $input = label_phone_score ]; then
                ground_truth_duration=false
            else
                ground_truth_duration=true
            fi
            xrun nnsvs-synthesis question_path=conf/jp_qst001_nnsvs.hed \
            timelag.checkpoint=$expdir/timelag/latest.pth \
            timelag.in_scaler_path=$dump_norm_dir/in_timelag_scaler.joblib \
            timelag.out_scaler_path=$dump_norm_dir/out_timelag_scaler.joblib \
            timelag.model_yaml=$expdir/timelag/model.yaml \
            duration.checkpoint=$expdir/duration/latest.pth \
            duration.in_scaler_path=$dump_norm_dir/in_duration_scaler.joblib \
            duration.out_scaler_path=$dump_norm_dir/out_duration_scaler.joblib \
            duration.model_yaml=$expdir/duration/model.yaml \
            acoustic.checkpoint=$expdir/acoustic/latest.pth \
            acoustic.in_scaler_path=$dump_norm_dir/in_acoustic_scaler.joblib \
            acoustic.out_scaler_path=$dump_norm_dir/out_acoustic_scaler.joblib \
            acoustic.model_yaml=$expdir/acoustic/model.yaml \
            utt_list=./data/list/$s.list \
            in_dir=data/acoustic/$input/ \
            out_dir=$expdir/synthesis/$s/latest/$input \
            ground_truth_duration=$ground_truth_duration
        done
    done
fi