#!/bin/bash

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
NNSVS_ROOT=$script_dir/../../../

spk="yoko"

dumpdir=dump

# HTS-style question used for extracting musical/linguistic context from musicxml files
question_path=./conf/jp_qst001_nnsvs.hed

# speficy if you have it locally, otherwise it will be downloaded at stage -1
hts_demo_root=downloads/HTS-demo_NIT-SONG070-F001

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

# NSF related settings.
sample_rate=16000
# The bap(Band Aperiodic-ity) dimension of 16kHz is 3 and differs from that of 48kHz(15).
acoustic_model_stream_sizes=[180,3,1,3]
# 180+3+1+3
acoustic_model_out_dim=187

nsf_root_dir=downloads/project-NN-Pytorch-scripts/
nsf_save_model_dir=$expdir/nsf/train_outputs
nsf_pretrained_model=$expdir/nsf/train_outputs/trained_network.pt

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    if [ ! -e downloads/HTS-demo_NIT-SONG070-F001 ]; then
        echo "stage -1: Downloading data"
        mkdir -p downloads
        cd downloads
        curl -LO http://hts.sp.nitech.ac.jp/archives/2.3/HTS-demo_NIT-SONG070-F001.tar.bz2
        tar jxvf HTS-demo_NIT-SONG070-F001.tar.bz2
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    # the following three directories will be created
    # 1) data/timelag 2) data/duration 3) data/acoustic
    python utils/data_prep.py $hts_demo_root ./data --gain-normalize --sample-rate $sample_rate

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

    for s in ${datasets[@]};
    do
      nnsvs-prepare-features utt_list=data/list/$s.list out_dir=$dump_org_dir/$s/  \
			     question_path=$question_path \
			     acoustic.use_harvest=false
      
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
        resume.checkpoint=$resume_checkpoint \
        model.stream_sizes=$acoustic_model_stream_sizes \
        model.netG.params.out_dim=$acoustic_model_out_dim
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
    if [ ! -e $nsf_pretrained_model ]; then
	echo "No NSF pretrained model found."
	echo "Please run stage 7-9 or download pretrained model from somewhare."
	exit 1
    fi

    for s in ${testsets[@]}; do
        for input in label_phone_score label_phone_align; do
            if [ $input = label_phone_score ]; then
                ground_truth_duration=false
            else
                ground_truth_duration=true
            fi
	    #            xrun nnsvs-synthesis question_path=conf/jp_qst001_nnsvs.hed \
            xrun python bin/synthesis_nsf.py question_path=conf/jp_qst001_nnsvs.hed \
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
            acoustic.stream_sizes=$acoustic_model_stream_sizes \
            utt_list=./data/list/$s.list \
            in_dir=data/acoustic/$input/ \
            out_dir=$expdir/synthesis/$s/latest/$input \
            ground_truth_duration=$ground_truth_duration \
	    nsf_root_dir=downloads/project-NN-Pytorch-scripts/ \
	    nsf.args.save_model_dir=$nsf_save_model_dir \
        nsf.args.trained_model=$nsf_pretrained_model
        done
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    if [ ! -e $nsf_root_dir ]; then
	echo "stage 7: Downloading NSF"
        mkdir -p downloads
        cd downloads
	git clone https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts
	cd $script_dir
    fi
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Data preparation for NSF"
    out_dir=$expdir/nsf
    mkdir -p $out_dir
    for s in ${datasets[@]};
    do
        if [ $s = $eval_set ]; then
	    xrun python bin/prepare_nsf_data.py in_dir=$dump_org_dir/$s/out_acoustic out_dir=$out_dir test_set=true
        else
	    xrun python bin/prepare_nsf_data.py in_dir=$dump_org_dir/$s/out_acoustic out_dir=$out_dir
	fi
    done
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: Training NSF model"
    if [ ! -e $nsf_root_dir ]; then
	echo "No NSF files found. Please set nsf_root_dir properly or run stage 7."
	exit 1
    fi

    input_dirs=$expdir/nsf/input_dirs
    output_dirs=$expdir/nsf/output_dirs
    mkdir -p $output_dirs
    mkdir -p $nsf_save_model_dir
    xrun python bin/train_nsf.py \
	 nsf_root_dir=$nsf_root_dir \
	 nsf_type=hn-sinc-nsf \
	 nsf.args.batch_size=1 \
	 nsf.args.epochs=100 \
	 nsf.args.no_best_epochs=5 \
	 nsf.args.lr=0.00003 \
	 nsf.args.save_model_dir=$nsf_save_model_dir \
	 nsf.args.trained_model=$nsf_pretrained_model \
	 nsf.model.input_dirs=["$input_dirs","$input_dirs","$input_dirs"]\
	 nsf.model.output_dirs=["$output_dirs"]
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10: Evaluating NSF model"
    if [ ! -e $nsf_root_dir ]; then
	echo "No NSF files found. Please set nsf_root_dir properly or run stage 7."
	exit 1
    fi

    # for inference
    test_input_dirs=$expdir/nsf/test_input_dirs
    test_output_dirs=$expdir/nsf/test_output_dirs
    mkdir -p $test_output_dirs
    xrun python bin/train_nsf.py \
	 nsf_root_dir=$nsf_root_dir \
	 nsf_type=hn-sinc-nsf \
	 nsf.args.batch_size=1 \
	 nsf.args.save_model_dir=$nsf_save_model_dir \
	 nsf.args.trained_model=$nsf_pretrained_model \
	 nsf.args.inference=true \
	 nsf.model.test_input_dirs=["$test_input_dirs","$test_input_dirs","$test_input_dirs"]\
	 nsf.model.test_output_dirs=$test_output_dirs

fi
