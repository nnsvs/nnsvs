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
NNSVS_COMMON_ROOT=$NNSVS_ROOT/recipes/_common/spsvs
. $NNSVS_ROOT/utils/yaml_parser.sh || exit 1;

eval $(parse_yaml "./config.yaml" "")

train_set="train_no_dev"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)
testsets=($dev_set $eval_set)

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
    if [ ! -e downloads/kiritan_singing ]; then
        echo "stage -1: Downloading data"
        mkdir -p downloads
        git clone https://github.com/r9y9/kiritan_singing downloads/kiritan_singing
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    kiritan_singing=downloads/kiritan_singing
    cd $kiritan_singing && git checkout .
    if [ ! -z "${kiritan_wav_root}" ]; then
        echo "" >> config.py
        echo "wav_dir = \"$kiritan_wav_root\"" >> config.py
    fi
    ./run.sh
    cd -
    mkdir -p data/list
    ln -sfn $PWD/$kiritan_singing/kiritan_singing_extra/timelag data/timelag
    ln -sfn $PWD/$kiritan_singing/kiritan_singing_extra/duration data/duration
    ln -sfn $PWD/$kiritan_singing/kiritan_singing_extra/acoustic data/acoustic

    # Normalize audio if sv56 is available
    if command -v sv56demo &> /dev/null; then
        echo "Normalize audio gain with sv56"
        python $NNSVS_COMMON_ROOT/sv56.py $out_dir/acoustic/wav $out_dir/acoustic/wav
    fi

    echo "train/dev/eval split"
    find data/acoustic/ -type f -name "*.wav" -exec basename {} .wav \; \
        | sort > data/list/utt_list.txt
    grep 05_ data/list/utt_list.txt > data/list/$eval_set.list
    grep 01_ data/list/utt_list.txt > data/list/$dev_set.list
    grep -v 01_ data/list/utt_list.txt | grep -v 05_ > data/list/$train_set.list
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
    . $NNSVS_COMMON_ROOT/train_resf0_acoustic.sh
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Generate features from timelag/duration/acoustic models"
    . $NNSVS_COMMON_ROOT/generate.sh
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Synthesis waveforms"
    . $NNSVS_COMMON_ROOT/synthesis_resf0.sh
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Generate static features"
    . $NNSVS_COMMON_ROOT/gen_static_features.sh
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Compute statistics of vocoder's input features"
    xrun python $NNSVS_COMMON_ROOT/scaler_joblib2npy_voc.py \
        $dump_norm_dir/out_acoustic_scaler.joblib $dump_norm_dir/
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: Training vocoder using parallel_wavegan"
    if [ ! -z ${pretrained_vocoder_checkpoint} ]; then
        extra_args="--resume $pretrained_vocoder_checkpoint"
    else
        extra_args=""
    fi
    # NOTE: copy normalization stats to expdir for convenience
    mkdir -p $expdir/$vocoder_model
    cp -v $dump_norm_dir/in_vocoder*.npy $expdir/$vocoder_model
    xrun parallel-wavegan-train --config conf/parallel_wavegan/${vocoder_model}.yaml \
        --train-dumpdir $dump_norm_dir/$train_set/out_acoustic_static \
        --dev-dumpdir $dump_norm_dir/$dev_set/out_acoustic_static/ \
        --outdir $expdir/$vocoder_model $extra_args
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10: Synthesis waveforms by parallel_wavegan"
    if [ -z "${vocoder_eval_checkpoint}" ]; then
        vocoder_eval_checkpoint="$(ls -dt "${expdir}/${vocoder_model}"/*.pkl | head -1 || true)"
    fi
    outdir="${expdir}/$vocoder_model/wav/$(basename "${vocoder_eval_checkpoint}" .pkl)"
    for s in ${testsets[@]}; do
        xrun parallel-wavegan-decode --dumpdir $dump_norm_dir/$s/out_acoustic_static/ \
            --checkpoint $vocoder_eval_checkpoint \
            --outdir $outdir
    done
fi

if [ ${stage} -le 99 ] && [ ${stop_stage} -ge 99 ]; then
    echo "Pack models for SVS"
    if [[ -z "${vocoder_eval_checkpoint}" && -d ${expdir}/${vocoder_model}/config.yml ]]; then
        vocoder_eval_checkpoint="$(ls -dt "$expdir/$vocoder_model"/*.pkl | head -1 || true)"
    fi
    # Determine the directory name of a packed model
    if [ -e "$vocoder_eval_checkpoint" ]; then
        # PWG's expdir or packed model's dir
        voc_dir=$(dirname $vocoder_eval_checkpoint)
        # PWG's expdir
        if [ -e ${voc_dir}/config.yml ]; then
            voc_config=${voc_dir}/config.yml
        # Packed model's dir
        elif [ -e ${voc_dir}/vocoder_model.yaml ]; then
            voc_config=${voc_dir}/vocoder_model.yaml
        else
            echo "ERROR: vocoder config is not found!"
            exit 1
        fi
        vocoder_config_name=$(basename $(grep config: ${voc_config} | awk '{print $2}'))
        vocoder_config_name=${vocoder_config_name/.yaml/}
        dst_dir=packed_models/${expname}_${timelag_model}_${duration_model}_${acoustic_model}_${vocoder_config_name}
    else
        dst_dir=packed_models/${expname}_${timelag_model}_${duration_model}_${acoustic_model}
    fi

    mkdir -p $dst_dir
    # global config file
    # NOTE: New residual F0 prediction models require relative_f0 to be false.
    cat > ${dst_dir}/config.yaml <<EOL
# Global configs
sample_rate: ${sample_rate}
frame_period: 5
log_f0_conditioning: true

# Model-specific synthesis configs
timelag:
    allowed_range: [-20, 20]
    allowed_range_rest: [-40, 40]
    force_clip_input_features: true
duration:
    force_clip_input_features: true
acoustic:
    subphone_features: "coarse_coding"
    force_clip_input_features: true
    relative_f0: false
    post_filter: true

# Model definitions
timelag_model: ${timelag_model}
duration_model: ${duration_model}
acoustic_model: ${acoustic_model}
EOL

    . $NNSVS_COMMON_ROOT/pack_model.sh
 fi
