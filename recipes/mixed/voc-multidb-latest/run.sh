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
    if [ ! -e downloads/HTS-demo_NIT-SONG070-F001 ]; then
        echo "stage -1: Downloading data for nit-song070"
        mkdir -p downloads
        cd downloads
        curl -LO http://hts.sp.nitech.ac.jp/archives/2.3/HTS-demo_NIT-SONG070-F001.tar.bz2
        tar jxvf HTS-demo_NIT-SONG070-F001.tar.bz2
        cd $script_dir
    fi
    if [ ! -e downloads/kiritan_singing ]; then
        echo "stage -1: Downloading data for kiritan_singing"
        mkdir -p downloads
        git clone https://github.com/r9y9/kiritan_singing downloads/kiritan_singing
    fi
    if [ ! -d downloads/jsut-song_ver1 ]; then
        echo "stage -1: Downloading JSUT-song"
        cd downloads
        curl -LO https://ss-takashi.sakura.ne.jp/corpus/jsut-song_ver1.zip
        unzip jsut-song_ver1.zip
        cd -
    fi
    if [ ! -d downloads/todai_child ]; then
        echo "stage -1: Downloading JSUT-song labels"
        cd downloads
        curl -LO https://ss-takashi.sakura.ne.jp/corpus/jsut-song_label.zip
        unzip jsut-song_label.zip
        cd -
    fi
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
    # This step will create the following directories
    # data/${dbname}/{acoustic,duration,timelag,list}

    ###########################################################
    #                nit-song070                              #
    ###########################################################
    dbname=$(echo $dbnames | awk '{print $1}')
    python local/nit_data_prep.py $nit_db_root $out_dir/$dbname

    echo "train/dev/eval split for nit-song070"
    mkdir -p data/$dbname/list
    find data/$dbname/acoustic/ -type f -name "nitech*.wav" -exec basename {} .wav \; \
        | sort > data/$dbname/list/utt_list.txt
    grep _003 data/$dbname/list/utt_list.txt > data/$dbname/list/$eval_set.list
    grep _004 data/$dbname/list/utt_list.txt > data/$dbname/list/$dev_set.list
    grep -v _003 data/$dbname/list/utt_list.txt | grep -v _004 > data/$dbname/list/$train_set.list

    ###########################################################
    #                kiritan_singing                          #
    ###########################################################
    kiritan_singing=downloads/kiritan_singing
    cd $kiritan_singing && git checkout .
    if [ ! -z "${kiritan_wav_root}" ]; then
        echo "" >> config.py
        echo "wav_dir = \"$kiritan_wav_root\"" >> config.py
    fi
    ./run.sh
    cd -

    dbname=$(echo $dbnames | awk '{print $2}')
    mkdir -p $out_dir/$dbname
    ln -sfn $PWD/$kiritan_singing/kiritan_singing_extra/timelag $out_dir/$dbname/timelag
    ln -sfn $PWD/$kiritan_singing/kiritan_singing_extra/duration $out_dir/$dbname/duration
    ln -sfn $PWD/$kiritan_singing/kiritan_singing_extra/acoustic $out_dir/$dbname/acoustic

    echo "train/dev/eval split for kiritan_singing"
    mkdir -p data/$dbname/list
    find -L data/$dbname/acoustic/ -type f -name "kiritan_singing_*.wav" -exec basename {} .wav \; \
        | sort > data/$dbname/list/utt_list.txt
    grep 05_ data/$dbname/list/utt_list.txt >> data/$dbname/list/$eval_set.list
    grep 01_ data/$dbname/list/utt_list.txt >> data/$dbname/list/$dev_set.list
    grep -v 01_ data/$dbname/list/utt_list.txt | grep -v 05_ >> data/$dbname/list/$train_set.list

    ###########################################################
    #                jsut-song                                #
    ###########################################################
    dbname=$(echo $dbnames | awk '{print $3}')
    python local/jsut_data_prep.py ./downloads/jsut-song_ver1 \
        ./downloads/todai_child/ \
        ./downloads/HTS-demo_NIT-SONG070-F001/ data/$dbname

    echo "train/dev/eval split for jsut-song"
    mkdir -p data/$dbname/list
    # exclude 045 since the label file is not available
    find data/$dbname/acoustic/ -type f -name "*.wav" -exec basename {} .wav \; \
        | grep -v 045 | sort > data/$dbname/list/utt_list.txt
    grep 003 data/$dbname/list/utt_list.txt > data/$dbname/list/$eval_set.list
    grep 004 data/$dbname/list/utt_list.txt > data/$dbname/list/$dev_set.list
    grep -v 003 data/$dbname/list/utt_list.txt | grep -v 004 > data/$dbname/list/$train_set.list

    ###########################################################
    #                PJS                                      #
    ###########################################################
    dbname=$(echo $dbnames | awk '{print $4}')
    python local/pjs_data_prep.py downloads/PJS_corpus_ver1.1 data/$dbname
    echo "train/dev/eval split for PJS"
    mkdir -p data/$dbname/list
    # exclude utts that are not strictly aligned
    find data/$dbname/acoustic/ -type f -name "*.wav" -exec basename {} .wav \; \
        | grep -v 030 | sort > data/$dbname/list/utt_list.txt
    grep 056 data/$dbname/list/utt_list.txt > data/$dbname/list/$eval_set.list
    grep 055 data/$dbname/list/utt_list.txt > data/$dbname/list/$dev_set.list
    grep -v 056 data/$dbname/list/utt_list.txt | grep -v 056    > data/$dbname/list/$train_set.list

    # Normalize audio if sv56 is available
    for dbname in $dbnames;
    do
        if command -v sv56demo &> /dev/null; then
            echo "Normalize audio gain with sv56"
            python $NNSVS_COMMON_ROOT/sv56.py $out_dir/$dbname/acoustic/wav $out_dir/$dbname/acoustic/wav
        fi
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature generation"
    . $NNSVS_COMMON_ROOT/multidb_feature_generation.sh
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Generate static features"
    . $NNSVS_COMMON_ROOT/multidb_gen_static_features.sh
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Compute statistics of vocoder's input features"
    xrun python $NNSVS_COMMON_ROOT/scaler_joblib2npy_voc.py \
        $dump_norm_dir/out_acoustic_scaler.joblib $dump_norm_dir/
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Training vocoder using parallel_wavegan"
    if [ ! -z ${pretrained_vocoder_checkpoint} ]; then
        extra_args="--resume $pretrained_vocoder_checkpoint"
    else
        extra_args=""
    fi
    xrun parallel-wavegan-train --config conf/parallel_wavegan/${vocoder_model}.yaml \
        --train-dumpdir $dump_norm_dir/$train_set/out_acoustic_static \
        --dev-dumpdir $dump_norm_dir/$dev_set/out_acoustic_static/ \
        --outdir $expdir/$vocoder_model $extra_args
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Synthesis waveforms by parallel_wavegan"
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
    echo "Pack vocoder for SVS"
    dst_dir=packed_models/${expname}_${vocoder_model}
    mkdir -p $dst_dir

    if [ -z "${vocoder_eval_checkpoint}" ]; then
        vocoder_eval_checkpoint="$(ls -dt "$expdir/$vocoder_model"/*.pkl | head -1 || true)"
    fi
    python $NNSVS_COMMON_ROOT/clean_checkpoint_state.py $vocoder_eval_checkpoint \
        $dst_dir/vocoder_model.pth
    cp $expdir/${vocoder_model}/config.yml $dst_dir/vocoder_model.yaml
    cp $dump_norm_dir/in_vocoder*.npy $dst_dir/

    echo "All the files are ready for SVS!"
    echo "Please check the $dst_dir directory"
 fi
