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
NO2_ROOT=$NNSVS_ROOT/recipes/_common/no2
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
    if [ ! -e $db_root ]; then
	cat<<EOF
stage -1: Downloading

This recipe does not download the archive of singing voice database automatically to
provide you the opportunity to read the original license.

Please visit https://drive.google.com/drive/folders/1XA2cm3UyRpAk_BJb1LTytOWrhjsZKbSN
and read the term of services, and then download the singing voice database manually.
EOF
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    sh $NO2_ROOT/utils/data_prep.sh ./config.yaml ust
    mkdir -p data/list

    echo "train/dev/eval split"
    # NOTE: 110 songs in total
    find data/acoustic/ -type f -name "*.wav" -exec basename {} .wav \; \
	 | sort > data/list/utt_list.txt
    # # 5 songs for dev/eval
    grep -e 1st_color -e ARROW -e BC -e Closetoyou -e ERROR data/list/utt_list.txt > data/list/$eval_set.list
    grep -e Baptism -e COZMIC_HEART -e Choir -e BRD -e Creuzer data/list/utt_list.txt > data/list/$dev_set.list
    # NOTE: exclude namine_ritsu_hana_seg12 to avoid alignment and audio length mitmatch. Probably a bug of data_prep.sh
    grep -v -e 1st_color -e ERROR -e ARROW -e BC -e Closetoyou -e Baptism -e COZMIC_HEART -e Choir -e BRD -e Creuzer -e namine_ritsu_hana_seg12 data/list/utt_list.txt > data/list/$train_set.list
fi

# Run the rest of the steps
# Please check the script file for more details
. $NNSVS_COMMON_ROOT/run_common_steps_dev.sh
