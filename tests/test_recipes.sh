#!/bin/bash

set -e

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
NNSVS_ROOT=$script_dir/..

# Use nit070 public dataset for testing on CI
cd $NNSVS_ROOT/recipes/nit-song070/test-latest

# Normal setup
./run.sh --stage -1 --stop-stage 6 \
    --timelag-model timelag_mdn \
    --duration-model duration_mdn \
    --acoustic_model acoustic_resf0conv

# Generate training data for post-filtering
./run.sh --stage 7 --stop-stage 7 \
    --timelag-model timelag_mdn \
    --duration-model duration_mdn \
    --acoustic_model acoustic_resf0conv \
    --postfilter_model postfilter_mgc --postfilter_train mgc

# Train post-filter for mgc/bap separately
# NOTE: we must use specific model/train configs for mgc/bap respectively.
for typ in mgc bap;
do
    ./run.sh --stage 8 --stop-stage 8 \
        --timelag-model timelag_mdn \
        --duration-model duration_mdn \
        --acoustic_model acoustic_resf0conv \
        --postfilter_model postfilter_${typ} --postfilter_train ${typ}
done
# Merge post-filters
python $NNSVS_ROOT/utils/merge_postfilters.py \
    exp/yoko/postfilter_mgc/latest.pth \
    exp/yoko/postfilter_bap/latest.pth \
    exp/yoko/postfilter_merged

# Train neural vocoder
./run.sh --stage 9 --stop-stage 12 \
    --timelag-model timelag_mdn \
    --duration-model duration_mdn \
    --acoustic_model acoustic_resf0conv \
    --vocoder_model hn-sinc-nsf_sr48k_pwgD

# Run the packaging step
./run.sh --stage 99 --stop-stage 99 \
    --timelag-model timelag_mdn \
    --duration-model duration_mdn \
    --acoustic_model acoustic_resf0conv \
    --postfilter_model postfilter_merged \
    --vocoder_model hn-sinc-nsf_sr48k_pwgD
