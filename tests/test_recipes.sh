#!/bin/bash

set -e

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
NNSVS_ROOT=$script_dir/..

###########################################################
#                Dev (melf0, 48k)                         #
###########################################################

# Use nit-song070 public dataset for testing on CI
cd $NNSVS_ROOT/recipes/nit-song070/test-48k-melf0
rm -rf dump exp outputs tensorboard packed_models

# # Normal setup
./run.sh --stage -1 --stop-stage 4 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_melf0_test


# Train post-filter
./run.sh --stage 7 --stop-stage 8 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_melf0_test \
    --postfilter_model postfilter_mel_test --postfilter_train mel_test

# Train neural vocoder
./run.sh --stage 9 --stop-stage 10 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_melf0_test \
    --vocoder_model hn-sinc-nsf_sr48k_melf0_pwgD_test

# Run the packaging step
./run.sh --stage 99 --stop-stage 99 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_melf0_test \
    --vocoder_model hn-sinc-nsf_sr48k_melf0_pwgD_test

###########################################################
#                Dev (world), 48k)                        #
###########################################################

# Use nit-song070 public dataset for testing on CI
cd $NNSVS_ROOT/recipes/nit-song070/test-48k-nodyn
rm -rf dump exp outputs tensorboard packed_models

# Normal setup
./run.sh --stage -1 --stop-stage 6 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_test

# Multi-stream model
./run.sh --stage 4 --stop-stage 4 --acoustic_model acoustic_multistream_test

# Generate training data for post-filtering
./run.sh --stage 7 --stop-stage 7 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_test

# Train post-filter for mgc/bap separately
# NOTE: we must use specific model/train configs for mgc/bap respectively.
for typ in sp_test bap_test;
do
    ./run.sh --stage 8 --stop-stage 8 \
        --timelag-model timelag_test \
        --duration-model duration_test \
        --acoustic_model acoustic_test \
        --postfilter_model postfilter_${typ} --postfilter_train ${typ}
done
# Merge post-filters
python $NNSVS_ROOT/utils/merge_postfilters.py \
    exp/yoko/postfilter_sp_test/latest.pth \
    exp/yoko/postfilter_bap_test/latest.pth \
    exp/yoko/postfilter_merged

# Train neural vocoder
./run.sh --stage 9 --stop-stage 11 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_test \
    --vocoder_model hn-sinc-nsf_sr48k_pwgD_test

# Run the packaging step
./run.sh --stage 99 --stop-stage 99 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_test \
    --postfilter_model postfilter_merged \
    --vocoder_model hn-sinc-nsf_sr48k_pwgD_test


###########################################################
#                Dev (w/ dyn feats, 24k)                  #
###########################################################

# # Use nit-song070 public dataset for testing on CI
# cd $NNSVS_ROOT/recipes/nit-song070/test-24k-nodyn
# rm -rf dump exp outputs tensorboard packed_models

# # Normal setup
# ./run.sh --stage -1 --stop-stage 6 \
#     --timelag-model timelag_test \
#     --duration-model duration_test \
#     --acoustic_model acoustic_test

# # Multi-stream model
# ./run.sh --stage 4 --stop-stage 4 --acoustic_model acoustic_multistream_test
