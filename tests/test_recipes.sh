#!/bin/bash

set -e

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
NNSVS_ROOT=$script_dir/..

# NOTE: This environmental variable is only used for testing purpose
export RUNNING_TEST_RECIPES=1

# hts.sp.nitech.ac.jp is a university lab server that is occasionally
# unreachable. Probe it once so we can skip the nit-song070 recipe tests
# gracefully instead of failing the whole CI job on an unrelated outage.
nitech_reachable() {
    curl -fsS --connect-timeout 10 --max-time 15 --head \
        https://hts.sp.nitech.ac.jp/archives/2.3/HTS-demo_NIT-SONG070-F001.tar.bz2 >/dev/null 2>&1
}

###########################################################
#                Dev (melf0, 48k)                         #
###########################################################

if nitech_reachable; then

# Use nit-song070 public dataset for testing on CI
cd $NNSVS_ROOT/recipes/nit-song070/test-48k-melf0
rm -rf dump exp outputs tensorboard packed_models

# # Normal setup
./run.sh --stage -1 --stop-stage 4 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_melf0_test

# Multi-stream model
./run.sh --stage 4 --stop-stage 4 --acoustic_model acoustic_nnsvs_melf0_multi_test

# Train post-filter
./run.sh --stage 7 --stop-stage 8 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_melf0_test \
    --postfilter_model postfilter_mel_test --postfilter_train mel_test

# Train neural vocoder
./run.sh --stage 9 --stop-stage 10 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_melf0_test \
    --vocoder_model hn-sinc-nsf_sr48k_melf0_pwgD_test

# Train hn-uSFGAN
# NOTE: conf/usfgan/generator/${vocoder_model}.yaml must exist
./run.sh --stage 11 --stop-stage 11 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_melf0_test \
    --vocoder_model nnsvs_melf0_parallel_hn_usfgan_sr48k
# Synthesize waveforms with hn-uSFGAN
# NOTE: very slow
# ./run.sh --stage 6 --stop-stage 6 --testsets eval \
#     --synthesis melf0_gv_usfgan \
#     --timelag-model timelag_test \
#     --duration-model duration_test \
#     --acoustic_model acoustic_nnsvs_melf0_test \
#     --vocoder_model nnsvs_melf0_parallel_hn_usfgan_sr48k

# Train SiFi-GAN
# NOTE: conf/sifigan/generator/${vocoder_model}.yaml must exist
./run.sh --stage 13 --stop-stage 13 \
    --synthesis melf0_gv_usfgan \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_melf0_test \
    --vocoder_model nnsvs_melf0_sifigan_sr48k
# Synthesize waveforms with SiFi-GAN
./run.sh --stage 6 --stop-stage 6 --testsets eval \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_melf0_test \
    --vocoder_model nnsvs_melf0_sifigan_sr48k

# Run the packaging step
./run.sh --stage 99 --stop-stage 99 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_melf0_test \
    --vocoder_model hn-sinc-nsf_sr48k_melf0_pwgD_test

# Run the packaging step with hn-uSFGAN
./run.sh --stage 99 --stop-stage 99 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_melf0_test \
    --vocoder_model nnsvs_melf0_parallel_hn_usfgan_sr48k

# Run the packaging step with SiFi-GAN
./run.sh --stage 99 --stop-stage 99 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_melf0_test \
    --vocoder_model nnsvs_melf0_sifigan_sr48k

else
    echo "WARNING: hts.sp.nitech.ac.jp is unreachable; skipping nit-song070 melf0 recipe test"
fi

###########################################################
#                Dev (world), 48k)                        #
###########################################################

if nitech_reachable; then

# Use nit-song070 public dataset for testing on CI
cd $NNSVS_ROOT/recipes/nit-song070/test-48k-world
rm -rf dump exp outputs tensorboard packed_models

# Normal setup
./run.sh --stage -1 --stop-stage 6 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_world_test

# Sinsy baseline
./run.sh --stage 4 --stop-stage 4 --acoustic_model acoustic_sinsy_world_resf0convlstm
# Multi-stream model
./run.sh --stage 4 --stop-stage 4 --acoustic_model acoustic_nnsvs_world_multi_test

# Generate training data for post-filtering
./run.sh --stage 7 --stop-stage 7 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_world_test

# Train post-filter for mgc/bap separately
# NOTE: we must use specific model/train configs for mgc/bap respectively.
for typ in sp_test bap_test;
do
    ./run.sh --stage 8 --stop-stage 8 \
        --timelag-model timelag_test \
        --duration-model duration_test \
        --acoustic_model acoustic_nnsvs_world_test \
        --postfilter_model postfilter_${typ} --postfilter_train ${typ}
done
# Merge post-filters
python $NNSVS_ROOT/utils/merge_postfilters.py \
    exp/yoko/postfilter_sp_test/latest.pth \
    exp/yoko/postfilter_bap_test/latest.pth \
    exp/yoko/postfilter_merged

# Train neural vocoder
./run.sh --stage 9 --stop-stage 10 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_world_test \
    --vocoder_model hn-sinc-nsf_sr48k_pwgD_test

# Train hn-uSFGAN
# NOTE: conf/usfgan/generator/${vocoder_model}.yaml must exist
./run.sh --stage 11 --stop-stage 11 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_world_test \
    --vocoder_model nnsvs_world_parallel_hn_usfgan_sr48k
# Synthesize waveforms with hn-uSFGAN
# NOTE: very slow
# ./run.sh --stage 6 --stop-stage 6 --testsets eval \
#     --synthesis world_gv_usfgan \
#     --timelag-model timelag_test \
#     --duration-model duration_test \
#     --acoustic_model acoustic_nnsvs_world_test \
#     --vocoder_model nnsvs_world_parallel_hn_usfgan_sr48k

# Train SiFi-GAN
# NOTE: conf/sifigan/generator/${vocoder_model}.yaml must exist
./run.sh --stage 13 --stop-stage 13 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_world_test \
    --vocoder_model nnsvs_world_sifigan_sr48k
# Synthesize waveforms with SiFi-GAN
./run.sh --stage 6 --stop-stage 6 --testsets eval \
    --synthesis world_gv_usfgan \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_world_test \
    --vocoder_model nnsvs_world_sifigan_sr48k

# Analysis-by-synthesis
# TODO: needs to add synthesis configs to support neural vocoders
# specifically, needs to specify vocoder_type and feature_type
./run.sh --stage 12 --stop-stage 12 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_world_test \
    --vocoder_model hn-sinc-nsf_sr48k_pwgD_test

# Run the packaging step
./run.sh --stage 99 --stop-stage 99 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_world_test \
    --postfilter_model postfilter_merged \
    --vocoder_model hn-sinc-nsf_sr48k_pwgD_test

# Run the packaging step with hn-uSFGAN
./run.sh --stage 99 --stop-stage 99 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_world_test \
    --postfilter_model postfilter_merged \
    --vocoder_model nnsvs_world_parallel_hn_usfgan_sr48k

# Run the packaging step with SiFi-GAN
./run.sh --stage 99 --stop-stage 99 \
    --timelag-model timelag_test \
    --duration-model duration_test \
    --acoustic_model acoustic_nnsvs_world_test \
    --postfilter_model postfilter_merged \
    --vocoder_model nnsvs_world_sifigan_sr48k

else
    echo "WARNING: hts.sp.nitech.ac.jp is unreachable; skipping nit-song070 world recipe test"
fi
