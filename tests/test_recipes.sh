#!/bin/bash

set -e

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
NNSVS_ROOT=$script_dir/..

cd $NNSVS_ROOT/recipes/nit-song070/test-latest
./run.sh --stage -1 --stop-stage 99 \
    --timelag-model timelag_mdn \
    --duration-model duration_mdn \
    --acoustic_model acoustic_resf0conv
