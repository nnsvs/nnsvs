# NOTE: The script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

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

if [ ${stage} -le 99 ] && [ ${stop_stage} -ge 99 ]; then
    echo "Pack models for SVS"
    if [[ -z "${vocoder_eval_checkpoint}" && -f ${expdir}/${vocoder_model}/config.yml ]]; then
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
