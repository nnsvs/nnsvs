# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

if [ ! -z ${pretrained_vocoder_checkpoint} ]; then
    extra_args="--resume $pretrained_vocoder_checkpoint"
else
    extra_args=""
fi
if [[ -z "${vocoder_model}" ]]; then
    echo "ERROR: vocoder_model is not specified."
    echo "Please specify a vocoder config name"
    echo "Note that conf/train_usfgan/generator/\${vocoder_model}.yaml must exist."
    exit 1
fi

if [[ ${acoustic_features} == *"melf0"* ]]; then
    feature_type="melf0"
else
    feature_type="world"
fi

# Convert NNSVS's data to usfgan's format
if [ ! -d dump_usfgan ]; then
    python $NNSVS_ROOT/utils/nnsvs2usfgan.py config.yaml dump_usfgan --feature_type $feature_type
fi

# NOTE: copy normalization stats to expdir for convenience
mkdir -p $expdir/$vocoder_model
cp -v $dump_norm_dir/in_vocoder*.npy $expdir/$vocoder_model

# NOTE: To get the maximum performance, it is highly recommended to configure
# training options in detail
cmdstr="usfgan-train --config-dir conf/train_usfgan/ \
    data=nnsvs_${feature_type}_sr48k \
    discriminator=nnsvs_hifigan \
    train=nnsvs_hn_usfgan_sr48k_test \
    generator=$vocoder_model \
    data.train_audio=dump_usfgan/scp/${spk}_sr${sample_rate}_train_no_dev.scp \
    data.train_feat=dump_usfgan/scp/${spk}_sr${sample_rate}_train_no_dev.list \
    data.valid_audio=dump_usfgan/scp/${spk}_sr${sample_rate}_dev.scp \
    data.valid_feat=dump_usfgan/scp/${spk}_sr${sample_rate}_dev.list \
    data.eval_feat=dump_usfgan/scp/${spk}_sr${sample_rate}_eval.list \
    data.stats=dump_usfgan/stats/scaler.joblib \
    data.sample_rate=${sample_rate} out_dir=$expdir/$vocoder_model $extra_args"
echo $cmdstr
eval $cmdstr
