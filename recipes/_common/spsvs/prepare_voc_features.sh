# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

for s in ${datasets[@]};
do
    if [ -d conf/prepare_static_features ]; then
        ext="--config-dir conf/prepare_static_features"
    else
        ext=""
    fi
    xrun python $NNSVS_ROOT/nnsvs/bin/prepare_voc_features.py $ext acoustic=$acoustic_features \
        in_dir=$dump_norm_dir/$s/out_acoustic/ \
        out_dir=$dump_norm_dir/$s/in_vocoder \
        utt_list=data/list/$s.list
done


if [[ ${acoustic_features} == *"static_deltadelta_sinevib"* ]]; then
    ext="--num_windows 3 --vibrato_mode sine"
elif [[ ${acoustic_features} == *"static_deltadelta_diffvib"* ]]; then
    ext="--num_windows 3 --vibrato_mode diff"
elif [[ ${acoustic_features} == *"static_only_sinevib"* ]]; then
    ext="--num_windows 1 --vibrato_mode sine"
elif [[ ${acoustic_features} == *"static_only_diffvib"* ]]; then
    ext="--num_windows 1 --vibrato_mode diff"
elif [[ ${acoustic_features} == *"static_deltadelta"* ]]; then
    ext="--num_windows 3 --vibrato_mode none"
elif [[ ${acoustic_features} == *"static_only"* ]]; then
    ext="--num_windows 1 --vibrato_mode none"
else
    ext=""
fi

# TODO: should be documnented
if [[ ${acoustic_features} == *"melf0"* ]]; then
    feature_type="melf0"
else
    feature_type="world"
fi
ext="$ext --feature_type $feature_type"

local_config_path=conf/prepare_static_features/acoustic/${acoustic_features}.yaml
global_config_path=$NNSVS_ROOT/nnsvs/bin/conf/prepare_static_features/acoustic/${acoustic_features}.yaml
if [ -e $local_config_path ]; then
    mgc_order=$(grep mgc_order $local_config_path | awk '{print $2}')
    use_mcep_aperiodicity=$(grep use_mcep_aperiodicity $local_config_path | awk '{print $2}')
    mcep_aperiodicity_order=$(grep mcep_aperiodicity_order $local_config_path | awk '{print $2}')
    ext="$ext --mgc_order $mgc_order --mcep_aperiodicity_order $mcep_aperiodicity_order"
    if [ $use_mcep_aperiodicity == "true" ]; then
        ext="$ext --use_mcep_aperiodicity"
    fi
elif [ -e $global_config_path ]; then
    mgc_order=$(grep mgc_order $global_config_path | awk '{print $2}')
    use_mcep_aperiodicity=$(grep use_mcep_aperiodicity $global_config_path | awk '{print $2}')
    mcep_aperiodicity_order=$(grep mcep_aperiodicity_order $global_config_path | awk '{print $2}')
    ext="$ext --mgc_order $mgc_order --mcep_aperiodicity_order $mcep_aperiodicity_order"
    if [ $use_mcep_aperiodicity == "true" ]; then
        ext="$ext --use_mcep_aperiodicity"
    fi
else
    echo "config file not found: $local_config_path or $global_config_path"
    exit 1
fi

# Compute statistics of vocoder's input features
# NOTE: no-op if the acoustic features don't have dynamic features
xrun python $NNSVS_COMMON_ROOT/scaler_joblib2npy_voc.py \
    $dump_norm_dir/out_acoustic_scaler.joblib $dump_norm_dir/ \
    --sample_rate $sample_rate $ext
