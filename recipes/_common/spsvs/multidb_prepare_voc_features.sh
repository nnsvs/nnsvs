# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

for dbname in ${dbnames};
do
    for s in ${datasets[@]};
    do
        if [ -d conf/prepare_static_features ]; then
            ext="--config-dir conf/prepare_static_features"
        else
            ext=""
        fi
        xrun nnsvs-prepare-voc-features $ext acoustic=$acoustic_features \
            in_dir=$dump_norm_dir/$s/out_acoustic/ \
            out_dir=$dump_norm_dir/$s/in_vocoder \
            utt_list=data/$dbname/list/$s.list
    done
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

# Compute statistics of vocoder's input features
# NOTE: no-op if the acoustic features don't have dynamic features
xrun python $NNSVS_COMMON_ROOT/scaler_joblib2npy_voc.py \
    $dump_norm_dir/out_acoustic_scaler.joblib $dump_norm_dir/ \
    --sample_rate $sample_rate $ext
