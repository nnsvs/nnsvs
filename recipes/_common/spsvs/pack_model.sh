# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

# Hed file
cp -v $question_path $dst_dir/qst.hed

# Table file for utaupy
if [[ ${utaupy_table_path+x} ]]; then
    cp -v $utaupy_table_path $dst_dir/kana2phonemes.table
fi

# Stats
for typ in "timelag" "duration" "acoustic"; do
    for inout in "in" "out"; do
        python $NNSVS_COMMON_ROOT/scaler_joblib2npy.py $dump_norm_dir/${inout}_${typ}_scaler.joblib $dst_dir
    done
done

# Timelag model
python $NNSVS_COMMON_ROOT/clean_checkpoint_state.py $expdir/${timelag_model}/$timelag_eval_checkpoint \
    $dst_dir/timelag_model.pth
cp $expdir/${timelag_model}/model.yaml $dst_dir/timelag_model.yaml

# Duration model
python $NNSVS_COMMON_ROOT/clean_checkpoint_state.py $expdir/${duration_model}/$duration_eval_checkpoint \
    $dst_dir/duration_model.pth
cp $expdir/${duration_model}/model.yaml $dst_dir/duration_model.yaml

# Acoustic model
python $NNSVS_COMMON_ROOT/clean_checkpoint_state.py $expdir/${acoustic_model}/$acoustic_eval_checkpoint \
    $dst_dir/acoustic_model.pth
cp $expdir/${acoustic_model}/model.yaml $dst_dir/acoustic_model.yaml

# Post-filter model
if [[ ${postfilter_model+x} ]]; then
    if [ -e $expdir/${postfilter_model}/${postfilter_eval_checkpoint} ]; then
        python $NNSVS_COMMON_ROOT/clean_checkpoint_state.py $expdir/${postfilter_model}/$postfilter_eval_checkpoint \
            $dst_dir/postfilter_model.pth
        cp $expdir/${postfilter_model}/model.yaml $dst_dir/postfilter_model.yaml
        python $NNSVS_COMMON_ROOT/scaler_joblib2npy.py $dump_norm_dir/out_postfilter_scaler.joblib $dst_dir
    else
        echo "WARN: Post-filter model checkpoint is not found. Skipping."
    fi
fi

# Vocoder model
if [[ ${vocoder_model+x} || ${vocoder_eval_checkpoint+x} ]]; then
    # PWG & uSFGAN
    if [[ -z "${vocoder_eval_checkpoint}" && ! -z ${vocoder_model} && -d ${expdir}/${vocoder_model} ]]; then
        vocoder_eval_checkpoint="$(ls -dt "$expdir/$vocoder_model"/*.pkl | head -1 || true)"
    fi
    if [ -e "$vocoder_eval_checkpoint" ]; then
        python $NNSVS_COMMON_ROOT/clean_checkpoint_state.py $vocoder_eval_checkpoint \
            $dst_dir/vocoder_model.pth
        # PWG's expdir or packed model's dir
        voc_dir=$(dirname $vocoder_eval_checkpoint)
        # PWG's expdir
        if [ -e ${voc_dir}/config.yml ]; then
            cp ${voc_dir}/config.yml $dst_dir/vocoder_model.yaml
        # uSFGAN's expdir
        elif [ -e ${voc_dir}/config.yaml ]; then
            cp ${voc_dir}/config.yaml $dst_dir/vocoder_model.yaml
        # Packed model's dir
        elif [ -e ${voc_dir}/vocoder_model.yaml ]; then
            cp ${voc_dir}/vocoder_model.yaml $dst_dir/vocoder_model.yaml
        else
            echo "ERROR: vocoder config is not found!"
            exit 1
        fi
        # NOTE: assuming statistics are copied to the checkpoint directory
        cp $voc_dir/in_vocoder*.npy $dst_dir/
    else
        echo "WARN: Vocoder model checkpoint is not found. Skipping."
    fi
fi

echo "All the files are ready for SVS!"
echo "Please check the $dst_dir directory"
