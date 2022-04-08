# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

# Hed file
cp -v $question_path $dst_dir/qst.hed

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

echo "All the files are ready for SVS!"
echo "Please check the $dst_dir directory"