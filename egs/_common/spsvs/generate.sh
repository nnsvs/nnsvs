# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

for s in ${testsets[@]}; do
    for typ in timelag duration acoustic; do
        if [ $typ = "timelag" ]; then
            eval_checkpoint=$timelag_eval_checkpoint
        elif [ $typ = "duration" ]; then
            eval_checkpoint=$duration_eval_checkpoint
        else
            eval_checkpoint=$acoustic_eval_checkpoint
        fi

        checkpoint=$expdir/$typ/${eval_checkpoint}
        name=$(basename $checkpoint)
        xrun nnsvs-generate model.checkpoint=$checkpoint \
            model.model_yaml=$expdir/$typ/model.yaml \
            out_scaler_path=$dump_norm_dir/out_${typ}_scaler.joblib \
            in_dir=$dump_norm_dir/$s/in_${typ}/ \
            out_dir=$expdir/$typ/predicted/$s/${name%.*}/
    done
done
