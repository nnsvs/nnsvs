# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

if [ -d conf/train ]; then
    ext="--config-dir conf/train_resf0/acoustic"
else
    ext=""
fi

if [ ! -z "${pretrained_expdir}" ]; then
    resume_checkpoint=$pretrained_expdir/${acoustic_model}/latest.pth
else
    resume_checkpoint=
fi
xrun nnsvs-train-resf0 $ext \
    model=$acoustic_model train=$acoustic_train data=$acoustic_data \
    data.train_no_dev.in_dir=$dump_norm_dir/$train_set/in_acoustic/ \
    data.train_no_dev.out_dir=$dump_norm_dir/$train_set/out_acoustic/ \
    data.dev.in_dir=$dump_norm_dir/$dev_set/in_acoustic/ \
    data.dev.out_dir=$dump_norm_dir/$dev_set/out_acoustic/ \
    data.in_scaler_path=$dump_norm_dir/in_acoustic_scaler.joblib \
    data.out_scaler_path=$dump_norm_dir/out_acoustic_scaler.joblib \
    train.out_dir=$expdir/${acoustic_model} \
    train.log_dir=tensorboard/${expname}_${acoustic_model} \
    train.resume.checkpoint=$resume_checkpoint