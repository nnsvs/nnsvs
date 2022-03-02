# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

if [ -d conf/train ]; then
    ext="--config-dir conf/train/timelag"
else
    ext=""
fi

if [ ! -z "${pretrained_expdir}" ]; then
    resume_checkpoint=$pretrained_expdir/timelag/latest.pth
else
    resume_checkpoint=
fi
xrun nnsvs-train $ext \
    model=$timelag_model train=$timelag_train data=$timelag_data \
    data.train_no_dev.in_dir=$dump_norm_dir/$train_set/in_timelag/ \
    data.train_no_dev.out_dir=$dump_norm_dir/$train_set/out_timelag/ \
    data.dev.in_dir=$dump_norm_dir/$dev_set/in_timelag/ \
    data.dev.out_dir=$dump_norm_dir/$dev_set/out_timelag/ \
    train.out_dir=$expdir/timelag \
    train.resume.checkpoint=$resume_checkpoint