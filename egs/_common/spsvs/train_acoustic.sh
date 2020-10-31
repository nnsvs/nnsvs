# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directry.

if [ ! -z "${pretrained_expdir}" ]; then
    resume_checkpoint=$pretrained_expdir/acoustic/latest.pth
else
    resume_checkpoint=
fi
xrun nnsvs-train data.train_no_dev.in_dir=$dump_norm_dir/$train_set/in_acoustic/ \
    data.train_no_dev.out_dir=$dump_norm_dir/$train_set/out_acoustic/ \
    data.dev.in_dir=$dump_norm_dir/$dev_set/in_acoustic/ \
    data.dev.out_dir=$dump_norm_dir/$dev_set/out_acoustic/ \
    model=acoustic train.out_dir=$expdir/acoustic \
    data.batch_size=$batch_size \
    resume.checkpoint=$resume_checkpoint