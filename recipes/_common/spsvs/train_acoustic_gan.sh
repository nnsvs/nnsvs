# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

if [ -d conf/train_acoustic ]; then
    ext="--config-dir conf/train_acoustic_gan"
else
    ext=""
fi


if [ ! -z "${pretrained_expdir}" ]; then
    resume_checkpoint_g=$pretrained_expdir/${acoustic_model}/latest.pth
    if [ -e $pretrained_expdir/${acoustic_model}/latest_D.pth ]; then
        resume_checkpoint_d=$pretrained_expdir/${acoustic_model}/latest_D.pth
    else
        resume_checkpoint_d=
    fi
else
    resume_checkpoint_g=
    resume_checkpoint_d=
fi

# Hyperparameter search with Hydra + optuna
# mlflow is used to log the results of the hyperparameter search
if [[ ${acoustic_hydra_optuna_sweeper_args+x} && ! -z $acoustic_hydra_optuna_sweeper_args ]]; then
    hydra_opt="-m ${acoustic_hydra_optuna_sweeper_args}"
    post_args="mlflow.enabled=true mlflow.experiment=${expname}_${acoustic_model} hydra.sweeper.n_trials=${acoustic_hydra_optuna_sweeper_n_trials}"
else
    hydra_opt=""
    post_args=""
fi

xrun python $NNSVS_ROOT/nnsvs/bin/train_acoustic_gan.py $ext $hydra_opt \
    model=$acoustic_model train=$acoustic_train data=$acoustic_data \
    data.train_no_dev.in_dir=$dump_norm_dir/$train_set/in_acoustic/ \
    data.train_no_dev.out_dir=$dump_norm_dir/$train_set/out_acoustic/ \
    data.dev.in_dir=$dump_norm_dir/$dev_set/in_acoustic/ \
    data.dev.out_dir=$dump_norm_dir/$dev_set/out_acoustic/ \
    data.in_scaler_path=$dump_norm_dir/in_acoustic_scaler.joblib \
    data.out_scaler_path=$dump_norm_dir/out_acoustic_scaler.joblib \
    data.sample_rate=$sample_rate \
    train.out_dir=$expdir/${acoustic_model} \
    train.log_dir=tensorboard/${expname}_${acoustic_model} \
    train.pretrained_vocoder_checkpoint=$pretrained_vocoder_checkpoint \
    train.resume.netG.checkpoint=$resume_checkpoint_g \
    train.resume.netD.checkpoint=$resume_checkpoint_d $post_args
