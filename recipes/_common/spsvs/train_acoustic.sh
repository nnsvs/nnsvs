# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

if [ -d conf/train ]; then
    ext="--config-dir conf/train/acoustic"
else
    ext=""
fi

if [ ! -z "${pretrained_expdir}" ]; then
    resume_checkpoint=$pretrained_expdir/${acoustic_model}/latest.pth
else
    resume_checkpoint=
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

xrun nnsvs-train $ext $hydra_opt \
    model=$acoustic_model train=$acoustic_train data=$acoustic_data \
    data.train_no_dev.in_dir=$dump_norm_dir/$train_set/in_acoustic/ \
    data.train_no_dev.out_dir=$dump_norm_dir/$train_set/out_acoustic/ \
    data.dev.in_dir=$dump_norm_dir/$dev_set/in_acoustic/ \
    data.dev.out_dir=$dump_norm_dir/$dev_set/out_acoustic/ \
    train.out_dir=$expdir/${acoustic_model} \
    train.log_dir=tensorboard/${expname}_${acoustic_model} \
    train.resume.checkpoint=$resume_checkpoint $post_args