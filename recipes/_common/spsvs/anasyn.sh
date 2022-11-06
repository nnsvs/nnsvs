# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

if [ -d conf/synthesis ]; then
    ext="--config-dir conf/synthesis"
else
    ext=""
fi

if [ -z $timelag_eval_checkpoint ]; then
    timelag_eval_checkpoint=best_loss.pth
fi
if [ -z $duration_eval_checkpoint ]; then
    duration_eval_checkpoint=best_loss.pth
fi
if [ -z $acoustic_eval_checkpoint ]; then
    acoustic_eval_checkpoint=latest.pth
fi

if [ -z "${vocoder_eval_checkpoint}" ]; then
    if [[ -z "${vocoder_eval_checkpoint}" && ! -z ${vocoder_model} ]]; then
        vocoder_eval_checkpoint="$(ls -dt "$expdir/$vocoder_model"/*.pkl | head -1 || true)"
    fi
fi

if [ -z "${vocoder_eval_checkpoint}" ]; then
    dst_name=anasyn_world
else
    if [ ! -z "${vocoder_model}" ]; then
        dst_name=anasyn_${vocoder_model}
    else
        vocoder_name=$(dirname ${vocoder_eval_checkpoint})
        vocoder_name=$(basename $vocoder_name)
        dst_name=anasyn_${vocoder_name}
    fi
fi

for s in ${testsets[@]}; do
    xrun python $NNSVS_ROOT/nnsvs/bin/anasyn.py $ext \
        synthesis=$synthesis \
        synthesis.sample_rate=$sample_rate \
        acoustic.model_yaml=$expdir/${acoustic_model}/model.yaml \
        vocoder.checkpoint=$vocoder_eval_checkpoint \
        utt_list=./data/list/$s.list \
        in_dir=$dump_org_dir/$s/out_acoustic \
        out_dir=$expdir/$dst_name/$s/
done
