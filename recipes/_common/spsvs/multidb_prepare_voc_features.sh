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
