# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

for s in ${datasets[@]};
do
    if [ -d conf/prepare_features ]; then
        ext="--config-dir conf/extract_static_features"
    else
        ext=""
    fi
    xrun nnsvs-extract-static-features $ext acoustic=$acoustic_features \
        in_dir=$dump_norm_dir/$s/out_acoustic/ \
        out_dir=$dump_norm_dir/$s/out_acoustic_static \
        utt_list=data/list/$s.list
done
