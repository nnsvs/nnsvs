# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

for s in ${datasets[@]};
do
    if [ -d conf/prepare_features ]; then
        ext="--config-dir conf/prepare_features"
    else
        ext=""
    fi
    xrun nnsvs-prepare-features $ext \
        utt_list=data/list/$s.list out_dir=$dump_org_dir/$s/  \
        question_path=$question_path \
        timelag=$timelag_features duration=$duration_features acoustic=$acoustic_features
done

# Compute normalization stats for each input/output
mkdir -p $dump_norm_dir
for inout in "in" "out"; do
    if [ $inout = "in" ]; then
        scaler_class="sklearn.preprocessing.MinMaxScaler"
    else
        scaler_class="sklearn.preprocessing.StandardScaler"
    fi
    for typ in timelag duration acoustic;
    do
        find $dump_org_dir/$train_set/${inout}_${typ} -name "*feats.npy" > train_list.txt
        scaler_path=$dump_org_dir/${inout}_${typ}_scaler.joblib
        xrun nnsvs-fit-scaler list_path=train_list.txt scaler._target_=$scaler_class \
            out_path=$scaler_path
        rm -f train_list.txt
        cp -v $scaler_path $dump_norm_dir/${inout}_${typ}_scaler.joblib
    done
done

# apply normalization
for s in ${datasets[@]}; do
    for inout in "in" "out"; do
        for typ in timelag duration acoustic;
        do
            xrun nnsvs-preprocess-normalize in_dir=$dump_org_dir/$s/${inout}_${typ}/ \
                scaler_path=$dump_org_dir/${inout}_${typ}_scaler.joblib \
                out_dir=$dump_norm_dir/$s/${inout}_${typ}/
        done
    done
done
