# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

for dbname in ${dbnames};
do
    for s in ${datasets[@]};
    do
        if [ -d conf/prepare_features ]; then
            ext="--config-dir conf/prepare_features"
        else
            ext=""
        fi
        xrun python $NNSVS_ROOT/nnsvs/bin/prepare_features.py $ext \
            utt_list=data/$dbname/list/$s.list out_dir=$dump_org_dir/$s/  \
            question_path=$question_path \
            timelag=$timelag_features duration=$duration_features acoustic=$acoustic_features \
            timelag.label_phone_score_dir=data/$dbname/timelag/label_phone_score/ \
            timelag.label_phone_align_dir=data/$dbname/timelag/label_phone_align/ \
            duration.label_dir=data/$dbname/duration/label_phone_align/ \
            acoustic.wav_dir=data/$dbname/acoustic/wav \
            acoustic.label_dir=data/$dbname/acoustic/label_phone_align \
            acoustic.sample_rate=$sample_rate
    done
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
        if [[ ${base_dump_norm_dir+x} && ! -z $base_dump_norm_dir ]]; then
            ext="external_scaler=${base_dump_norm_dir}/${inout}_${typ}_scaler.joblib"
        else
            ext=""
        fi
        find $dump_org_dir/$train_set/${inout}_${typ} -name "*feats.npy" > train_list.txt
        scaler_path=$dump_org_dir/${inout}_${typ}_scaler.joblib
        xrun nnsvs-fit-scaler list_path=train_list.txt scaler._target_=$scaler_class \
            out_path=$scaler_path ${ext}
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
