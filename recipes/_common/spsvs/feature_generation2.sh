# NOTE: the script is supposed to be used called from nnsvs recipes.
# Please don't try to run the shell script directory.

if [ -z ${sample_rate+x} ]; then
    echo "ERROR: sampler_rate must be set explicitly. Please add 'sample_rate: <value>' to config.yaml."
    echo "e.g., sample_rate: 48000"
    exit -1
fi

if [[ -z ${trajectory_smoothing+x} ]]; then
    trajectory_smoothing=false
    trajectory_smoothing_cutoff=50
fi

for s in ${datasets[@]};
do
    if [ -d conf/prepare_features ]; then
        ext="--config-dir conf/prepare_features"
    else
        ext=""
    fi
    xrun python $NNSVS_ROOT/nnsvs/bin/prepare_features.py $ext \
        utt_list=data/list/$s.list out_dir=$dump_org_dir/$s/ \
        question_path=$question_path \
        timelag=$timelag_features duration=$duration_features acoustic=$acoustic_features \
        acoustic.sample_rate=$sample_rate \
        acoustic.trajectory_smoothing=${trajectory_smoothing} \
        acoustic.trajectory_smoothing_cutoff=${trajectory_smoothing_cutoff}
done

# Pitch-shift data augmentation
for s in ${datasets[@]}; do
    for typ in in "in" "out"; do
        for shift_in_cent in -100 100; do
            python $NNSVS_ROOT/utils/pitch_augmentation.py data/list/$s.list \
                $dump_org_dir/$s/${typ}_acoustic $dump_org_dir/$s/${typ}_acoustic_aug \
                $question_path $typ $shift_in_cent
        done
        cp $dump_org_dir/$s/${typ}_acoustic/*.npy $dump_org_dir/$s/${typ}_acoustic_aug/
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
    for typ in timelag duration acoustic postfilter;
    do
        if [ ! -d $dump_org_dir/$train_set/${inout}_${typ} ]; then
            continue
        fi
        if [[ ${base_dump_norm_dir+x} && ! -z $base_dump_norm_dir ]]; then
            ext="external_scaler=${base_dump_norm_dir}/${inout}_${typ}_scaler.joblib"
        else
            ext=""
        fi
        find $dump_org_dir/$train_set/${inout}_${typ} -name "*feats.npy" > train_list.txt
        scaler_path=$dump_org_dir/${inout}_${typ}_scaler.joblib
        xrun python $NNSVS_ROOT/nnsvs/bin/fit_scaler.py \
            list_path=train_list.txt scaler._target_=$scaler_class \
            out_path=$scaler_path ${ext}
        rm -f train_list.txt
        cp -v $scaler_path $dump_norm_dir/${inout}_${typ}_scaler.joblib
    done
done

# apply normalization
for s in ${datasets[@]}; do
    for inout in "in" "out"; do
        for typ in timelag duration acoustic postfilter;
        do
            if [ -e $dump_org_dir/${inout}_${typ}_scaler.joblib ]; then
                xrun nnsvs-preprocess-normalize in_dir=$dump_org_dir/$s/${inout}_${typ}/ \
                    scaler_path=$dump_org_dir/${inout}_${typ}_scaler.joblib \
                    out_dir=$dump_norm_dir/$s/${inout}_${typ}/
                if [ -d $dump_org_dir/$s/${inout}_${typ}_aug ]; then
                    xrun nnsvs-preprocess-normalize in_dir=$dump_org_dir/$s/${inout}_${typ}_aug/ \
                        scaler_path=$dump_org_dir/${inout}_${typ}_scaler.joblib \
                        out_dir=$dump_norm_dir/$s/${inout}_${typ}_aug/
                fi
            fi
        done
    done
done
