# Notes for reproducing experiments

Paper: https://arxiv.org/abs/2210.15987.

## Data

Please normalize data to -26dB before running recipes. You can use sv56 to normalize audio signals.

## Recipes

- `icassp2023-24k-world`: Recipe using WORLD features. NNSVS-WORLD v1, v2, v3, v4, and Sinsy without vibrato modeling were trained with this recipe.
- `icassp2023-24k-mel`: Recipe using mel-spectrogram. NNSVS-Mel v1, v2, and v3 were trained with this recipe.
- `icassp2023-24k-mel-diffsinger-compat`: Recipe using mel-spectrogram. The feature extraction settings (e.g. FFT size) are the same as the [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger). This recipe was used for training DiffSinger-compatible hn-HiFi-GAN.
- `icassp2023-24k-world-sinevib`: Recipe using WORLD features with extra sine-based vibrato parameters. This recipe was used to build Sinsy with explicit vibrato modeling.
- `icassp2023-24k-world-diffvib`: Recipe using WORLD features with extra diff-based vibrato parameters. We used this recipe only in our preliminary experiments.

## Acoustic model configs

The following table summarizes the names of acoustic model configurations used in our experiments.
All the config files are stored inside the recipe directories.

| System                      | Recipe                       | Acoustic model config                               |
|-----------------------------|------------------------------|-----------------------------------------------------|
| Sinsy                       | icassp2023-24k-world         | acoustic_sinsy_world_novib.yaml                     |
| Sinsy (w/ pitch correction) | icassp2023-24k-world         | acoustic_sinsy_world_novib_pitchreg.yaml            |
| Sinsy (w/ vibrato modeling) | icassp2023-24k-world-sinevib | acoustic_sinsy_world_sinevib_pitchreg.yaml          |
| NNSVS-Mel v1                | icassp2023-24k-mel           | acoustic_nnsvs_melf0_multi_nonar.yaml               |
| NNSVS-Mel v2                | icassp2023-24k-mel           | acoustic_nnsvs_melf0_multi_ar_f0.yaml               |
| NNSVS-Mel v3                | icassp2023-24k-mel           | acoustic_nnsvs_melf0_multi_ar_melf0_prenet0.yaml    |
| NNSVS-WORLD v1              | icassp2023-24k-world         | acoustic_nnsvs_world_multi_nonar.yaml               |
| NNSVS-WORLD v2              | icassp2023-24k-world         | acoustic_nnsvs_world_multi_ar_f0.yaml               |
| NNSVS-WORLD v3              | icassp2023-24k-world         | acoustic_nnsvs_world_multi_ar_mgcf0_prenet0.yaml    |
| NNSVS-WORLD v4              | icassp2023-24k-world         | acoustic_nnsvs_world_multi_ar_mgcf0bap_npss_v1.yaml |


## How to run recipes

The same as the other recipes.

## Run acoustic model training

For example, to train NNSVS-WORLD v4, you can try the following command:

```
CUDA_VISIBLE_DEVICES=0 ./run.sh  --stage 4 --stop-stage 4 --tag 20220926_icassp_v3 --acoustic-model acoustic_nnsvs_world_multi_ar_mgcf0bap_npss_v1 --acoustic-data myconfig
```

If you have a pre-trained uSFGAN model checkpoint, you could try:

```
CUDA_VISIBLE_DEVICES=0 ./run.sh  --stage 4 --stop-stage 4 --tag 20220926_icassp_v3 --acoustic-model acoustic_nnsvs_world_multi_ar_mgcf0bap_npss_v1 --acoustic-data myconfig  --vocoder-eval-checkpoint $PWD/path/to/usfgan/checkpoint-600000steps.pkl
```

The above two commands are based on my command line history.

## How to train uSFGAN vocoder

### WORLD ver.

```
CUDA_VISIBLE_DEVICES=0 ./run.sh  --stage 98 --stop-stage 98 --vocoder-model nnsvs_world_parallel_hn_usfgan_sr24k
```

### MEL ver.

```
CUDA_VISIBLE_DEVICES=0 ./run.sh  --stage 98 --stop-stage 98 --vocoder-model nnsvs_melf0_parallel_hn_usfgan_sr24k
```

## How to synthesize waveforms

Please do make sure to set `--synthesis ${name}` property. For neural vocoders, please also specify `--vocoder-eval-checkpoint` explicitly.
Examples are as follows:

### WORLD vocoder

`--synthesis world_gv`

```
CUDA_VISIBLE_DEVICES=0 ./run.sh  --stage 6 --stop-stage 6 --acoustic-model acoustic_nnsvs_world_multi_ar_mgcf0bap_npss_v1 --synthesis world_gv
```

### uSFGAN (WORLD ver.)

`--synthesis world_gv_usfgan`

```
CUDA_VISIBLE_DEVICES=0 ./run.sh  --stage 6 --stop-stage 6 --acoustic-model acoustic_nnsvs_world_multi_ar_mgcf0bap_npss_v1 --synthesis world_gv_usfgan \
    --vocoder-model nnsvs_world_parallel_hn_usfgan_sr24k --vocoder-eval-checkpoint /path/to/checkpoint.pkl
```

### uSFGAN (MEL ver.)

`--synthesis melf0_gv_usfgan`

```
CUDA_VISIBLE_DEVICES=0 ./run.sh  --stage 6 --stop-stage 6 --acoustic-model acoustic_nnsvs_melf0_multi_ar_f0 --synthesis melf0_gv_usfgan \
    --vocoder-model nnsvs_melf0_parallel_hn_usfgan_sr24k --vocoder-eval-checkpoint /path/to/checkpoint.pkl
```

## Subjective tests

Please check mos_utt_list.txt for the list of utterance IDs used for our subjective evaluations.


## How to train DiffSinger

Code: https://github.com/nnsvs/DiffSinger (branch: namine_ritsu)

In the following guide, we assume NNSVS and DiffSinger repositories are placed at $HOME/nnsvs and $HOME/DiffSinger, respectively.

The basic strategy is to convert NNSVS's data to one of the DiffSinger's supported databases (i.e., Opencpop), so that the DiffSinger's code can be used with minimal changes. Once the conversion is property done, we can just use DiffSinger's codebase.

### Convert NNSVS's dataset to Opencpop's format

- You first need to run stage 0 for one of the icassp2023-* recipes (e.g., icassp2023-24k-mel-diffsinger-compat). Please make sure that `data/acoustic` directory is created.
- After the stage 0 is finished, you can convert the NNSVS' data to Opencpop's style by the following command:

```
python $HOME/nnsvs/utils/nnsvs2opencpop.py data/acoustic/ $HOME/DiffSinger/data/raw/ritsu_24k_diffsinger/segments
```

Note that we only tested Namine Ritsu's database, but it should be possible to use other databases as long as the same data preparation as in the NNSVS's stage 0 is used.

### Feature extraction

Please make sure to change the working directory to the DiffSinger directory.

```
cd $HOME/DiffSinger
```

Then, run the following command:

```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config usr/configs/midi/cascade/opencs/aux_rel_ritsu.yaml
```

### Training pitch extarctor

```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/pe_ritsu.yaml --exp_name 0923_ritsu_pe --reset
```

### Training DiffSinger MIDI-B version

There are two configs for training DiffSinger's acoustic model:

- usr/configs/midi/e2e/opencpop/ds100_adj_rel_ritsu_v3_ritsu_pe.yaml
- usr/configs/midi/e2e/opencpop/ds100_adj_rel_ritsu_v4_ritsu_pe.yaml

#### Using the pre-trained vocoder for Namine Ritsu (ds100_adj_rel_ritsu_v4_ritsu_pe.yaml)

NOTE: To train a DiffSinger model used in our experiments, you must need a pre-trained vocoder trained on Namine Ritsu's database. If you want to get a pre-trained model, please contact [@r9y9](https://github.com/r9y9).

```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/e2e/opencpop/ds100_adj_rel_ritsu_v4_ritsu_pe.yaml --exp_name 0923_ds100_adj_rel_ritsu_v4_ritsu_pe --reset
```

#### Using the pre-trained universal vocoder provided by the DiffSinger's authors (ds100_adj_rel_ritsu_v3_ritsu_pe.yaml)

Alternatively, if you are fine with the pre-trained universal vocoder provided by the DiffSinger's authors, you can train the DiffSinger by the following command:

```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/e2e/opencpop/ds100_adj_rel_ritsu_v3_ritsu_pe.yaml --exp_name 0923_ds100_adj_rel_ritsu_v3_ritsu_pe --reset
```

The results get slightly worse in my experience, but it should work fine.

### Synthesizing waveforms

Add ``--infer`` to the training command. e.g.,

```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/midi/e2e/opencpop/ds100_adj_rel_ritsu_v4_ritsu_pe.yaml --exp_name 0923_ds100_adj_rel_ritsu_v4_ritsu_pe --reset --infer
```

Please also check the DiffSinger's documentation for the detailed usage.
