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

TBD
