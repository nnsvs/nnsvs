# svs-world-conv

A recipe to build a statistical parametric singing voice synthesis system. The system consists of three trainable networks:

1. Time-lag model
2. Duration model
3. Acoustic model

For waveform analysis/synthesis, WORLD vocoder is used. The basic architecture of the system is similar to that described in the following paper:

Y. Hono et al, "Recent Development of the DNN-based Singing Voice Synthesis System — Sinsy," Proc. of APSIPA, 2017. ([PDF](http://www.apsipa.org/proceedings/2018/pdfs/0001003.pdf))

Note that details are much different (e.g., mixture density networks are not used in our code for now). Please see the code for the complete setup.

## Steps

Please download kiritan database first: https://zunko.jp/kiridev/login.php, and set `wav_root` in `run.sh` accordingly.

### Data download

```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage -1 --stop-stage -1
```

### Data preparation

```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 0 --stop-stage 0
```

### Feature extraction

```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 1 --stop-stage 1
```

### Training timelag/duration/acoustic models

```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 2 --stop-stage 4
```

### Synthesis


```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 5 --stop-stage 6
```

You can find generated samples in the `exp/kiritan/synthesis` directory.

## Advanced usage

### Pretrained model

If you want to utilize an external pretrained models, please specify `--pretrained-expdir` like:

```
CUDA_VISIBLE_DEVICES=0  run.sh --stage 2 --stop-stage 4 --pretrained-expdir /path/to/expdir
```

It is expected that the pretrained exp directory contains three sub-directires:

1. timelag
2. duration
3. acoustic

where each directory has a pre-trained model for timelag/duration/acoustic model, respectively.
