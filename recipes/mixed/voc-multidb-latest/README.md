# voc-multidb-latest

A recipe to build a universal neural vocoder on multiple databases. So far the following academic-friendly databases are used:

1. nit-song070
2. kiritan_singing
3. jsut-song
4. PJS

If you want to maximize the performane of a vocoder for your database, please consider the two options below:
1. Add your database and train a universal neural vocoder
2. Train a singer-dependent vocoder for your database

If you want to avoid re-training vocoders for every database, the former would be better. If you want to maximizez the performance specifically for your database, the later would be better.

## Notice

Training will likely take a few days with a high-performance GPU (such as Tesla V100). If you use strong discriminators such as ones in HiFiGAN, it will take more. I don't recommend training NSFs on google colab. If you see GPU out of memory errors, try smaller `batch_size` and `batch_max_steps`.

If you don't have enough compute resources, please consider using a pre-trained universal model (TBD: I'll share it when training is finished), or try fine-tuning strategy.

## Requirements

```
pip install git+https://github.com/r9y9/ParallelWaveGAN@nnsvs
```

The fork of ParallelWaveGAN supports neural source filter (NSF) models, which work nicely for singing voice synthesis. Furthermore, thanks to the flexible design of ParallelWaveGAN, it is possible to train NSF with adversarial training.


## Configs

NSF configs are found in conf directory. So far configs for 48 kHz sampling is only included.
