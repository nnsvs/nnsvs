# DNN SVS

PyTorch-based singing voice synthesis (SVS) library for research purposes.

## Samples

https://soundcloud.com/r9y9/sets/dnn-based-singing-voice

## Installation

- Python 3.5 or newer
- [nnmnkwii](https://github.com/r9y9/nnmnkwii): **development** version (master branch) is required
- [pysinsy](https://github.com/r9y9/pysinsy) **development** version is required. Please have a look at the repostiory for installation.
- Pytorch >= 1.x

Note taht packages listed above should be manually installed. After installing them, you can run:

```
python setup.py develop
```

to install the rest of dependencies.

## Recipes

Please see the [egs](egs) directory for recipes. Recipes include all the necessary steps to reproduce experiments, as similar to ones in [kaldi](https://github.com/kaldi-asr/kaldi) and [ESPnet](https://github.com/espnet/espnet). Note that the directory structure is different from the kaldi's ones.


## What

PyTorch-based singing voice synthesis (SVS) library for research purposes. We’ll focus on the research advancement of the singing voice synthesis technology.

## Background

As of Feb. 2020, [NEUTRINO](https://n3utrino.work/), a DNN-based singing voice synthesis tool, has started gaining its popularity in Japan. Because of the powerful DNN-based approach, users can create expressive and natural singing voices even without manual tuning which is typically required to achieve satisfactory quality using the existing tools.

While NEUTRINO is a great tool for creative purposes, it is not open-source software. In fact, there are only a few open-source toolkits to the best of our knowledge. To advance the singing voice synthesis research, we aim to provide a modern DNN-based singing voice synthesis tool for researchers and developers.

That being said, I was just curious to see if I can make a better one than NEUTRINO. We’ll see :)

## History

See [HISTORY.md](HISTORY.md)

## References

- A fork of sinsy: https://github.com/r9y9/sinsy
- Python wrapper for sinsy: https://github.com/r9y9/pysinsy
- NEUTRINO: https://n3utrino.work/