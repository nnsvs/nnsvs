# NN-SVS

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)
![Python CI](https://github.com/r9y9/nnsvs/workflows/Python%20CI/badge.svg)

Neural network-based singing voice synthesis library for research.

## Demo

### Neural network-based singing voice synthesis demo using kiritan_singing database (Japanese)

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r9y9/Colaboratory/blob/master/Neural_network_based_singing_voice_synthesis_demo_using_kiritan_singing_database_(Japanese).ipynb)
- [![Nbviewer](https://github.com/jupyter/design/blob/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/gist/r9y9/79705665ed5a94f0028839ca40992751)

## Audio samples

- Kiritan samples: https://soundcloud.com/r9y9/sets/dnn-based-singing-voice

## Installation

```
python setup.py develop
```

## Repository structure

- Core library: [nnsvs/](nnsvs/)
- Command line programs: [nnsvs/bin/](nnsvs/bin) and its configurations [nnsvs/bin/conf/](nnsvs/bin/conf/)
- Recipes: [egs/](egs/)

## Python docstring style

https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

## Recipes

A recipe is a set of scripts and configuraitons that are used to reproduce experiments. All the steps used to conduct experiments are provided in a self-contained way. Please have a look at the [egs](egs) directory if you want to build your singing voice systems.

## Background

As of Feb. 2020, [NEUTRINO](https://n3utrino.work/), a DNN-based singing voice synthesis tool, has started gaining its popularity in Japan. Because of the powerful DNN-based approach, users can create expressive and natural singing voices even without manual tuning which is typically required to achieve satisfactory quality using the existing tools.

While NEUTRINO is a great tool for creative purposes, it is not open-source software. In fact, there are only a few open-source toolkits to the best of our knowledge. To advance the singing voice synthesis research, we aim to provide a modern DNN-based singing voice synthesis tool for researchers and developers.

That being said, I was just curious to see if I can make a better one than NEUTRINO. We’ll see :)

## History

See [HISTORY.md](HISTORY.md)

## Related projects

- English support for nnsvs: https://github.com/DynamiVox/nnsvs-english-support
- NNSVSのモデルをUTAUで使えるようにするツール (UTAU plugin software powered by NNSVS): https://github.com/oatsu-gh/ENUNU

## References

- Y. Hono et al, "Sinsy: A Deep Neural Network-Based Singing Voice Synthesis System", Journal of IEEE/ACM TASLP https://arxiv.org/abs/2108.02776
- Y. Hono et al, "Recent Development of the DNN-based Singing Voice Synthesis System — Sinsy," Proc. of APSIPA, 2017. ([PDF](http://www.apsipa.org/proceedings/2018/pdfs/0001003.pdf))
- A fork of sinsy: https://github.com/r9y9/sinsy
- Python wrapper for sinsy: https://github.com/r9y9/pysinsy
- NEUTRINO: https://n3utrino.work/

