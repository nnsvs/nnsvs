Recipes
========

What is a recipe?
-----------------

A recipe is a set of scripts and configuraitons to create singing voice synthesis (SVS) systems.
A recipe describes all the necessary steps including data preprocessing, training, and synthesis.
Since NNSVS provides recipes in a self-contained way, it is easy for anyone to create SVS systems.

Recipes have been adopted for reproducibility in several research projects such as `Kaldi <https://github.com/kaldi-asr/kaldi>`_ and `ESPnet <https://github.com/espnet/espnet>`_.
NNSVS follows a similar approach [1]_.

How to run a recipe
-------------------

All the recipes are stored in the ``recipes`` directory.
There are three important parts for every recipe:

- ``run.sh``: The entry point of the recipe.
- ``config.yaml``: A YAML based config file for recipe-specific configurations
- ``conf``: A directory that contains detailed configurations for each model. YAML-based configuration files for time-lag/duration/acoustic models are stored in this directory.


An example of ``conf`` directory is shown below. You can find model-specific configurations.

.. code-block:: bash

    conf
    ├── train
    │   ├── duration
    │   │   ├── data
    │   │   │   └── myconfig.yaml
    │   │   ├── model
    │   │   │   └── duration_mdn.yaml
    │   │   └── train
    │   │       └── myconfig.yaml
    │   └── timelag
    │       ├── data
    │       │   └── myconfig.yaml
    │       ├── model
    │       │   └── timelag_mdn.yaml
    │       └── train
    │           └── myconfig.yaml
    └── train_resf0
        └── acoustic
            ├── data
            │   └── myconfig.yaml
            ├── model
            │   └── acoustic_resf0convlstm.yaml
            └── train
                └── myconfig.yaml


A basic workflow to run a recipe is to run the ``run.sh`` from the command-line as follows:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 0 --stop-stage 99

The ``CUDA_VISIBLE_DEVICES=0`` denotes that the script uses the 0-th GPU. If you have only one GPU, you can omit the ``CUDA_VISIBLE_DEVICES``.
The last ``--stage 0 --stop-stage 99`` denotes that the script runs from stage 0 to stage 99.

To understand what's going on when running recipes, it is strongly recommended to run the recipe step-by-step as follows:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 0 --stop-stage 0

and then

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 1 --stop-stage 1

and so on.

Step -1 is typically reserved for optional data downloading step.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage -1 --stop-stage -1

Customizing recipes
--------------------

1. Copy a recipe directory and rename it
2. Edit config files

That's it. Pull requests for adding new recipes are welcome.


Packing models
----------------

As explained in the :doc:`overview`, NNSVS's SVS system is composed of multiple modules.
NNSVS provides functionality to pack the multiple models into a single directory, which can then be shared/loaded easily.

Some recipes have special step at 99 for the model packaging purpose. Please check the recipes for ``kiritan_singing`` database for example.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 99 --stop-stage 99

You can find a packed model in the ``packed_model`` directory.

A packed model directory will have the following files. Note that ``*postfilter_*`` and ``*vocoder_*`` files are optional.

.. code-block::

    $ ls -1
    acoustic_model.pth
    acoustic_model.yaml
    config.yaml
    duration_model.pth
    duration_model.yaml
    in_acoustic_scaler_min.npy
    in_acoustic_scaler_scale.npy
    in_duration_scaler_min.npy
    in_duration_scaler_scale.npy
    in_timelag_scaler_min.npy
    in_timelag_scaler_scale.npy
    in_vocoder_scaler_mean.npy
    in_vocoder_scaler_scale.npy
    in_vocoder_scaler_var.npy
    out_acoustic_scaler_mean.npy
    out_acoustic_scaler_scale.npy
    out_acoustic_scaler_var.npy
    out_duration_scaler_mean.npy
    out_duration_scaler_scale.npy
    out_duration_scaler_var.npy
    out_postfilter_scaler_mean.npy
    out_postfilter_scaler_scale.npy
    out_postfilter_scaler_var.npy
    out_timelag_scaler_mean.npy
    out_timelag_scaler_scale.npy
    out_timelag_scaler_var.npy
    postfilter_model.pth
    postfilter_model.yaml
    qst.hed
    timelag_model.pth
    timelag_model.yaml
    vocoder_model.pth
    vocoder_model.yaml

Some notes:

- ``*.pth`` files contain parameters of neural networks.
- ``*_model.yaml`` files contain definitions of neural networks such as the name of the PyTorch model (e.g., ``nnsvs.model.MDN``), number of layers, number of hidden units, etc.
- ``*.npy`` files contain parameters of scikit-learn's scalers that are used to normalize/denormalize features.
- ``qst.hed`` is the HED file used for training models.
- ``config.yaml`` is the global config file. It specifies sampling rate for example.

Once the packaging step is done, you can use the packaged model by the :doc:`svs` module. An example of using packed models can be found at :doc:`notebooks/Demos`.

.. [1] Recipes in NNSVS and Kaldi are technically different. For example, NNSVS does't use ``text``, ``feats.scp``, ``wav.scp`` and ``segments`` that are traditionally used in Kaldi.