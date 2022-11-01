Getting started with recipes
============================

.. warning::

    This section needs to be re-written according to the recent major updates. Will be updated soon.

This page describes how to use a recipe to create a singing voice synthesis (SVS) system.

What is a recipe?
-----------------

A recipe is a set of scripts and configuraitons to create SVS systems.
A recipe describes all the necessary steps including data preprocessing, training, and synthesis.

Recipes have been adopted for reproducibility in several research projects such as `Kaldi <https://github.com/kaldi-asr/kaldi>`_ and `ESPnet <https://github.com/espnet/espnet>`_.
NNSVS follows a similar approach [1]_.

Structure of a recipe
---------------------

All the recipes are stored in the ``recipes`` directory. Recipes are usually made per database. e.g., ``recipes/nit-song070`` contains recipes for nit-song070 database.

There are three important parts for every recipe:

- ``run.sh``: The entry point of the recipe.
- ``config.yaml``: A YAML based config file for recipe-specific configurations
- ``conf``: A directory that contains detailed configurations for each model. YAML-based configuration files for time-lag/duration/acoustic models are stored in this directory.


An example of ``conf`` directory is shown below. You can find model-specific configurations.

.. code-block::

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
    └── train_acoustic
        └── acoustic
            ├── data
            │   └── myconfig.yaml
            ├── model
            │   └── acoustic_resf0convlstm.yaml
            └── train
                └── myconfig.yaml

.. note::

    The contents of the rest of this page is based on ``recipes/conf/spsvs/run_common_steps_stable.sh``.

How to run a recipe
-------------------

A basic workflow to run a recipe is to run the ``run.sh`` from the command-line as follows:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 0 --stop-stage 99

The ``CUDA_VISIBLE_DEVICES=0`` denotes that the script uses the 0-th GPU. If you have only one GPU, you can omit the ``CUDA_VISIBLE_DEVICES``.
The last ``--stage 0 --stop-stage 99`` denotes that the script runs from stage 0 to stage 99.

Stage -1 is reserved for optional data downloading step. Some databases require you to sign a contract in advance. In that case, you need to manually download database.


.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage -1 --stop-stage -1


To understand what's going on when running recipes, it is strongly recommended to run the recipe step-by-step.

.. note::

    If you are new to recipes, please start with a test recipe ``nnsvs/recipes/nit-song070/dev-test``.

Note that every item in ``config.yaml`` can be customized by the command-line interface.
For example, the following command

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 0 --stop-stage 99 \
        --timelag_model timelag_mdn

is equivalent to manually changing the ``config.yaml`` for the ``nnsvs/recipes/nit-song070/dev-test`` recipe as follows:

.. code-block:: diff

    -timelag_model: timelag_test
    +timelag_model: timelag_mdn

Recipes can be arbitrary configured depending on your purpose, but the followings are some common steps for recipes.


Stage 0: Data preparation
-------------------------

Stage 0 for the most recipes does the following three things:

- Convert MusicXML or UST to HTS-style full-context labels.
- Segment singing data into small segments.
- Split the data into train/dev/test sets.

The second step is optional but is helpful to avoid GPU out-of-memory errors.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 0 --stop-stage 0

Stage 1: Feature generation
---------------------------

This step performs all the feature extraction steps needed to train time-lag/duration/acoustic models.
HTS-style full-context label files and wav files are processed together to prepare inputs/outputs for neural networks.

Note that errors will happen when your wav files and label files are not aligned correctly.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 1 --stop-stage 1

After running the step, you can find extracted features in ``dump`` directory.

.. code-block::

    $ tree -L 4  dump/

    dump/
    └── yoko
        ├── norm
        │   ├── dev
        │   │   ├── in_acoustic
        │   │   ├── in_duration
        │   │   ├── in_timelag
        │   │   ├── out_acoustic
        │   │   ├── out_duration
        │   │   └── out_timelag
        │   ├── eval
        │   │   ├── in_acoustic
        │   │   ├── in_duration
        │   │   ├── in_timelag
        │   │   ├── out_acoustic
        │   │   ├── out_duration
        │   │   └── out_timelag
        │   ├── in_acoustic_scaler.joblib
        │   ├── in_duration_scaler.joblib
        │   ├── in_timelag_scaler.joblib
        │   ├── out_acoustic_scaler.joblib
        │   ├── out_duration_scaler.joblib
        │   ├── out_timelag_scaler.joblib
        │   └── train_no_dev
        │       ├── in_acoustic
        │       ├── in_duration
        │       ├── in_timelag
        │       ├── out_acoustic
        │       ├── out_duration
        │       └── out_timelag
       └── org
       ...

Some notes:

- ``norm`` and ``org`` directories contain normalized and unnormalized features. Normalized features are used for training neural networks.
- ``*_scaler.joblib`` files are used to normalize/de-normalize features and contain statistics of the training data (e.g., mean and varaince). The file format follows `joblib <https://joblib.readthedocs.io/en/latest/>`_.
- ``in_*`` and ``out_*`` directories contain input and output features.

All the features are saved in numpy format. You can inspect features by a simple python script like:

.. code-block::

    import numpy as np
    feats = np.load("path/to/your/features.npy")


Stage 2: Train time-lag model
-----------------------------

Once the feature generation is completed, you are ready to train neural networks.

You can train a time-lag model by:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 2 --stop-stage 2

Or, you may want to explicltly specify a model by:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 2 --stop-stage 2 \
        --timelag-model timelag_test

You can find available model configs in ``conf/train/timelag/model`` directory, or you can create your own model config.

After training is finished, you can find model checkpoints in ``exp`` directory.

.. code-block::

    exp/yoko/timelag_test/
    ├── best_loss.pth
    ├── config.yaml
    ├── epoch0002.pth
    ├── latest.pth
    └── model.yaml

Some notes:

- ``*.pth`` files are the model checkpoints where the parameters of neural networks are stored.
- ``*.yaml`` are the configuration files.
- ``model.yaml`` is a model-specific config. This file can be used to instantiate a model by `hydra <https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_.
- ``config.yaml`` contains all the training details.
- ``best_loss.pth`` is the checkpoint when the model hit the best development loss.
- ``latest.pth`` is the latest checkpoint.
- ``epoch*.pth`` are intermediate checkpoints at a specific epoch.

Stage 3: Train duration model
-----------------------------

Similarly, you can train a duration model by:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 3 --stop-stage 3

You can explicltly specify a model type by:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 3 --stop-stage 3 \
        --duration-model duration_test

You can find available model configs in ``conf/train/duration/model``.

After training is finished, you can find model checkpoints in ``exp`` directory.

.. code-block::

    exp/yoko/duration_test/
    ├── best_loss.pth
    ├── config.yaml
    ├── epoch0002.pth
    ├── latest.pth
    └── model.yaml

Stage 4: Train acoustic model
-----------------------------

The acoustic model is the most important part of the SVS system. You are likely to run this step multiple times until you get a good model.
You can train an acoustic model by:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 4 --stop-stage 4

You can explicltly specify a model type by:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 4 --stop-stage 4 \
        --acoustic-model acoustic_test

You can find available model configs in ``conf/train_acoustic/acoustic/model``, or you can create your own model config.

.. note::

    Training aoustic models requires several hours or a whole day depending on training configurations.
    During training, it is useful to monitor training progress using Tensorboard. See :doc:`tips` for more details.

After training is finished, you can find model checkpoints in ``exp`` directory.

.. code-block::

    exp/yoko/acoustic_test/
    ├── best_loss.pth
    ├── config.yaml
    ├── epoch0002.pth
    ├── latest.pth
    └── model.yaml

Stage 5: Generate features
---------------------------

One you have trained all the models, you can genearte features by your models.
You can ignore this step if you want to listen to audio samples rather than inspecting intermedieate features.

If you use your custom model types at the training steps, you must specify these models at this step.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 5 --stop-stage 5 \
        --timelag-model timelag_test \
        --duration-model duration_test \
        --acoustic-model acoustic_test

Stage 6: Synthesis waveforms
----------------------------

Stage 6 generates waveforms using the trained models. To run this step, please make sure to specify your model types when you train custom models.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 5 --stop-stage 5 \
        --timelag-model timelag_test \
        --duration-model duration_test \
        --acoustic-model acoustic_test

You can find generated wav files in ``exp/${speaker name}/synthesis_*`` directory.

Packing models
---------------

As explained in the :doc:`overview`, NNSVS's SVS system is composed of multiple modules.
NNSVS provides functionality to pack the multiple models into a single directory, which can then be shared/used easily.

Recipes have special step at 99 for the model packaging purpose.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 99 --stop-stage 99

Note that you must specify model types if you use custom models. e.g.,

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 99 --stop-stage 99 \
        --timelag-model timelag_test \
        --duration-model duration_test \
        --acoustic-model acoustic_test


After running the command above, you can find a packed model in the ``packed_model`` directory.

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

Once the packaging step is done, you can use the packaged model by the :doc:`modules/svs` module. An example of using packed models can be found at :doc:`notebooks/Demos`.

With the packed model, you can easily generate singing voice by inputting MusicXML or UST files.

Customizing recipes
--------------------

Not just running existing recipes, you may want to make your own ones. e.g., adding your custom models, customizing steps, using your own data, etc.

If you want to make your own recipe, the easiest way is to copy an existing recipe and modify it accordingly.
Please check one of recipes in the NNSVS repostiry and start modifying part of them.

.. [1] Recipes in NNSVS and Kaldi are technically different. For example, NNSVS does't use ``text``, ``feats.scp``, ``wav.scp`` and ``segments`` that are traditionally used in Kaldi.

