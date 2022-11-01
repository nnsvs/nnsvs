How to train neural vocoders with ParallelWaveGAN
=================================================

Please check :doc:`recipes` and :doc:`overview` first.

NNSVS v0.0.3 and later supports neural vocoders. This page summarizes how to train neural vocoders.

.. warning::

    This section needs to be re-written according to the recent major updates. Will be updated soon.

.. note::

    The contents in this page is based on ``recipes/conf/spsvs/run_common_steps_dev.sh``.
    Also, before you make your custom recipes, it is recommenced to start with a test recipe ``recipes/nit-song070/dev-test``.

Pre-requisites
--------------

What is NSF?
^^^^^^^^^^^^

NSF is an abbreviation of a neural source-filter model for speech synthesis (:cite:t:`wang2019neural`).

Input/output of a neural vocoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Input: acoustic features containing only static features
- Output: waveform

The current implementation of NNSVS does not use dynamic features (i.e., delta and delta-delta features) for neural vocoders.
If you enabled dynamic features, you must need to extract static features before training neural vocoders.

Install a fork of parallel_wavegan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NNSVS's neural vocoder integration is done by the external `r9y9/parallel_wavegan <https://github.com/r9y9/ParallelWaveGAN>`_ repository that provides various GAN-based neural vocoders.
To enable the neural vocoder support for NNSVS, you must need to install the nnsvs branch of the parallel_wavegan repository in advance:

.. code::

    pip install git+https://github.com/r9y9/ParallelWaveGAN@nnsvs

The nnsvs branch includes code for NSF.
After installation, please make sure that you can access ``HnSincNSF`` class. The following code should throw no errors if the installtaion is property done.

Jupyter:

.. code::

    In [1]: from parallel_wavegan.models.nsf import HnSincNSF
    In [2]: HnSincNSF(1,1)

Command-line:

.. code::

    python -c "from parallel_wavegan.models.nsf import HnSincNSF; print(HnSincNSF(1,1))"


Vocoder settings in ``config.yaml``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following settings are related to neural vocoders:

.. code-block::

    # NOTE: conf/parallel_wavegan/${vocoder_model}.yaml must exist.
    vocoder_model: hn-sinc-nsf_sr48k_pwgD_test
    # Pretrained checkpoint path for the vocoder model
    # NOTE: if you want to try fine-tuning, please specify the path here
    pretrained_vocoder_checkpoint:
    # absolute/relative path to the checkpoint
    # NOTE: the checkpoint is used for synthesis and packing
    # This doesn't have any effect on training
    vocoder_eval_checkpoint:

You can manually edit them or set them by command line like ``---vocoder-model hn-sinc-nsf_sr48k_pwgD_test``.

Stage 9: Prepare features for neural vocoders
---------------------------------------------

The pre-processing for neural vocoders (i.e., extract static features from acoustic features) is implemented as stage 9.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 9 --stop-stage 9

After running the above command, you will see the following output:

.. code-block:: text

    Converting dump/yoko/norm/out_acoustic_scaler.joblib mean/scale npy files
    [mean] dim: (206,) -> (67,)
    [scale] dim: (206,) -> (67,)
    [var] dim: (206,) -> (67,)

    If you are going to train NSF-based vocoders, please set the following parameters:

    out_lf0_mean: 5.9012025218118325
    out_lf0_scale: 0.2378365181913869

    NOTE: If you are using the same data for training acoustic/vocoder models, the F0 statistics
    for those models should be the same. If you are using different data for training
    acoustic/vocoder models (e.g., training a vocoder model on a multiple DBs),
    you will likely need to set different F0 statistics for acoustic/vocoder models.

After the pre-processing is property done, you can find the all the necessary features for training neural vocoders:

.. code-block::

    tree -L 3 dump/yoko/

    dump/yoko/
    ├── norm
    │   ├── dev
    │   │   ├── in_acoustic
    │   │   ├── in_duration
    │   │   ├── in_timelag
    │   │   ├── in_vocoder
    │   │   ├── out_acoustic
    │   │   ├── out_duration
    │   │   ├── out_postfilter
    │   │   └── out_timelag
    │   ├── eval
    │   │   ├── in_acoustic
    │   │   ├── in_duration
    │   │   ├── in_timelag
    │   │   ├── in_vocoder
    │   │   ├── out_acoustic
    │   │   ├── out_duration
    │   │   ├── out_postfilter
    │   │   └── out_timelag
    │   ├── in_acoustic_scaler.joblib
    │   ├── in_duration_scaler.joblib
    │   ├── in_timelag_scaler.joblib
    │   ├── in_vocoder_scaler_mean.npy
    │   ├── in_vocoder_scaler_scale.npy
    │   ├── in_vocoder_scaler_var.npy
    │   ├── out_acoustic_scaler.joblib
    │   ├── out_duration_scaler.joblib
    │   ├── out_postfilter_scaler.joblib
    │   ├── out_timelag_scaler.joblib
    │   └── train_no_dev
    │       ├── in_acoustic
    │       ├── in_duration
    │       ├── in_timelag
    │       ├── in_vocoder
    │       ├── out_acoustic
    │       ├── out_duration
    │       ├── out_postfilter
    │       └── out_timelag

Some notes:

- ``norm/${spk}/*/in_vocoder`` directory contains features for neural vocoders. Note that the directory contains both the input and output features. Specifically, ``*-feats.npy`` contains static features consisting of ``mgc``, ``lf0``, ``vuv`` and ``bap``; ``*-wave.npy`` contains raw waveform, respectively.
- ``norm/in_vocoder_scaler_*.npy`` contains statistics used to normalize/de-normalize the input features for neural vocoders.

Stage 10: Training vocoder using parallel_wavegan
-------------------------------------------------

.. warning::

    You must configure vocoder configs according to the sampling rate of the waveform and your feature extraction settings. It is strongly recommenced to go though the vocoder config before training your model. Vocoder configs for 24khz and 48kHz are available in the NNSVS repository, but can be extended for other sampling rates (e.g., 16kHz).

Once the pre-processing is done, you can train a neural vocoder by:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 10 --stop-stage 10 \
        --vocoder-model hn-sinc-nsf_sr48k_pwgD_test

You can find available model configs in ``conf/parlalel_wavegan`` directory, or you can create your own model config. Please do make sure to set ``out_lf0_mean`` and ``out_lf0_scale`` parameters correctly.

.. code-block::

    $ tree conf/parallel_wavegan
    conf/parallel_wavegan
    ├── hn-sinc-nsf_sr24k_pwgD.yaml
    ├── hn-sinc-nsf_sr48k_hifiganD.yaml
    ├── hn-sinc-nsf_sr48k_pwgD.yaml
    └── hn-sinc-nsf_sr48k_pwgD_test.yaml

Training progress can be monitored by tensorboard. During training you can check generated waveforms in ``exp/${speaker name}/${vocoder config name}/predictions`` directory.

Stage 11: Synthesis waveforms by parallel_wavegan
-------------------------------------------------

Stage 11 generates waveforms using the trained neural vocoder. Please make sure to specify your model type explicitly.

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 11 --stop-stage 11 \
        --vocoder-model hn-sinc-nsf_sr48k_pwgD_test

Generated wav files can be found in ``exp/${speaker name}/${vocoder config name}/wav`` directory.
To generate waveforms from a specific checkpoint, please specify the checkpoint path by ``--vocoder-eval-checkpoint /path/to/checkpoint``.

Packing models with neural vocoder
----------------------------------

To package all the models together, you can run the following command:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 99 --stop-stage 99 \
        --timelag-model timelag_test \
        --duration-model duration_test \
        --acoustic-model acoustic_test \
        --vocoder_model hn-sinc-nsf_sr48k_pwgD_test

Please make sure to add ``--vocoder_model ${vocoder config name}`` to package the trained vocoder as well.
You can also specify the explicit path of the trained model by ``--vocoder-eval-checkpoint /path/to/checkpoint``.

How to use the packed model with the trained vocoder?
-----------------------------------------------------

Please specify ``vocder_type="pwg"`` with the :doc:`modules/svs` module. An example:

.. code-block::

    import numpy as np
    import pysinsy
    from nnmnkwii.io import hts
    from nnsvs.pretrained import retrieve_pretrained_model
    from nnsvs.svs import SPSVS
    from nnsvs.util import example_xml_file

    model_dir = "/path/to/your/packed/model_dir"
    engine = SPSVS(model_dir)

    contexts = pysinsy.extract_fullcontext(example_xml_file(key="get_over"))
    labels = hts.HTSLabelFile.create_from_contexts(contexts)

    wav, sr = engine.svs(labels, vocoder_type="pwg")


Available neural vocoders
-------------------------

In addition to NSF, *any* models implemented in `parallel_wavegan <https://github.com/kan-bayashi/ParallelWaveGAN>`_ can be used with NNSVS. For example, Parallel WaveGAN, HiFiGAN, MelGAN, etc.
However, to get the best performance in singing synthesis, I'd recommend using ``HnSincNSF`` model (:cite:t:`wang2019hnsincnsf`), which is an advanced version of the original NSF (:cite:t:`wang2019neural`).

How to train universal vocoders?
--------------------------------

It is possible to make an *universal* vocoder that generalizes well on unseen speaker's data by training a vocoder on a large amount of mixed databases.

Training on mixed singing databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose you have multiple singing databases to train an neural vocodder on. Steps to train an universal vocoder are like:

- Run NNSVS' pre-processing for each database and combine them
- Run vocoder training

That's it. Please check the recipes in ``recipes/mixed`` for example.

Training on mixed speech and singing databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This should be easily implemented but not yet done by myself (r9y9). I may add code and docs for this in the future.
