How to train post-filters
=========================

Please check :doc:`recipes` and :doc:`overview` first.

NNSVS v0.0.3 and later supports an optional trainable post-filter to enhance the acoustic model's prediction.
This page summarizes how to train post-filters.

.. warning::

    As of 2022/10/15, I concluded that GV-post filter works better than trainalbe post-filters in most cases.
    Please consider using GV-post filter insteaad.

.. note::

    The contents in this page is based on ``recipes/conf/spsvs/run_common_steps_dev.sh``.
    Also, before you make your custom recipes, it is recommenced to start with a test recipe ``recipes/nit-song070/dev-test``.

Pre-requisites
--------------

Input/output of a post-filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Input and output of a post-filter are as follows:

- Input: acoustic features predicted by an acoustic model
- Output: enhanced acoustic features

Note that post-filters do not use delta and delta-delta features.
If your acoustic model's output contains delta and delta-delta features, the parameter generation algorithm (a.k.a. MLPG) is performed to prepare input/output features for post-filters.

You must train an acoustic model first
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You must train an acoustic model first since the input of a post-filter depends the output of an acoustic model.
Furthermore, please be aware that you need to re-train a post-filter whenever you re-train your acoustic model.
Therefore, it is highly recommended to train a good acoustic model before training a post-filter.

Train a good acoustic model first
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is better to train a good acoustic model. This is because the post-filter is trained on the features predicted by the acoustic model.
If the acoustic model's prediction is not accurate enough, the post-filter is likely to have a bad performance.

In addition to the steps described in :doc:`recipes`,  the following are the steps related to post-filters.

Stage 7: Prepare features for post-filter
-----------------------------------------

Once your acoustic model is ready, you can run the stage 7 to prepare input and output features for training post-filters.


.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 7 --stop-stage 7 \
        --acoustic-model acoustic_test

After running the above command, you can find the input features for post-filters in the acoustic model's checkpoint directory:

.. code-block::

    $ tree -L 3 exp/yoko/acoustic_test/

    exp/yoko/acoustic_test/
    ├── best_loss.pth
    ├── config.yaml
    ├── epoch0002.pth
    ├── latest.pth
    ├── model.yaml
    ├── norm
    │   ├── dev
    │   │   └── in_postfilter
    │   ├── eval
    │   │   └── in_postfilter
    │   ├── in_postfilter_scaler.joblib
    │   └── train_no_dev
    │       └── in_postfilter
    └── predicted
        └── eval
            └── latest

Some notes:

- ``norm/*/in_postfilter`` directory contains the input features for post-filters.
- ``norm/in_postfilter_scaler.joblib`` contains the scaler used to normalize/de-normalize the input features for post-filters.

As for the output features, you can find them in the ``dump`` directory.

.. code-block::

    $ tree -L 4 dump/

    dump/
    └── yoko
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
        └── org


Some notes:

- ``dump/*/norm/*/out_postfilter`` directory contains the output features for post-filters. Again, remember that these features don't contain delta and delta-delta features.
- ``dump/*/norm/out_postfilter_scaler.joblib`` contains the scaler used to normalize/de-normalize the output features for post-filters.


Stage 8: Train post-filters
---------------------------

Once you generated input/output features, you are ready to train post-filters. The current NNSVS's post-filter is based on generative adversarial networks (GANs). So you need to train generator and discrimiantor together.

There are number of different ways to train post-filters by NNSVS. However, the following is the recommended way to get the best performance (based on r9y9's experience):

1. Train a post-filter only for ``mgc``
2. Train a post-filter only for ``bap``
3. Merge the two post-filters into one post-filter

Pre-tuned config files are stored in ``recipes/_common/jp_dev_latest/conf/train_postfilter``.

Train post-filter for ``mgc``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train a post-filter for ``mgc``, you can run the following command:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 8 --stop-stage 8 \
        --acoustic-model acoustic_test \
        --postfilter-model postfilter_mgc_test \
        --postfilter-train mgc

Note that you must specify ``--postfilter-train mgc``. This tells the training script to only use the ``mgc`` feature stream. Other streams such as ``lf0`` and ``bap`` are ignored.

.. warning::

    Training a post-filter for ``mgc`` requires larger amount of GPU VRAM than the normal acoustic model training at the moment. Try using a smaller batch size.

Once the training is finished, you can find model checkpoints in the ``exp`` directory:

.. code-block::

    $ tree exp/yoko/postfilter_mgc_test

    exp/yoko/postfilter_mgc_test
    ├── best_loss.pth
    ├── best_loss_D.pth
    ├── config.yaml
    ├── epoch0002.pth
    ├── epoch0002_D.pth
    ├── latest.pth
    ├── latest_D.pth
    └── model.yaml

Some notes:

- ``*_D.pth`` is the model checkpoint for the discriminator. D stands for discriminators.
- ``model.yaml`` includes configs for both generator and discrimiantor.

Train post-filter for ``bap``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 8 --stop-stage 8 \
        --acoustic-model acoustic_test \
        --postfilter-model postfilter_bap \
        --postfilter-train bap

Note that you must specify ``--postfilter-train bap``. This tells the training script to only use the ``bap`` feature stream.


Merge the two post-filters
^^^^^^^^^^^^^^^^^^^^^^^^^^

This step is not included in the recipe. So you need to manually run the following command to merge the two post-filters:


.. code-block::

    python ../../../utils/merge_postfilters.py exp/yoko/postfilter_mgc_test/latest.pth \
        exp/yoko/postfilter_bap_test/latest.pth \
        exp/yoko/postfilter_merged

Then, you can see the merged post-filter in the ``exp/yoko/postfilter_merged`` directory.

.. code-block::

    $ tree exp/yoko/postfilter_merged/

    exp/yoko/postfilter_merged/
    ├── latest.pth
    └── model.yaml

Packing models with post-filter
--------------------------------

As the same as in :doc:`recipes`, you can pack the models into a single directory by running stage 99. Please make sure to specify the merged post-filter like:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 99 --stop-stage 99 \
        --timelag-model timelag_test \
        --duration-model duration_test \
        --acoustic-model acoustic_test \
        --postfilter-model postfilter_merged

The above command should make a packed model directory with your trained post-filter.

How to use the packed model with the trained post-filter?
----------------------------------------------------------

Please specify ``post_filter_type="nnsvs"`` with the :doc:`modules/svs` module. An example:

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

    wav, sr = engine.svs(labels, post_filter_type="nnsvs")


Tips for training post-filters
------------------------------

If you look into the post-filter configs, you will find many parameters. Here are the tips if you want to turn by yourself:

Train configs
^^^^^^^^^^^^^

- ``fm_weight``: The weight of the feature matching loss. By increasing the weight, you may get more stable results with a possible loss of naturalness. By setting ``fm_weight`` to zero, training will get unstable.
- ``adv_weight``: The weight of the adversarial loss. By increasing the weight, you may get better naturalness.
- ``mse_weight``: The weight of the MSE loss. If you set non-zero value, you will get smoother output features.

Model configs
^^^^^^^^^^^^^^
- ``smoothed_width``: The width of the smoothing window. If you set non-zero value, you will get smoother outputs. This is useful to reduce audible artifacts. Only used for inference.

Details of post-filter implementation
-------------------------------------

You don't need to understand the details if you just want to try, but please look into :cite:t:`Kaneko2017Interspeech`, :cite:t:`kaneko2017generative` if you are interested.
