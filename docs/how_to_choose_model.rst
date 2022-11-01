How to choose model
====================

There are number of models available in NNSVS. This page describes how to choose a model if you are unsure what to use.

Best models for Namine Ritsu's database are listed for your reference.


Time-lag model
---------------

Use :class:`nnsvs.model.MDNv2` or :class:`nnsvs.model.VariancePredictor` (with ``use_mdn=True``). The latter works better at least for the Namine Ritsu's database.

.. note::

    The best model for Namine Ritsu's database: :class:`nnsvs.model.VariancePredictor`

Duration model
---------------

Use :class:`nnsvs.model.MDNv2` or :class:`nnsvs.model.VariancePredictor` (with ``use_mdn=True``). The latter works better at least for the Namine Ritsu's database.

.. note::

    The best model for Namine Ritsu's database: :class:`nnsvs.model.VariancePredictor`

Acoustic model
--------------

- Use static features only. We found that dynamic features are less beneficial (at least for Namine Ritsu's database).
- Use :class:`nnsvs.model.Conv1dResnet` (``use_mdn=True``) if you like old-style MDN-based acoustic model that was used in `Namine Ritsu's V2 model <https://www.youtube.com/watch?v=pKeo9IE_L1I>`_.
- If you have a large amount of data, use multi-stream models (e.g., :class:`nnsvs.acoustic_models.NPSSMultistreamParametricModel`) where each feature stream is modeled by an autoregressive decoder except V/UV feature stream. Specifically, use autoregressive models for MGCs, log-F0, and BAP features. Please refer to the Namine Ritsu's recipes to find example model configurations. Note that autoregresive models tend to require a larger amount of training data.


Multi-stream models
~~~~~~~~~~~~~~~~~~~

- For F0 feature stream: use :class:`nnsvs.acoustic_models.BiLSTMNonAttentiveDecoder`. We found that autoregressive F0 models worked better than non-autoregressive alternatives. There are a number of model parameters, but please be aware that ``reduction_factor`` has great impact on the modeling capability. Smaller values allow the model to capture finer grained temporal information with training instability, wheareas larger values allow the model to capture coarser grained temporal information. We recommend to use ``reduction_factor=4`` for F0 feature stream. You may also try ``reduction_factor=2``.
- For MGC/BAP feature streams: Use non-MDN autoregressive models (e.g., :class:`nnsvs.acoustic_models.BiLSTMNonAttentiveDecoder`) over MDN-based autoregressive models (e.g., :class:`nnsvs.acoustic_models.BiLSTMMDNNonAttentiveDecoder`). As reported in Tacotorn 2 paper (:cite:t:`shen2018natural`), we empirically found that non-MDN version worked better than the MDN-version.

.. note::

    The best model for Namine Ritsu's database: :class:`nnsvs.acoustic_models.NPSSMultistreamParametricModel`

Vocoder
-------

.. note::

    The best model for Namine Ritsu's database: uSFGAN

- Use WORLD first. WORLD can achieve reasonably good-quality synthesis with pitch robustness. It also generalizes well on unseen speakers (singers) with no training.
- If you want to maximize the quality, use `uSFGAN <https://github.com/chomeyama/HN-UnifiedSourceFilterGAN>`_.
