Change log
==========

v0.0.3 <2022-xx-xx>
-------------------

New features
^^^^^^^^^^^^^

- Recipe-level integration of hyperparameter optimization with Optuna `#43`_ :doc:`optuna`
- Speech parameter trajectory smoothing (:cite:t:`takamichi2015naist`). Disabled by default.
- Objective metrics (such as mel-cepstrum distortion and RMSE) are now logged to tensorboard. `#41`_
- Spectrogram, aperiodicity, F0, and generated audio is now logged to tensorboard if ``train_resf0.py`` is used.
- A heuristic trick is added to prevent serious V/UV prediction errors (hardcoded for Japanese for now). `#95`_
- GAN-based post-filters (:cite:t:`Kaneko2017Interspeech`, :cite:t:`kaneko2017generative`) `#85`_
- GV post-filter (:cite:t:`silen2012ways`)
- Number of training iterations can be now specified by either epochs or steps.
- Mixed precision training `#106`_
- Added VariancePredictor (:cite:t:`ren2020fastspeech`).

Bug fixes
~~~~~~~~~

- Add a heuristic trick to prevent non-negative durations at synthesis time

Changes
~~~~~~~

- ```nnsvs.model.MDN`` now support dropout by the ``dropout`` argument. The ``dropout`` argument existed before but it was no-op for a long time.

Deprecations
^^^^^^^^^^^^^

- ``dropout`` for ``nnsvs.model.Conv1dResnet`` is deprecated. Please consider removing the parameter as it has no effect.
- ``FeedForwardNet`` is renamed to ``FFN`` to be consistent with other names (such as MDN)
- ``ResF0Conv1dResnetMDN`` is deprecated. You can use ``ResF0Conv1dResnet`` with ``use_mdn=True``.
- ``Conv1dResnetMDN`` is deprecated. You can use ``Conv1dResnet`` with ``use_mdn=True``.

Documentation
^^^^^^^^^^^^^

Added documentations as mush as possible.

Experimental features
^^^^^^^^^^^^^^^^^^^^^

Some features that are available but not yet tested or documented

- CycleGAN-based post-filter
- Support for neural vocoders `#72`_
- Tacotron-like autoregressive models `#15`_
- WaveNet `#100`_
- GAN-based acoustic models `#85`_

v0.0.2 (2022-04-29)
-------------------

A version that should work with `ENUNU v0.4.0 <https://github.com/oatsu-gh/ENUNU/releases/tag/v0.4.0>`_

New features
^^^^^^^^^^^^

- Improved timings with MDN duration models `#80`_
- Improved acoustic models with residual F0 prediction `#76`_

Bug fixes
^^^^^^^^^

- numpy.linalg.LinAlgError in MDN models `#94`_

v0.0.1 <2022-03-11>
-------------------

The first release

The initial version of nnsvs (with some experimental features like vibrato modeling and data augmentation). This version should be compatible with currently available tools around nnsvs (e.g., ENUNU). Hydra >=v1.0.0, <v1.2.0 is supported.
PyPi release is also available. So you can install the core library by pip install nnsvs.

.. _#15: https://github.com/r9y9/nnsvs/issues/15
.. _#41: https://github.com/r9y9/nnsvs/issues/41
.. _#43: https://github.com/r9y9/nnsvs/issues/43
.. _#72: https://github.com/r9y9/nnsvs/issues/72
.. _#76: https://github.com/r9y9/nnsvs/issues/76
.. _#80: https://github.com/r9y9/nnsvs/issues/80
.. _#85: https://github.com/r9y9/nnsvs/issues/85
.. _#94: https://github.com/r9y9/nnsvs/issues/94
.. _#95: https://github.com/r9y9/nnsvs/issues/95
.. _#100: https://github.com/r9y9/nnsvs/issues/100
.. _#106: https://github.com/r9y9/nnsvs/issues/106
