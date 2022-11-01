Change log
==========

v0.1.0 <2022-xx-xx>
-------------------

WILL BE MOVED TO GITHUB RELEASES.

New features
^^^^^^^^^^^^

- Dynamic batch size support (by ``batch_max_frames``).
- New acoustic models based on duration informed Tacotron.
- New F0 prediction models based on duration informed Tacotron.
- Multi-stream model implementations
- Support mel-spectrogram as acoustic features.
- Integrate uSFGAN vocoder

v0.0.3 <2022-10-15>
-------------------

Moved the repository to https://github.com/nnsvs organization.

New features
^^^^^^^^^^^^

- Mixed precision training `#106`_
- Recipe-level integration of hyperparameter optimization with Optuna `#43`_ :doc:`optuna`
- Added VariancePredictor (:cite:t:`ren2020fastspeech`).
- Spectrogram, aperiodicity, F0, and generated audio is now logged to tensorboard if ``train_resf0.py`` is used.
- Objective metrics (such as mel-cepstrum distortion and RMSE) are now logged to tensorboard. `#41`_
- Added MDNv2 (MDN + dropout) `#118`_
- Correct V/UV (``correct_vuv``) option is added to feature processing.
- Support training non-resf0 models with ``train_resf0.py`` `#125`_
- Global-variance (GV)-based post-filter

Bug fixes
^^^^^^^^^

- Add a heuristic trick to prevent non-negative durations at synthesis time
- Fix error when no dynamic features are used `#128`_
- Add a workaround for WORLD's segfaults issue when ``min_f0`` is too high.
- Fix bug of computing pitch regularization weights
- Fix continuous F0 for rest

Improvements
^^^^^^^^^^^^

- ``nnsvs.model.MDN`` now support dropout by the ``dropout`` argument. The ``dropout`` argument existed before but it was no-op for a long time.
- Number of training iterations can be now specified by either epochs or steps.
- A heuristic trick is added to prevent serious V/UV prediction errors . `#95`_ `#119`_
- Speech parameter trajectory smoothing (:cite:t:`takamichi2015naist`). Disabled by default.
- Added recipe tests on CI `#116`_
- Add option to allow filtering of long segments `#135`_
- Stream-wise flags to enable/disable dynamic features
- Pre-processing: Tweaked min_f0/max_f0 threshold
- Pre-processing: Add resampling if necessary
- Pre-processing: Allow users to specify expliciti F0 range
- Expose decay_size for pitch reguralization
- Support Codecov

Deprecations
^^^^^^^^^^^^

- ``dropout`` for ``nnsvs.model.MDN`` is deprecated. Please consider removing the parameter as it has no effect.
- ``dropout`` for ``nnsvs.model.Conv1dResnet`` is deprecated. Please consider removing the parameter as it has no effect.
- ``FeedForwardNet`` is renamed to ``FFN`` to be consistent with other names (such as MDN)
- ``ResF0Conv1dResnetMDN`` is deprecated. You can use ``ResF0Conv1dResnet`` with ``use_mdn=True``.
- ``Conv1dResnetMDN`` is deprecated. You can use ``Conv1dResnet`` with ``use_mdn=True``.

Breaking changes
^^^^^^^^^^^^^^^^

- Update d4c threshold to prevent serious voiced -> unvoiced errors from 0.85 to 0.15. If you prefer the old default, please set `d4c_threshold` to 0.85.
- Default values of functions in ``gen.py`` and ``svs.py`` are changed while refactoring. Please explicitly set the function arguments to avoid unexpected behavior.

Documentation
^^^^^^^^^^^^^

Added documentations as mush as possible.

Experimental features
^^^^^^^^^^^^^^^^^^^^^

Some features that are available but not yet tested or documented

- GAN-based post-filters (:cite:t:`Kaneko2017Interspeech`, :cite:t:`kaneko2017generative`) `#85`_ and GV post-filter (:cite:t:`silen2012ways`)
- CycleGAN-based post-filter
- Support for neural vocoders `#72`_
- Add ``ResF0NonAttentiveTacotron`` acoustic model. `#129`_ `#15`_
- WaveNet `#100`_
- GAN-based acoustic models `#85`_
- Make :doc:`modules/svs` to support trainable post-filters and neural vocoders.

v0.0.2 <2022-04-29>
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
.. _#116: https://github.com/r9y9/nnsvs/pull/116
.. _#118: https://github.com/r9y9/nnsvs/pull/118
.. _#119: https://github.com/r9y9/nnsvs/pull/119
.. _#125: https://github.com/r9y9/nnsvs/pull/125
.. _#128: https://github.com/r9y9/nnsvs/pull/128
.. _#129: https://github.com/r9y9/nnsvs/pull/129
.. _#135: https://github.com/r9y9/nnsvs/pull/135

