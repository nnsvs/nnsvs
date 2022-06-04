Update guide
==============

v0.0.2 to master
----------------

Please check the following changes and consider changing your config files accordingly.

config.yaml
^^^^^^^^^^^^

- New parameter: ``trajectory_smoothing`` specifies if we apply trajectory smoothing proposed in :cite:t:`takamichi2015naist`. Default is false. It is likely to have little effects unless if you use (very experimental) learned post-filters.
- New parameter: ``trajectory_smoothing_cutoff`` specifies the cuttoff frequency for the trajectory smoothing. Default is 50 Hz. `This slide <https://www.slideshare.net/ShinnosukeTakamichi/apsipa2017-trajectory-smoothing-for-vocoderfree-speech-synthesis>`_ is useful to know the effects of the cutoff frequency.
- Changed: ``sample_rate`` became mandatory parameter while it was optional.
- New parameter: ``*_sweeper_args`` and ``*_sweeper_n_trials`` specifies configurations for hyperparameter optimization. See :doc:`optuna` for details.

train.py: train config
^^^^^^^^^^^^^^^^^^^^^^

- New parameter: ``use_amp`` specifies if we use mixed precision training or not. Default is false. If you have GPUs/CUDA that supports mixed precision training, you can get performance gain by setting it to true.
- New parameter: ``max_train_steps`` specifies maximum number of training steps (not epoch). Default is -1, which means maximum number of epochs is used to check if training is finished.
- New parameter: ``feats_criterion`` specifiefs where we use MSE loss or L1 loss. You can use L1 loss if you want while it was hardcoded to use MSE loss.

train.py: data config
^^^^^^^^^^^^^^^^^^^^^^

- New parameter: ``max_time_frames`` specifies maximum number of time frames. You can set non-negative values to limit the maximum time frames for making a mini-batch. It would be useful to workaround GPU OOM issues.


train_resf0.py: train config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- New parameter: ``use_amp`` specifies if we use mixed precision training or not. Default is false. If you have GPUs/CUDA that supports mixed precision training, you can get performance gain by setting it to true.
- New parameter: ``max_train_steps`` specifies maximum number of training steps (not epoch). Default is -1, which means maximum number of epochs is used to check if training is finished.
- New parameter: ``feats_criterion`` specifiefs where we use MSE loss or L1 loss. You can use L1 loss if you want while it was hardcoded to use MSE loss.

train_resf0.py: data config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- New parameter: ``max_time_frames`` specifies maximum number of time frames. You can set non-negative values to limit the maximum time frames for making a mini-batch. It would be useful to workaround GPU OOM issues.


MDN configs
^^^^^^^^^^^^

- Deprecated: ``dropout`` for ``nnsvs.model.MDN`` is deprecated. Please consider removing the parameter as it has no effect.
- Deprecated: ``FeedForwardNet`` is renamed to ``FFN``.
- Deprecated: ``ResF0Conv1dResnetMDN`` is deprecated. You can use ``ResF0Conv1dMDN`` with ``use_mdn=True``.