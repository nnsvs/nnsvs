Update guide
==============

This page summarizes some guides when you want to use the updated version of NNSVS.

v0.0.2 to master
----------------

.. note::

    The master branch is the development version of NNSVS. It is ready for developers to try out new features but use it on your own.

Please check the following changes and consider changing your config files accordingly.

Models
^^^^^^^

- New parameter: All models now accept new argument ``init_type`` that specifies the initialization method for model parameters. Setting ``init_type`` to ``kaiming_normal`` or ``xavier_normal`` may improve convergence a bit for deep networks. Defaults to ``none``. The implementation was taken by `junyanz/pytorch-CycleGAN-and-pix2pix <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>`_.
- Deprecated: ``FeedForwardNet`` is renamed to ``FFN``.
- Deprecated: ``ResF0Conv1dResnetMDN`` is deprecated. You can use ``ResF0Conv1dMDN`` with ``use_mdn=True``.

config.yaml
^^^^^^^^^^^^

- New parameter: ``trajectory_smoothing`` specifies if we apply trajectory smoothing proposed in :cite:t:`takamichi2015naist`. Default is false. It is likely to have little effects unless if you use (very experimental) learned post-filters.
- New parameter: ``trajectory_smoothing_cutoff`` specifies the cuttoff frequency for the trajectory smoothing. Default is 50 Hz. `This slide <https://www.slideshare.net/ShinnosukeTakamichi/apsipa2017-trajectory-smoothing-for-vocoderfree-speech-synthesis>`_ is useful to know the effects of the cutoff frequency.
- Changed: ``sample_rate`` became mandatory parameter while it was optional.
- New parameter: ``*_sweeper_args`` and ``*_sweeper_n_trials`` specifies configurations for hyperparameter optimization. See :doc:`optuna` for details.

Run.sh
^^^^^^^

- Consider adding model packing stage 99 by following ``kiritan_singing`` recipes.
- Consider adding post-filter related steps on your own by following ``kiritan_singing`` recipes.

train.py: train config
^^^^^^^^^^^^^^^^^^^^^^

- New parameter: ``use_amp`` specifies if we use mixed precision training or not. Default is false. If you have GPUs/CUDA that supports mixed precision training, you can get performance gain by setting it to true.
- New parameter: ``max_train_steps`` specifies maximum number of training steps (not epoch). Default is -1, which means maximum number of epochs is used to check if training is finished.
- New parameter: ``feats_criterion`` specifies where we use MSE loss or L1 loss. You can use L1 loss if you want while it was hardcoded to use MSE loss.

train.py: data config
^^^^^^^^^^^^^^^^^^^^^^

- New parameter: ``max_time_frames`` specifies maximum number of time frames. You can set non-negative values to limit the maximum time frames for making a mini-batch. It would be useful to workaround GPU OOM issues.


train_resf0.py: train config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- New parameter: ``use_amp`` specifies if we use mixed precision training or not. Default is false. If you have GPUs/CUDA that supports mixed precision training, you can get performance gain by setting it to true.
- New parameter: ``max_train_steps`` specifies maximum number of training steps (not epoch). Default is -1, which means maximum number of epochs is used to check if training is finished.
- New parameter: ``feats_criterion`` specifies where we use MSE loss or L1 loss. You can use L1 loss if you want while it was hardcoded to use MSE loss.

train_resf0.py: data config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- New parameter: ``max_time_frames`` specifies maximum number of time frames. You can set non-negative values to limit the maximum time frames for making a mini-batch. It would be useful to workaround GPU OOM issues.
