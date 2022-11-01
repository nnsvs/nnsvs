Update guide
==============

This page summarizes guides when you use the updated version of NNSVS.

v0.0.3 to master
----------------

.. warning::

    The master branch is the development version of NNSVS. It is ready for developers to try out new features but use it on your own.

Breaking changes
~~~~~~~~~~~~~~~~

- train_resf0.py is renamed to train_acoustic.py. Please use train_acoustic.py for training acoustic models.

Hed
~~~

NNSVS now uses the rest note information in the preprocessing and synthesis stages.
In addition, NNSVS aasumes that the rest note (or equivalent phoneme) context is on the first feature in your hed file. For example, the first feature of a JP hed file should look like:

.. code-block::

    QS "C-Phone_Muon"     {*-sil+*,*-pau+*}

Please do make sure to have the rest note context on the top of the hed file.

Models
^^^^^^^

- We've found that multi-stream models generally worked better than single-stream models. Please consider using multi-stream models. See also :doc:`acoustic_models` and :doc:`how_to_choose_model`.

config.yaml
^^^^^^^^^^^^

TBD

Run.sh
^^^^^^^

TBD

train.py: train config
^^^^^^^^^^^^^^^^^^^^^^

TBD

train.py: data config
^^^^^^^^^^^^^^^^^^^^^^

- New parameter: ``batch_max_frames`` if specified, the batch size will be automatically adjusted to fit the specified number of frames. To allow efficient use of GPU memory, please do set this value.

train_acoustic.py: train config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TBD

train_resf0.py: data config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- New parameter: ``batch_max_frames`` if specified, the batch size will be automatically adjusted to fit the specified number of frames. To allow efficient use of GPU memory, please do set this value.


v0.0.2 to v0.0.3
----------------

Hed
~~~

If your hed file does not contain QS features that specify voiced/unvoiced phonemes, consider adding them to tell NNSVS to generate stable V/UV sounds. For example, add the followings for a JP hed:

.. code-block::

    QS "C-VUV_Voiced" {*-a+*,*-i+*,*-u+*,*-e+*,*-o+*,*-v+*,*-b+*,*-by+*,*-m+*,*-my+*,*-w+*,*-z+*,*-j+*,*-d+*,*-dy+*,*-n+*,*-ny+*,*-N+*,*-r+*,*-ry+*,*-g+*,*-gy+*,*-y+*}
    QS "C-VUV_Unvoiced"  {*-A+*,*-I+*,*-U+*,*-E+*,*-O+*,*-f+*,*-p+*,*-py+*,*-s+*,*-sh+*,*-ts+*,*-ch+*,*-t+*,*-ty+*,*-k+*,*-ky+*,*-h+*,*-hy+*}

Then NNSVS will check the flags to correct V/UV at synthesis time by default.

If your hed file contains the following QS features,

.. code-block::

    QS "C-Phone_Yuuseion" {*-a+*,*-i+*,*-u+*,*-e+*,*-o+*,*-v+*,*-b+*,*-by+*,*-m+*,*-my+*,*-w+*,*-z+*,*-j+*,*-d+*,*-dy+*,*-n+*,*-ny+*,*-N+*,*-r+*,*-ry+*,*-g+*,*-gy+*,*-y+*}
    QS "C-Phone_Museion"  {*-A+*,*-I+*,*-U+*,*-E+*,*-O+*,*-f+*,*-p+*,*-py+*,*-s+*,*-sh+*,*-ts+*,*-ch+*,*-t+*,*-ty+*,*-k+*,*-ky+*,*-h+*,*-hy+*}

please rename them to ``C-VUV_Voiced`` and ``C-VUV_Unvoiced``.

Models
^^^^^^^

- Use ``MDNv2`` (MDN + dropout) instead of ``MDN``.
- New parameter: All models now accept new argument ``init_type`` that specifies the initialization method for model parameters. Setting ``init_type`` to ``kaiming_normal`` or ``xavier_normal`` may improve convergence a bit for deep networks. Defaults to ``none``. The implementation was taken by `junyanz/pytorch-CycleGAN-and-pix2pix <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>`_.
- Deprecated: ``FeedForwardNet`` is renamed to ``FFN``.
- Deprecated: ``ResF0Conv1dResnetMDN`` is deprecated. You can use ``ResF0Conv1dMDN`` with ``use_mdn=True``.

config.yaml
^^^^^^^^^^^^

- New parameter: ``trajectory_smoothing`` specifies if we apply trajectory smoothing proposed in :cite:t:`takamichi2015naist`. Default is false. It is likely to have little effects unless if you use (very experimental) learned post-filters.
- New parameter: ``trajectory_smoothing_cutoff`` specifies the cuttoff frequency for the trajectory smoothing. Default is 50 Hz. `This slide <https://www.slideshare.net/ShinnosukeTakamichi/apsipa2017-trajectory-smoothing-for-vocoderfree-speech-synthesis>`_ is useful to know the effects of the cutoff frequency.
- Changed: ``sample_rate`` became mandatory parameter while it was optional.
- New parameter: ``*_sweeper_args`` and ``*_sweeper_n_trials`` specifies configurations for hyperparameter optimization. See :doc:`optuna` for details.

Consider adding the following to enable vocoder training:

.. code-block::

    # NOTE: conf/parallel_wavegan/${vocoder_model}.yaml must exist.
    vocoder_model:
    # Pretrained checkpoint path for the vocoder model
    # NOTE: if you want to try fine-tuning, please specify the path here
    pretrained_vocoder_checkpoint:
    # absolute/relative path to the checkpoint
    # NOTE: the checkpoint is used for synthesis and packing
    # This doesn't have any effect on training
    vocoder_eval_checkpoint:

Run.sh
^^^^^^^

- Consider adding model packing stage 99. See :doc:`recipes` for details.
- Consider adding post-filter related steps. See :doc:`train_postfilters` for details.
- Consider adding vocoder related steps. See :doc:`train_vocoders` for details.

train.py: train config
^^^^^^^^^^^^^^^^^^^^^^

- New parameter: ``use_amp`` specifies if we use mixed precision training or not. Default is false. If you have GPUs/CUDA that supports mixed precision training, you can get performance gain by setting it to true.
- New parameter: ``max_train_steps`` specifies maximum number of training steps (not epoch). Default is -1, which means maximum number of epochs is used to check if training is finished.
- New parameter: ``feats_criterion`` specifies where we use MSE loss or L1 loss. You can use L1 loss if you want while it was hardcoded to use MSE loss.

train.py: data config
^^^^^^^^^^^^^^^^^^^^^^

- New parameter: ``max_time_frames`` specifies maximum number of time frames. You can set non-negative values to limit the maximum time frames for making a mini-batch. It would be useful to workaround GPU OOM issues.
- New parameter: ``filter_long_segments`` specifies if long segments are filtered or not. Consider to set it True when you have GPU OOM issues. Default is False.
- New parameter: ``filter_num_frames`` specifies the threshold for filtering long segments. Default is 6000, which means segments longer than 30 sec will not be used for training.


train_resf0.py: train config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- New parameter: ``use_amp`` specifies if we use mixed precision training or not. Default is false. If you have GPUs/CUDA that supports mixed precision training, you can get performance gain by setting it to true.
- New parameter: ``max_train_steps`` specifies maximum number of training steps (not epoch). Default is -1, which means maximum number of epochs is used to check if training is finished.
- New parameter: ``feats_criterion`` specifies where we use MSE loss or L1 loss. You can use L1 loss if you want while it was hardcoded to use MSE loss.
- New parameter: ``pitch_reg_decay_size`` specifies the decay size for the pitch regularization. The larger the decay size, the smoother pitch transitions between notes are allowed during training. See :cite:t:`hono2021sinsy` for details.

train_resf0.py: data config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- New parameter: ``max_time_frames`` specifies maximum number of time frames. You can set non-negative values to limit the maximum time frames for making a mini-batch. It would be useful to workaround GPU OOM issues.
