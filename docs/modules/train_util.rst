nnsvs.train_util
====================

Specific utilities for training

.. automodule:: nnsvs.train_util


Helper for training
--------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    setup
    save_checkpoint
    num_trainable_params

DataLoader
----------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    batch_by_size
    collate_fn_default
    collate_fn_random_segments
    get_data_loaders

Plotting
---------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    plot_spsvs_params

Misc
----

.. autosummary::
    :toctree: generated/
    :nosignatures:

    note_segments
    compute_pitch_regularization_weight
    compute_batch_pitch_regularization_weight
    compute_distortions
    ensure_divisible_by
