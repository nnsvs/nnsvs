nnsvs.model
============

.. automodule:: nnsvs.model

Generic models
--------------

Generic models that can be used for time-lag/duration/acoustic models.

.. autoclass:: FFN
    :members:

.. autoclass:: LSTMRNN
    :members:

.. autoclass:: Conv1dResnet
    :members:

.. autoclass:: MDN
    :members:

.. autoclass:: RMDN
    :members:

.. autoclass:: FFConvLSTM
    :members:

.. autoclass:: VariancePredictor
    :members:

Acoustic models
---------------

Models that can only be used for acoustic models.

.. autoclass:: ResF0Conv1dResnet
    :members:

.. autoclass:: ResSkipF0FFConvLSTM
    :members:

.. autoclass:: ResF0VariancePredictor
    :members:
