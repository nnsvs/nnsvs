
Defining your custom model
==========================

*Your PyTorch models can be used with NNSVS.*

NNSVS allows you to define your custom model easily. If you want your custom model to be used with NNSVS, you can implement your own by inheriting the :class:`nnsvs.base.BaseModel` class.

Write your PyTorch model
------------------------

.. note::

    If you are not familiar with PyTorch, please check the `PyTorch's documentation <https://pytorch.org/>`_ first.


A simplest example is shown below.

.. code-block:: python

    from nnsvs.base import BaseModel
    from torch import nn

    class MyModel(BaseModel):
        """My awesome neural network

        Args:
            in_dim (int): input dimension
            hidden_dim (int): hidden dimension
            out_dim (int): output dimension
        """

        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )

        def forward(self, x, lengths=None, y=None):
            """Forward pass

            Args:
                x (torch.Tensor): input tensor
                lengths (torch.Tensor): input sequence lengths
                y (torch.Tensor): target tensor (optional)

            Returns:
                torch.Tensor: output tensor
            """
            return self.model(x)

The above is a toy example defining a model with simple two-layer feed-forward neural networks with ReLU activation function. ``lengths`` and ``y`` are optional arguments.
The model name, number of arguments, and model architecture are totally customizable.

If you follow the :class:`nnsvs.base.BaseModel` interface, your model can be used as time-lag/duration/acoustic models.


Specify your model in model configs
-----------------------------------

Once you implement your model, you can use your model by changing your model configs like:


.. code-block:: yaml

    netG:
    _target_: ${path.to.your.model}
    # the followings are arguments passed to your model's __init__ method
    in_dim: 331
    hidden_dim: 32
    out_dim: 1


Note that your model must be in the ``PYTHONPATH``. If you edit ``nnsvs/model.py`` directly, you can specify your new model as:

.. code-block:: yaml

    netG:
    _target_: nnsvs.model.MyModel
    # the followings are arguments passed to your model's __init__ method
    in_dim: 331
    hidden_dim: 32
    out_dim: 1

If you add a new file at ``nnsvs/test.py`` for example, you can refer your model by:

.. code-block:: yaml

    netG:
    _target_: nnsvs.test.MyModel
    # the followings are arguments passed to your model's __init__ method
    in_dim: 331
    hidden_dim: 32
    out_dim: 1

That's it.

Available model types
---------------------

You may want to know what models are implemented and what are missing? Please check the following docs for the available models:

- Generic models: :doc:`modules/model`
- Acoustic models: :doc:`modules/acoustic_models`
- Post-filteres: :doc:`modules/postfilters`

If you find you model works well, please feel to free to make pull requests to the NNSVS repository.
