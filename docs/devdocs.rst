Development guide
=================

This page summarizes docs for developers of NNSVS. If you want to contribute to NNSVS itself, please check the document below.

Installation
------------

For development purposes, it is recommended to install full requirements with editiable mode  (``-e`` with pip) enabled:

.. code::

   pip install -e ".[dev,lint,test,docs]"

This allows your local changes available to your python environment without manually re-installing NNSVS.


Repository structure
---------------------

Here's the list of important components of the NNSVS repository:

- ``nnsvs``: The core Python library. Neural network implementations for SVS systems can be found here.
- ``recipes``: Recipes.  The recipes are written mostly in bash and YAML-style configs. Some recipes use small Python scripts.
- ``docs``: Documentation. It is written by `Sphinx <https://www.sphinx-doc.org/>`_.
- ``notebooks``: Jupyter notebooks. Notebooks are helpful for interactive debugging and development.
- ``utils``: Utility scripts that are used by the recipes.
- ``tests``: Tests

Python docstring style
----------------------

NNSVS follows the `Google's style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.
If you write a docstrings for your new functinoality, please follow the same style.

Formatting and linting
----------------------

https://github.com/astral-sh/ruff is used for formatting and linting. Please run the following commands when you make a PR.

Formatting
^^^^^^^^^^^

.. code::

   ruff format .

Linting
^^^^^^^

.. code::

   ruff check .

Tests
-----

To prevent unintentional bugs, it is better to write tests as much as possible. If you propose a new function, please consdier to write tests.
You can run the tests by the following command:

.. code::

    pytest -v -s

Please make sure tests are all passing before making a PR.

``torch.load`` and ``weights_only``
------------------------------------

Since PyTorch 2.6, ``torch.load`` defaults to ``weights_only=True``, which restricts
unpickling to a small allow-list of "safe" types. NNSVS checkpoints used to fail this
check because optimizers were instantiated with ``**optim_config.optimizer.params``,
an OmegaConf ``DictConfig``. Unpacking it with ``**`` doesn't convert nested containers
to plain Python types, so list-valued hyperparameters (e.g. Adam's
``betas: [0.9, 0.999]``) were passed through as ``omegaconf.listconfig.ListConfig``
objects, which then got pickled into ``optimizer_state`` -- and ``ListConfig`` is not on
the ``weights_only=True`` allow-list. (``add_safe_globals`` cannot work around this:
PyTorch's unpickler separately hard-codes that ``SETITEM`` may only target
``dict``/``OrderedDict``/``Counter``, and ``ListConfig`` uses a ``defaultdict``
internally.)

This has been fixed at the source: ``nnsvs/train_util.py`` now converts optimizer/
scheduler params to plain Python types via ``OmegaConf.to_container(resolve=True)``
before instantiation, so checkpoints saved by current code never contain a
``ListConfig``. All first-party ``torch.load`` call sites use ``weights_only=True``.
**Checkpoints saved before this fix still need migrating**, since they may carry a raw
``ListConfig`` in ``optimizer_state``.

Migrating old checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    # single checkpoint file
    python utils/migrate_checkpoint_weights_only.py exp/foo/latest.pth exp/foo/latest_migrated.pth

    # a whole pretrained_expdir (only latest.pth / latest_D.pth are touched,
    # since that's all the resume path reads)
    python utils/migrate_expdir_weights_only.py /path/to/old_pretrained_expdir /path/to/new_pretrained_expdir

Both convert embedded ``ListConfig``/``DictConfig`` objects to plain types and verify
the result loads under ``weights_only=True``. They are one-off, manually-run tools, not
wired into any recipe stage. Only run them against checkpoints whose provenance you
trust -- they need ``weights_only=False`` to read the old file in the first place.

Building docs locally
---------------------

Run the following command at the top of nnsvs directory:

.. code::

    sphinx-autobuild docs docs/_build/html/
