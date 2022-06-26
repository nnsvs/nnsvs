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

https://github.com/pfnet/pysen is used for formatting and linting. Please run the following commands when you make a PR.

Formatting
^^^^^^^^^^^

.. code::

   pysen run format

Linting
^^^^^^^

.. code::

   pysen run lint

Tests
-----

To prevent unintentional bugs, it is better to write tests as much as possible. If you propose a new function, please consdier to write tests.
You can run the tests by the following command:

.. code::

    pytest -v -s

Please make sure tests are all passing before making a PR.

Building docs locally
---------------------

Run the following command at the top of nnsvs directory:

.. code::

    sphinx-autobuild docs docs/_build/html/
