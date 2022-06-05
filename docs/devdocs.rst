Notes for developers
====================

This page summarizes docs for developers of NNSVS. If you want to contribute to NNSVS itself, please check the document below.

Installation
---------------

It is recommended to install full requirements with editiable mode  (``-e`` with pip) enabled:

.. code::

   pip install -e ".[dev,lint.test,docs]"


Repository structure
---------------------

- ``nnsvs``: The core Python library. Neural network implementations for SVS systems can be found here.
- ``recipes``: Recipes.  The recipes are written mostly in bash and YAML-style configs. Some recipes use small Python scripts.
- ``docs``: Documentation. It is written by `Sphinx <https://www.sphinx-doc.org/>`_.
- ``notebooks``: Jupyter notebooks. Notebooks are helpful for interactive debugging and development.
- ``utils``: Utility scripts

Python docstring style
----------------------

NNSVS follows the Google's style: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

Formatting and linting
----------------------

https://github.com/pfnet/pysen is used for formatting and linting.

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

.. code::

    pytest -v -s

Building docs locally
---------------------

Run the following command at the top of nnsvs directory:

.. code::

    sphinx-autobuild docs docs/_build/html/
