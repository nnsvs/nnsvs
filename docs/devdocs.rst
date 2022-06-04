Notes for developers
====================

This is a note for developers of NNSVS. If you want to contribute to NNSVS itself, please check the document below.

Installation
---------------

It is recommended to install full requirements with editiable mode  (``-e`` with pip) enabled:

.. code::

   pip install -e ".[dev,lint.test,docs]"


Repository structure
---------------------

- Core library: ``nnsvs``
- Recipes: ``recipes``
- Documentation: ``docs``
- Jupyter notebooks: ``notebooks``
- Utility: ``utils``

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