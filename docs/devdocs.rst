Development notes
=================

Installation
---------------

It is recommended to install full requirements with editiable mode  (``-e`` with pip) enabled:

.. code::

   pip install -e ".[lint.test,docs]"


Repository structure
---------------------

- Core library: ``nnsvs``
- Command line programs: ``nnsvs/bin`` (configs: ``nnsvs/bin/conf``)
- Recipes: ``recipes`` (previously ``egs``)


Python docstring style
----------------------

https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

Formatting and linting
----------------------

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