.. nnsvs documentation master file, created by
   sphinx-quickstart on Fri Apr 24 00:51:13 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NN-SVS
======

Neural network based singing voice synthesis library

Audio samples
-------------

- Samples by r9y9: https://soundcloud.com/r9y9/sets/dnn-based-singing-voice
- Samples by NNSVS/ENUNU community: https://www.youtube.com/results?search_query=nnsvs+enunu

Installation
-------------

.. code::

   pip install -e ".[lint.test]"

Note: adding ``[lint,test]`` to the end of the command above will installk test/lint requirements as well.
If you don't want the extra requirements to be installed, you can run:

.. code::

      pip install -e .


.. toctree::
   :maxdepth: 1
   :caption: Demos

   notebooks/Demos.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Notes

   devdocs
   recipes

.. toctree::
   :maxdepth: 1
   :caption: Resources

   related
   papers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
