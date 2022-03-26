.. nnsvs documentation master file, created by
   sphinx-quickstart on Fri Apr 24 00:51:13 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NNSVS
=====

Neural network based singing voice synthesis library

Audio samples
-------------

- Samples by r9y9: https://soundcloud.com/r9y9/sets/dnn-based-singing-voice
- Samples by NNSVS/ENUNU community: https://www.youtube.com/results?search_query=nnsvs+enunu

Installation
-------------


For development
^^^^^^^^^^^^^^^^

.. code::

   git clone https://github.com/r9y9/nnsvs.git && cd nnsvs
   pip install -e ".[lint.test]"

Note: adding ``[lint,test]`` to the end of the command above will install test/lint requirements as well.

For inference only
^^^^^^^^^^^^^^^^^^

.. code::

   pip install nnsvs

If you don't need to train your models by yourself (I guess it's unlikely though), this should be enough.

.. toctree::
   :maxdepth: 1
   :caption: Demos

   notebooks/Demos.ipynb
   demo_server

.. toctree::
   :maxdepth: 1
   :caption: Notes

   devdocs
   recipes

.. toctree::
   :maxdepth: 1
   :caption: Package reference

   svs
   pretrained
   mdn
   pitch

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
