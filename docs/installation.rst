Installation
=============

For development
----------------

.. code::

   git clone https://github.com/r9y9/nnsvs.git && cd nnsvs
   pip install -e ".[lint.test]"

Note: adding ``[lint,test]`` to the end of the command above will install test/lint requirements as well.

For inference only
--------------------

.. code::

   pip install nnsvs

If you don't need to train your models by yourself (I guess it's unlikely though), this should be enough.