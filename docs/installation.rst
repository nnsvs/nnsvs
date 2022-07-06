Installation guide
===================

Supported platform
------------------

- Linux
- Mac OS X
- Windows

NNSVS is tested on all these platforms by `GitHub actions <https://github.com/nnsvs/nnsvs/actions>`_.
It is strongly recommended to use Linux for development purposes.


C/C++ compiler
---------------

You must need to install C/C++ compiler in advance. You can use `GCC <https://gcc.gnu.org/>`_, `Clang <https://clang.llvm.org/>`_, `Visual Studio <https://visualstudio.microsoft.com/>`_, or `MinGW <https://mingw.org/>`_.

For Linux/Mac OS X users, it is likely that you already have C/C++ compiler installed. For Windows users, you'd need to install Visual Studio with C++ compiler support.

GPU/CUDA
--------

GPU/CUDA are strongly recommended to get the best performance of neural networks. If you don't have GPUs, it is possible to run NNSVS on `Google Colab <https://colab.research.google.com/>`_.

If you have recent NVIDIA GPUs, you can accelerate training by using `mixed precision training <https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html>`_.
Please check the GPU driver and CUDA installation in advance.

Single GPU is good enough for training typical neural networks for singing voice synthesis, but you might need more if you want to try fancy/big models.

Python
-------

Python 3.7 or later.
Because NNSVS is written by `PyTorch <https://pytorch.org/>`_, it is recommended to check the Pytorch installation before testing NNSVS.

Installation commands
---------------------

Once the above setup is done, you can install NNSVS as follows.

For development
^^^^^^^^^^^^^^^

.. code::

   git clone https://github.com/nnsvs/nnsvs.git && cd nnsvs
   pip install -e ".[dev,lint,test]"

Note: adding ``[dev,lint,test]`` to the end of the command above will install dev/test/lint requirements as well.

For inference only
^^^^^^^^^^^^^^^^^^

.. code::

   pip install nnsvs

If you don't need to train your models by yourself (I guess it's unlikely though), this should be enough.


Google Colab
^^^^^^^^^^^^

If you are on Google colab, you may want to copy the following command into a cell.

.. code-block::

   %%capture
   try:
      import nnsvs
   except ImportError:
      ! pip install git+https://github.com/nnsvs/nnsvs
