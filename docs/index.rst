.. nnsvs documentation master file, created by
   sphinx-quickstart on Fri Apr 24 00:51:13 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NNSVS
=====

Neural network based singing voice synthesis library

Features
--------

- **Open-source**: NNSVS is fully open-source. You can create your own voicebanks with your dataset.
- **Multiple languages**: NNSVS has been used for creating singing voice synthesis (SVS) systems for multiple languages by VocalSynth comminities (8+ as far as I know).
- **Research friendly**: NNSVS comes with reproducible Kaldi/ESPnet-style recipes. You can use NNSVS to create baseline systems for your research.

Note that NNSVS was originally designed for research purposes. Please check out more user-friendly `ENUNU <https://github.com/oatsu-gh/ENUNU>`_ for creative purposes.
You can find a practical guide for NNSVS/ENUNU at https://nnsvs.carrd.co/.

Audio samples
-------------

Samples by r9y9: https://soundcloud.com/r9y9/sets/dnn-based-singing-voice

Selected videos
---------------

Demo by https://github.com/DYVAUX

..  youtube:: 0sSd31TUVCU

You can find more from the NNSVS/ENUNU community: `YouTube <https://www.youtube.com/results?search_query=nnsvs+enunu>`_, `NicoNico <https://www.nicovideo.jp/search/nnsvs?ref=nicotop_search>`_


.. toctree::
   :maxdepth: 1
   :caption: Demos

   notebooks/Demos
   notebooks/NNSVS_vs_Sinsy
   demo_server

.. toctree::
   :maxdepth: 1
   :caption: Notes

   installation
   tutorial
   devdocs
   recipes

.. toctree::
   :maxdepth: 1
   :caption: Package reference

   pretrained
   svs
   dsp
   gen
   mdn
   pitch
   multistream
   postfilters
   discriminators

.. toctree::
   :maxdepth: 1
   :caption: Resources

   papers
   related


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
