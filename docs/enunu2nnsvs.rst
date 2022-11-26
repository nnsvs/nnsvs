How to convert ENUNU models to NNSVS' ones
==========================================

If you are comfortable with NNSVS's command line interface and want to use ENUNU models with NNSVS, you can use ``utils/enunu2nnsvs.py``.

An example of ENUNU model: https://drive.google.com/file/d/1ue3fjaRN8KnUJGL06z14qj7IzT6KLvPa/view?usp=sharing


Usage
-----

.. code-block:: bash

    python utils/enunu2nnsvs.py ENUNU_RITSU_Ver2_1202 output_directory


Using ENUNU model with NNSVS
----------------------------

Once you convert a ENUNU model to NNSVS's format, you can create an SVS engine as follows:

.. code-block:: python

    from nnsvs.svs import SPSVS
    engine = SPSVS("/path/to/output_directory")

Please check :doc:`notebooks/Demos` for more details.
