Change log
==========

v0.1.0 <2022-xx-xx>
-------------------

TBD

v0.0.3 <2022-xx-xx>
-------------------

TBD

v0.0.2 (2022-04-29)
-------------------

A version that should work with `ENUNU v0.4.0 <https://github.com/oatsu-gh/ENUNU/releases/tag/v0.4.0>`_

New features
~~~~~~~~~~~~~

- Improved timings with MDN duration models `#80`_
- Improved acoustic models with residual F0 prediction `#76`_

Bug fixes
~~~~~~~~~~~~~

- numpy.linalg.LinAlgError in MDN models `#94`_

v0.0.1 <2022-03-11>
-------------------

The first release

The initial version of nnsvs (with some experimental features like vibrato modeling and data augmentation). This version should be compatible with currently available tools around nnsvs (e.g., ENUNU). Hydra >=v1.0.0, <v1.2.0 is supported.
PyPi release is also available. So you can install the core library by pip install nnsvs.

.. _#76: https://github.com/r9y9/nnsvs/issues/76
.. _#80: https://github.com/r9y9/nnsvs/issues/80
.. _#94: https://github.com/r9y9/nnsvs/issues/94
