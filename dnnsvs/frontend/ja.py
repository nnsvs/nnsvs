# coding: utf-8

import os
from nnmnkwii.io import hts


try:
    import pysinsy
except ImportError as e:
    print("Pysinsy must be manually installed!")
    raise e

# TODO: consider replacing pysinsy to pure python implementation

_global_sinsy = None


def _lazy_init():
    global _global_sinsy
    if _global_sinsy is None:
        _global_sinsy = pysinsy.sinsy.Sinsy()
        # TODO: remove hardcode
        assert _global_sinsy.setLanguages("j", "/usr/local/lib/sinsy/dic")


def xml2lab(xml):
    """Convert musicxml to HTS full context labels

    Args:
        xml (str): Path to musicxml file.

    Returns:
        HTS full context labels
    """
    _lazy_init()
    global _global_sinsy

    _global_sinsy.loadScoreFromMusicXML(xml)
    label = _global_sinsy.createLabelData(False, 1, 1)
    label = hts.load(lines=label.getData())
    _global_sinsy.clearScore()
    return label