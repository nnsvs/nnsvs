import pysinsy
from nnmnkwii.io import hts

# TODO: consider replacing pysinsy to pure python implementation

_global_sinsy = None


def _lazy_init(dic_dir=None):
    if dic_dir is None:
        dic_dir = pysinsy.get_default_dic_dir()
    global _global_sinsy
    if _global_sinsy is None:
        _global_sinsy = pysinsy.sinsy.Sinsy()
        assert _global_sinsy.setLanguages("j", dic_dir)


def xml2lab(xml):
    """Convert musicxml to HTS full context labels

    Args:
        xml (str): Path to musicxml file.

    Returns:
        HTS full context labels
    """
    _lazy_init()
    _global_sinsy.loadScoreFromMusicXML(xml)
    label = _global_sinsy.createLabelData(False, 1, 1)
    label = hts.load(lines=label.getData())
    _global_sinsy.clearScore()
    return label
