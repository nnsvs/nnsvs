import numpy as np
import pysinsy
import pytest
from nnmnkwii.io import hts
from nnsvs.pretrained import retrieve_pretrained_model
from nnsvs.svs import SPSVS
from nnsvs.util import example_xml_file


@pytest.mark.parametrize("segmented_synthesis", [False, True])
@pytest.mark.parametrize("post_filter_type", ["merlin", "gv", "nnsvs"])
@pytest.mark.parametrize("vocoder_type", ["world"])
def test_svs(segmented_synthesis, post_filter_type, vocoder_type):
    model_dir = retrieve_pretrained_model("r9y9/yoko_latest")
    engine = SPSVS(model_dir)

    contexts = pysinsy.extract_fullcontext(example_xml_file(key="get_over"))
    labels = hts.HTSLabelFile.create_from_contexts(contexts)

    wav, sr = engine.svs(
        labels,
        segmented_synthesis=segmented_synthesis,
        post_filter_type=post_filter_type,
        vocoder_type=vocoder_type,
    )
    assert sr == 48000
    assert np.isfinite(wav).all()
