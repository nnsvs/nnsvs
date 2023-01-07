import numpy as np
from nnsvs.io.hts import full_to_mono
from nnsvs.neutrino import NEUTRINO
from nnsvs.pretrained import retrieve_pretrained_model
from nnsvs.util import example_xml_file


def test_neutrino():
    model_dir = retrieve_pretrained_model("r9y9/yoko_latest")
    engine = NEUTRINO(model_dir)

    musicxml = example_xml_file(key="get_over")
    full_labels, mono_labels = engine.musicxml2label(musicxml)

    assert len(full_labels) == len(mono_labels)

    num_phrases = engine.get_num_phrases(mono_labels)
    assert num_phrases >= 1

    timing_labels = full_to_mono(engine.predict_timing(full_labels))
    f0, mgc, bap = engine.predict_acoustic(full_labels, timing_labels)
    assert len(f0) == len(mgc) == len(bap)

    wav = engine.predict_waveform(f0, mgc, bap, vocoder_type="world")
    assert np.isfinite(wav).all()


def test_neutrino_phrase():
    model_dir = retrieve_pretrained_model("r9y9/yoko_latest")
    engine = NEUTRINO(model_dir)

    musicxml = example_xml_file(key="get_over")
    full_labels, mono_labels = engine.musicxml2label(musicxml)
    num_phrases = engine.get_num_phrases(mono_labels)
    assert num_phrases >= 1

    timing_labels = full_to_mono(engine.predict_timing(full_labels))

    phraselist = engine.get_phraselist(full_labels, timing_labels)
    print(phraselist)

    f0, mgc, bap = engine.predict_acoustic(full_labels, timing_labels, phrase_num=1)
    assert len(f0) == len(mgc) == len(bap)

    wav = engine.predict_waveform(f0, mgc, bap, vocoder_type="world")
    assert np.isfinite(wav).all()
