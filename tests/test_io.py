from os.path import dirname, join

import numpy as np
import pytest
from nnmnkwii.io import hts
from nnsvs.io.hts import (
    full_to_mono,
    get_note_indices,
    label2phrases,
    label2phrases_str,
    merge_sil,
    overwrite_phoneme_flags_,
    segment_labels,
)


def test_full_to_mono():
    full_labels = hts.load(
        join(dirname(__file__), "data", "neutrino", "sample1_full.lab")
    )
    mono_labels = hts.load(
        join(dirname(__file__), "data", "neutrino", "sample1_mono.lab")
    )

    assert str(mono_labels) == str(full_to_mono(full_labels))


@pytest.mark.parametrize("filename", ["sample1", "sample2", "sample3", "meltdown_main"])
def test_get_note_indices(filename):
    mono_labels = hts.load(
        join(dirname(__file__), "data", "neutrino", f"{filename}_mono.lab")
    )
    note_indices1 = get_note_indices(mono_labels)

    full_labels = hts.load(
        join(dirname(__file__), "data", "neutrino", f"{filename}_full.lab")
    )
    note_indices2 = get_note_indices(full_labels)

    assert (np.array(note_indices1) == np.array(note_indices2)).all()


@pytest.mark.parametrize("filename", ["sample1", "sample2", "sample3", "meltdown_main"])
def test_label2phrases_str(filename):
    mono_labels = hts.load(
        join(dirname(__file__), "data", "neutrino", f"{filename}_mono.lab")
    )
    note_indices = get_note_indices(mono_labels)

    timing_labels = hts.load(
        join(dirname(__file__), "data", "neutrino", f"{filename}_timing.lab")
    )
    nnsvs_str = label2phrases_str(timing_labels, note_indices)

    with open(
        join(dirname(__file__), "data", "neutrino", f"{filename}-phraselist.txt"), "r"
    ) as f:
        neutrino_str = f.read()

    assert nnsvs_str == neutrino_str


@pytest.mark.parametrize("filename", ["sample1", "sample2", "sample3", "meltdown_main"])
@pytest.mark.parametrize("fix_offset", [True, False])
def test_label2phrases(filename, fix_offset):
    timing_labels = hts.load(
        join(dirname(__file__), "data", "neutrino", f"{filename}_timing.lab")
    )
    phrases = label2phrases(timing_labels, fix_offset=fix_offset)
    assert len(phrases) > 0
    assert isinstance(phrases, list)
    for idx, phrase in enumerate(phrases):
        print(f"Phrase {idx}:", " ".join(phrase.contexts))
        assert isinstance(phrase, hts.HTSLabelFile)

    # Check if the number of labels are the same
    assert len(timing_labels) == sum(len(phrase) for phrase in phrases)


@pytest.mark.parametrize(
    "filename", ["sample1", "sample2", "sample3", "meltdown_main", "sample4"]
)
def test_segment_label(filename):
    timing_labels = hts.load(
        join(dirname(__file__), "data", "neutrino", f"{filename}_timing.lab")
    )
    timing_labels = merge_sil(timing_labels)
    phrases = segment_labels(timing_labels)
    assert len(phrases) > 0
    assert isinstance(phrases, list)
    for idx, phrase in enumerate(phrases):
        print(f"Phrase {idx}:", " ".join(phrase.contexts))
        assert isinstance(phrase, hts.HTSLabelFile)

    # Check if the number of labels are the same
    assert len(timing_labels) == sum(len(phrase) for phrase in phrases)


def test_overwrite_phoneme_flags():
    labels = hts.HTSLabelFile()
    labels.append((0, 50000, "v@xx^pau-o+r=e_xx%xx^xx_xx~xx-1!1[xx$xx]xx/A:"))
    overwrite_phoneme_flags_(labels, "soft")
    assert labels.contexts[0] == "v@xx^pau-o+r=e_xx%xx^soft_xx~xx-1!1[xx$xx]xx/A:"
    overwrite_phoneme_flags_(labels, "xx")
    assert labels.contexts[0] == "v@xx^pau-o+r=e_xx%xx^xx_xx~xx-1!1[xx$xx]xx/A:"
    overwrite_phoneme_flags_(labels, "loud")
    assert labels.contexts[0] == "v@xx^pau-o+r=e_xx%xx^loud_xx~xx-1!1[xx$xx]xx/A:"
