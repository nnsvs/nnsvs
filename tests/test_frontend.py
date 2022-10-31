def test_ja():
    from nnsvs.frontend import ja

    print("num vocab:", ja.num_vocab())
    seq = ja.text_to_sequence(["a", "i", "u", "e", "o"])
    assert " ".join(ja.sequence_to_text(seq)) == "a i u e o"


def test_zh():
    from nnsvs.frontend import zh

    print("num vocab:", zh.num_vocab())
    seq = zh.text_to_sequence(["AP", "SP"])
    assert " ".join(zh.sequence_to_text(seq)) == "AP SP"
