phonemes = [
    "A",
    "E",
    "I",
    "N",
    "O",
    "U",
    "a",
    "b",
    "br",
    "by",
    "ch",
    "cl",
    "d",
    "dy",
    "e",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "p",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "v",
    "w",
    "y",
    "z",
    "pau",
    "sil",
    "fy",
    "vy",
    "GlottalStop",
    "Edge",
]

_pad = "~"

symbols = [_pad] + phonemes


_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def num_vocab():
    return len(symbols)


def text_to_sequence(text):
    return [_symbol_to_id[s] for s in text]


def sequence_to_text(seq):
    return [_id_to_symbol[s] for s in seq]
