# NOTE: Opencpop uses SP as silence and AP as aspire (breath)
# for consistency with the other JP database, adding sil, pau and br to the list.
phonemes = [
    "AP",
    "SP",
    "sil",
    "pau",
    "br",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
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
