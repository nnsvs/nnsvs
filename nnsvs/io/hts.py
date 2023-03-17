import re
from copy import deepcopy

import numpy as np
from nnmnkwii.io import hts

_flag_re = re.compile(r"\^([A-Za-z0-9]+)\_")


def full_to_mono(labels):
    """Convert full-context labels to mono labels

    Args:
        labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels

    Returns:
        nnmnkwii.io.hts.HTSLabelFile: Mono HTS labels
    """
    is_full_context = "@" in labels.contexts[0]
    if not is_full_context:
        return labels

    mono_labels = deepcopy(labels)
    mono_labels.contexts = [c.split("-")[1].split("+")[0] for c in labels.contexts]

    return mono_labels


def get_note_frame_indices(binary_dict, numeric_dict, in_feats):
    """Get note frame indices from frame-level input features

    Note that the F0 in the input features must be discrete F0.

    Args:
        binary_dict (dict): Dictionary of binary features
        numeric_dict (dict): Dictionary of numeric features
        in_feats (np.ndarray): Input features

    Returns:
        np.ndarray: Note frame indices
    """
    pitch_idx = get_pitch_index(binary_dict, numeric_dict)
    score_f0 = in_feats[:, pitch_idx]
    note_frame_indices = np.where(score_f0 > 0)[0]
    return note_frame_indices


def get_pitch_index(binary_dict, numeric_dict):
    """Get pitch index from binary and numeric feature dictionaries

    Args:
        binary_dict (dict): Dictionary of binary features
        numeric_dict (dict): Dictionary of numeric features

    Returns:
        int: Pitch index
    """
    idx = 0
    pitch_idx = len(binary_dict)
    while idx < len(numeric_dict):
        if numeric_dict[idx][1].pattern.startswith("/E"):
            pitch_idx = pitch_idx + idx
            break
        idx += 1
    return pitch_idx


def get_pitch_indices(binary_dict, numeric_dict):
    """Get pitch indices from binary and numeric feature dictionaries

    Args:
        binary_dict (dict): Dictionary of binary features
        numeric_dict (dict): Dictionary of numeric features

    Returns:
        list: Pitch indices
    """
    idx = 0
    pitch_idx = len(binary_dict)
    assert np.any(
        [numeric_dict[idx][1].pattern.startswith(p) for p in ["/D", "/E", "/F"]]
    )
    pitch_indices = [pitch_idx]
    while True:
        idx += 1
        if np.any(
            [numeric_dict[idx][1].pattern.startswith(p) for p in ["/D", "/E", "/F"]]
        ):
            pitch_indices.append(pitch_idx + idx)
        else:
            break
    return pitch_indices


def get_note_indices(labels):
    """Get note start indices from HTS labels

    Args:
        labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels

    Returns:
        list: Note start indices
    """
    note_indices = [0]
    last_start_time = labels.start_times[0]
    for idx in range(1, len(labels)):
        if labels.start_times[idx] != last_start_time:
            note_indices.append(idx)
            last_start_time = labels.start_times[idx]
        else:
            pass
    return note_indices


def merge_sil(labels):
    N = len(labels)
    f = hts.HTSLabelFile()
    f.append(labels[0], strict=False)
    is_full_context = "@" in labels[0][-1]
    for i in range(1, N):
        if (is_full_context and "-sil" in f[-1][-1] and "-sil" in labels[i][-1]) or (
            not is_full_context and f[-1][-1] == "sil" and labels[i][-1] == "sil"
        ):
            # extend sil
            f.end_times[-1] = labels[i][1]
        else:
            f.append(labels[i], strict=False)
    return f


def _is_br(label):
    is_full_context = "@" in label
    if is_full_context:
        is_br = "-br" in label
    else:
        is_br = label == "br"
    return is_br


def _is_silence(label):
    is_full_context = "@" in label
    if is_full_context:
        is_silence = "-sil" in label or "-pau" in label
    else:
        is_silence = label == "sil" or label == "pau"
    return is_silence


def compute_nosil_duration(labels, threshold=5.0):
    is_full_context = "@" in labels[0][-1]
    sum_d = 0
    for s, e, label in labels:
        d = (e - s) * 1e-7
        if is_full_context:
            is_silence = "-sil" in label or "-pau" in label
        else:
            is_silence = label == "sil" or label == "pau"
        if is_silence and d > threshold:
            pass
        else:
            sum_d += d
    return sum_d


def segment_labels(
    labels,
    strict=True,
    silence_threshold=0.1,
    min_duration=5.0,
    force_split_threshold=5.0,
):
    """Segment labels based on sil/pau

    Example:

    [a b c sil d e f pau g h i sil j k l]
    ->
    [a b c] [d e f] [g h i] [j k l]

    """
    seg = hts.HTSLabelFile()
    start_indices = []
    end_indices = []
    si = 0

    done_last_label = False
    for idx, (s, e, label) in enumerate(labels):
        d = (e - s) * 1e-7
        is_silence = _is_silence(label)

        if len(seg) > 0:
            # Compute duration except for long silences
            seg_d = compute_nosil_duration(seg)
        else:
            seg_d = 0

        # let's try to split
        if (is_silence and d > force_split_threshold) or (
            is_silence and d > silence_threshold and seg_d > min_duration
        ):
            if idx == len(labels) - 1:
                pass
            elif len(seg) > 0:
                start_indices.append(si)
                if is_silence and d > force_split_threshold:
                    end_indices.append(idx - 1)
                    start_indices.append(idx)
                    end_indices.append(idx)
                    seg = hts.HTSLabelFile()
                else:
                    seg.append((s, e, label), strict)
                    end_indices.append(idx)
                    seg = hts.HTSLabelFile()
                si = idx + 1
            elif len(seg) == 0:
                seg.append((s, e, label), strict)
                start_indices.append(si)
                end_indices.append(idx)
                seg = hts.HTSLabelFile()
        else:
            if len(seg) == 0:
                si = idx
            if idx == len(labels) - 1:
                done_last_label = True
            seg.append((s, e, label), strict)

    if len(seg) > 0:
        seg_d = compute_nosil_duration(seg)
        # If the last segment is short, combine with the previous segment.
        if seg_d < min_duration and len(end_indices) > 1:
            end_indices[-1] = si + len(seg) - 1
        else:
            start_indices.append(si)
            end_indices.append(si + len(seg) - 1)

        # Handle last label
        if not done_last_label:
            s, e, label = labels[-1]
            d = (e - s) * 1e-7
            if _is_silence(label) and d > silence_threshold:
                start_indices.append(end_indices[-1])
                end_indices.append(end_indices[-1])

    segments = []
    for s, e in zip(start_indices, end_indices):
        seg = labels[s : e + 1]

        offset = seg.start_times[0]
        seg.start_times = np.asarray(seg.start_times) - offset
        seg.end_times = np.asarray(seg.end_times) - offset

        segments.append(seg)

    return segments


def _label2phrases_neutrino(labels):
    """Segment HTS labels to phrases (NETRINO compatible)

    Args:
        labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels

    Returns:
        list: phrases (i.e., list of HTS labels)
    """
    start_indices = []
    end_indices = []

    started = True
    start_indices.append(0)
    is_sil_phrase = _is_silence(labels.contexts[0])

    for idx, (_, _, label) in enumerate(labels):
        # sil or pau shouldn't be placed right before the br
        if idx > 0 and _is_br(label):
            assert not _is_silence(labels.contexts[idx - 1])

        # we are in the same phrase group
        if started:
            if is_sil_phrase:
                if _is_silence(label):
                    continue
            else:
                if (
                    not _is_silence(label)
                    and (idx > 0 and not _is_br(labels.contexts[idx - 1]))
                    or (idx == 0 and not _is_silence(label))
                ):
                    continue

        # reached the end of phrase
        end_indices.append(idx)
        started = False

        # start new phrase
        if not started:
            started = True
            is_sil_phrase = _is_silence(label)
            start_indices.append(idx)

    # handle last phrase
    if len(end_indices) == len(start_indices) - 1:
        end_indices.append(len(labels))

    # Make a list of HTS labels
    phrases = [labels[s:e] for (s, e) in zip(start_indices, end_indices)]
    return phrases, start_indices, end_indices


def fix_label_offset_to_zero(labels):
    """Fix label offset to zero

    Args:
        labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels

    Returns:
        nnmnkwii.io.hts.HTSLabelFile: HTS labels with fixed offset
    """
    offset = labels.start_times[0]
    if offset > 0:
        labels.start_times = np.asarray(labels.start_times) - offset
        labels.end_times = np.asarray(labels.end_times) - offset
    return labels


def _label2phoneme_for_phrases(labels, s, e, note_indices=None):
    if s == e:
        r = labels.contexts[s]
    elif note_indices is None:
        r = " ".join(labels[s:e].contexts)
    else:
        rs = []
        for i in range(s, e):
            if i not in [s, e] and i in note_indices:
                rs.append(",")
            rs.append(labels.contexts[i])
        r = " ".join(rs).replace(" ,", ",")
    return r


def label2phrases_str(labels, note_indices):
    """Convert labels to NEUTRINO-format phraselist

    Note that timing labels should be used as input to get the same
    output as the NEUTRINO.

    Args:
        labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels
        note_indices (list): indices of notes. This is needed to
            insert ``,`` at note boundaries.

    Returns:
        str: NEUTRINO-format phraselist
    """
    _, start_indices, end_indices = _label2phrases_neutrino(labels)

    out = ""
    for idx in range(len(end_indices)):
        s, e = start_indices[idx], end_indices[idx]
        start_time = int(labels.start_times[s] // 10000)
        ph = _label2phoneme_for_phrases(labels, s, e, note_indices)
        is_voiced_phrase = not ("sil" in ph or "pau" in ph)
        out += f"{idx} {start_time} {int(is_voiced_phrase)} {ph}\n"
    return out


def label2phrases(labels, fix_offset=True):
    """Convert labels to phrases

    The definision of a phrase is the same as NEUTRINO.
    See https://studio-neutrino.com/332/ for details.

    Args:
        labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels
        fix_offset (bool): If True, fix label offset to zero

    Returns:
        list: phrases (i.e., list of HTS labels)
    """
    phrases = _label2phrases_neutrino(labels)[0]
    if fix_offset:
        phrases = [fix_label_offset_to_zero(p) for p in phrases]
    return phrases


def overwrite_phoneme_flags_(labels, flag):
    """Overwrite phoneme flags

    Args:
        labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels
        flag (str): phoneme flag to overwrite

    Returns:
        nnmnkwii.io.hts.HTSLabelFile: modified HTS labels
    """

    contexts = labels.contexts
    for i in range(len(contexts)):
        n = len(_flag_re.findall(contexts[i]))
        if n == 0:
            print(i, contexts[i])
            print("Warn: it is likely to have a wrong input format. Ignoring.")
        elif n != 1:
            print(i, contexts[i])
            raise RuntimeError("More than two flags are found")
        contexts[i] = _flag_re.sub(f"^{flag}_", contexts[i])
    labels.contexts = contexts

    return labels
