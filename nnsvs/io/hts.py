import numpy as np
from nnmnkwii.io import hts


def get_pitch_index(binary_dict, numeric_dict):
    idx = 0
    pitch_idx = len(binary_dict)
    while idx < len(numeric_dict):
        if numeric_dict[idx][1].pattern.startswith("/E"):
            pitch_idx = pitch_idx + idx
            break
        idx += 1
    return pitch_idx


def get_pitch_indices(binary_dict, numeric_dict):
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
    silence_threshold=0.5,
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
            seg.append((s, e, label), strict)

    if len(seg) > 0:
        seg_d = compute_nosil_duration(seg)

        # If the last segment is short, combine with the previous segment.
        if seg_d < min_duration:
            end_indices[-1] = si + len(seg) - 1
        else:
            start_indices.append(si)
            end_indices.append(si + len(seg) - 1)

        # Handle last label
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
