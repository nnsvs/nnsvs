# coding: utf-8
from os.path import join
import jaconv
import numpy as np
from nnmnkwii.io import hts
from tqdm import tqdm

import yaml
with open('config.yaml', 'r') as yml:
    config = yaml.load(yml, Loader=yaml.FullLoader)

def merge_sil(lab):
    N = len(lab)
    f = hts.HTSLabelFile()
    f.append(lab[0], strict=False)
    is_full_context = "@" in lab[0][-1]
    for i in range(1, N):
        if (is_full_context and "-sil" in f[-1][-1] and "-sil" in lab[i][-1]) \
            or (not is_full_context and f[-1][-1] == "sil" and lab[i][-1] == "sil"):
            # extend sil
            f.end_times[-1] = lab[i][1]
        else:
            f.append(lab[i], strict=False)
    return f

def _is_silence(l):
    is_full_context = "@" in l
    if is_full_context:
        is_silence = ("-sil" in l or "-pau" in l)
    else:
        is_silence = (l == "sil" or l == "pau")
    return is_silence


def trim_long_sil_and_pau(lab, return_indices=False, threshold=10.0):
    forward = 0
    while True:
        d  = (lab.end_times[forward] - lab.start_times[forward]) * 1e-7
        if _is_silence(lab.contexts[forward]) and d > threshold:
            forward += 1
        else:
            break

    backward = len(lab) - 1
    while True:
        d  = (lab.end_times[backward] - lab.start_times[backward]) * 1e-7
        if _is_silence(lab.contexts[backward]) and d > threshold:
            backward -= 1
        else:
            break

    if return_indices:
        return lab[forward:backward+1], forward, backward
    else:
        return lab[forward:backward+1]


def compute_nosil_duration(lab, threshold=5.0):
    is_full_context = "@" in lab[0][-1]
    sum_d = 0
    for s,e,l in lab:
        d = (e - s) * 1e-7
        if is_full_context:
            is_silence = ("-sil" in l or "-pau" in l)
        else:
            is_silence = (l == "sil" or l == "pau")
        if is_silence and d > threshold:
            pass
        else:
            sum_d += d
    return sum_d


def segment_labels(lab, strict=True, threshold=1.0, min_duration=5.0,
        force_split_threshold=10.0):
    """Segment labels based on sil/pau

    Example:

    [a b c sil d e f pau g h i sil j k l]
    ->
    [a b c] [d e f] [g h i] [j k l]

    """
    segments = []
    seg = hts.HTSLabelFile()
    start_indices = []
    end_indices = []
    si = 0
    large_silence_detected = False

    for idx, (s, e, l) in enumerate(lab):
        d = (e-s) * 1e-7
        is_silence = _is_silence(l)

        if len(seg) > 0:
            # Compute duration except for long silences
            seg_d = compute_nosil_duration(seg)
        else:
            seg_d = 0

        # let's try to split
        # if we find large silence, force split regardless min_duration
        if (d > force_split_threshold) or (is_silence and d > threshold and seg_d > min_duration):
            if idx == len(lab)-1:
                continue
            elif len(seg) > 0:
                if d > force_split_threshold:
                    large_silence_detected = True
                else:
                    large_silence_detected = False
                start_indices.append(si)
                si = 0
                end_indices.append(idx - 1)
                segments.append(seg)
                seg = hts.HTSLabelFile()
            continue
        else:
            if len(seg) == 0:
                si = idx
            seg.append((s, e, l), strict)

    if len(seg) > 0:
        seg_d = compute_nosil_duration(seg)
        # If the last segment is short, combine with the previous segment.
        if seg_d < min_duration and not large_silence_detected:
            end_indices[-1] = si + len(seg) - 1
        else:
            start_indices.append(si)
            end_indices.append(si + len(seg) - 1)

    #  Trim large sil for each segment
    segments2 = []
    start_indices_new, end_indices_new = [], []
    for s, e in zip(start_indices, end_indices):
        seg = lab[s:e+1]

        # ignore "sil" or "pau" only segment
        if len(seg) ==1 and _is_silence(seg.contexts[0]):
            continue
        seg2, forward, backward = trim_long_sil_and_pau(seg, return_indices=True)

        start_indices_new.append(s+forward)
        end_indices_new.append(s+backward)

        segments2.append(seg2)

    return segments2, start_indices_new, end_indices_new


def prep_ph2num():
    kiritan_phone_mapping = {}
    with open(join(config["sinsy_dic"], "japanese.table"), encoding="UTF-8") as f:
        for l in f:
            s = l.strip().split()
            key = jaconv.hira2kata(s[0])
            kiritan_phone_mapping[key] = s[1:]
    sinsy_phone_mapping = {}
    with open(join(config["sinsy_dic"], "japanese.utf_8.table"), encoding="UTF-8") as f:
        for l in f:
            s = l.strip().split()
            key = jaconv.hira2kata(s[0])
            sinsy_phone_mapping[key] = s[1:]
    ph2num = {}
    counter = 0
    for p in ["sil", "pau", "br"]:
        ph2num[p] = counter
        counter += 1
    for k, v in sinsy_phone_mapping.items():
        for p in v:
            if p not in ph2num:
                ph2num[p] = counter
                counter += 1
    for k, v in kiritan_phone_mapping.items():
        for p in v:
            if p not in ph2num:
                ph2num[p] = counter
                counter += 1
    # undef
    ph2num["xx"] = counter

    return ph2num


def ph2numeric(contexts, ph2num):
    return [ph2num[k] for k in contexts]


def fix_offset(lab):
    offset = lab.start_times[0]
    lab.start_times = np.asarray(lab.start_times) - offset
    lab.end_times = np.asarray(lab.end_times) - offset
    return lab


def trim_sil_and_pau(lab, return_indices=False):
    forward = 0
    while "-sil" in lab.contexts[forward] or "-pau" in lab.contexts[forward]:
        forward += 1

    backward = len(lab)-1
    while "-sil" in lab.contexts[backward] or "-pau" in lab.contexts[backward]:
        backward -= 1

    if return_indices:
        return lab[forward:backward+1], forward, backward
    else:
        return lab[forward:backward+1]


def get_note_indices(lab):
    note_indices = [0]
    last_start_time = lab.start_times[0]
    for idx in range(1, len(lab)):
        if lab.start_times[idx] != last_start_time:
            note_indices.append(idx)
            last_start_time = lab.start_times[idx]
        else:
            pass
    return note_indices

def fix_mono_lab_before_align(lab):
    # There is nothing to do
    return lab

def fix_mono_lab_after_align(lab):
    f = hts.HTSLabelFile()
    f.append(lab[0])
    for i in range(1, len(lab)):
        # fix contigous pau
        if(f.contexts[-1] == "pau" and lab.contexts[i] == "pau"
           and f.start_times[-1] == lab.start_times[i]
           and f.end_times[-1] == lab.end_times[i]):
            d = round((lab.end_times[i] - lab.start_times[i]) / 2)
            f.end_times[-1] = f.start_times[-1] + d
            f.append((f.end_times[-1], lab.end_times[i], lab.contexts[i]))
        else:        
            f.append(lab[i], strict=False)
    return(f)
