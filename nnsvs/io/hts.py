# coding: utf-8


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
