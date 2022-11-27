"""Convert Opencpop's segmented data to NNSVS's structure
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from nnmnkwii.io import hts
from nnsvs.io.hts import get_note_indices
from scipy.io import wavfile
from tqdm.auto import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert Opencpop's segmented data to NNSVS's structure",
    )
    parser.add_argument("in_dir", type=str, help="Path to input dir")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument(
        "--long_sil_threshold",
        type=float,
        default=1000,
        help="Long silence threshold (sec)",
    )
    return parser


def create_label_score(phs, notes, note_durs, ph_durs, is_slurs, round=False):
    labels = hts.HTSLabelFile()
    # e7: sec -> 0.001 sec, and rounding to int
    note_durs_001sec = np.rint(np.array(note_durs) / 0.01)

    # phoneme-level positional features
    p12 = 0
    prev_note_dur = None
    prev_note_dur_001sec = None
    start_time = 0
    end_time = 0

    if phs[0] in ["sil", "br"]:
        print("Warning: first phoneme is sil or br")

    for idx, (ph, note, note_dur, note_dur_001sec, _, is_slur) in enumerate(
        zip(phs, notes, note_durs, note_durs_001sec, ph_durs, is_slurs)
    ):
        if prev_note_dur is not None and note_dur != prev_note_dur:
            p12 = 1
            if round:
                start_time += prev_note_dur_001sec * 0.01
            else:
                start_time += prev_note_dur
        else:
            p12 += 1
        if round:
            end_time = start_time + (note_dur_001sec * 0.01)
        else:
            end_time = start_time + note_dur
        prev_note_dur = note_dur
        prev_note_dur_001sec = note_dur_001sec

        # prev pitch
        d1 = notes[idx - 1] if idx > 0 else "xx"
        # current pitch
        e1 = note
        # next pitch
        f1 = notes[idx + 1] if idx < len(notes) - 1 else "xx"

        if e1 != "xx" and ph in ["sil", "pau", "br", "SP", "AP"]:
            print("Warning: phoneme is sil or br, but note is not xx")
            print(f"{utt_id}: ph={ph}, note={note}, note_dur={note_dur}")

        # p3,p12,d1,e1,e7,f1,e26
        context = f"xx@xx^xx-{ph}+xx=xx_xx%-{p12}!/D:{d1}!/E:{e1}]@{int(note_dur_001sec)}#|{is_slur}]/F:{f1}#/J:xx~xx@xx"  # noqa
        assert p12 < 5, "must be a bug"

        label = (1e7 * start_time, 1e7 * end_time, context)
        labels.append(label, strict=False)

    return labels


def round_phoneme_durations(ph_durs):
    new_ph_durs = np.asarray(ph_durs).copy()

    for i in range(len(new_ph_durs) - 1):
        offset = 0.005 - new_ph_durs[i] % 0.005
        new_ph_durs[i] += offset
        new_ph_durs[i + 1] -= offset

    # TODO: adjust last phoneme

    new_ph_durs = np.round(new_ph_durs, 5)

    return new_ph_durs


def create_label_align(phs, notes, note_durs, ph_durs, is_slurs, round=True):
    labels = hts.HTSLabelFile()
    # e7: sec -> 0.001 sec, and rounding to int
    note_durs_001sec = np.rint(np.array(note_durs) / 0.01)

    # phoneme-level positional features
    p12 = 0
    prev_note_dur = None
    prev_ph_dur = None
    start_time = 0
    end_time = 0

    # Round phoneme duraitons to 0.005 sec unit
    if round:
        ph_durs = round_phoneme_durations(ph_durs)

    for idx, (ph, note, note_dur, note_dur_001sec, ph_dur, is_slur) in enumerate(
        zip(phs, notes, note_durs, note_durs_001sec, ph_durs, is_slurs)
    ):
        if prev_ph_dur is not None:
            start_time += prev_ph_dur
        if prev_note_dur is not None and note_dur != prev_note_dur:
            p12 = 1
        else:
            p12 += 1
        end_time = start_time + ph_dur
        prev_note_dur = note_dur
        prev_ph_dur = ph_dur

        # prev pitch
        d1 = notes[idx - 1] if idx > 0 else "xx"
        # current pitch
        e1 = note
        # next pitch
        f1 = notes[idx + 1] if idx < len(notes) - 1 else "xx"

        # p3,p12,d1,e1,e7,f1,e26
        context = f"xx@xx^xx-{ph}+xx=xx_xx%-{p12}!/D:{d1}!/E:{e1}]@{int(note_dur_001sec)}#|{is_slur}]/F:{f1}#/J:xx~xx@xx"  # noqa
        assert p12 < 5, "must be a bug"

        label = (1e7 * start_time, 1e7 * end_time, context)
        labels.append(label, strict=True)

    return labels


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    long_sil_threshold = args.long_sil_threshold

    in_wav_dir = in_dir / "wavs"
    transcriptions_txt = in_dir / "transcriptions.txt"
    train_txt = in_dir / "train.txt"
    test_txt = in_dir / "test.txt"

    acoustic_label_score_dir = out_dir / "acoustic" / "label_phone_score"
    acoustic_label_align_dir = out_dir / "acoustic" / "label_phone_align"
    acoustic_out_wav_dir = out_dir / "acoustic" / "wav"
    duration_label_align_dir = out_dir / "duration" / "label_phone_align"
    timelag_label_score_dir = out_dir / "timelag" / "label_phone_score"
    timelag_label_align_dir = out_dir / "timelag" / "label_phone_align"
    list_dir = out_dir / "list"

    for d in [
        acoustic_label_score_dir,
        acoustic_label_align_dir,
        acoustic_out_wav_dir,
        duration_label_align_dir,
        timelag_label_score_dir,
        timelag_label_align_dir,
        list_dir,
    ]:
        d.mkdir(exist_ok=True, parents=True)

    for name, path in [
        ("train_no_dev", train_txt),
        ("dev", test_txt),
        ("eval", test_txt),
    ]:
        with open(path) as f:
            train_ids = [line.strip().split("|")[0] for line in f]
            with open(list_dir / f"{name}.list", "w") as of:
                for utt_id in train_ids:
                    of.write(utt_id + "\n")

    with open(transcriptions_txt) as f:
        transcriptions = f.readlines()

    def _pitch(s):
        ss = s.split("/")
        p = ss[0] if len(ss) == 1 else ss[1]
        return p

    for line in tqdm(transcriptions):
        song_info = line.split("|")
        utt_id = song_info[0].strip()
        phs = [
            s.replace("SP", "sil").replace("AP", "br")
            for s in song_info[2].strip().split(" ")
        ]
        notes = [_pitch(x) if x != "rest" else "xx" for x in song_info[3].split(" ")]
        note_durs = [float(x) for x in song_info[4].split(" ")]
        ph_durs = [float(x) for x in song_info[5].split(" ")]
        # e26
        is_slurs = [int(x) for x in song_info[6].split(" ")]

        assert len(phs) == len(notes) == len(note_durs) == len(ph_durs) == len(is_slurs)

        sils = [s in ["sil", "pau", "br"] for s in phs]

        is_long_sil = np.array(note_durs)[sils] > long_sil_threshold
        sil_remove_regions = []
        if is_long_sil.any():
            start = 0
            new_note_durs = []
            new_ph_durs = []
            prev_note_dur = None
            for ph, note_dur, ph_dur in zip(phs, note_durs, ph_durs):
                # NOTE: assuming single phones are assigned to a single note
                if ph in ["sil", "pau", "br"] and note_dur > long_sil_threshold:
                    cut_length = note_dur - long_sil_threshold
                    center = start + note_dur / 2
                    sil_remove_regions.append(
                        (center - cut_length / 2, center + cut_length / 2)
                    )
                    new_note_durs.append(long_sil_threshold)
                    new_ph_durs.append(long_sil_threshold)
                else:
                    new_note_durs.append(note_dur)
                    new_ph_durs.append(ph_dur)
                if prev_note_dur is not None and note_dur != prev_note_dur:
                    start += note_dur
                prev_note_dur = note_dur
            note_durs = new_note_durs
            ph_durs = new_ph_durs

        # label_score
        label_score = create_label_score(phs, notes, note_durs, ph_durs, is_slurs)
        # label_align
        label_align = create_label_align(phs, notes, note_durs, ph_durs, is_slurs)

        assert len(label_score) == len(label_align)
        # Labels for time-lag model
        # NOTE: since opencpop's annotations in MIDI format, there's no time lag between
        # MIDI and recordings.
        # Save the labels for consistency with MusicXML or UST format.
        note_indices = get_note_indices(label_score)
        note_label_score = label_score[note_indices]
        note_label_align = label_align[note_indices]

        # Save data for acoustic
        with open(acoustic_label_score_dir / f"{utt_id}.lab", "w") as f:
            f.write(str(label_score))
        with open(acoustic_label_align_dir / f"{utt_id}.lab", "w") as f:
            f.write(str(label_align))

        sr, x = wavfile.read(in_wav_dir / f"{utt_id}.wav")

        if len(sil_remove_regions) > 0:
            assert len(sil_remove_regions) == 1
            print(f"{utt_id}: trim long silence to {long_sil_threshold} sec")
            new_x = []
            for start, end in sil_remove_regions:
                start = int(start * sr)
                end = int(end * sr)
                new_x.append(x[:start])
                new_x.append(x[end:])
            x = np.concatenate(new_x)

        wavfile.write(acoustic_out_wav_dir / f"{utt_id}.wav", sr, x)
        # Save data for duration
        with open(duration_label_align_dir / f"{utt_id}.lab", "w") as f:
            f.write(str(label_align))

        # Save data for timelag
        with open(timelag_label_score_dir / f"{utt_id}.lab", "w") as f:
            f.write(str(note_label_score))
        with open(timelag_label_align_dir / f"{utt_id}.lab", "w") as f:
            f.write(str(note_label_align))
