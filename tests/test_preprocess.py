from pathlib import Path

import pytest
from nnmnkwii.datasets import FileSourceDataset
from nnsvs.data import MelF0AcousticSource, WORLDAcousticSource


@pytest.mark.skipif(
    not (Path(__file__).parent / "data" / "utt_list.txt").exists(),
    reason="test data not found",
)
@pytest.mark.parametrize("f0_extractor", ["dio", "harvest", "parselmouth"])
def test_world(f0_extractor):
    cwd = Path(__file__).parent
    utt_list = cwd / "data" / "utt_list.txt"
    wav_root = cwd / "data" / "wav"
    label_root = cwd / "data" / "label_phone_align"
    hed_path = cwd / "data" / "test.hed"
    data_source = WORLDAcousticSource(
        utt_list=utt_list,
        wav_root=wav_root,
        label_root=label_root,
        question_path=hed_path,
        dynamic_features_flags=[False, False, False, False],
        relative_f0=False,
        sample_rate=24000,
        f0_extractor=f0_extractor,
        num_windows=1,
    )

    dataset = FileSourceDataset(data_source)
    for data in dataset:
        y, wave, y_pf = data
        print(y.shape, wave.shape, y_pf.shape)
        break


@pytest.mark.skipif(
    not (Path(__file__).parent / "data" / "utt_list.txt").exists(),
    reason="test data not found",
)
@pytest.mark.parametrize("f0_extractor", ["dio", "harvest", "parselmouth"])
def test_melf0(f0_extractor):
    cwd = Path(__file__).parent
    utt_list = cwd / "data" / "utt_list.txt"
    wav_root = cwd / "data" / "wav"
    label_root = cwd / "data" / "label_phone_align"
    hed_path = cwd / "data" / "test.hed"
    data_source = MelF0AcousticSource(
        utt_list=utt_list,
        wav_root=wav_root,
        label_root=label_root,
        question_path=hed_path,
        sample_rate=24000,
        fft_size=512,
        hop_size=120,
        win_length=480,
        f0_extractor=f0_extractor,
    )

    dataset = FileSourceDataset(data_source)
    for data in dataset:
        y, wave, y_pf = data
        print(y.shape, wave.shape, y_pf.shape)
        break
