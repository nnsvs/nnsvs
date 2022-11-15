from nnsvs.util import get_world_stream_info


def test_get_world_stream_info():
    assert get_world_stream_info(24000, 59, 1, vibrato_mode="none") == [60, 1, 1, 3]
    assert get_world_stream_info(24000, 59, 1, vibrato_mode="diff") == [60, 1, 1, 3, 1]
    assert get_world_stream_info(24000, 59, 1, vibrato_mode="sine") == [
        60,
        1,
        1,
        3,
        3,
        1,
    ]

    assert get_world_stream_info(44100, 59, 1, vibrato_mode="none") == [60, 1, 1, 5]
    assert get_world_stream_info(48000, 59, 1, vibrato_mode="none") == [60, 1, 1, 5]


def test_get_world_stream_info_mcep_aperiodicity():
    assert (
        get_world_stream_info(
            44100,
            59,
            1,
            vibrato_mode="none",
            use_mcep_aperiodicity=True,
            mcep_aperiodicity_order=24,
        )
        == [60, 1, 1, 25]
    )
    assert (
        get_world_stream_info(
            48000,
            59,
            1,
            vibrato_mode="none",
            use_mcep_aperiodicity=True,
            mcep_aperiodicity_order=24,
        )
        == [60, 1, 1, 25]
    )
