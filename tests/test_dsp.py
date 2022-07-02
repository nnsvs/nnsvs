import numpy as np
import pytest
from nnsvs.dsp import lowpass_filter


@pytest.mark.parametrize("cutoff", [3, 5, 8])
def test_lowpass_filter_noraise(cutoff):
    sr = int(1 / 0.005)

    # special case: 0-length array
    lowpass_filter(np.zeros(10)[0:0], sr, cutoff=cutoff)
    for n in range(sr * 2):
        lowpass_filter(np.random.rand(n), sr, cutoff=cutoff)
