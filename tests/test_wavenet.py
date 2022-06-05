import torch
from nnsvs.wavenet import WaveNet


def test_wavenet():
    x = torch.rand(16, 200, 206)
    c = torch.rand(16, 200, 300)

    model = WaveNet(in_dim=300, out_dim=206, layers=2)
    y = model(c, x)

    assert y.shape == x.shape

    model.eval()
    for T in [10, 20, x.shape[1]]:
        y = model.inference(c, num_time_steps=T)
        assert y.shape == (16, T, 206)
