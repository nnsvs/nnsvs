from os.path import dirname, join

import hydra
import torch
from omegaconf import OmegaConf

from nnsvs.model import MDN


# https://github.com/r9y9/nnsvs/pull/114#issuecomment-1156631058
def test_mdn_compat():
    config = OmegaConf.load(join(dirname(__file__), "data", "mdn_test.yaml"))
    model = hydra.utils.instantiate(config.netG)
    checkpoint = torch.load(
        join(dirname(__file__), "data", "mdn_test.pth"), weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])
    assert isinstance(model, MDN)
