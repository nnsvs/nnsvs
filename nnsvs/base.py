# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F
from abc import ABC, abstractmethod


class TimeLagModel(ABC):
    @abstractmethod
    def forward(self, feats, feats_lens=None):
        pass

class DurationModel(ABC):
    @abstractmethod
    def forward(self, feats, feats_lens=None):
        pass


class AcousticModel(ABC):
    @abstractmethod
    def forward(self, feats, feats_lens=None):
        pass


class BaseModel(nn.Module):
    def inference(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def preprocess_target(self, y):
        """Preprocess target signals at training time

        This is useful for shallow AR models in which a FIR filter
        is used for the target signals.

        Args:
            y (tensor): target features
        """
        return y