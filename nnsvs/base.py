from abc import ABC, abstractmethod
from enum import Enum

from torch import nn


class PredictionType(Enum):
    DETERMINISTIC = 1
    PROBABILISTIC = 2


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
    """Base class for all models"""

    def inference(self, *args, **kwargs):
        """Inference method"""
        return self.forward(*args, **kwargs)

    def preprocess_target(self, y):
        """Preprocess target signals at training time

        This is useful for shallow AR models in which a FIR filter
        is used for the target signals.

        Args:
            y (tensor): target features
        """
        return y

    def prediction_type(self):
        """Prediction type

        Returns:
            PredictionType: prediction type. Determisitic or probabilistic
        """
        return PredictionType.DETERMINISTIC

    def is_autoregressive(self):
        """Is autoregressive or not

        Returns:
            bool: True if autoregressive
        """
        return False
