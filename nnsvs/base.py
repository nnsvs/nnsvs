from enum import Enum

from torch import nn


class PredictionType(Enum):
    DETERMINISTIC = 1
    PROBABILISTIC = 2


class BaseModel(nn.Module):
    """Base class for all models"""

    def forward(self, x, lengths=None, y=None):
        """Forward pass

        Args:
            x (tensor): input features
            lengths (tensor): lengths of the input features
            y (tensor): optional target features

        Returns:
            tensor: output features
        """
        pass

    def inference(self, *args, **kwargs):
        """Inference method"""
        return self(*args, **kwargs)

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
