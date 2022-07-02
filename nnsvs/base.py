from enum import Enum

from torch import nn


class PredictionType(Enum):
    DETERMINISTIC = 1
    PROBABILISTIC = 2


class BaseModel(nn.Module):
    """Base class for all models

    If you want to implement your new model, you should inherit from this class.
    You must need to implement the forward method. Other methods are optional.
    """

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
        """Inference method

        If you want to implement custom inference method such as autoregressive sampling,
        please override this method.

        Defaults to call the forward method.
        """
        return self(*args, **kwargs)

    def preprocess_target(self, y):
        """Preprocess target signals at training time

        This is useful for shallow AR models in which a FIR filter
        is used for the target signals. For other types of model, you don't need to
        implement this method.

        Defaults to do nothing.

        Args:
            y (tensor): target features

        Returns:
            tensor: preprocessed target features
        """
        return y

    def prediction_type(self):
        """Prediction type.

        If your model has a MDN layer, please return ``PredictionType.PROBABILISTIC``.

        Returns:
            PredictionType: Determisitic or probabilistic. Default is deterministic.
        """
        return PredictionType.DETERMINISTIC

    def is_autoregressive(self):
        """Is autoregressive or not

        Returns:
            bool: True if autoregressive. Default is False.
        """
        return False
