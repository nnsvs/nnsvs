from enum import Enum

from torch import nn


class PredictionType(Enum):
    """Prediction types"""

    DETERMINISTIC = 1
    """Deterministic prediction

    Non-MDN single-stream models should use this type.

    Pseudo code:

    .. code-block::

        # training
        y = model(x)
        # inference
        y = model.inference(x)
    """

    PROBABILISTIC = 2
    """Probabilistic prediction with mixture density networks

    MDN-based models should use this type.

    Pseudo code:

    .. code-block::

        # training
        mdn_params = model(x)
        # inference
        mu, sigma = model.inference(x)
    """

    MULTISTREAM_HYBRID = 3
    """Multi-stream preodictions where each prediction can be
    detereministic or probabilistic

    Multi-stream models should use this type.

    Pseudo code:

    .. code-block::

        # training
        feature_streams = model(x) # e.g. (mgc, lf0, vuv, bap) or (mel, lf0, vuv)
        # inference
        y = model.inference(x)

    Note that concatenated features are assumed to be returned during inference.
    """

    DIFFUSION = 4
    """Diffusion model's prediction

    NOTE: may subject to change in the future

    Pseudo code:

    .. code-block::

        # training
        noise, x_recon = model(x)

        # inference
        y = model.inference(x)
    """


class BaseModel(nn.Module):
    """Base class for all models

    If you want to implement your custom model, you should inherit from this class.
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

    def inference(self, x, lengths=None):
        """Inference method

        If you want to implement custom inference method such as autoregressive sampling,
        please override this method.

        Defaults to call the forward method.

        Args:
            x (tensor): input features
            lengths (tensor): lengths of the input features

        Returns:
            tensor: output features
        """
        return self(x, lengths)

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

        If your custom model is an autoregressive model, please return ``True``. In that case,
        you would need to implement autoregressive sampling in :py:meth:`inference`.

        Returns:
            bool: True if autoregressive. Default is False.
        """
        return False

    def has_residual_lf0_prediction(self):
        """Whether the model has residual log-F0 prediction or not.

        This should only be used for acoustic models.

        Returns:
            bool: True if the model has residual log-F0 prediction. Default is False.
        """
        return False
