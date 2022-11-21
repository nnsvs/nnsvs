import numpy as np
import torch
from nnsvs.base import PredictionType
from nnsvs.mdn import mdn_get_most_probable_sigma_and_mu
from torch.nn import functional as F


def predict_lf0_with_residual(
    in_feats,
    out_feats,
    in_lf0_idx=300,
    in_lf0_min=5.3936276,
    in_lf0_max=6.491111,
    out_lf0_idx=180,
    out_lf0_mean=5.953093881972361,
    out_lf0_scale=0.23435173188961034,
    residual_f0_max_cent=600,
):
    """Predict log-F0 with residual.

    Args:
        in_feats (np.ndarray): input features
        out_feats (np.ndarray): output of an acoustic model
        in_lf0_idx (int): index of LF0 in input features
        in_lf0_min (float): minimum value of LF0 in the training data of input features
        in_lf0_max (float): maximum value of LF0 in the training data of input features
        out_lf0_idx (int): index of LF0 in output features
        out_lf0_mean (float): mean of LF0 in the training data of output features
        out_lf0_scale (float): scale of LF0 in the training data of output features
        residual_f0_max_cent (int): maximum value of residual LF0 in cent

    Returns:
        tuple: (predicted log-F0, residual log-F0)
    """
    # Denormalize lf0 from input musical score
    lf0_score = in_feats[:, :, in_lf0_idx].unsqueeze(-1)
    lf0_score_denorm = lf0_score * (in_lf0_max - in_lf0_min) + in_lf0_min

    # To avoid unbounded residual f0 that would potentially cause artifacts,
    # let's constrain the residual F0 to be in a certain range by the scaled tanh
    max_lf0_ratio = residual_f0_max_cent * np.log(2) / 1200

    if len(out_feats.shape) == 4:
        # MDN case (B, T, num_gaussians, C) -> (B, T, num_gaussians)
        lf0_residual = out_feats[:, :, :, out_lf0_idx]
    else:
        # (B, T, C) -> (B, T, 1)
        lf0_residual = out_feats[:, :, out_lf0_idx].unsqueeze(-1)
    lf0_residual = max_lf0_ratio * torch.tanh(lf0_residual)

    # Residual connection in the denormalized f0 domain
    lf0_pred_denorm = lf0_score_denorm + lf0_residual

    # Back to normalized f0
    lf0_pred = (lf0_pred_denorm - out_lf0_mean) / out_lf0_scale

    return lf0_pred, lf0_residual


def pad_inference(
    model, x, lengths, reduction_factor, mode="replicate", y=None, mdn=False
):
    mod = max(lengths) % reduction_factor
    pad = reduction_factor - mod

    # Pad zeros to the end of the input features
    # so that the length of the input features is a multiple of the reduction factor
    if pad != 0:
        x_pad = F.pad(x, (0, 0, 0, pad), mode=mode)
        y_pad = F.pad(y, (0, 0, 0, pad), mode=mode) if y is not None else None
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.clone()
        else:
            lengths = lengths.copy()
        lengths = [length + pad for length in lengths]
    else:
        x_pad = x
        y_pad = y if y is not None else None
    y = model(x_pad, lengths, y_pad)
    # Residual F0 prediction: (out, lf0)
    if mdn:
        assert isinstance(y, tuple) and len(y) == 2
        # NOTE: need to parse per-stream output
        if (
            model.prediction_type() == PredictionType.MULTISTREAM_HYBRID
            and y_pad is not None
        ):
            if len(y[0]) == 4:
                mgc, lf0, vuv, bap = y[0]
                if isinstance(mgc, tuple) and len(mgc) == 3:
                    mgc = mdn_get_most_probable_sigma_and_mu(*mgc)[1]
                elif isinstance(mgc, tuple) and len(mgc) == 2:
                    mgc = mgc[1]
                if isinstance(bap, tuple) and len(bap) == 3:
                    bap = mdn_get_most_probable_sigma_and_mu(*bap)[1]
                elif isinstance(bap, tuple) and len(bap) == 2:
                    bap = bap[1]
                if pad != 0:
                    mgc = mgc[:, :-pad] if mgc.shape[1] > x.shape[1] else mgc
                    bap = bap[:, :-pad] if bap.shape[1] > x.shape[1] else bap
                    lf0 = lf0[:, :-pad] if lf0.shape[1] > x.shape[1] else lf0
                    vuv = vuv[:, :-pad] if vuv.shape[1] > x.shape[1] else vuv
                mu = torch.cat([mgc, lf0, vuv, bap], dim=-1)
            elif len(y[0]) == 3:
                mel, lf0, vuv = y[0]
                if isinstance(mel, tuple) and len(mel) == 3:
                    mel = mdn_get_most_probable_sigma_and_mu(*mel)[1]
                elif isinstance(mel, tuple) and len(mel) == 2:
                    mel = mel[1]
                if pad != 0:
                    mel = mel[:, :-pad] if mel.shape[1] > x.shape[1] else mel
                    lf0 = lf0[:, :-pad] if lf0.shape[1] > x.shape[1] else lf0
                    vuv = vuv[:, :-pad] if vuv.shape[1] > x.shape[1] else vuv
                mu = torch.cat([mel, lf0, vuv], dim=-1)
            else:
                raise ValueError("Invalid number of streams: {}".format(len(y)))
            sigma = mu
        else:
            mu, sigma = y
            if pad != 0:
                mu = mu[:, :-pad]
                sigma = sigma[:, :-pad]
        y = (mu, sigma)
    else:
        if model.has_residual_lf0_prediction():
            y = y[0]
        # Multiple output: (out, out_fine)
        if isinstance(y, list):
            y = y[-1]
        if pad != 0:
            y = y[:, :-pad]

    return y
