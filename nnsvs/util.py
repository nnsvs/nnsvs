import importlib
import random
from os.path import join
from typing import Any

import numpy as np
import pkg_resources
import pyworld
import torch
from torch import nn

# mask-related functions were adapted from https://github.com/espnet/espnet

EXAMPLE_DIR = "_example_data"


# Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Args:
        net (torch.nn.Module): network to initialize
        init_type (str): the name of an initialization method:
            normal | xavier | kaiming | orthogonal | none.
        init_gain (float): scaling factor for normal, xavier and orthogonal.
    """
    if init_type == "none":
        return

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def get_world_stream_info(sr, mgc_order, num_windows=3, vibrato_mode="none"):
    """Get stream sizes for WORLD-based acoustic features

    Args:
        sr (int): sampling rate
        mgc_order (int): order of mel-generalized cepstrum
        num_windows (int): number of windows
        vibrato_mode (str): vibrato analysis mode

    Returns:
        tuple: stream sizes and flags for dynamic features
    """
    # [mgc, lf0, vuv, bap]
    stream_sizes = [
        (mgc_order + 1) * num_windows,
        num_windows,
        1,
        pyworld.get_num_aperiodicities(sr) * 3,
    ]
    has_dynamic_features = [True, True, False, True]
    if vibrato_mode == "diff":
        # vib
        stream_sizes.append(num_windows)
        has_dynamic_features.append(True)
    elif vibrato_mode == "sine":
        # vib + vib_flags
        stream_sizes.append(3 * num_windows)
        has_dynamic_features.append(True)
        stream_sizes.append(1)
        has_dynamic_features.append(False)
    elif vibrato_mode == "none":
        pass
    else:
        raise RuntimeError("Unknown vibrato mode: {}".format(vibrato_mode))

    return stream_sizes, has_dynamic_features


def load_utt_list(utt_list):
    """Load a list of utterances.

    Args:
        utt_list (str): path to a file containing a list of utterances

    Returns:
        List[str]: list of utterances
    """
    with open(utt_list) as f:
        utt_ids = f.readlines()
    utt_ids = map(lambda utt_id: utt_id.strip(), utt_ids)
    utt_ids = filter(lambda utt_id: len(utt_id) > 0, utt_ids)
    return list(utt_ids)


def example_xml_file(key="haruga_kita"):
    """Get the path to an included xml file.

    Args:
        key (str): key of the file

    Returns:
        str: path to an example xml file

    Raises:
        FileNotFoundError: if the file is not found
    """
    return pkg_resources.resource_filename(__name__, join(EXAMPLE_DIR, f"{key}.xml"))


def init_seed(seed):
    """Initialize random seed.

    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dynamic_import(name: str) -> Any:
    """Dynamic import

    Args:
        name (str): module_name + ":" + class_name

    Returns:
        Any: class object
    """
    mod_name, class_name = name.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, class_name)


def pad_2d(x, max_len, constant_values=0):
    """Pad a 2d-tensor.

    Args:
        x (torch.Tensor): tensor to pad
        max_len (int): maximum length of the tensor
        constant_values (int, optional): value to pad with. Default: 0

    Returns:
        torch.Tensor: padded tensor
    """
    x = np.pad(
        x,
        [(0, max_len - len(x)), (0, 0)],
        mode="constant",
        constant_values=constant_values,
    )
    return x


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor. If set, masks will be
            the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    """
    return ~make_pad_mask(lengths, xs, length_dim, maxlen)


class PyTorchStandardScaler(nn.Module):
    """PyTorch module for standardization.

    Args:
        mean (torch.Tensor): mean
        scale (torch.Tensor): scale
    """

    def __init__(self, mean, scale):
        super().__init__()
        self.mean_ = nn.Parameter(mean, requires_grad=False)
        self.scale_ = nn.Parameter(scale, requires_grad=False)

    def transform(self, x):
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x):
        return x * self.scale_ + self.mean_


class StandardScaler:
    """sklearn.preprocess.StandardScaler like class with only
    transform functionality

    Args:
        mean (np.ndarray): mean
        var (np.ndarray): variance
        scale (np.ndarray): scale
    """

    def __init__(self, mean, var, scale):
        self.mean_ = mean
        self.var_ = var
        # NOTE: scale may not exactly same as np.sqrt(var)
        self.scale_ = scale

    def transform(self, x):
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x):
        return x * self.scale_ + self.mean_


class MinMaxScaler:
    """sklearn.preprocess.MinMaxScaler like class with only
    transform functionality

    Args:
        min (np.ndarray): minimum
        scale (np.ndarray): scale
        data_min (np.ndarray): minimum of input data
        data_max (np.ndarray): maximum of input data
        feature_range (tuple): (min, max)
    """

    def __init__(self, min, scale, data_min=None, data_max=None, feature_range=(0, 1)):
        self.min_ = min
        self.scale_ = scale
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.feature_range = feature_range

    def transform(self, x):
        return self.scale_ * x + self.min_

    def inverse_transform(self, x):
        return (x - self.min_) / self.scale_
