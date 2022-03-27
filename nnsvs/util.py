import importlib
import random
from os.path import join
from typing import Any

import numpy as np
import pkg_resources
import torch

# mask-related functions were adapted from https://github.com/espnet/espnet

EXAMPLE_DIR = "_example_data"


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


class StandardScaler:
    """sklearn.preprocess.StandardScaler like class with only
    transform functionality

    Args:
        mean (np.ndarray): mean
        std (np.ndarray): standard deviation
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
        feature_range (tuple): (min, max)
    """

    def __init__(self, min, scale, feature_range=(0, 1)):
        self.min_ = min
        self.scale_ = scale
        self.feature_range = feature_range

    def transform(self, x):
        return self.scale_ * x + self.min_

    def inverse_transform(self, x):
        return (x - self.min_) / self.scale_
