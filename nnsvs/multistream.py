# Utils for multi-stream features

import numpy as np
import torch
from nnmnkwii import paramgen


def get_windows(num_window=1):
    """Get windows for MLPG.

    Args:
        num_window (int): number of windows

    Returns:
        list: list of windows
    """
    windows = [(0, 0, np.array([1.0]))]
    if num_window >= 2:
        windows.append((1, 1, np.array([-0.5, 0.0, 0.5])))
    if num_window >= 3:
        windows.append((1, 1, np.array([1.0, -2.0, 1.0])))

    if num_window >= 4:
        raise ValueError(f"Not supported num windows: {num_window}")

    return windows


def select_streams(
    inputs,
    stream_sizes=None,
    streams=None,
    concat=True,
):
    """Select streams from multi-stream features

    Args:
        inputs (array like): input 3-d or 2-d array
        stream_sizes (list): stream sizes
        streams (list): Streams of interests. Returns all streams if streams is None.
        concat (bool): Concatenate streams. Defaults to True.

    Returns:
        array like: selected streams
    """
    if stream_sizes is None:
        stream_sizes = [60, 1, 1, 1]
    if streams is None:
        streams = [True] * len(stream_sizes)
    ret = []
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    for start_idx, size, enabled in zip(start_indices, stream_sizes, streams):
        if not enabled:
            continue
        if len(inputs.shape) == 3:
            s = inputs[:, :, start_idx : start_idx + size]
        else:
            s = inputs[:, start_idx : start_idx + size]
        ret.append(s)

    if not concat:
        return ret

    if isinstance(inputs, torch.Tensor):
        return torch.cat(ret, dim=-1)
    else:
        return np.concatenate(ret, -1)


def split_streams(inputs, stream_sizes=None):
    """Split streams from multi-stream features

    Args:
        inputs (array like): input 3-d array
        stream_sizes (list): sizes for each stream

    Returns:
        list: list of stream features
    """
    if stream_sizes is None:
        stream_sizes = [60, 1, 1, 1]
    ret = []
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    for start_idx, size in zip(start_indices, stream_sizes):
        if len(inputs.shape) == 3:
            s = inputs[:, :, start_idx : start_idx + size]
        else:
            s = inputs[:, start_idx : start_idx + size]
        ret.append(s)

    return ret


def get_static_stream_sizes(stream_sizes, has_dynamic_features, num_windows):
    """Get stream sizes for static features

    Args:
        inputs (array like): input 3-d or 2-d array
        num_windows (int): number of windows
        stream_sizes (list): stream sizes
        has_dynamic_features (list): binary flags that indicates if steams have dynamic features
        streams (list, optional): Streams of interests. Returns all streams if streams is None.
            Defaults to None.

    Returns:
        list: stream sizes
    """
    static_stream_sizes = np.array(stream_sizes)
    static_stream_sizes[has_dynamic_features] = (
        static_stream_sizes[has_dynamic_features] / num_windows
    )

    return static_stream_sizes


def get_static_features(
    inputs,
    num_windows,
    stream_sizes=None,
    has_dynamic_features=None,
    streams=None,
):
    """Get static features from static+dynamic features

    Args:
        inputs (array like): input 3-d or 2-d array
        num_windows (int): number of windows
        stream_sizes (list): stream sizes
        has_dynamic_features (list): binary flags that indicates if steams have dynamic features
        streams (list, optional): Streams of interests. Returns all streams if streams is None.
            Defaults to None.

    Returns:
        list: list of static features
    """
    if stream_sizes is None:
        stream_sizes = [180, 3, 1, 15]
    if has_dynamic_features is None:
        has_dynamic_features = [True, True, False, True]
    if streams is None:
        streams = [True] * len(stream_sizes)
    _, _, D = inputs.shape
    if stream_sizes is None or (len(stream_sizes) == 1 and has_dynamic_features[0]):
        return inputs[:, :, : D // num_windows]
    if len(stream_sizes) == 1 and not has_dynamic_features[0]:
        return inputs

    # Multi stream case
    ret = []
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    for start_idx, size, v, enabled in zip(
        start_indices, stream_sizes, has_dynamic_features, streams
    ):
        start_idx = int(start_idx)
        size = int(size)
        if not enabled:
            continue
        if v:
            static_features = inputs[:, :, start_idx : start_idx + size // num_windows]
        else:
            static_features = inputs[:, :, start_idx : start_idx + size]
        ret.append(static_features)
    return ret


def multi_stream_mlpg(
    inputs,
    variances,
    windows,
    stream_sizes=None,
    has_dynamic_features=None,
    streams=None,
):
    """Split streams and do apply MLPG if stream has dynamic features

    Args:
        inputs (array like): input 3-d or 2-d array
        variances (array like): variances of input features
        windows (list): windows for parameter generation
        stream_sizes (list): stream sizes
        has_dynamic_features (list): binary flags that indicates if steams have dynamic features
        streams (list, optional): Streams of interests. Returns all streams if streams is None.
            Defaults to None.

    Raises:
        RuntimeError: if stream sizes are wrong

    Returns:
        array like: generated static features
    """
    if stream_sizes is None:
        stream_sizes = [180, 3, 1, 3]
    if has_dynamic_features is None:
        has_dynamic_features = [True, True, False, True]
    if streams is None:
        streams = [True] * len(stream_sizes)
    T, D = inputs.shape
    if D != sum(stream_sizes):
        raise RuntimeError("You probably have specified wrong dimension params.")

    # Straem indices for static+delta features
    # [0,   180, 183, 184]
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    # [180, 183, 184, 199]
    end_indices = np.cumsum(stream_sizes)

    ret = []
    for in_start_idx, in_end_idx, v, enabled in zip(
        start_indices,
        end_indices,
        has_dynamic_features,
        streams,
    ):
        if not enabled:
            continue
        x = inputs[:, in_start_idx:in_end_idx]
        if inputs.shape == variances.shape:
            var_ = variances[:, in_start_idx:in_end_idx]
        else:
            var_ = np.tile(variances[in_start_idx:in_end_idx], (T, 1))
        y = paramgen.mlpg(x, var_, windows) if v else x
        ret.append(y)

    return np.concatenate(ret, -1)
