from .wavenet import WaveNet

__all__ = ["WaveNet", "receptive_field_size"]


def receptive_field_size(
    total_layers, num_cycles, kernel_size, dilation=lambda x: 2 ** x
):
    """Compute receptive field size of WaveNet

    Args:
        total_layers (int): total layers
        num_cycles (int): cycles
        kernel_size (int): kernel size
        dilation (lambda): lambda to compute dilation factor. ``lambda x : 1``
          to disable dilated convolution.

    Returns:
        int: receptive field size in sample

    Examples:

    .. ipython::

        In [1]: from ttslearn.wavenet import receptive_field_size

        In [2]: receptive_field_size(30, 3, 2)
        Out[2]: 3070
    """
    assert total_layers % num_cycles == 0
    layers_per_cycle = total_layers // num_cycles
    dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
    return (kernel_size - 1) * sum(dilations) + 1
