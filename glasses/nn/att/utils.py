import torch.nn as nn
from .se import SpatialSE


class WithAtt:
    """Utility class that adds an attention module after `.block`.

    :Usage:

        >>> WithAtt(ResNetBottleneckBlock, att=SpatialSE)
        >>> WithAtt(ResNetBottleneckBlock, att=ECA)
        >>> from functools import partial
        >>> WithAtt(ResNetBottleneckBlock, att=partial(SpatialSE, reduction=8))
    """

    def __init__(self, block: nn.Module, att: nn.Module = SpatialSE):
        self.block = block
        self.att = att

    def __call__(
        self, in_features: int, out_features: int, *args, **kwargs
    ) -> nn.Module:
        b = self.block(in_features, out_features, *args, **kwargs)
        b.block.add_module("se", self.att(out_features))
        return b


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v