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
