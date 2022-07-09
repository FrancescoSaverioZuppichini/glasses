import torch
from typing import List
from torch import nn
from torch import Tensor
from math import sqrt


class SpatialPyramidPool(nn.Module):
    """Implementation of `Spatial Pyramid Pooling in Deep Convolutional Networks  for Visual Recognition <https://arxiv.org/pdf/1406.4729.pdf>`_


    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/SPP.png?raw=true

    It generate fixed length representation regardless of image dimensions.

    Examples:
        >>> x = torch.randn((4, 256, 14, 14))
        >>> SpatialPyramidPool()(x).shape
        >>> # torch.Size([4, 256, 21])

    Args:
        num_pools (List[int], optional): The number of pooling output size. Defaults to [1, 4, 16].
        pool (nn.Module, optional): The pooling layer. Defaults to nn.AdaptiveMaxPool2d.
    """

    def __init__(
        self, num_pools: List[int] = None, pool: nn.Module = nn.AdaptiveMaxPool2d
    ):
        super().__init__()

        if num_pools is None:
            num_pools = [1, 4, 16]

        self.pools = nn.ModuleList([])
        for p in num_pools:
            # the output of 2d pool is B X C X P X P
            # since we want  B X C X P features, we have to
            # set an output size of sqrt(P)
            output_size = int(sqrt(p))
            self.pools.append(pool(output_size))

    def forward(self, x: Tensor) -> Tensor:
        pooled = []
        for p in self.pools:
            pooled.append(p(x).flatten(2))
        return torch.cat(pooled, dim=2)
