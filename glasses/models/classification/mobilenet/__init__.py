from __future__ import annotations
from torch import nn
from torch import Tensor
from glasses.nn.blocks.residuals import ResidualAdd, Residual
from glasses.nn.blocks import Conv2dPad, ConvBnAct
from collections import OrderedDict
from typing import List
from functools import partial
from ..efficientnet import EfficientNet
from glasses.utils.PretrainedWeightsProvider import Config, pretrained



class MobileNet(EfficientNet):
    """Implementation of MobileNet v2 proposed in `MobileNetV2: Inverted Residuals and Linear Bottlenecks <https://arxiv.org/pdf/1801.04381.pdf>`_

    MobileNet is a special case of EfficientNet.

    Examples:
    
        Default model

        >>> MobileNet.mobilenet_v2()

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    @classmethod
    def mobilenet_v2(cls, *args, **kwargs) -> EfficientNet:
        return cls(depths=[1, 2, 3, 4, 3, 3, 1],
                   widths=[32, 16, 24, 32, 64, 96, 160, 320, 1280],
                   strides=[2, 1, 2, 2, 2, 1, 2, 1],
                   expansions=[1, 6, 6, 6, 6, 6, 6],
                   kernel_sizes=[3, 3, 3, 3, 3, 3, 3],
                   se=[False, False, False, False, False, False, False],
                   drop_rate=0,
                   activation=nn.ReLU6,  *args, **kwargs)