from __future__ import annotations
from torch import nn
from torch import Tensor
from glasses.nn.blocks.residuals import ResidualAdd
from glasses.nn.blocks import Conv2dPad
from collections import OrderedDict
from typing import List
from functools import partial
from ..resnet import ResNet, ResNetBottleneckBlock
from glasses.utils.PretrainedWeightsProvider import pretrained

ReLUInPlace = partial(nn.ReLU, inplace=True)


class WideResNetBottleNeckBlock(ResNetBottleneckBlock):
    """Wide resnet bottle neck block, you can control the width of the inner features with the width_factor parameter
    Args:
        in_features ([type]): [description]
        out_features ([type]): [description]
        width_factor (int, optional): Scales the 3x3 conv features in the bottle neck block. Defaults to 2.
    """

    def __init__(self, in_features: int, out_features: int, width_factor: int = 2, reduction: int = 4, **kwargs):
        features = int(out_features * width_factor // reduction)
        
        super().__init__(in_features, out_features,
                         features=features,  reduction=reduction, **kwargs)


class WideResNet(ResNet):
    """Implementation of Wide ResNet proposed in `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    Create a default model

    Examples:
        >>> WideResNet.wide_resnet50_2()
        >>> WideResNet.wide_resnet101_2()
        >>> # create a wide_resnet18_4
        >>> WideResNet.resnet18(block=WideResNetBottleNeckBlock, width_factor=4)

        You can easily customize your model

        >>> # change activation
        >>> WideResNet.resnext50_32x4d(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> WideResNet.resnext50_32x4d(n_classes=100)
        >>> # pass a different block
        >>> WideResNet.resnext50_32x4d(block=SENetBasicBlock)
        >>> # change the initial convolution
        >>> model = WideResNet.resnext50_32x4d
        >>> model.encoder.gate.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        >>> # store each feature
        >>> x = torch.rand((1, 3, 224, 224))
        >>> model = WideResNet.wide_resnet50_2()
        >>> features = []
        >>> x = model.encoder.gate(x)
        >>> for block in model.encoder.layers:
        >>>     x = block(x)
        >>>     features.append(x)
        >>> print([x.shape for x in features])
        >>> # [torch.Size([1, 64, 56, 56]), torch.Size([1, 128, 28, 28]), torch.Size([1, 256, 14, 14]), torch.Size([1, 512, 7, 7])]

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    @classmethod
    @pretrained('wide_resnet50_2')
    def wide_resnet50_2(cls, *args, **kwargs) -> WideResNet:
        """Creates a wide_resnet50_2 model

        Returns:
            ResNet: A wide_resnet50_2 model
        """
        return cls.resnet50(*args, **kwargs, block=WideResNetBottleNeckBlock)

    @classmethod
    @pretrained('wide_resnet101_2')
    def wide_resnet101_2(cls, *args, **kwargs) -> WideResNet:
        """Creates a wide_resnet50_2 model

        Returns:
            ResNet: A wide_resnet50_2 model
        """
        return cls.resnet101(*args, **kwargs, block=WideResNetBottleNeckBlock)
