from __future__ import annotations
from torch import nn
from torch import Tensor
from ....blocks.residuals import ResidualAdd
from ....blocks import Conv2dPad
from collections import OrderedDict
from typing import List
from functools import partial
from ..resnet import ResNet, ResNetBottleneckBlock
from glasses.utils.PretrainedWeightsProvider import Config, pretrained

ReLUInPlace = partial(nn.ReLU, inplace=True)


class ResNetXtBottleNeckBlock(ResNetBottleneckBlock):
    def __init__(self, in_features: int, out_features: int, groups: int = 32, base_width: int = 4, reduction: int = 4, **kwargs):
        """Basic ResNetXt block build on top of ResNetBottleneckBlock. 
        It uses `base_width` to compute the inner features of the 3x3 conv.

        Args:
            in_features (int): [description]
            out_features (int): [description]
            groups (int, optional): [description]. Defaults to 32.
            base_width (int, optional): width factor uses to compute the inner features in the 3x3 conv. Defaults to 4.
        """
        features = (int(out_features * (base_width / 64) / reduction) * groups)
        super().__init__(in_features, out_features,
                         features=features, groups=groups, **kwargs)


class ResNetXt(ResNet):
    """Implementation of ResNetXt proposed in `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Create a default model

    Examples:
        >>> ResNetXt.resnext50_32x4d()
        >>> ResNetXt.resnext101_32x8d()
        >>> # create a resnetxt18_32x4d
        >>> ResNetXt.resnet18(block=ResNetXtBottleNeckBlock, groups=32, base_width=4)

    Customization

    You can easily customize your model

    Examples:
        >>> # change activation
        >>> ResNetXt.resnext50_32x4d(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> ResNetXt.resnext50_32x4d(n_classes=100)
        >>> # pass a different block
        >>> ResNetXt.resnext50_32x4d(block=SENetBasicBlock)
        >>> # change the initial convolution
        >>> model = ResNetXt.resnext50_32x4d
        >>> model.encoder.gate.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        >>> # store each feature
        >>> x = torch.rand((1, 3, 224, 224))
        >>> model = ResNetXt.resnext50_32x4d()
        >>> features = []
        >>> x = model.encoder.gate(x)
        >>> for block in model.encoder.blocks:
        >>>     x = block(x)
        >>>     features.append(x)
        >>> print([x.shape for x in features])
        >>> # [torch.Size([1, 64, 56, 56]), torch.Size([1, 128, 28, 28]), torch.Size([1, 256, 14, 14]), torch.Size([1, 512, 7, 7])]

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    @classmethod
    @pretrained('resnext50_32x4d')
    def resnext50_32x4d(cls, *args, **kwargs) -> ResNetXt:
        """Creates a resnext50_32x4d model

        Returns:
            ResNet: A resnext50_32x4d model
        """
        return cls.resnet50(*args, block=ResNetXtBottleNeckBlock, **kwargs)

    @classmethod
    @pretrained('resnext101_32x8d')
    def resnext101_32x8d(cls, *args, **kwargs) -> ResNetXt:
        """Creates a resnext101_32x8d model

        Returns:
            ResNet: A resnext101_32x8d model
        """
        return cls.resnet101(*args, **kwargs, block=ResNetXtBottleNeckBlock, groups=32, base_width=8)
