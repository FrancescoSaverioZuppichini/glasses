from __future__ import annotations
from torch import nn
from torch import Tensor
from glasses.nn.blocks.residuals import ResidualAdd
from glasses.nn.blocks import Conv2dPad
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
            in_features (int): Number of input features
            out_features (int): Number of output features
            groups (int, optional): [description]. Defaults to 32.
            base_width (int, optional): width factor uses to compute the inner features in the 3x3 conv. Defaults to 4.
        """
        self.features = (int(out_features * (base_width / 64) / reduction) * groups)
        super().__init__(in_features, out_features,
                         features=self.features, groups=groups, reduction=reduction, **kwargs)


class ResNetXt(ResNet):
    """Implementation of ResNetXt proposed in `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Create a default model

    Examples:
        >>> ResNetXt.resnext50_32x4d()
        >>> ResNetXt.resnext101_32x8d()
        >>> # create a resnetxt18_32x4d
        >>> ResNetXt.resnet18(block=ResNetXtBottleNeckBlock, groups=32, base_width=4)

        You can easily customize your model

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
        >>> # first call .features, this will activate the forward hooks and tells the model you'll like to get the features
        >>> model.encoder.features
        >>> model(torch.randn((1,3,224,224)))
        >>> # get the features from the encoder
        >>> features = model.encoder.features
        >>> print([x.shape for x in features])
        >>> #[torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 128, 28, 28]), torch.Size([1, 256, 14, 14])]

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


    @classmethod
    # @pretrained('resnext101_32x16d')
    def resnext101_32x16d(cls, *args, **kwargs) -> ResNetXt:
        """Creates a resnext101_32x16d model

        Returns:
            ResNet: A resnext101_32x16d model
        """
        return cls.resnet101(*args, **kwargs, block=ResNetXtBottleNeckBlock, groups=32, base_width=16)


    @classmethod
    # @pretrained('resnext101_32x32d')
    def resnext101_32x32d(cls, *args, **kwargs) -> ResNetXt:
        """Creates a resnext101_32x32d model

        Returns:
            ResNet: A resnext101_32x32d model
        """
        return cls.resnet101(*args, **kwargs, block=ResNetXtBottleNeckBlock, groups=32, base_width=32)

    @classmethod
    # @pretrained('resnext101_32x48d')
    def resnext101_32x48d(cls, *args, **kwargs) -> ResNetXt:
        """Creates a resnext101_32x48d model

        Returns:
            ResNet: A resnext101_32x48d model
        """
        return cls.resnet101(*args, **kwargs, block=ResNetXtBottleNeckBlock, groups=32, base_width=48)

