from __future__ import annotations
from torch import nn
from ..resnet import ResNetBasicBlock, ResNetBottleneckBlock, ResNet
from glasses.nn.att import SpatialSE, WithAtt


SENetBasicBlock = WithAtt(ResNetBasicBlock, att=SpatialSE)
SENetBottleneckBlock = WithAtt(ResNetBottleneckBlock, att=SpatialSE)


class SEResNet(ResNet):
    """Implementation of Squeeze and Excitation ResNet using booth the original spatial se
    and the channel se proposed in
    `Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks <https://arxiv.org/abs/1803.02579>`_
    The models with the channel se are labeldb with prefix `c`
    """

    @classmethod
    def se_resnet18(cls, *args, **kwargs) -> ResNet:
        """Original SE resnet18 with Spatial Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet18(*args, block=SENetBasicBlock, **kwargs)

    @classmethod
    def se_resnet34(cls, *args, **kwargs) -> ResNet:
        """Original SE resnet34 with Spatial Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet34(*args, block=SENetBasicBlock, **kwargs)

    @classmethod
    def se_resnet50(cls, *args, **kwargs) -> ResNet:
        """Original SE resnet50 with Spatial Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet50(*args, block=SENetBottleneckBlock, **kwargs)

    @classmethod
    def se_resnet101(cls, *args, **kwargs) -> ResNet:
        """Original SE resnet101 with Spatial Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet101(*args, block=SENetBottleneckBlock, **kwargs)

    @classmethod
    def se_resnet152(cls, *args, **kwargs) -> ResNet:
        """Original SE resnet152 with Spatial Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet152(*args, block=SENetBottleneckBlock, **kwargs)
