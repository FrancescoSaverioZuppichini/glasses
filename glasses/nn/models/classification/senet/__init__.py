from __future__ import annotations
from torch import nn
from ..resnet import ResNetBasicBlock, ResNetBottleneckBlock, ResNet
from ..se import SpatialSE, ChannelSE
from glasses.utils.PretrainedWeightsProvider import Config

class WithSE:
    def __init__(self, block: nn.Module, se: nn.Module = SpatialSE):
        self.block = block
        self.se = se

    def __call__(self, in_features: int, out_features: int, squeeze: int = 16, *args, **kwargs) -> nn.Module:
        b = self.block(in_features, out_features, *args, **kwargs)
        b.block.add_module('se', self.se(out_features, reduction=squeeze))
        return b

SENetBasicBlock = WithSE(ResNetBasicBlock, se=SpatialSE)
SENetBottleneckBlock = WithSE(ResNetBottleneckBlock, se=SpatialSE)

CSENetBasicBlock = WithSE(ResNetBasicBlock, se=ChannelSE)
CSENetBottleneckBlock = WithSE(ResNetBottleneckBlock, se=ChannelSE)

class SEResNet(ResNet):
    """Implementation of Squeeze and Excitation ResNet using booth the original spatial se 
    and the channel se proposed in  
    `Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks <https://arxiv.org/abs/1803.02579>`_
    The models with the channel se are labelab with prefix `c` 

    Args:
        ResNet ([type]): [description]

    Returns:
        [type]: [description]
    """

    configs = {
        'se_resnet18': Config(),
        'se_resnet34': Config(),
        'se_resnet50': Config(),
        'se_resnet101': Config(),
        'se_resnet152': Config(),
        'se_resnet200': Config(),
        'cse_resnet18': Config(),
        'cse_resnet34': Config(),
        'cse_resnet50': Config(interpolation='bicubic'),
        'cse_resnet101': Config(),
        'cse_resnet152': Config(),
        'cse_resnet200': Config()
    }

    @classmethod
    def se_resnet18(cls, *args, **kwargs) -> SEResNet:
        """Original SE resnet18 with Spatial Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet18(*args, block=SENetBasicBlock, **kwargs)

    @classmethod
    def se_resnet34(cls, *args, **kwargs) -> SEResNet:
        """Original SE resnet34 with Spatial Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet34(*args, block=SENetBasicBlock, **kwargs)

    @classmethod
    def se_resnet50(cls, *args, **kwargs) -> SEResNet:
        """Original SE resnet50 with Spatial Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet50(*args, block=SENetBottleneckBlock, **kwargs)

    @classmethod
    def se_resnet101(cls, *args, **kwargs) -> SEResNet:
        """Original SE resnet101 with Spatial Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet101(*args, block=SENetBottleneckBlock, **kwargs)

    @classmethod
    def se_resnet152(cls, *args, **kwargs) -> SEResNet:
        """Original SE resnet152 with Spatial Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet152(*args, block=SENetBottleneckBlock, **kwargs)

    @classmethod
    def cse_resnet18(cls, *args, **kwargs) -> SEResNet:
        """SE resnet18 with Channel Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet18(*args, block=CSENetBasicBlock, **kwargs)

    @classmethod
    def cse_resnet34(cls, *args, **kwargs) -> SEResNet:
        """SE resnet34 with Channel Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet34(*args, block=CSENetBasicBlock, **kwargs)

    @classmethod
    def cse_resnet50(cls, *args, **kwargs) -> SEResNet:
        """SE resnet50 with Channel Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet50(*args, block=CSENetBottleneckBlock, **kwargs)

    @classmethod
    def cse_resnet101(cls, *args, **kwargs) -> SEResNet:
        """SE resnet101 with Channel Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet101(*args, block=CSENetBottleneckBlock, **kwargs)

    @classmethod
    def cse_resnet152(cls, *args, **kwargs) -> SEResNet:
        """SE resnet152 with Channel Squeeze and Excitation

        Returns:
            SEResNet: [description]
        """
        return ResNet.resnet152(*args, block=CSENetBottleneckBlock, **kwargs)

