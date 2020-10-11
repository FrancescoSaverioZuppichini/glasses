from __future__ import annotations
from ..resnet import ResNetBasicBlock, ResNetBottleneckBlock, ResNet
from ..se import SpatialSE


class SENetBasicBlock(ResNetBasicBlock):
    def __init__(self, in_features: int, out_features: int, squeeze: int = 16, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)
        self.block.block.add_module('se', SpatialSE(out_features, reduction=squeeze))


class SENetBottleneckBlock(ResNetBottleneckBlock):
    def __init__(self, in_features: int, out_features: int, squeeze: int = 16, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)
        self.block.block.add_module('se', SpatialSE(out_features, reduction=squeeze))


class SEResNet(ResNet):

    @classmethod
    def resnet18(cls, *args, **kwargs) -> SEResNet:
        return ResNet.resnet18(*args, block=SENetBasicBlock, **kwargs)

    @classmethod
    def resnet34(cls, *args, **kwargs) -> SEResNet:
        return ResNet.resnet34(*args, block=SENetBasicBlock, **kwargs)

    @classmethod
    def resnet50(cls, *args, **kwargs) -> SEResNet:
        return ResNet.resnet50(*args, block=SENetBottleneckBlock, **kwargs)

    @classmethod
    def resnet101(cls, *args, **kwargs) -> SEResNet:
        return ResNet.resnet101(*args, block=SENetBottleneckBlock, **kwargs)

    @classmethod
    def resnet152(cls, *args, **kwargs) -> SEResNet:
        return ResNet.resnet152(*args, block=SENetBottleneckBlock, **kwargs)
