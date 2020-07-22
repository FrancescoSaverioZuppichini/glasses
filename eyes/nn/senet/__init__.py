from ..resnet import ResNetBasicBlock, ResNetBottleNeckBlock, ResNet
from ..se import SEModule

class SENetBasicBlock(ResNetBasicBlock):
    def __init__(self, in_features: int, out_features: int, reduction: int =16, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)
        self.block.add_module('se', SEModule(out_features))
        


class SENetBottleNeckBlock(ResNetBottleNeckBlock):
    def __init__(self, in_features: int, out_features: int, reduction: int =16, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)
        self.block.add_module('se', SEModule(out_features))
        

def se_resnet18(*args, **kwargs) -> ResNet:
    return ResNet(*args, **kwargs, block=SENetBasicBlock, deepths=[2, 2, 2, 2])


def se_resnet34(*args, **kwargs) -> ResNet:
    return ResNet(*args, **kwargs, block=SENetBasicBlock, deepths=[3, 4, 6, 3])


def se_resnet50(*args, **kwargs) -> ResNet:
    return ResNet(*args, **kwargs, block=SENetBottleNeckBlock, deepths=[3, 4, 6, 3])


def se_resnet101(*args, **kwargs) -> ResNet:
    return ResNet(*args, **kwargs, block=SENetBottleNeckBlock, deepths=[3, 4, 23, 3])


def se_resnet152(*args, **kwargs) -> ResNet:
    return ResNet(*args, **kwargs, block=SENetBottleNeckBlock, deepths=[3, 8, 36, 3])
