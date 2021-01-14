from __future__ import annotations
from torch import nn
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce
from torch import Tensor
from glasses.nn.blocks import ConvBnAct, Conv2dPad
from ..resnetxt import ResNetXtBottleNeckBlock
from ..resnet import ReLUInPlace, ResNet, ResNetStemC

class SplitAtt(nn.Module):
    def __init__(self, in_features: int, features: int, radix: int, groups: int):
        """Implementation of Split Attention proposed in `"ResNeSt: Split-Attention Networks" <https://arxiv.org/abs/2004.08955>`_

        Grouped convolution have been proved to be impirically better (ResNetXt). The main idea is to apply an attention group-wise. 

        Einops is used to improve the readibility of this module

        Args:
            in_features (int): number of input features
            features (int): attention's features
            radix (int): number of subgroups (`radix`) in the groups
            groups (int): number of groups, each group contains `radix` subgroups
        """
        super().__init__()
        self.radix, self.groups = radix, groups
        self.att = nn.Sequential(
            # this produces U^{/hat}
            Reduce('b (r k c) h w -> b (k c) h w',
                   reduction='mean', r=radix, k=groups),
            # eq 1
            nn.AdaptiveAvgPool2d(1),
            # the two following conv layers are G in the paper
            ConvBnAct(in_features, features, kernel_size=1,
                      groups=groups, activation=ReLUInPlace, bias=True),
            nn.Conv2d(features, in_features * radix,
                      kernel_size=1, groups=groups),
            Rearrange('b (r k c) h w -> b r (k c) h w', r=radix, k=groups),
            nn.Softmax(dim=1) if radix > 1 else nn.Sigmoid,
            Rearrange('b r (k c) h w -> b (r k c) h w', r=radix, k=groups)
        )

    def forward(self, x: Tensor) -> Tensor:
        att = self.att(x)
        # eq 2, scale using att and sum-up over the radix axis
        x *= att 
        x = reduce(x, 'b (r k c) h w -> b (k c) h w',
                   reduction='mean', r=self.radix, k=self.groups)
        return x


class ResNeStBottleneckBlock(ResNetXtBottleNeckBlock):
    def __init__(self, in_features, out_features, stride: int = 1, radix: int = 2, groups: int = 1,
                fast: bool = True, reduction: int = 4, activation: nn.Module = ReLUInPlace, **kwargs):
        """Implementation of ResNetSt Bottleneck Block proposed in proposed in `"ResNeSt: Split-Attention Networks" <https://arxiv.org/abs/2004.08955>`_. 
        It subclasses `ResNetXtBottleNeckBlock` to use the inner features calculation based on the reduction and groups widths.

        It uses `SplitAttention` after the 3x3 conv and an `AvgPool2d` layer instead of a strided 3x3 convolution to downsample the input.

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNetStBlock.png?raw=true

        Args:
            in_features ([type]): [description]
            out_features ([type]): [description]
            reduction (int, optional): [description]. Defaults to 4.
            stride (int, optional): [description]. Defaults to 1.
            radix (int, optional): [description]. Defaults to 2.
            groups (int, optional): [description]. Defaults to 1.
            fast (bool, optional): If True, the pooling is applied before the 3x3 conv, this improves performance. Defaults to True.
            activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
        """
        super().__init__(in_features, out_features, reduction=reduction,
                         activation=activation, stride=stride, groups=groups, **kwargs)
        att_features = max(self.features * radix // reduction, 32)
        pool = nn.AvgPool2d(kernel_size=3, stride=2,
                            padding=1) if stride == 2 else nn.Identity()
        self.block = nn.Sequential(
            ConvBnAct(in_features, self.features,
                      kernel_size=1, activation=activation),
            pool if fast else nn.Identity(),
            ConvBnAct(self.features, self.features * radix,
                      activation=activation, kernel_size=3, groups=groups * radix),
            SplitAtt(self.features, att_features, radix, groups),
            pool if not fast else nn.Identity(),
            ConvBnAct(self.features, out_features,
                      kernel_size=1, activation=None)
        )

class ResNetSt(ResNet):
    """Implementation of ResNetSt proposed in `"ResNeSt: Split-Attention Networks" <https://arxiv.org/abs/2004.08955>`_. 

    Create a default model

    Examples:
        >>> ResNetSt.resnext50_32x4d()
        >>> ResNetSt.resnext101_32x8d()
        >>> # create a ResNetSt18_32x4d
        >>> ResNetSt.resnet18(block=ResNetStBottleNeckBlock, groups=32, base_width=4)

        You can easily customize your model

        >>> # change activation
        >>> ResNetSt.resnext50_32x4d(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> ResNetSt.resnext50_32x4d(n_classes=100)
        >>> # pass a different block
        >>> ResNetSt.resnext50_32x4d(block=SENetBasicBlock)
        >>> # change the initial convolution
        >>> model = ResNetSt.resnext50_32x4d
        >>> model.encoder.gate.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        >>> # store each feature
        >>> x = torch.rand((1, 3, 224, 224))
        >>> model = ResNetSt.resnext50_32x4d()
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
    # 'resnest14d',
    # 'resnest26d',
    # 'resnest50d',
    # 'resnest50d_1s4x24d',
    # 'resnest50d_4s2x40d',
    # 'resnest101e',
    # 'resnest200e',
    # 'resnest269e
    
    @classmethod
    def resnest14d(cls, *args, **kwargs) -> ResNetSt:
        return ResNet(*args, stem=ResNetStemC, block=ResNeStBottleneckBlock, widths=[256, 512, 1024, 2048],  depths=[1,1,1,1], base_width=64, **kwargs)

    @classmethod
    def resnest26d(cls, *args, **kwargs) -> ResNetSt:
        return cls.resnet26d(*args, block=ResNeStBottleneckBlock, base_width=64, **kwargs)

    @classmethod
    def resnest50d(cls, *args, **kwargs) -> ResNetSt:
        return cls.resnet50d(*args, block=ResNeStBottleneckBlock, base_width=64,  **kwargs)
    
    @classmethod
    def resnest50d_fast(cls, *args, **kwargs) -> ResNetSt:
        return cls.resnet50d(*args, block=ResNeStBottleneckBlock, fast=True, base_width=64, **kwargs)

    @classmethod
    def resnest50d_1s4x24d(cls, *args, **kwargs) -> ResNetSt:
        return cls.resnet50d(*args, block=ResNeStBottleneckBlock, radix=1, groups=4, base_width=24, **kwargs)    

    @classmethod
    def resnest50d_4s2x40d(cls, *args, **kwargs) -> ResNetSt:
        return cls.resnet50d(*args, block=ResNeStBottleneckBlock, radix=4, groups=2, base_width=40, **kwargs)    

    @classmethod
    def resnest101e(cls, *args, **kwargs) -> ResNetSt:
        return cls.resnet101(*args, stem=ResNetStemC, start_features=128, block=ResNeStBottleneckBlock, base_width=64, **kwargs)

    @classmethod
    def resnest200e(cls, *args, **kwargs) -> ResNetSt:
        return cls.resnet200(*args, stem=ResNetStemC, start_features=128, block=ResNeStBottleneckBlock, base_width=64, **kwargs)

    @classmethod
    def resnest269e(cls, *args, **kwargs) -> ResNetSt:
        return cls(*args, stem=ResNetStemC, start_features=128, block=ResNeStBottleneckBlock, depths=[3, 30, 48, 8], base_width=64, **kwargs)
