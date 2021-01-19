from __future__ import annotations
from ast import parse
from collections import OrderedDict
from torch import nn
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce
from torch import Tensor
from glasses.nn.blocks import ConvBnAct, Conv2dPad, ConvBnDropAct, Lambda
from ..resnetxt import ResNetXtBottleNeckBlock
from ..resnet import ReLUInPlace, ResNet, ResNetStemC, ResNetLayer, ResNetEncoder
from typing import List
from glasses.nn.regularization import DropBlock


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
            Reduce('b r (k c) h w-> b (k c) h w',
                   reduction='sum', r=radix, k=groups),
            # eq 1
            nn.AdaptiveAvgPool2d(1),
            # the two following conv layers are G in the paper
            ConvBnAct(in_features, features, kernel_size=1,
                      groups=groups, activation=ReLUInPlace, bias=True),
            nn.Conv2d(features, in_features * radix,
                      kernel_size=1, groups=groups),
            Rearrange('b (r k c) h w -> b r k c h w', r=radix, k=groups),
            nn.Softmax(dim=1) if radix > 1 else nn.Sigmoid(),
            Rearrange('b r k c h w -> b r (k c) h w', r=radix, k=groups)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, 'b (r k c) h w -> b r (k c) h w', r=self.radix, k=self.groups)
        att = self.att(x)
        # eq 2, scale using att and sum-up over the radix axis
        x *= att
        x = reduce(x, 'b r (k c) h w -> b (k c) h w',
                   reduction='sum', r=self.radix, k=self.groups)
        return x

class ResNeStBottleneckBlock(ResNetXtBottleNeckBlock):
    def __init__(self, in_features: int, out_features: int, stride: int = 1, radix: int = 2, groups: int = 1,
                 fast: bool = True, reduction: int = 4, activation: nn.Module = ReLUInPlace, drop_block_p: float = 0, **kwargs):
        """Implementation of ResNeSt Bottleneck Block proposed in proposed in `"ResNeSt: Split-Attention Networks" <https://arxiv.org/abs/2004.08955>`_. 
        It subclasses `ResNetXtBottleNeckBlock` to use the inner features calculation based on the reduction and groups widths.

        It uses `SplitAttention` after the 3x3 conv and an `AvgPool2d` layer instead of a strided 3x3 convolution to downsample the input.

        DropBlock is added after every convolution operation.

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNeStBlock.jpg?raw=true

        Args:
            in_features (int): [description]
            out_features (int): [description]
            reduction (int, optional): Reduction used in the bottleneck. Defaults to 4.
            stride (int, optional): Stride that was originally applied to the 3x3 conv. Defaults to 1.
            radix (int, optional): Number of subgroups. Defaults to 2.
            groups (int, optional): Number of groups. Defaults to 1.
            fast (bool, optional): If True, the pooling is applied before the 3x3 conv, this improves performance. Defaults to True.
            activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
        """
        super().__init__(in_features, out_features, reduction=reduction,
                         activation=activation, stride=stride, groups=groups, **kwargs)
        att_features = max(self.features * radix // reduction, 32)
        pool = nn.AvgPool2d(kernel_size=3, stride=2,
                            padding=1) if stride == 2 else nn.Identity()
        self.block = nn.Sequential(
            ConvBnDropAct(in_features, self.features, activation=activation,
                          p=drop_block_p, kernel_size=1),
            pool if fast else nn.Identity(),
            ConvBnDropAct(self.features, self.features * radix, activation=activation,
                          p=drop_block_p,  kernel_size=3, groups=groups * radix),
            SplitAtt(self.features, att_features, radix, groups),
            pool if not fast else nn.Identity(),
            ConvBnDropAct(self.features, out_features, activation=activation,
                          p=drop_block_p,  kernel_size=1),
        )


class ResNeStEncoder(ResNetEncoder):
    def __init__(self, *args, start_features: int = 64,  widths: List[int] = [64, 128, 256, 512], depths: List[int] = [2, 2, 2, 2],
                 stem: nn.Module = ResNetStemC, activation: nn.Module = ReLUInPlace, block: nn.Module = ResNeStBottleneckBlock,
                 downsample_first: bool = False, drop_block_p: float = 0.2, **kwargs):

        super().__init__(*args, start_features=start_features, widths=widths, depths=depths, stem=stem,
                         activation=activation, block=block, downsample_first=downsample_first, **kwargs)

        self.layers = nn.ModuleList([
            ResNetLayer(start_features, widths[0], depth=depths[0], activation=activation,
                        block=block, stride=2 if downsample_first else 1, **kwargs),
            *[ResNetLayer(in_features,
                          out_features, depth=n, activation=activation,
                          block=block, 
                          # add drop block in the last two stages
                          drop_block_p=0 if i < 1 else drop_block_p, **kwargs)
              for i, ((in_features, out_features), n) in enumerate(zip(self.in_out_widths, depths[1:]))]
        ])


class ResNeSt(ResNet):
    """Implementation of ResNeSt proposed in `"ResNeSt: Split-Attention Networks" <https://arxiv.org/abs/2004.08955>`_. 

    This model beats EfficientNet both in speed and accuracy.

    Create a default model

    Examples:
        >>> ResNeSt.resnest14d()
        >>> ResNeSt.resnest26d()
        >>> ResNeSt.resnest50d()
        >>> ResNeSt.resnest50d_1s4x24d()
        >>> ResNeSt.resnest50d_4s2x40d()
        >>> # 'e' models have a bigger start_features (128), resulting in a 64 stem width
        >>> ResNeSt.resnest101e()
        >>> ResNeSt.resnest200e()
        >>> ResNeSt.resnest269e()
        >>> # create a ResNeSt50_2s4s40d 
        >>> ResNeSt.resnet50d(radix=2, groups=4, base_width=80)

        You can easily customize your model

        >>> # change activation
        >>> ResNeSt.resnest50d(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> ResNeSt.resnest50d(n_classes=100)
        >>> # pass a different block
        >>> ResNeSt.resnest50d(block=SENetBasicBlock)
        >>> # store each feature
        >>> x = torch.rand((1, 3, 224, 224))
        >>> model = ResNeSt.resnest50d()
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

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__(in_channels, *args, n_classes=n_classes, **kwargs)
        self.encoder = ResNeStEncoder(in_channels, *args, **kwargs)

    @classmethod
    def resnest14d(cls, *args, **kwargs) -> ResNeSt:
        return cls(*args, stem=ResNetStemC, block=ResNeStBottleneckBlock, widths=[256, 512, 1024, 2048],  depths=[1, 1, 1, 1], base_width=64, **kwargs)

    @classmethod
    def resnest26d(cls, *args, **kwargs) -> ResNeSt:
        return cls.resnet26d(*args, block=ResNeStBottleneckBlock, base_width=64, **kwargs)

    @classmethod
    def resnest50d(cls, *args, **kwargs) -> ResNeSt:
        return cls.resnet50d(*args, block=ResNeStBottleneckBlock, base_width=64,  **kwargs)

    @classmethod
    def resnest50d_fast(cls, *args, **kwargs) -> ResNeSt:
        return cls.resnet50d(*args, block=ResNeStBottleneckBlock, fast=True, base_width=64, **kwargs)

    @classmethod
    def resnest50d_1s4x24d(cls, *args, **kwargs) -> ResNeSt:
        return cls.resnet50d(*args, block=ResNeStBottleneckBlock, radix=1, groups=4, base_width=24, **kwargs)

    @classmethod
    def resnest50d_4s2x40d(cls, *args, **kwargs) -> ResNeSt:
        return cls.resnet50d(*args, block=ResNeStBottleneckBlock, radix=4, groups=2, base_width=40, **kwargs)

    @classmethod
    def resnest101e(cls, *args, **kwargs) -> ResNeSt:
        return cls.resnet101(*args, stem=ResNetStemC, start_features=128, block=ResNeStBottleneckBlock, base_width=64, **kwargs)

    @classmethod
    def resnest200e(cls, *args, **kwargs) -> ResNeSt:
        return cls.resnet200(*args, stem=ResNetStemC, start_features=128, block=ResNeStBottleneckBlock, base_width=64, **kwargs)

    @classmethod
    def resnest269e(cls, *args, **kwargs) -> ResNeSt:
        return cls(*args, stem=ResNetStemC, start_features=128, block=ResNeStBottleneckBlock, depths=[3, 30, 48, 8], base_width=64, **kwargs)
