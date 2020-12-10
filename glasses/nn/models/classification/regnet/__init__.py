from __future__ import annotations
import numpy as np
from torch import nn
from torch import Tensor
from ....blocks.residuals import ResidualAdd
from ....blocks import Conv2dPad, BnActConv, ConvBnAct
from collections import OrderedDict
from typing import List
from functools import partial
from glasses.utils.PretrainedWeightsProvider import Config, pretrained
from ....models.base import Encoder, VisionModule
from ..resnet import ResNet, ResNetBottleneckBlock
from ..resnetxt import ResNetXtBottleNeckBlock

"""Implementation of RegNet proposed in `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>_`
"""

class RegNetScaler:
    """Generates per stage widths and depths from RegNet parameters. 
        Code borrowed from the original implementation.
    """
    
    def __call__(self, w_a: float, w_0: float, w_m: float, depth: int, groups_width: int =8):
        assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
        # Generate continuous per-block ws
        ws_cont = np.arange(d) * w_a + w_0
        # Generate quantized per-block ws
        ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
        ws_all = w_0 * np.power(w_m, ks)
        ws_all = np.round(np.divide(ws_all, q)).astype(int) * q
        # Generate per stage ws and ds (assumes ws_all are sorted)
        ws, ds = np.unique(ws_all, return_counts=True)
        # Convert numpy arrays to lists and return
        ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
        return ws, ds

class RegNetXBotteneckBlock(ResNetBottleneckBlock):

    def __init__(self, in_features: int, out_features: int, groups_width: int = 1, reduction: int = 1, **kwargs):
        super().__init__(in_features, out_features, reduction=reduction, groups= out_features // groups_width,  **kwargs)

RegNetStem = partial(ConvBnAct, kernel_size=3, stride=2)

class RegNet(ResNet):
    """"Implementation of RegNet proposed in `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>_`

    Create a default model

    Examples:

    Vanilla models

        >>> ResNet.resnet18()
        >>> ResNet.resnet26()
        >>> ResNet.resnet34()
        >>> ResNet.resnet50()
        >>> ResNet.resnet101()
        >>> ResNet.resnet152()
        >>> ResNet.resnet200()

    Customization

    You can easily customize your model

    Examples:
        >>> # change activation
        >>> ResNet.resnet18(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> ResNet.resnet18(n_classes=100)
        >>> # pass a different block
        >>> ResNet.resnet18(block=SENetBasicBlock)
        >>> # change the steam
        >>> model = ResNet.resnet18(stem=ResNetStemC)
        >>> change shortcut
        >>> model = ResNet.resnet18(block=partial(ResNetBasicBlock, shortcut=ResNetShorcutD))
        >>> # store each feature
        >>> x = torch.rand((1, 3, 224, 224))
        >>> # get features
        >>> model = ResNet.resnet18()
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
    def regnetx_002(cls, *args, **kwargs):
        return cls(start_features=32, stem=RegNetStem, depths=[1, 1, 4, 7], widths=[24, 56, 152, 368], downsample_first=True, block=RegNetXBotteneckBlock, groups_width=8, **kwargs)

    @classmethod
    def regnetx_004(cls, *args, **kwargs):
        return cls(start_features=32, stem=RegNetStem, depths=[1, 2, 7, 12],widths= [32, 64, 160, 384], downsample_first=True, block=RegNetXBotteneckBlock, groups_width=16, **kwargs)

    @classmethod
    def regnetx_006(cls, *args, **kwargs):
        return cls(start_features=32, stem=RegNetStem, depths=[1, 3, 5, 7], widths=[48, 96, 240, 528], downsample_first=True, block=RegNetXBotteneckBlock, groups_width=24, **kwargs)

    @classmethod
    def regnetx_008(cls, *args, **kwargs):
        return cls(start_features=32, stem=RegNetStem, depths=[1, 3, 7, 5], widths=[64, 128, 288, 672], downsample_first=True, block=RegNetXBotteneckBlock, groups_width=16, **kwargs)

    @classmethod
    def regnetx_016(cls, *args, **kwargs):
        return cls(start_features=32, stem=RegNetStem, depths=[2, 4, 10, 2],widths= [72, 168, 408, 912], downsample_first=True, block=RegNetXBotteneckBlock, groups_width=24, **kwargs)

    @classmethod
    def regnetx_032(cls, *args, **kwargs):
        return cls(start_features=32, stem=RegNetStem, depths=[2, 6, 15, 2],widths= [96, 192, 432, 1008], downsample_first=True, block=RegNetXBotteneckBlock, groups_width=48, **kwargs)

    @classmethod
    def regnetx_040(cls, *args, **kwargs):
        return cls(start_features=32, stem=RegNetStem, depths=[2, 5, 14, 2],widths= [80, 240, 560, 1360], downsample_first=True, block=RegNetXBotteneckBlock, groups_width=40, **kwargs)

    @classmethod
    def regnetx_064(cls, *args, **kwargs):
        return cls(start_features=32, stem=RegNetStem, depths=[2, 4, 10, 1],widths= [168, 392, 784, 1624], downsample_first=True, block=RegNetXBotteneckBlock, groups_width=56, **kwargs)

    @classmethod
    def regnetx_080(cls, *args, **kwargs):
        return cls(start_features=32, stem=RegNetStem, depths=[2, 5, 15, 1],widths= [80, 240, 720, 1920], downsample_first=True, block=RegNetXBotteneckBlock, groups_width=120, **kwargs)

    @classmethod
    def regnetx_120(cls, *args, **kwargs):
        return cls(start_features=32, stem=RegNetStem, depths=[2, 5, 11, 1],widths= [224, 448, 896, 2240], downsample_first=True, block=RegNetXBotteneckBlock, groups_width=112, **kwargs)

    @classmethod
    def regnetx_160(cls, *args, **kwargs):
        return cls(start_features=32, stem=RegNetStem, depths=[2, 6, 13, 1],widths= [256, 512, 896, 2048], downsample_first=True, block=RegNetXBotteneckBlock, groups_width=128, **kwargs)

    @classmethod
    def regnetx_320(cls, *args, **kwargs):
        return cls(start_features=32, stem=RegNetStem, depths=[2, 7, 13, 1], widths= [336, 672, 1344, 2520], downsample_first=True, block=RegNetXBotteneckBlock, groups_width=168, **kwargs)