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
from glasses.nn.att import ChannelSE

"""Implementation of RegNet proposed in `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>_`
"""


class RegNetScaler:
    """Generates per stage widths and depths from RegNet parameters. 
        Code borrowed from the original implementation.
    """

    def __call__(self, w_a: float, w_0: float, w_m: float, depth: int, groups_width: int = 8):
        assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % groups_width == 0
        # Generate continuous per-block ws
        ws_cont = np.arange(depth) * w_a + w_0
        # Generate quantized per-block ws
        ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
        ws_all = w_0 * np.power(w_m, ks)
        ws_all = np.round(np.divide(ws_all, groups_width)).astype(int) * groups_width
        # Generate per stage ws and ds (assumes ws_all are sorted)
        ws, ds = np.unique(ws_all, return_counts=True)
        # Convert numpy arrays to lists and return
        ws, ds, ws_all, ws_cont = (x.tolist()
                                   for x in (ws, ds, ws_all, ws_cont))
        return ws, ds


class RegNetXBotteneckBlock(ResNetBottleneckBlock):
    """RegNet modified block from ResNetXt, bottleneck reduction (b in the paper) if fixed to 1.
    """

    def __init__(self, in_features: int, out_features: int, groups_width: int = 1, **kwargs):
        super().__init__(in_features, out_features, reduction=1,
                         groups=out_features // groups_width,  **kwargs)


class RegNetYBotteneckBlock(RegNetXBotteneckBlock):
    """RegNetXBotteneckBlock with Squeeze and Exitation. Differently from SEResNet, The SE module is applied after the 3x3 conv.
    """
    def __init__(self,  in_features: int, out_features: int,  **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        # se is added after the 3x3 conv
        self.block[2] = nn.Sequential(ChannelSE(self.features, reduction=4),
                                      self.block[2],
                                      )


RegNetStem = partial(ConvBnAct, kernel_size=3, stride=2)


class RegNet(ResNet):
    """Implementation of RegNet proposed in `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_

    The main idea is to start with a high dimensional search space and iteratively reduce the search space by empirically apply
    constrains based on the best performing models sampled by the current search space. 

    The resulting models are light, accurate, and faster than EfficientNets (up to 5x times!)
    
    For example, to go from :math:`AnyNet_A` to :math:`AnyNet_B` they fixed the bottleneck ratio :math:`b_i` for all stage :math:`i`. The following table
    shows all the restrictions applied from one search space to the next one.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/RegNetDesignSpaceTable.png?raw=true


    The paper is really well written and very interesting paper, I highly recommended to read it.

    Examples:
        Vanilla models

        >>> ResNet.regnetx_002()
        >>> ResNet.regnetx_004()
        >>> ResNet.regnetx_006()
        >>> ResNet.regnetx_008()
        >>> ResNet.regnetx_016()
        >>> ResNet.regnetx_040()
        >>> ResNet.regnetx_064()
        >>> ResNet.regnetx_080()
        >>> ResNet.regnetx_120()
        >>> ResNet.regnetx_160()
        >>> ResNet.regnetx_320()
        >>> # Y variants (with SE)
        >>> ResNet.regnety_002()
        >>> # ...
        >>> ResNet.regnetx_320()
            
    Customization

    You can easily customize your model

    Examples:
        >>> # change activation
        >>> RegNet.regnetx_004(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> RegNet.regnetx_004(n_classes=100)
        >>> # pass a different block
        >>> RegNet.regnetx_004(block=RegNetYBotteneckBlock)
        >>> # change the steam
        >>> model = RegNet.regnetx_004(stem=ResNetStemC)
        >>> change shortcut
        >>> model = RegNet.regnetx_004(block=partial(RegNetYBotteneckBlock, shortcut=ResNetShorcutD))
        >>> # store each feature
        >>> x = torch.rand((1, 3, 224, 224))
        >>> # get features
        >>> model = RegNet.regnetx_004()
        >>> # first call .features, this will activate the forward hooks and tells the model you'll like to get the features
        >>> model.encoder.features
        >>> model(torch.randn((1,3,224,224)))
        >>> # get the features from the encoder
        >>> features = model.encoder.features
        >>> print([x.shape for x in features])
        >>> #[torch.Size([1, 32, 112, 112]), torch.Size([1, 32, 56, 56]), torch.Size([1, 64, 28, 28]), torch.Size([1, 160, 14, 14])]

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    @classmethod
    def regnetx_002(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetXBotteneckBlock, **kwargs):
        return cls(*args, start_features=start_features, stem=stem, depths=[1, 1, 4, 7], widths=[24, 56, 152, 368], downsample_first=True, block=block, groups_width=8, **kwargs)

    @classmethod
    def regnetx_004(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetXBotteneckBlock, **kwargs):
        return cls(*args, start_features=start_features, stem=stem, depths=[1, 2, 7, 12], widths=[32, 64, 160, 384], downsample_first=True, block=block, groups_width=16, **kwargs)

    @classmethod
    def regnetx_006(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetXBotteneckBlock, **kwargs):
        return cls(*args, start_features=start_features, stem=stem, depths=[1, 3, 5, 7], widths=[48, 96, 240, 528], downsample_first=True, block=block, groups_width=24, **kwargs)

    @classmethod
    def regnetx_008(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetXBotteneckBlock, **kwargs):
        return cls(*args, start_features=start_features, stem=stem, depths=[1, 3, 7, 5], widths=[64, 128, 288, 672], downsample_first=True, block=block, groups_width=16, **kwargs)

    @classmethod
    def regnetx_016(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetXBotteneckBlock, **kwargs):
        return cls(*args, start_features=start_features, stem=stem, depths=[2, 4, 10, 2], widths=[72, 168, 408, 912], downsample_first=True, block=block, groups_width=24, **kwargs)

    @classmethod
    def regnetx_032(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetXBotteneckBlock, **kwargs):
        return cls(*args, start_features=start_features, stem=stem, depths=[2, 6, 15, 2], widths=[96, 192, 432, 1008], downsample_first=True, block=block, groups_width=48, **kwargs)

    @classmethod
    def regnetx_040(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetXBotteneckBlock, **kwargs):
        return cls(*args, start_features=start_features, stem=stem, depths=[2, 5, 14, 2], widths=[80, 240, 560, 1360], downsample_first=True, block=block, groups_width=40, **kwargs)

    @classmethod
    def regnetx_064(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetXBotteneckBlock, **kwargs):
        return cls(*args, start_features=start_features, stem=stem, depths=[2, 4, 10, 1], widths=[168, 392, 784, 1624], downsample_first=True, block=block, groups_width=56, **kwargs)

    @classmethod
    def regnetx_080(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetXBotteneckBlock, **kwargs):
        return cls(*args, start_features=start_features, stem=stem, depths=[2, 5, 15, 1], widths=[80, 240, 720, 1920], downsample_first=True, block=block, groups_width=120, **kwargs)

    @classmethod
    def regnetx_120(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetXBotteneckBlock, **kwargs):
        return cls(*args, start_features=start_features, stem=stem, depths=[2, 5, 11, 1], widths=[224, 448, 896, 2240], downsample_first=True, block=block, groups_width=112, **kwargs)

    @classmethod
    def regnetx_160(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetXBotteneckBlock, **kwargs):
        return cls(*args, start_features=start_features, stem=stem, depths=[2, 6, 13, 1], widths=[256, 512, 896, 2048], downsample_first=True, block=block, groups_width=128, **kwargs)

    @classmethod
    def regnetx_320(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetXBotteneckBlock, **kwargs):
        return cls(*args, start_features=start_features, stem=stem, depths=[2, 7, 13, 1], widths=[336, 672, 1344, 2520], downsample_first=True, block=block, groups_width=168, **kwargs)

    @classmethod
    def regnety_002(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetYBotteneckBlock, **kwargs):
        return cls.regnety_002(*args,  start_features=start_features, stem=RegNetStem, block=block, **kwargs)

    @classmethod
    def regnety_004(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetYBotteneckBlock, **kwargs):
        return cls.regnety_004(*args,  start_features=start_features, stem=RegNetStem, block=block, **kwargs)

    @classmethod
    def regnety_006(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetYBotteneckBlock, **kwargs):
        return cls.regnety_006(*args,  start_features=start_features, stem=RegNetStem, block=block, **kwargs)

    @classmethod
    def regnety_008(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetYBotteneckBlock, **kwargs):
        return cls.regnetx_008(*args,  start_features=start_features, stem=RegNetStem, block=block, **kwargs)

    @classmethod
    def regnety_016(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetYBotteneckBlock, **kwargs):
        return cls.regnety_016(*args,  start_features=start_features, stem=RegNetStem, block=block, **kwargs)

    @classmethod
    def regnety_032(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetYBotteneckBlock, **kwargs):
        return cls.regnety_032(*args,  start_features=start_features, stem=RegNetStem, block=block, **kwargs)

    @classmethod
    def regnety_040(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetYBotteneckBlock, **kwargs):
        return cls.regnety_040(*args,  start_features=start_features, stem=RegNetStem, block=block, **kwargs)

    @classmethod
    def regnety_064(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetYBotteneckBlock, **kwargs):
        return cls.regnety_064(*args,  start_features=start_features, stem=RegNetStem, block=block, **kwargs)

    @classmethod
    def regnety_080(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetYBotteneckBlock, **kwargs):
        return cls.regnety_080(*args,  start_features=start_features, stem=RegNetStem, block=block, **kwargs)

    @classmethod
    def regnety_120(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetYBotteneckBlock, **kwargs):
        return cls.regnety_120(*args,  start_features=start_features, stem=RegNetStem, block=block, **kwargs)

    @classmethod
    def regnety_160(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetYBotteneckBlock, **kwargs):
        return cls.regnety_160(*args,  start_features=start_features, stem=RegNetStem, block=block, **kwargs)

    @classmethod
    def regnety_320(cls, *args,  start_features=32, stem=RegNetStem, block=RegNetYBotteneckBlock, **kwargs):
        return cls.regnety_320(*args,  start_features=start_features, stem=RegNetStem, block=block, **kwargs)
