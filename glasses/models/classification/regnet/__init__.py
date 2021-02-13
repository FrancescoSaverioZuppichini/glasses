from __future__ import annotations
import numpy as np
from torch import nn
from torch import Tensor
from glasses.nn.blocks.residuals import ResidualAdd
from glasses.nn.blocks import Conv2dPad, BnActConv, ConvBnAct
from collections import OrderedDict
from typing import List
from functools import partial
from glasses.utils.PretrainedWeightsProvider import Config, pretrained
from ....models.base import Encoder, VisionModule
from ..resnet import ResNet, ResNetEncoder, ResNetBottleneckBlock
from ..resnetxt import ResNetXtBottleNeckBlock
from glasses.nn.att import ChannelSE

"""Implementation of RegNet proposed in `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>_`
"""

class RegNetScaler:
    """Generates per stage widths and depths from RegNet parameters. 
        Code borrowed from the original implementation.
    """

    def adjust_block_compatibility(self, ws, bs, gs):
        """Adjusts the compatibility of widths, bottlenecks, and groups."""
        assert len(ws) == len(bs) == len(gs)
        assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
        vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
        gs = [int(min(g, v)) for g, v in zip(gs, vs)]
        ms = [np.lcm(g, b) if b > 1 else g for g, b in zip(gs, bs)]
        vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
        ws = [int(v / b) for v, b in zip(vs, bs)]
        assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
        return ws, bs, gs

    def __call__(self, w_0: float, w_a: float,  w_m: float, depth: int,  group_w: int, q: int = 8):
        assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
        # Generate continuous per-block ws
        ws_cont = np.arange(depth) * w_a + w_0
        # Generate quantized per-block ws
        ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
        ws_all = w_0 * np.power(w_m, ks)
        ws_all = np.round(np.divide(ws_all, q)).astype(int) * q
        # Generate per stage ws and ds (assumes ws_all are sorted)
        ws, ds = np.unique(ws_all, return_counts=True)
        # Convert numpy arrays to lists and return
        ws, ds, ws_all, ws_cont = (x.tolist()
                                   for x in (ws, ds, ws_all, ws_cont))

        bs = [1] * len(ws)
        gs = [group_w] * len(ws)
        ws, _, gs = self.adjust_block_compatibility(ws, bs, gs)
        return ds, ws, gs[0]


class RegNetXBotteneckBlock(ResNetBottleneckBlock):
    """RegNet modified block from ResNetXt, bottleneck reduction (b in the paper) if fixed to 1.
    """

    def __init__(self, in_features: int, out_features: int, groups_width: int = 1, **kwargs):
        super().__init__(in_features, out_features, reduction=1,
                         groups=out_features // groups_width,  **kwargs)


class RegNetYBotteneckBlock(RegNetXBotteneckBlock):
    """RegNetXBotteneckBlock with Squeeze and Exitation. Differently from SEResNet, The SE module is applied after the 3x3 conv.

    .. note::
        This block is wrong but it follows the official doc where the inner features of the SE module are `in_features // reduction`,
        a correct implementation of SE should have the inner features computed from `self.features`.
        """

    def __init__(self,  in_features: int, out_features: int,  reduction: int = 4, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        # se is added after the 3x3 conv
        self.block[2] = nn.Sequential(ChannelSE(self.features, reduced_features=in_features // reduction),
                                      self.block[2],
                                      )


RegNetStem = partial(ConvBnAct, kernel_size=3, stride=2)
"""RegNet's Stem, just a Conv-Bn-Act with `stride=2`
"""

RegNetEncoder = partial(ResNetEncoder, start_features=32, stem=RegNetStem, downsample_first=True,
                        block=RegNetXBotteneckBlock)
"""RegNet's Encoder is a variant of the `ResNet` encoder. The original stem is replaced by `RegNetStem`, 
the first layer also applies a `stride=2` and the starting features are always set to 32.
"""


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

        Default models

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

        You can easily customize your model

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
    # depths, widths, group with
    models_config = {'regnetx_002': ([1, 1, 4, 7], [24, 56, 152, 368], 8),
                     'regnetx_004': ([1, 2, 7, 12], [32, 64, 160, 384], 16),
                     'regnetx_006': ([1, 3, 5, 7], [48, 96, 240, 528], 24),
                     'regnetx_008': ([1, 3, 7, 5], [64, 128, 288, 672], 16),
                     'regnetx_016': ([2, 4, 10, 2], [72, 168, 408, 912], 24),
                     'regnetx_032': ([2, 6, 15, 2], [96, 192, 432, 1008], 48),
                     'regnetx_040': ([2, 5, 14, 2], [80, 240, 560, 1360], 40),
                     'regnetx_064': ([2, 4, 10, 1], [168, 392, 784, 1624], 56),
                     'regnetx_080': ([2, 5, 15, 1], [80, 240, 720, 1920], 80),
                     'regnetx_120': ([2, 5, 11, 1], [224, 448, 896, 2240], 112),
                     'regnetx_160': ([2, 6, 13, 1], [256, 512, 896, 2048], 128),
                     'regnetx_320': ([2, 7, 13, 1], [336, 672, 1344, 2520], 168),
                     'regnety_002': ([1, 1, 4, 7], [24, 56, 152, 368], 8),
                     'regnety_004': ([1, 3, 6, 6], [48, 104, 208, 440], 8),
                     'regnety_006': ([1, 3, 7, 4], [48, 112, 256, 608], 16),
                     'regnety_008': ([1, 3, 8, 2], [64, 128, 320, 768], 16),
                     'regnety_016': ([2, 6, 17, 2], [48, 120, 336, 888], 24),
                     'regnety_032': ([2, 5, 13, 1], [72, 216, 576, 1512], 24),
                     'regnety_040': ([2, 6, 12, 2], [128, 192, 512, 1088], 64),
                     'regnety_064': ([2, 7, 14, 2], [144, 288, 576, 1296], 72),
                     'regnety_080': ([2, 4, 10, 1], [168, 448, 896, 2016], 56),
                     'regnety_120': ([2, 5, 11, 1], [224, 448, 896, 2240], 112),
                     'regnety_160': ([2, 4, 11, 1], [224, 448, 1232, 3024], 112),
                     'regnety_320': ([2, 5, 12, 1], [232, 696, 1392, 3712], 232)
                     }

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__(in_channels, n_classes, widths=kwargs['widths'])
        self.encoder = RegNetEncoder(in_channels, *args, **kwargs)

    @classmethod
    @pretrained()
    def regnetx_002(cls, *args, **kwargs):
        depths, widths, groups_width = cls.models_config['regnetx_002']
        return cls(*args, depths=depths, widths=widths, groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnetx_004(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnetx_004']
        return cls(*args, depths=depths, widths=widths,  groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnetx_006(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnetx_006']
        return cls(*args, depths=depths, widths=widths,  groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnetx_008(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnetx_008']
        return cls(*args, depths=depths, widths=widths,  groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnetx_016(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnetx_016']
        return cls(*args, depths=depths, widths=widths,  groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnetx_032(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnetx_032']
        return cls(*args, depths=depths, widths=widths,  groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnetx_040(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnetx_040']
        return cls(*args, depths=depths, widths=widths,  groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnetx_064(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnetx_064']
        return cls(*args, depths=depths, widths=widths,  groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnetx_080(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnetx_080']
        return cls(*args, depths=depths, widths=widths,  groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnetx_120(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnetx_120']
        return cls(*args, depths=depths, widths=widths,  groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnetx_160(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnetx_160']
        return cls(*args, depths=depths, widths=widths,  groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnetx_320(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnetx_320']
        return cls(*args, depths=depths, widths=widths,  groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnety_002(cls, *args,  **kwargs):
        depths, widths, groups_width = cls.models_config['regnety_002']
        return cls(*args, depths=depths, widths=widths,  block=RegNetYBotteneckBlock, groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnety_004(cls, *args, **kwargs):
        depths, widths, groups_width = cls.models_config['regnety_004']
        return cls(*args, depths=depths, widths=widths,  block=RegNetYBotteneckBlock, groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnety_006(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnety_006']
        return cls(*args, depths=depths, widths=widths,  block=RegNetYBotteneckBlock, groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnety_008(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnety_008']
        return cls(*args, depths=depths, widths=widths,  block=RegNetYBotteneckBlock, groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnety_016(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnety_016']
        return cls(*args, depths=depths, widths=widths,  block=RegNetYBotteneckBlock, groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnety_032(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnety_032']
        return cls(*args, depths=depths, widths=widths,  block=RegNetYBotteneckBlock, groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnety_040(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnety_040']
        return cls(*args, depths=depths, widths=widths,  block=RegNetYBotteneckBlock, groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnety_064(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnety_064']
        return cls(*args, depths=depths, widths=widths,  block=RegNetYBotteneckBlock, groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnety_080(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnety_080']
        return cls(*args, depths=depths, widths=widths,  block=RegNetYBotteneckBlock, groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnety_120(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnety_120']
        return cls(*args, depths=depths, widths=widths,  block=RegNetYBotteneckBlock, groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnety_160(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnety_160']
        return cls(*args, depths=depths, widths=widths,  block=RegNetYBotteneckBlock, groups_width=groups_width, **kwargs)

    @classmethod
    @pretrained()
    def regnety_320(cls, *args,   **kwargs):
        depths, widths, groups_width = cls.models_config['regnety_320']
        return cls(*args, depths=depths, widths=widths,  block=RegNetYBotteneckBlock, groups_width=groups_width, **kwargs)
