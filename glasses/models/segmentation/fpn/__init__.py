from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from collections import OrderedDict
from typing import List, Callable, Union
from functools import partial
from glasses.nn.blocks import ConvBnAct
from ..base import SegmentationModule
from ...base import Encoder
from ....models.classification.resnet import ResNetEncoder


class FPNSegmentationBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, block: nn.Module = ConvBnAct, **kwargs):
        """FPN segmentation (smooth) layer used to smooth and upsample the decoder features

        Args:
            in_features (int): [description]
            out_features (int): [description]
            block (nn.Module, optional): [description]. Defaults to ConvBnAct.
        """
        super().__init__()
        self.block = block(in_features, out_features,
                           kernel_size=3, **kwargs)
        self.up = nn.UpsamplingNearest2d(
            scale_factor=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        x = self.up(x)
        return x


class FPNUpLayer(nn.Module):
    """FPN up layer (right side).

    Args:
        out_features (int): Number of input features
        out_features (int): Number of output features
        block (nn.Module, optional): Block used. Defaults to UpBlock.

    """

    def __init__(self, in_features: int, out_features: int,  block: nn.Module = ConvBnAct, upsample: bool = True,  **kwargs):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(
            scale_factor=2) if upsample else nn.Identity()
        self.block = block(in_features, out_features,
                           kernel_size=1, **kwargs)

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        out = self.up(x)
        if res is not None:
            res = self.block(res)
            out += res
        return out


class PFPNSegmentationLayer(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, depth: int = 1, **kwargs):
        super().__init__(FPNSegmentationBlock(in_features, out_features, **kwargs),
                         *[FPNSegmentationBlock(out_features, out_features, **kwargs) for _ in range(depth)])


class FPNSegmentationLayer(PFPNSegmentationLayer):
    def __init__(self, *args, depth: int = 0, **kwargs):
        super().__init__(*args, depth=0, **kwargs)


class FPNSegmentationBranch(nn.Module):
    def __init__(self, in_features: int = 256, out_features: int = 128, depth: int = 3, layer=FPNSegmentationLayer, block: nn.Module = ConvBnAct,  **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            # we iterate backward using the number of layers to keep track
            # of how many times we have to upsample
            *[layer(in_features, out_features,
                    depth=i - 1, block=block, **kwargs)
              for i in range(depth, 0, -1)],
            # latest one doesn't need to upsample
            block(in_features, out_features, kernel_size=3, **kwargs),
        ])

    def forward(self, x: Tensor, residuals: List[Tensor]) -> Tensor:
        features = []
        for layer, p in zip(self.layers, residuals):
            x = layer(p)
            features.append(x)
        return features


class FPNDecoder(nn.Module):
    """
    FPN Decoder composed of several layer of upsampling layers aimed to decrease the features space and increase the resolution.
    """

    def __init__(self, start_features: int = 512, pyramid_width: int = 256,  prediction_width: int = 128, lateral_widths: List[int] = None, segmentation_branch: nn.Module = FPNSegmentationBranch, block: nn.Module = ConvBnAct,  **kwargs):
        super().__init__()
        # we start from c_2
        self.lateral_widths = lateral_widths[:-1]
        self.widths = [prediction_width] * len(self.lateral_widths)
        self.in_out_block_sizes = list(zip(self.lateral_widths, self.widths))

        self.middle = block(
            start_features, pyramid_width, kernel_size=1, **kwargs)

        self.layers = nn.ModuleList([
            FPNUpLayer(lateral_features, pyramid_width, **kwargs)
            for lateral_features in self.lateral_widths
        ])

        self.segmentation_branch = segmentation_branch(
            pyramid_width, prediction_width, depth=len(self.layers), block=block, **kwargs)

    def forward(self, x: Tensor, residuals: List[Tensor]) -> Tensor:
        x = self.middle(x)
        features = [x]
        for layer, res in zip(self.layers, residuals):
            x = layer(x, res)
            features.append(x)
        features = self.segmentation_branch(x, features)
        return features


class Merge(nn.Module):
    """This layer merges all the features by summing them.

    Args:
        policy (str, optional): [description]. Defaults to 'sum'.
    """

    def __init__(self, policy: str = 'sum'):
        super().__init__()
        assert policy in ['sum'], f'Policy {policy} is not supported.'
        self.policy = policy

    def forward(self, features):
        if type(features) == list:
            features = torch.stack(features, dim=1)
        x = torch.sum(features, dim=1)
        return x


class FPN(SegmentationModule):
    """Implementation of Feature Pyramid Networks proposed in `Feature Pyramid Networks for Object Detection <https://arxiv.org/abs/1612.03144>`_

    .. warning::
        This model should be used only to extract features from an image, the output is a vector of shape [B, N, <prediction_width>, :math:`S_i`, :math:`S_i`].
        Where :math:`S_i` is the spatial shape of the :math:`i-th` stage of the encoder. 
        For image segmentation please use `PFPN`.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/PPFN.png?raw=true

    Examples:

        Default models

        >>> FPN()

        You can easily customize your model

        >>> # change activation
        >>> FPN(activation=nn.SELU)
        >>> # change number of classes (default is 2 )
        >>> FPN(n_classes=2)
        >>> # change encoder
        >>> FPN = FPN(encoder=lambda *args, **kwargs: ResNet.resnet26(*args, **kwargs).encoder,)
        >>> FPN = FPN(encoder=lambda *args, **kwargs: EfficientNet.efficientnet_b2(*args, **kwargs).encoder,)
        >>> # change decoder
        >>> FPN(decoder=partial(FPNDecoder, pyramid_width=64, prediction_width=32))
        >>> # pass a different block to decoder
        >>> FPN(encoder=partial(ResNetEncoder, block=SENetBasicBlock))
        >>> # all *Decoder class can be directly used
        >>> FPN = FPN(encoder=partial(ResNetEncoder, block=ResNetBottleneckBlock, depths=[2,2,2,2]))

    Args:

       in_channels (int, optional): [description]. Defaults to 1.
       n_classes (int, optional): [description]. Defaults to 2.
       encoder (Encoder, optional): [description]. Defaults to ResNetEncoder.
       ecoder (nn.Module, optional): [description]. Defaults to FPNDecoder.
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 2,
                 encoder: Encoder = ResNetEncoder,
                 decoder: nn.Module = FPNDecoder,
                 **kwargs):

        super().__init__(in_channels, n_classes, encoder, decoder, **kwargs)
        self.head = nn.Identity()


PFPNSegmentationBranch = partial(FPNSegmentationBranch, layer=PFPNSegmentationLayer)
r"""Panoptic FPN Segmentation Branch that upsample every features to match :math:`\frac{1}{4}` of the spatial dimension of the input"""
PFPNDecoder = partial(FPNDecoder, segmentation_branch=PFPNSegmentationBranch)
"""Panoptic FPN Decoder that uses  :func:`~PFPNSegmentationBranch` as segmentation branch"""


class PFPN(FPN):
    r"""Implementation of Panoptic Feature Pyramid Networks proposed in `Panoptic Feature Pyramid Networks <https://arxiv.org/pdf/1901.02446.pdf>`_

    Basically, each features obtained from the segmentation branch is upsampled to match :math:`\frac{1}{4}` of the input, in the `ResNet` case :math:`58`. 
    Then, the features are merged by summing them to obtain a single vector that is upsampled to the input spatial shape.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/PFPN.png?raw=true

    Examples:

       Create a default model

        >>> PFPN()

        You can easily customize your model

        >>> # change activation
        >>> PFPN(activation=nn.SELU)
        >>> # change number of classes (default is 2 )
        >>> PFPN(n_classes=2)
        >>> # change encoder
        >>> pfpn = PFPN(encoder=lambda *args, **kwargs: ResNet.resnet26(*args, **kwargs).encoder,)
        >>> pfpn = PFPN(encoder=lambda *args, **kwargs: EfficientNet.efficientnet_b2(*args, **kwargs).encoder,)
        >>> # change decoder
        >>> PFPN(decoder=partial(PFPNDecoder, pyramid_width=64, prediction_width=32))
        >>> # pass a different block to decoder
        >>> PFPN(encoder=partial(ResNetEncoder, block=SENetBasicBlock))
        >>> # all *Decoder class can be directly used
        >>> pfpn = PFPN(encoder=partial(ResNetEncoder, block=ResNetBottleneckBlock, depths=[2,2,2,2]))

    Args:

       in_channels (int, optional): [description]. Defaults to 1.
       n_classes (int, optional): [description]. Defaults to 2.
       encoder (Encoder, optional): [description]. Defaults to ResNetEncoder.
       ecoder (nn.Module, optional): [description]. Defaults to PFPNDecoder.
    """

    def __init__(self, *args, n_classes: int = 2, decoder: nn.Module = PFPNDecoder, **kwargs):
        super().__init__(*args, decoder=decoder, **kwargs)
        self.head = nn.Sequential(
            Merge(),
            nn.UpsamplingNearest2d(scale_factor=4),
            nn.Conv2d(self.decoder.widths[-1], n_classes, kernel_size=1))
