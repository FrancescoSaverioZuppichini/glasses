from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from collections import OrderedDict
from typing import List, Callable
from functools import partial
from glasses.nn.blocks import ConvBnAct
from glasses.utils.Storage import ForwardModuleStorage
from ..base import SegmentationModule
from ...base import Encoder


class UNetBasicBlock(nn.Sequential):
    """Basic Block for UNet. It is composed by a double 3x3 conv.
    """

    def __init__(self, in_features: int, out_features: int, activation: nn.Module = partial(nn.ReLU, inplace=True), *args, **kwargs):
        super().__init__(ConvBnAct(in_features, out_features, kernel_size=3, activation=activation, *args, **kwargs),
                         ConvBnAct(
            out_features, out_features, kernel_size=3, activation=activation, *args, **kwargs))


DownBlock = UNetBasicBlock
UpBlock = UNetBasicBlock


class DownLayer(nn.Module):
    """UNet down layer (left side). 

    Args:
        out_features (int): Number of input features
        out_features (int): Number of output features
        donwsample (bool, optional): If true maxpoll will be used to reduce the resolution of the input. Defaults to True.
        block (nn.Module, optional): Block used. Defaults to DownBlock.

    """

    def __init__(self, in_features: int, out_features: int, donwsample: bool = True, block: nn.Module = DownBlock, *args, **kwargs):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(2, stride=2) if donwsample else nn.Identity(),
            block(in_features, out_features, *args, **kwargs))

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class UpLayer(nn.Module):
    """UNet up layer (right side). 

    Args:
        out_features (int): Number of input features
        out_features (int): Number of output features
        block (nn.Module, optional): Block used. Defaults to UpBlock.

    """

    def __init__(self, in_features: int, out_features: int, lateral_features: int = None, block: nn.Module = UpBlock, *args, **kwargs):
        super().__init__()
        lateral_features = out_features if lateral_features is None else lateral_features
        self.up = nn.ConvTranspose2d(
            in_features, out_features, 2, 2)

        self.block = block(out_features + lateral_features,
                           out_features, *args, **kwargs)

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        x = self.up(x)
        if res is not None:
            x = torch.cat([res, x], dim=1)
        out = self.block(x)

        return out


class UNetEncoder(Encoder):
    """UNet Encoder composed of several layers of convolutions aimed to increased the features space and decrease the resolution.
    """

    def __init__(self, in_channels: int,  widths: List[int] = [64, 128, 256, 512, 1024], *args, **kwargs):
        super().__init__()
        self.in_out_block_sizes = list(zip(widths, widths[1:]))
        self.widths = widths
        self.stem = nn.Identity()

        self.layers = nn.ModuleList([
            DownLayer(in_channels, widths[0],
                      donwsample=False, *args, **kwargs),
            *[DownLayer(in_features,
                        out_features, *args, **kwargs)
              for (in_features, out_features) in self.in_out_block_sizes]
        ])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


class UNetDecoder(nn.Module):
    """
    UNet Decoder composed of several layer of upsampling layers aimed to decrease the features space and increase the resolution.
    """

    def __init__(self, start_features: int = 512, widths: List[int] = [256, 128, 64, 32], lateral_widths: List[int] = None, *args, **kwargs):
        super().__init__()
        widths = [start_features, *widths]
        self.widths = widths
        lateral_widths = widths if lateral_widths is None else lateral_widths
        lateral_widths.extend([0] * (len(widths) - len(lateral_widths)))

        self.in_out_block_sizes = list(zip(widths, widths[1:]))
        self.layers = nn.ModuleList([
            UpLayer(in_features,
                    out_features, lateral_features, **kwargs)
            for (in_features, out_features), lateral_features in zip(self.in_out_block_sizes, lateral_widths)
        ])

    def forward(self, x: Tensor, residuals: List[Tensor]) -> Tensor:
        for layer, res in zip(self.layers, residuals):
            x = layer(x, res)

        return x

class UNet(SegmentationModule):
    """Implementation of Unet proposed in `U-Net: Convolutional Networks for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/UNet.png?raw=true

    Examples:

        Default models

        >>> UNet()

        You can easily customize your model

        >>> # change activation
        >>> UNet(activation=nn.SELU)
        >>> # change number of classes (default is 2 )
        >>> UNet(n_classes=2)
        >>> # change encoder
        >>> unet = UNet(encoder=lambda *args, **kwargs: ResNet.resnet26(*args, **kwargs).encoder,)
        >>> unet = UNet(encoder=lambda *args, **kwargs: EfficientNet.efficientnet_b2(*args, **kwargs).encoder,)
        >>> # change decoder
        >>> UNet(decoder=partial(UNetDecoder, widths=[256, 128, 64, 32, 16]))
        >>> # pass a different block to decoder
        >>> UNet(encoder=partial(UNetEncoder, block=SENetBasicBlock))
        >>> # all *Decoder class can be directly used
        >>> unet = UNet(encoder=partial(ResNetEncoder, block=ResNetBottleneckBlock, depths=[2,2,2,2]))

    Args:

       in_channels (int, optional): [description]. Defaults to 1.
       n_classes (int, optional): [description]. Defaults to 2.
       encoder (Encoder, optional): [description]. Defaults to UNetEncoder.
       ecoder (nn.Module, optional): [description]. Defaults to UNetDecoder.
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 2,
                 encoder: Encoder = UNetEncoder,
                 decoder: nn.Module = UNetDecoder,
                 **kwargs):

        super().__init__(in_channels, n_classes, encoder, decoder, **kwargs)
        self.head = nn.Conv2d(
            self.decoder.widths[-1], n_classes, kernel_size=1)
