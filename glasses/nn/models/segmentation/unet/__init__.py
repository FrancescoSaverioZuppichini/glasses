from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from collections import OrderedDict
from typing import List
from functools import partial
from ....blocks import ConvBnAct


class UNetBasicBlock(nn.Module):
    """Basic Block for UNet. It is composed by a double 3x3 conv.
    """

    def __init__(self, in_features: int, out_features: int, activation: nn.Module = partial(nn.ReLU, inplace=True), *args, **kwargs):
        super().__init__()

        self.block = nn.Sequential(nn.Sequential(ConvBnAct(in_features, out_features, kernel_size=3, activation=activation, *args, **kwargs),
                                                 ConvBnAct(out_features, out_features, kernel_size=3, activation=activation, *args, **kwargs))
                                   )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


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

    def __init__(self, in_features: int, out_features: int, block: nn.Module = UpBlock, *args, **kwargs):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_features, out_features, 2, 2)

        self.block = block(out_features * 2, out_features, *args, **kwargs)

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        x = self.up(x)
        x = torch.cat([res, x], dim=1)
        out = self.block(x)

        return out


class UNetEncoder(nn.Module):
    """UNet Encoder composed of several layers of convolutions aimed to increased the features space and decrease the resolution.
    """

    def __init__(self, in_channels: int,  widths: List[int] = [64, 128, 256, 512, 1024], *args, **kwargs):
        super().__init__()
        self.gate = nn.Identity()
        self.in_out_block_sizes = list(zip(widths, widths[1:]))
        self.widths = widths
        self.blocks = nn.ModuleList([
            DownLayer(in_channels, widths[0],
                      donwsample=False, *args, **kwargs),
            *[DownLayer(in_features,
                        out_features, *args, **kwargs)
              for (in_features, out_features) in self.in_out_block_sizes]
        ])


class UNetDecoder(nn.Module):
    """
    UNet Decoder composed of several layer of upsampling layers aimed to decrease the features space and increase the resolution.
    """

    def __init__(self, widths: List[int] = [64, 128, 256, 512, 1024], *args, **kwargs):
        super().__init__()
        self.in_out_block_sizes = list(zip(widths, widths[1:]))
        self.blocks = nn.ModuleList([
            UpLayer(in_features,
                    out_features, *args, **kwargs)
            for (in_features, out_features) in self.in_out_block_sizes
        ])


class UNet(nn.Module):
    """Implementation of Unet proposed in `U-Net: Convolutional Networks for Biomedical Image Segmentation
 <https://arxiv.org/abs/1505.04597>`_

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/UNet.png?raw=true

    Create a default model

    Examples:
        >>> UNet()

    Customization

    You can easily customize your model

    Examples:

        >>> # change activation
        >>> UNet(activation=nn.SELU)
        >>> # change number of classes (default is 2 )
        >>> UNet(n_classes=2)
        >>> # pass a different block
        >>> UNet(encoder=partial(UNetEncoder, block=SENetBasicBlock))
        >>> # change the encoder
        >>> unet = UNet(encoder=partial(ResNetEncoder, block=ResNetBottleneckBlock, depths=[2,2,2,2]))


    Args:
            in_channels (int, optional): [description]. Defaults to 1.
            n_classes (int, optional): [description]. Defaults to 2.
            encoder (nn.Module, optional): Model's encoder (left part). It have a `.gate` and `.block : nn.ModuleList` fields. Defaults to UNetEncoder.
            decoder (nn.Module, optional): Model's decoder (left part). It must have a `.blocks : nn.ModuleList` field. Defaults to UNetDecoder.
            widths (List[int], optional): [description]. Defaults to [64, 128, 256, 512, 1024].
        """

    def __init__(self, in_channels: int = 1, n_classes: int = 2, encoder: nn.Module = UNetEncoder,
                 decoder: nn.Module = UNetDecoder, *args, **kwargs):

        super().__init__()
        self.encoder = encoder(
            in_channels, *args, **kwargs)
        self.decoder = decoder(
            widths=self.encoder.widths[::-1], *args, **kwargs)
        self.tail = nn.Conv2d(
            self.encoder.widths[0], n_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        self.residuals = []
        x = self.encoder.gate(x)
        for block in self.encoder.blocks:
            x = block(x)
            self.residuals.append(x)
        # reverse the residuals and skip the middle one
        self.residuals = self.residuals[::-1][1:]
        for block, res in zip(self.decoder.blocks, self.residuals):
            x = block(x, res)

        x = self.tail(x)
        return x
