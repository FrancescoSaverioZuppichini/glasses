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
        in_features (int): [description]
        out_features (int): [description]
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
        in_features (int): [description]
        out_features (int): [description]
        block (nn.Module, optional): Block used. Defaults to UpBlock.

    """

    def __init__(self, in_features: int, out_features: int, block: nn.Module = UpBlock, *args, **kwargs):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_features, out_features, 2, 2)

        self.block = block(out_features * 2, out_features, *args, **kwargs)

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        x = self.up(x)
        # we need to pad the input in order to have the same dimensions
        diffX = x.size()[2] - res.size()[2]
        diffY = x.size()[3] - res.size()[3]
        pad = (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2))
        res = F.pad(res, pad)

        x = torch.cat([res, x], dim=1)

        out = self.block(x)

        return out


class UNetEncoder(nn.Module):
    """UNet Encoder composed of several layers of convolutions aimed to increased the features space and decrease the resolution.
    """

    def __init__(self, in_channels: int,  blocks_sizes: List[int] = [64, 128, 256, 512, 1024], *args, **kwargs):
        super().__init__()

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks_sizes = blocks_sizes
        self.blocks = nn.ModuleList([
            DownLayer(in_channels, blocks_sizes[0],
                      donwsample=False, *args, **kwargs),
            *[DownLayer(in_features,
                        out_features, *args, **kwargs)
              for (in_features, out_features) in self.in_out_block_sizes]
        ])


class UNetDecoder(nn.Module):
    """
    UNet Decoder composed of several layer of upsampling layers aimed to decrease the features space and increase the resolution.
    """

    def __init__(self, blocks_sizes: List[int], *args, **kwargs):
        super().__init__()

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))

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
        >>> UNet(down_block=SENetBasicBlock, up_block=SENetBasicBlock)
        >>> # change the encoder
        >>> resnet34 = ResNet.resnet34()
        >>> # we need to change the first conv in order to accept a gray image
        >>> resnet34.encoder.blocks[0].block[0].block.block.conv1 = Conv2dPad(1, 64, kernel_size=1)
        >>> unet = UNet(1, n_classes=2, blocks_sizes=resnet34.encoder.blocks_sizes)
        >>> unet.encoder.blocks = resnet34.encoder.blocks 


    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 1.
        n_classes (int, optional): Number of classes. Defaults to 2.
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 2, down_block: nn.Module = DownBlock, up_block: nn.Module = UpBlock, blocks_sizes: List[int] = [64, 128, 256, 512, 1024], *args, **kwargs):
        super().__init__()
        self.encoder = UNetEncoder(
            in_channels, blocks_sizes, block=down_block, *args, **kwargs)
        self.decoder = UNetDecoder(
            self.encoder.blocks_sizes[::-1], block=up_block, *args, **kwargs)
        self.tail = nn.Conv2d(
            self.encoder.blocks_sizes[0], n_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        self.residuals = []
        for block in self.encoder.blocks:
            x = block(x)
            self.residuals.append(x)
        # reverse the residuals and skip the middle one
        self.residuals = self.residuals[::-1][1:]
        for block, res in zip(self.decoder.blocks, self.residuals):
            x = block(x, res)

        x = self.tail(x)
        return x
