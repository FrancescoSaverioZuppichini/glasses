from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from collections import OrderedDict
from typing import List
from functools import partial
from ....blocks import ConvBnAct


class DownBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, donwsample: bool = True):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(2, stride=2) if donwsample else nn.Identity(),
            ConvBnAct(in_features, out_features, kernel_size=3),
            ConvBnAct(out_features, out_features, kernel_size=3))

    def forward(self, x):
        x = self.block(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_features, out_features, 2, 2)

        self.block = nn.Sequential(nn.Sequential(ConvBnAct(out_features * 2, out_features, kernel_size=3),
                                                 ConvBnAct(out_features, out_features, kernel_size=3))
                                   )

    def forward(self, x: Tensor, res: Tensor) -> Tensor:

        x = self.up(x)
        # we need to pad the input
        diffX = x.size()[2] - res.size()[2]
        diffY = x.size()[3] - res.size()[3]
        pad = (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2))
        res = F.pad(res, pad)

        x = torch.cat([res, x], dim=1)

        out = self.block(x)

        return out


class UnetEncoder(nn.Module):
    def __init__(self, in_channels: int,  blocks_sizes: List[int] = [64, 128, 256, 512, 1024],
                 block: nn.Module = DownBlock, *args, **kwargs):
        super().__init__()

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks_sizes = blocks_sizes
        self.blocks = nn.ModuleList([
            DownBlock(in_channels, blocks_sizes[0], donwsample=False),
            *[DownBlock(in_features,
                        out_features, *args, **kwargs)
              for (in_features, out_features) in self.in_out_block_sizes]
        ])


class UnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(self, blocks_sizes: List[int], block: nn.Module = DownBlock, *args, **kwargs):
        super().__init__()

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))

        self.blocks = nn.ModuleList([
            UpBlock(in_features, 
                        out_features, *args, **kwargs)
              for (in_features, out_features) in self.in_out_block_sizes
        ])



class Unet(nn.Module):
    """Implementations of ResNet proposed in `U-Net: Convolutional Networks for Biomedical Image Segmentation
 <https://arxiv.org/abs/1505.04597>`_

    Create a default model

    Examples:
        >>> Unet()


    Customization

    You can easily customize your model

    Examples:

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 2.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = UnetEncoder(in_channels, *args, **kwargs)
        self.decoder = UnetDecoder(
            self.encoder.blocks_sizes[::-1], n_classes)

    def forward(self, x: Tensor) -> Tensor:
        self.residuals = []
        for block in self.encoder.blocks:
            x = block(x)
            self.residuals.append(x)
        # reverse the residuals and skip the middle one
        self.residuals = self.residuals[::-1][1:]
        for block, res in zip(self.decoder.blocks, self.residuals):
            x = block(x, res)
        return x
