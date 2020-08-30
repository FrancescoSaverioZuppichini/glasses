from __future__ import annotations
import torch
from torch import nn
from torch import Tensor
from ....blocks.residuals import ResidualAdd
from collections import OrderedDict
from typing import List
from functools import partial
from ..mobilenet import InvertedResidualBlock, DepthWiseConv2d, MobileNetEncoder
from ....blocks import Conv2dPad, ConvBnAct
from ..se import SEModuleConv


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEInvertedResidualBlock(InvertedResidualBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        se = SEModuleConv(self.expanded_features, self.expanded_features)
        # squeeze and excitation is applied after the depth wise conv
        self.block.block.conv[1] = nn.Sequential(
            se,
            self.block.block.conv[1]
        )


class EfficientNetLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, block: nn.Module = SEInvertedResidualBlock,
                 n: int = 1, downsampling: int = 2, *args, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            block(in_features, out_features, *args,
                  downsampling=downsampling,  **kwargs),
            *[block(out_features,
                    out_features, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class EfficientNetEncoder(MobileNetEncoder):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels: int = 3, *args, **kwargs):
        super().__init__(in_channels, activation=Swish, *args, **kwargs)

    

        self.blocks = nn.ModuleList([
            EfficientNetLayer(self.blocks_sizes[0], self.blocks_sizes[1], downsampling=1),
            EfficientNetLayer(self.blocks_sizes[1], self.blocks_sizes[2]),
            EfficientNetLayer(self.blocks_sizes[2], self.blocks_sizes[3], kernel_size=5),
            EfficientNetLayer(self.blocks_sizes[3], self.blocks_sizes[4]),
            EfficientNetLayer(self.blocks_sizes[4], self.blocks_sizes[5], kernel_size=5),
            EfficientNetLayer(self.blocks_sizes[5], self.blocks_sizes[6], kernel_size=5),
            EfficientNetLayer(self.blocks_sizes[6], self.blocks_sizes[7], kernel_size=3),
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet(nn.Module):
    """Implementations of ResNet proposed in `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_

    Create a default model

    Examples:
        >>> ResNet.resnet18()
        >>> ResNet.resnet34()
        >>> ResNet.resnet50()
        >>> ResNet.resnet101()
        >>> ResNet.resnet152()

    Customization

    You can easily customize your resnet

    Examples:


    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(
            self.encoder.blocks[-1].block[-1].expanded_features, n_classes)

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
