from __future__ import annotations
from torch import nn
from torch import Tensor
from ....blocks.residuals import ResidualAdd
from ....blocks import Conv2dPad, ConvBnAct
from collections import OrderedDict
from ..resnet import ResNetBasicBlock, ResNetEncoder, ResNetLayer
from typing import List
from functools import partial


"""Implementations of ResNet proposed in `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`
"""


ReLUInPlace = partial(nn.ReLU, inplace=True)


class DeepthWiseConv2d(Conv2dPad):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, groups=in_channels, **kwargs)


class MobileNetBasicBlock(nn.Module):
    expansion = 6

    def __init__(self, in_features: int, out_features: int,  activation: nn.Module = nn.ReLU6, downsampling: int = 1):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.expanded_features = in_features * self.expansion

        weights = nn.Sequential(
            ConvBnAct(in_features,  self.expanded_features,
                      activation=activation, kernel_size=1),
            ConvBnAct(self.expanded_features, self.expanded_features,
                      conv=DeepthWiseConv2d,
                      activation=activation,
                      kernel_size=3,
                      stride=downsampling),
            Conv2dPad(self.expanded_features, out_features, kernel_size=1),
            nn.BatchNorm2d(out_features)
        )
        # do not apply residual when downsamping and when features are different
        # in mobilenet we do not use a shortcut
        self.block = ResidualAdd(weights) if downsampling == 1 and in_features == out_features else weights

    
    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x

class MobileNetLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block: nn.Module = MobileNetBasicBlock, n: int = 1, downsampling: int = 1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'

        self.block = nn.Sequential(
            block(in_channels, out_channels, *args,
                  downsampling=downsampling,  **kwargs),
            *[block(out_channels,
                    out_channels, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x

class MobileNetEncoder(ResNetEncoder):
    """
    MobileNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels: int = 3, blocks_sizes: List[int] = [16, 24, 32, 64, 96, 160, 320],
                 depths: List[int] = [1, 2, 3, 4, 3, 3, 1],
                 strides: List[int] = [1, 2, 2, 2, 1, 2, 1, 1],
                 activation: nn.Module = nn.ReLU6, block: nn.Module = MobileNetBasicBlock, *args, **kwargs):
        super().__init__()
        # TODO mostly copied from resnet, we should find a way to re use the resnet one!
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            ConvBnAct(in_channels, 32, activation=activation,
                              kernel_size=3, stride=2),
                            
            ConvBnAct(32, 32,
                      conv=DeepthWiseConv2d,
                      activation=activation,
                      kernel_size=3),
            Conv2dPad(32,  blocks_sizes[0], kernel_size=1),
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))

        self.blocks = nn.ModuleList([
            *[MobileNetLayer(in_channels,
                          out_channels, n=n, downsampling=s, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n, s in zip(self.in_out_block_sizes, depths[1:], strides[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class MobileNetDecoder(nn.Module):
    """
    This class represents the tail of MobileNet. It performs a global pooling and maps the output to the
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


class MobileNet(nn.Module):
    """Implementations of MobileNet v2 proposed in `MobileNetV2: Inverted Residuals and Linear Bottlenecks <https://arxiv.org/pdf/1801.04381.pdf>`_

    Create a default model

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        # self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        # self.decoder = MobileNetDecoder(
        #     self.encoder.blocks[-1].block[-1].expanded_features, n_classes)

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
