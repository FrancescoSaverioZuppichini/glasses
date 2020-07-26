from __future__ import annotations
import torch
from torch import nn
from torch import Tensor
from ..resnet import ResNetEncoder, ResnetDecoder, ResNet
from collections import OrderedDict
from typing import List
from functools import partial
from ..resnet import ReLUInPlace


class DenseNetBasicBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, conv: nn.Module = nn.Conv2d, activation: nn.Module = ReLUInPlace, *args, **kwargs):
        super().__init__()
        self.block = nn.Sequential(OrderedDict({
            'bn': nn.BatchNorm2d(in_features),
            'act': activation(),
            'conv': conv(in_features, out_features, kernel_size=3, padding=1, *args, **kwargs)
        }))

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x = torch.cat([res, x], dim=1)
        return x


class DenseBottleNeckBlock(DenseNetBasicBlock):
    expansion: int = 4

    def __init__(self, in_features: int, out_features: int, conv: nn.Module = nn.Conv2d, activation: nn.Module = ReLUInPlace, *args, **kwargs):
        super().__init__(in_features, out_features, conv, activation, *args, **kwargs)
        self.expanded_features = out_features * self.expansion

        self.block = nn.Sequential(OrderedDict({
            'bn1': nn.BatchNorm2d(in_features),
            'act1': activation(),
            'conv1': conv(in_features, self.expanded_features, kernel_size=1, bias=False, *args, **kwargs),
            'bn2': nn.BatchNorm2d(self.expanded_features),
            'act2': activation(),
            'conv2': conv(self.expanded_features, out_features, kernel_size=3, padding=1,  bias=False, *args, **kwargs)
        }))


class TransitionBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.block = nn.Sequential(
            OrderedDict(
                {
                    'bn': nn.BatchNorm2d(in_features),
                    'act': ReLUInPlace(),
                    'conv': nn.Conv2d(in_features, out_features,
                                      kernel_size=1, bias=False),
                    'pool': nn.AvgPool2d(kernel_size=2, stride=2)
                }
            ))

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self, in_features: int, grow_rate: int = 32, n: int = 4, block: nn.Module = DenseNetBasicBlock, transition: bool = True):
        super().__init__()
        self.out_features = grow_rate * n + in_features
        self.block = nn.Sequential(
            *[block(grow_rate * i + in_features, grow_rate) for i in range(n)],
            TransitionBlock(self.out_features, self.out_features //
                            2) if transition else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class DenseNetEncoder(ResNetEncoder):
    """
    Dense encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels: int = 3, start_features: int = 64,  grow_rate: int = 32,
                 deepths: List[int] = [4, 4, 4, 4],
                 activation: nn.Module = ReLUInPlace, block: nn.Module = DenseNetBasicBlock, *args, **kwargs):
        super().__init__(in_channels, [64])

        self.blocks = nn.ModuleList([])

        in_features = start_features

        for deepth in deepths[:-1]:
            self.blocks.append(DenseLayer(
                in_features, grow_rate, deepth, block=block, *args, **kwargs))
            in_features += deepth * grow_rate
            in_features //= 2

        self.blocks.append(DenseLayer(
            in_features, grow_rate, deepths[-1], block=block, *args, transition=False, **kwargs))
        self.out_features = in_features + deepths[-1] * grow_rate
        self.bn = nn.BatchNorm2d(self.out_features)

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)

        x = self.bn(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, in_channels: int = 3,  n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = DenseNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(
            self.encoder.out_features, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @classmethod
    def densenet121(cls, in_channels: int = 3,  n_classes: int = 1000, **kwargs):
        return DenseNet(in_channels, grow_rate=32, deepths=[6, 12, 24, 16], n_classes=n_classes, block=DenseBottleNeckBlock, **kwargs)

    @classmethod
    def densenet161(cls, in_channels: int = 3,  n_classes: int = 1000, **kwargs):
        return DenseNet(in_channels, grow_rate=48, deepths=[6, 12, 36, 24], n_classes=n_classes, block=DenseBottleNeckBlock, **kwargs)

    @classmethod
    def densenet169(cls, in_channels: int = 3,  n_classes: int = 1000, **kwargs):
        return DenseNet(in_channels, grow_rate=32, deepths=[6, 12, 32, 32], n_classes=n_classes, block=DenseBottleNeckBlock, **kwargs)

    @classmethod
    def densenet201(cls, in_channels: int = 3,  n_classes: int = 1000, **kwargs):
        return DenseNet(in_channels, grow_rate=32, deepths=[6, 12, 48, 32], n_classes=n_classes, block=DenseBottleNeckBlock, **kwargs)