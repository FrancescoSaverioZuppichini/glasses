from __future__ import annotations
import torch
from torch import nn
from torch.functional import Tensor

from glasses.models.base import Encoder
from ..resnet import ReLUInPlace, ResNet, ResNetEncoder, ResNetStem3x3
from glasses.nn.blocks import Conv3x3BnAct, ConvBnAct
from typing import List
from functools import partial

class VoVNetBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, stage_features: int, n: int = 5, block=Conv3x3BnAct):
        super().__init__()
        self.blocks = nn.Sequential(
            block(in_features, stage_features),
            *[block(stage_features, stage_features) for _ in range(n - 1)]
        )
        self.aggregate = ConvBnAct(
            in_features + (stage_features * n),  out_features, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        features = [x]
        for block in self.blocks:
            x = block(x)
            features.append(x)
        x = torch.cat(features, dim=1)
        x = self.aggregate(x)
        return x


class VoVNetLayer(nn.Sequential):
    def __init__(self,  in_features: int, out_features: int, pool: nn.Module = nn.MaxPool2d, *args, **kwargs):
        super().__init__(
            VoVNetBlock(in_features, out_features, *args, **kwargs),
            pool(kernel_size=3, stride=2),
        )

VoVNetStem = partial(ResNetStem3x3, out_features=128, widths=[64, 64])

class VoVEncoder(Encoder):
    def __init__(self,
                 in_channels: int = 3,
                 start_features: int = 64, 
                 widths: List[int] = [256, 512, 768, 1024], 
                 depths: List[int] = [1, 1, 2, 2],
                 stages_widths: List[int] = [128, 160, 192, 224],
                 activation: nn.Module = ReLUInPlace, 
                 block: nn.Module = VoVNetBlock,
                 stem: nn.Module = ResNetStem3x3, **kwargs):
        super().__init__()
        self.widths = widths
        self.start_features = start_features
        self.in_out_widths = list(zip(widths, widths[1:]))
        self.stem = stem(in_channels, start_features, activation=activation)

        self.layers = nn.ModuleList(
            [
                VoVNetLayer(
                    start_features,
                    widths[0],
                    depth=depths[0],
                    activation=activation,
                    block=block,
                    **kwargs,
                ),
                *[
                    VoVNetLayer(
                        in_features,
                        out_features,
                        depth=n,
                        activation=activation,
                        block=block,
                        **kwargs,
                    )
                    for (in_features, out_features), n in zip(
                        self.in_out_widths, depths[1:]
                    )
                ],
            ]
        )

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def stages(self):
        return [self.stem[-2], *self.layers[:-1]]

    @property
    def features_widths(self):
        return [self.start_features, *self.widths[:-1]]


class VoVNet(ResNet):
    """Implementation of VoVNet, popular backbone also used in object-detection
    `An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection
 <https://arxiv.org/abs/1904.09730>`_
    The models with the channel se are labelab with prefix `c`
    """
