from __future__ import annotations
import torch
from torch import nn
from torch import Tensor
from ..resnet import ResNetEncoder, ResnetDecoder, ResNet
from collections import OrderedDict
from typing import List
from functools import partial
from ..resnet import ReLUInPlace
from ....blocks import Conv2dPad

class InceptionBlock(nn.Module):
    def __init__(self, in_features: int, conv: nn.Module = Conv2dPad, activation: nn.Module = ReLUInPlace, *args, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([])

    def forward(self, x: Tensor) -> Tensor:
        out = None
        for block in self.blocks:
            if out is None:
                out = block(x)
            else:
                res = block(x)
                print(res.shape)
                out = torch.cat([out, res], dim=1)
            print('out', out.shape)
        return out


class InceptionABlock(InceptionBlock):
    def __init__(self, in_features: int, conv: nn.Module = Conv2dPad, activation: nn.Module = ReLUInPlace, *args, **kwargs):
        super().__init__(in_features, conv, activation, *args, **kwargs)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                OrderedDict({
                    'conv1': conv(in_features, 64, kernel_size=1),
                    'conv2': conv(64, 96, kernel_size=3),
                    'conv3': conv(96, 96, kernel_size=3)
                })
            ),
            nn.Sequential(
                OrderedDict({
                    'conv1': conv(in_features, 64, kernel_size=1),
                    'conv2': conv(64, 96, kernel_size=3),
                })
            ),
            nn.Sequential(
                OrderedDict({
                    'pool': nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                    'conv': conv(in_features, 96, kernel_size=1)
                })

            ),
            nn.Sequential(
                OrderedDict({
                    'conv': conv(in_features, 96, kernel_size=1)
                })
            )
        ])


class Inception(nn.Module):
    """Implementations of Inception proposed in `Rethinking the Inception Architecture for Computer Vision
 <https://arxiv.org/abs/1512.00567>`_

    """
