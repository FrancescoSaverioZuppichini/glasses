from __future__ import annotations
import torch
from torch import nn
from torch import Tensor
from ....blocks.residuals import ResidualAdd
from collections import OrderedDict
from typing import List
from functools import partial
from ..mobilenet import InvertedResidualBlock, DepthWiseConv2d, MobileNetEncoder, MobileNetDecoder
from ....blocks import Conv2dPad, ConvBnAct
from ..se import SEModuleConv


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# A memory-efficient implementation of Swish function


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class SEInvertedResidualBlock(InvertedResidualBlock):
    def __init__(self, in_features: int, *args, activation: nn.Module = Swish,  **kwargs):
        super().__init__(in_features, *args, activation=activation, **kwargs)
        reduced_features = in_features // 4
        print(reduced_features)
        se = SEModuleConv(self.expanded_features,
                          reduced_features=reduced_features, activation=activation)
        # squeeze and excitation is applied after the depth wise conv
        self.block.block.conv[1] = nn.Sequential(
            se,
            self.block.block.conv[1]
        )


class EfficientNetLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, block: nn.Module = SEInvertedResidualBlock,
                 depth: int = 1, downsampling: int = 2, *args, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            block(in_features, out_features, *args,
                  downsampling=downsampling,  **kwargs),
            *[block(out_features,
                    out_features, *args, **kwargs) for _ in range(depth - 1)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class EfficientNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels: int = 3,
                 blocks_sizes: List[int] = [
                     32, 16, 24, 40, 80, 112, 192, 320, 1280],
                 depths: List[int] = [1, 2, 2, 3, 3, 4, 1],
                 strides: List[int] = [1, 2, 2, 2, 2, 1, 2],
                 expansions: List[int] = [1, 6, 6, 6, 6, 6, 6],
                 kernels_sizes: List[int] = [3, 3, 5, 3, 5, 5, 3],
                 activation: nn.Module = Swish, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(OrderedDict({
            'conv': Conv2dPad(in_channels, self.blocks_sizes[0], kernel_size=3, stride=2, bias=False),
            'bn': nn.BatchNorm2d(self.blocks_sizes[0])
        }))

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:-1]))


        self.blocks = nn.ModuleList([
            *[EfficientNetLayer(in_channels,
                                out_channels, depth=n, downsampling=s, *args,  expansion=t, kernel_size=k, **kwargs)
              for (in_channels, out_channels), n, s, t, k
                in zip(self.in_out_block_sizes, depths, strides, expansions, kernels_sizes)]
        ])

        self.blocks.append(
            ConvBnAct(self.blocks_sizes[-2], self.blocks_sizes[-1],
                      activation=Swish, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class EfficientNet(nn.Module):
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
        self.encoder = EfficientNetEncoder(in_channels, *args, **kwargs)
        self.decoder = MobileNetDecoder(
            self.encoder.blocks_sizes[-1], n_classes)

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

        # 'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        # 'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        # 'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        # 'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        # 'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        # 'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        # 'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        # 'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        # 'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        # 'efficientnet-l2': (4.3, 5.3, 800, 0.5),