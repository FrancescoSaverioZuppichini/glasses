from __future__ import annotations
import torch
from torch import nn
from torch import Tensor
from ....blocks.residuals import ResidualAdd
from ....blocks import Conv2dPad, ConvBnAct
from ..resnet import ResNetShorcut, ResNetLayer
from collections import OrderedDict
from typing import List
from functools import partial
from ..resnet import ResNetBottleneckBlock, ReLUInPlace, ResNetEncoder, ResNetShorcut
from ..se import ChannelSE


class GrowModuleList(nn.ModuleList):
    def __init__(self, block: nn.Module, start_features: int = 64, n: int = 4, *args, **kwargs):
        widths = [start_features]
        for _ in range(n):
            widths.append(widths[-1] * 2)
        self.in_out_widths = list(zip(widths, widths[1:]))
        blocks = [block(in_f, out_f, *args, **kwargs)
                  for in_f, out_f in self.in_out_widths]
        super().__init__(blocks)


class BnActConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, conv: nn.Module = Conv2dPad,
                 normalization: nn.Module = nn.BatchNorm2d, activation: nn.Module = nn.ReLU, *args, **kwargs):
        super().__init__()
        self.add_module('bn', normalization(in_features))
        self.add_module('act', activation())
        self.add_module('conv', conv(
            in_features, out_features, *args, **kwargs))


FishNetShortCut = partial(BnActConv, kernel_size=1, bias=False)


class FishNetChannelReductionShortcut(nn.Module):
    def __init__(self, k: int, *args, **kwargs):
        super().__init__()
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.size()
        x_red = x.view(n, c // self.k, self.k, h, w).sum(2)
        return x_red


class FishNetBottleNeck(nn.Module):
    """FishNetBottleNeck Bottleneck block based on a correct interpretation of the original resnet paper.

    Args:
        out_features (int): Number of input features
        out_features (int): Number of output features
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
        stride (int, optional): [description]. Defaults to 1.
        conv (nn.Module, optional): [description]. Defaults to nn.Conv2d.
        expansion (int, optional): [description]. Defaults to 4.
    """

    def __init__(self, in_features: int, out_features: int, activation: nn.Module = ReLUInPlace, reduction: int = 4, stride=1, shortcut: nn.Module = FishNetShortCut, **kwargs):
        super().__init__()
        self.reduction = reduction
        features = out_features // reduction

        self.block = nn.Sequential(BnActConv(in_features, features, activation=activation, kernel_size=1, bias=False),
                                   BnActConv(features, features, activation=activation, bias=False,
                                             kernel_size=3, stride=stride, **kwargs),
                                   BnActConv(
                                       features, out_features, activation=activation, bias=False, kernel_size=1)
                                   )

        self.shortcut = shortcut(
            in_features, out_features, stride=stride) if in_features != out_features else None

    def forward(self, x: Tensor) -> Tensor:
        res = x
        if self.shortcut is not None:
            res = self.shortcut(res)
        x = self.block(x)
        x += res
        return x


class FishNetURBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, trans_features: int, block: nn.Module = FishNetBottleNeck, n: int = 1, trans_n: int = 1, *args, **kwargs):
        super().__init__()
        self.k = in_features // out_features
        self.transfer = nn.Sequential(
            *[block(trans_features, trans_features) for _ in range(trans_n)])
        self.block = nn.Sequential(
            block(in_features,  out_features,
                  shortcut=FishNetChannelReductionShortcut, *args, **kwargs),
            *[block(out_features, out_features, *args, **kwargs)
              for _ in range(n-1)],
            nn.Upsample(scale_factor=2))

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        x = self.block(x)
        res = self.transfer(res)
        x = torch.cat([x, res], dim=1)
        return x


class FishNetDRBlock(FishNetURBlock):
    def __init__(self, in_features: int, out_features: int, trans_features: int, block: nn.Module = FishNetBottleNeck, n: int = 1,  trans_n: int = 1, *args, **kwargs):
        super().__init__(in_features, out_features,
                         trans_features, block, n, trans_n, *args, **kwargs)
        self.block = nn.Sequential(
            block(in_features,  out_features,
                  shortcut=ResNetShorcut, *args, **kwargs),
            *[block(out_features, out_features, *args, **kwargs)
              for _ in range(n-1)],
            nn.MaxPool2d(kernel_size=2, stride=2))


class FishNetBrigde(nn.Module):
    def __init__(self, in_features: int, out_features: int, block: nn.Module = FishNetBottleNeck, n: int = 1, *args, **kwargs):
        super().__init__()
        self.stem = nn.Sequential(
            BnActConv(in_features, in_features //
                      2, kernel_size=1, bias=False),
            BnActConv(in_features//2, in_features *
                      2, kernel_size=1),
        )

        self.block = nn.Sequential(FishNetBottleNeck(in_features*2, out_features),
                                   *[FishNetBottleNeck(out_features, out_features) for _ in range(n - 1)])
        # very wrong se implementation and application
        self.se = nn.Sequential(nn.BatchNorm2d(in_features * 2),
                                nn.ReLU(inplace=True),
                                nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(in_features*2, in_features //
                                          16, kernel_size=1),
                                nn.ReLU(True),
                                nn.Conv2d(in_features//16,
                                          out_features, kernel_size=1),
                                nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        att = self.se(x)
        x = self.block(x)

        return (x * att) + att


class FishNetTailBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, n: int = 1,
                 block: nn.Module = FishNetBottleNeck, *args, **kwargs):
        super().__init__()
        self.block = nn.Sequential(block(in_features, out_features, **kwargs),
                                   *[block(out_features, out_features, **kwargs)
                                     for _ in range(n-1)],
                                   nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class FishNetHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, block: nn.Module = FishNetURBlock, n: int = 1, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class FishNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels: int = 3, start_features: int = 64,
                 tail_depths: List[int] = [1, 1, 1],
                 body_depths: List[int] = [1, 1, 1],
                 body_trans_depths: List[int] = [1, 1, 1],
                 head_depths: List[int] = [1, 1, 1],
                 head_trans_depths: List[int] = [1, 1, 1],
                 bridge_depth: int = 1,

                 activation: nn.Module = ReLUInPlace,  *args, **kwargs):
        super().__init__()

        self.gate = nn.Sequential(
            ConvBnAct(
                in_channels, start_features // 2, activation=activation,  kernel_size=3,  stride=2),
            ConvBnAct(start_features // 2, start_features // 2,
                      activation=activation,  kernel_size=3, ),
            ConvBnAct(start_features // 2, start_features,
                      activation=activation,  kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.tail_widths, self.body_widths, self.head_widths = self.find_widths(
            start_features, len(tail_depths))

        self.tail = nn.ModuleList([
            FishNetTailBlock(in_features, out_features,
                             block=FishNetBottleNeck, n=n, *args, **kwargs)
            for (in_features, out_features), n in zip(self.tail_widths, tail_depths)]
        )

        self.bridge = FishNetBrigde(
            self.tail_widths[-1][-1], self.body_widths[0][0], n=bridge_depth)

        self.body = nn.ModuleList([])

        for i, (tail_w, (in_features, out_features), n, trans_n) in enumerate(zip(self.tail_widths[::-1], self.body_widths, body_depths, body_trans_depths)):
            self.body.append(FishNetURBlock(
                in_features, out_features, tail_w[0], n=n, trans_n=trans_n, dilation=2**i, padding=2**i))

        self.head = nn.ModuleList([])

        for body_w, (in_features, out_features), n, trans_n in zip(self.body_widths[::-1], self.head_widths, head_depths, head_trans_depths):
            self.head.append(FishNetDRBlock(
                in_features, out_features, body_w[0], n=n, trans_n=trans_n))

    def forward(self, x):
        x = self.gate(x)
        residuals = []
        # down
        for block in self.tail:
            residuals.append(x)
            x = block(x)
        x = self.bridge(x)
        # up
        residuals = residuals[::-1]
        for i, (block, res) in enumerate(zip(self.body, residuals)):
            residuals[i] = x
            x = block(x, res)
            print(x.shape)
        # down
        residuals = residuals[::-1]
        for block, res in zip(self.head, residuals):
            x = block(x, res)

        return x

    @staticmethod
    def find_widths(start_features: int = 64, n: int = 3) -> List[int]:
        # from
        n = 3
        start_features = 64
        tail_channels = [(start_features, start_features*2)]
        for i in range(n - 1):
            tail_channels.append(
                (tail_channels[-1][1], tail_channels[-1][1] * 2))

        in_c, transfer_c = tail_channels[-1][1], tail_channels[-2][1]
        body_channels = [
            (in_c, in_c), (in_c + transfer_c, (in_c + transfer_c)//2)]
        # First body module is not change feature map channel
        for i in range(1, n-1):
            transfer_c = tail_channels[-i-2][1]
            in_c = body_channels[-1][1] + transfer_c
            body_channels.append((in_c, in_c//2))

        in_c = body_channels[-1][1] + tail_channels[0][0]
        head_channels = [(in_c, in_c)]
        for i in range(n):
            transfer_c = body_channels[-i-1][0]
            in_c = head_channels[-1][1] + transfer_c
            head_channels.append((in_c, in_c))

        return tail_channels, body_channels, head_channels


class FishNetDecoder(nn.Sequential):
    """
    """

    def __init__(self, in_features: int, n_classes: int):
        super().__init__(
            nn.BatchNorm2d(in_features),
            nn.ReLU(True),
            nn.Conv2d(in_features, in_features//2, 1, bias=False),
            nn.BatchNorm2d(in_features//2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features // 2, n_classes, 1, bias=True)
        )


class FishNet(nn.Module):
    """Implementation of ResNet proposed in `FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction <https://arxiv.org/abs/1901.03495>`_

    Create a default model

    Examples:
        >>> FishNet.resnet18()

    Customization

    You can easily customize your model

    Examples:
        >>> # change activation
        >>> ResNet.resnet18(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> ResNet.resnet18(n_classes=100)
        >>> # pass a different block
        >>> ResNet.resnet18(block=SENetBasicBlock)
        >>> # change the initial convolution
        >>> model = ResNet.resnet18()
        >>> model.encoder.gate.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        >>> # store each feature
        >>> x = torch.rand((1, 3, 224, 224))
        >>> model = ResNet.resnet18()
        >>> features = []
        >>> x = model.encoder.gate(x)
        >>> for block in model.encoder.blocks:
        >>>     x = block(x)
        >>>     features.append(x)
        >>> print([x.shape for x in features])
        >>> # [torch.Size([1, 64, 56, 56]), torch.Size([1, 128, 28, 28]), torch.Size([1, 256, 14, 14]), torch.Size([1, 512, 7, 7])]

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = FishNetEncoder(in_channels, *args, **kwargs)
        self.decoder = FishNetDecoder(
            self.encoder.head_widths[-1][1], n_classes)

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

    @classmethod
    def fishnet99(self, *args, **kwargs) -> FishNet:
        start_features = 64

        tail_depths = [2, 2, 6]
        bridge_depth = 2

        body_depths = [1, 1, 1]
        body_trans_depths = [1, 1, 1]

        head_depths = [1, 2, 2]
        head_trans_depths = [1, 1, 4]

        return FishNet(*args, start_features=start_features,
                       tail_depths=tail_depths,
                       bridge_depth=bridge_depth,
                       body_depths=body_depths,
                       body_trans_depths=body_trans_depths,
                       head_depths=head_depths,
                       head_trans_depths=head_trans_depths, **kwargs)
