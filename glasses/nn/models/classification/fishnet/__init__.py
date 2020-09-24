from __future__ import annotations
from torch import nn
from torch import Tensor
from ....blocks.residuals import ResidualAdd
from ....blocks import Conv2dPad, ConvBnAct
from collections import OrderedDict
from typing import List
from functools import partial
from ..resnet import ResNetBottleneckBlock, ReLUInPlace, ResNetEncoder


class FishNetURBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, block: nn.Module = ResNetBottleneckBlock, *args, **kwargs):
        super().__init__()
        self.k = in_features // out_features
        self.up = nn.ConvTranspose2d(
            in_features, out_features, kernel_size=2, stride=2)
        self.block = block(out_features, out_features, *args, **kwargs)
        self.transfer = block(in_features, out_features)

    def channel_reduction(self, x: Tensor) -> Tensor:
        n, c, h, w = x.size()
        x_red = x.view(n, c // self.k, self.k, h, w).sum(2)
        return x_red

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        res = self.transfer(res)
        x = torch.cat([res, x], dim=1)
        x = self.channel_reduction(x) + self.block(x)
        x = self.up(x)
        return x


class FishNetDRBlock(FishNetURBlock):
    def __init__(self, in_features: int, out_features: int, block: nn.Module = ResNetBottleneckBlock, n: int = 1, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor, res: Tensor) -> Tensor:

        x = self.block(x)
        return x


class FishNetTail(ResNetEncoder):
    pass


class FishNetBody(nn.Module):
    def __init__(self, in_features: int, out_features: int, block: nn.Module = ResNetBottleneckBlock, n: int = 1, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:

        x = self.block(x)
        return x


class FishNetHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, block: nn.Module = ResNetBottleneckBlock, n: int = 1, *args, **kwargs):
        super().__init__()
        # 'We perform stride directly by convolutional layers that have a stride of 2.'

    def forward(self, x: Tensor) -> Tensor:

        x = self.block(x)
        return x


class FishNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels: int = 3, widths: List[int] = [64, 128, 256, 512], depths: List[int] = [2, 2, 2, 2],
                 activation: nn.Module = ReLUInPlace,  *args, **kwargs):
        super().__init__()

        self.gate = nn.Sequential(
            ConvBnAct(
                in_channels, widths[0] // 2, activation=activation,  kernel_size=3,  stride=2),
            ConvBnAct(widths[0] // 2, widths[0] // 2,
                      activation=activation,  kernel_size=3, ),
            ConvBnAct(widths[0] // 2, widths[0],
                      activation=activation,  kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.tail = FishNetTail(widths=widths, depths=[1, 1, 1, 1], * args, **kwargs)
        self.bridge = None
        # self.body = FishNetBody(widths, *args, **kwargs)
        # self.head = FishNetHead(widths, *args, **kwargs)

    def forward(self, x):
        x = self.gate(x)
        res = []
        # down
        for block in self.tail.blocks:
            x = block(x)
            res.append(x)
        x = self.bridge(x)
        # up
        for block in self.body.blocks:
            x = block(x)
            res.append(x)
        # down
        for block in self.head.blocks:
            x = block(x)
        return x


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
