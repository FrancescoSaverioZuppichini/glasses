from __future__ import annotations
from torch import nn
from torch import Tensor
from ..blocks.residuals import ResidualAdd
from collections import OrderedDict
from typing import List
from functools import partial


"""Implementations of ResNet proposed in `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`
"""


ReLUInPlace = partial(nn.ReLU, inplace=True)


class ResNetShorcut(nn.Module):
    """Shorcut function applied by ResNet to upsample the channel
    when residual and output features do not match

    Args:
        in_features (int): features (channels) of the input
        out_features (int): features (channels) of the desidered output
    """

    def __init__(self, in_features: int, out_features: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features,
                              kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBasicBlock(nn.Module):
    expansion: int = 1
    """Basic ResNet block composed by two 3x3 convs with residual connection. 

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNetBasicBlock.png?raw=true

    *The residual connection is showed as a black line*

    The output of the layer is defined as:

    :math:`x' = F(x) + x`

    Args:
        in_features (int): [description]
        out_features (int): [description]
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
        downsampling (int, optional): [description]. Defaults to 1.
        conv (nn.Module, optional): [description]. Defaults to nn.Conv2d.
    """

    def __init__(self, in_features: int, out_features: int,  activation: nn.Module = ReLUInPlace, downsampling: int = 1, conv: nn.Module = nn.Conv2d):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.expanded_channels = self.out_features * self.expansion
        self.should_apply_shortcut = self.in_features != self.expanded_channels

        self.block = nn.Sequential(
            OrderedDict(
                {
                    'conv1': conv(in_features, out_features, kernel_size=3, stride=downsampling, padding=1, bias=False),
                    'bn1': nn.BatchNorm2d(out_features),
                    'act1': activation(),
                    'conv2': conv(out_features, out_features, kernel_size=3, padding=1, bias=False),
                    'bn2': nn.BatchNorm2d(out_features),
                }
            ))
        self.shortcut = ResNetShorcut(
            in_features, out_features * self.expansion, downsampling) if self.should_apply_shortcut else None
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        # activation is applied after the residual
        x = self.act(x)
        return x


class ResNetBottleNeckBlock(ResNetBasicBlock):
    expansion: int = 4

    """ResNet BottleNeck block is composed by three convs layer. 
    The expensive 3x3 conv is computed after a cheap 1x1 conv donwsample the input resulting in less parameters. Later, another conv 1v1 upsample the output to the correct channel size

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNetBottleNeckBlock.png?raw=true

    *The residual connection is showed as a black line*

    Args:
        in_features (int): [description]
        out_features (int): [description]
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
        downsampling (int, optional): [description]. Defaults to 1.
        conv (nn.Module, optional): [description]. Defaults to nn.Conv2d.
        expansion (int, optional): [description]. Defaults to 4.
    """

    def __init__(self, in_features: int, out_features: int, activation: nn.Module = ReLUInPlace, downsampling: int = 1, conv: nn.Module = nn.Conv2d, expansion: int = 4):
        super().__init__(in_features, out_features, activation, downsampling)
        self.block = nn.Sequential(
            OrderedDict(
                {
                    'conv1': conv(in_features, out_features, kernel_size=1, bias=False),
                    'bn1': nn.BatchNorm2d(out_features),
                    'act1': activation(),
                    'conv2': conv(out_features, out_features, kernel_size=3, stride=downsampling, padding=1, bias=False),
                    'bn2': nn.BatchNorm2d(out_features),
                    'act2': activation(),
                    'conv3': conv(out_features, out_features * expansion, kernel_size=1, bias=False),
                    'bn3': nn.BatchNorm2d(out_features * expansion),
                }
            ))


class ResNetBasicPreActBlock(ResNetBottleNeckBlock):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module = ReLUInPlace, downsampling: int = 1, conv: nn.Module = nn.Conv2d, *args, **kwars):
        super().__init__(in_features, out_features, activation, downsampling, *args, **kwars)
        self.block = nn.Sequential(
            OrderedDict(
                {
                    'bn1': nn.BatchNorm2d(out_features),
                    'act1': activation(),
                    'conv1': conv(in_features, out_features, kernel_size=3, stride=downsampling, padding=1, bias=False),
                    'bn2': nn.BatchNorm2d(out_features),
                    'act2': activation(),
                    'conv2': conv(out_features, out_features, kernel_size=3, padding=1, bias=False),
                }
            ))

        self.act = nn.Identity()


class ResNetBottleNeckPreActBlock(ResNetBasicBlock):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module = ReLUInPlace, downsampling: int = 1, conv: nn.Module = nn.Conv2d, expansion: int = 4, *args, **kwars):
        super().__init__(in_features, out_features, activation,
                         downsampling, expansion, *args, **kwars)
        # TODO I am not sure it is correct
        self.block = nn.Sequential(
            OrderedDict(
                {
                    'bn1': nn.BatchNorm2d(out_features),
                    'act1': activation(),
                    'conv1': conv(in_features, out_features, kernel_size=1, bias=False),
                    'bn2': nn.BatchNorm2d(out_features),
                    'act2': activation(),
                    'conv2': conv(out_features, out_features, kernel_size=3, stride=downsampling, padding=1, bias=False),
                    'bn3': nn.BatchNorm2d(out_features * self.expansion),
                    'conv3': conv(out_features, out_features * self.expansion, kernel_size=1, bias=False),
                    'act3': activation(),
                }
            ))

        self.act = nn.Identity()


class ResNetLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block: nn.Module = ResNetBasicBlock, n: int = 1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.block = nn.Sequential(
            block(in_channels, out_channels, *args,
                  downsampling=downsampling,  **kwargs),
            *[block(out_channels * block.expansion,
                    out_channels, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels: int = 3, blocks_sizes: List[int] = [64, 128, 256, 512], deepths: List[int] = [2, 2, 2, 2],
                 activation: nn.Module = ReLUInPlace, block: nn.Module = ResNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            OrderedDict(
                {
                    'conv': nn.Conv2d(
                        in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
                    'bn': nn.BatchNorm2d(self.blocks_sizes[0]),
                    'act': activation(),
                    'pool': nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                }
            )
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
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

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(
            self.encoder.blocks[-1].block[-1].expanded_channels, n_classes)

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
    def resnet18(cls, *args, **kwargs) -> ResNet:
        """Create a resnet18 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNet18.png?raw=true

        Returns:
            ResNet: [description]
        """
        return cls(*args, **kwargs, block=ResNetBasicBlock, deepths=[2, 2, 2, 2])

    @classmethod
    def resnet34(cls, *args, **kwargs) -> ResNet:
        """Create a resnet34 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNet34.png?raw=true

        Returns:
            ResNet: [description]
        """
        return cls(*args, **kwargs, block=ResNetBasicBlock, deepths=[3, 4, 6, 3])

    @classmethod
    def resnet50(cls, *args, **kwargs) -> ResNet:
        """Create a resnet50 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNet50.png?raw=true

        Returns:
            ResNet: [description]
        """
        return cls(*args, **kwargs, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3])

    @classmethod
    def resnet101(cls, *args, **kwargs) -> ResNet:
        """Create a resnet101 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNet101.png?raw=true

        Returns:
            ResNet: [description]
        """
        return cls(*args, **kwargs, block=ResNetBottleNeckBlock, deepths=[3, 4, 23, 3])

    @classmethod
    def resnet152(cls, *args, **kwargs) -> ResNet:
        """Create a resnet152 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNet152.png?raw=true

        Returns:
            ResNet: [description]
        """
        return cls(*args, **kwargs, block=ResNetBottleNeckBlock, deepths=[3, 8, 36, 3])


ResNet.resnet18()
