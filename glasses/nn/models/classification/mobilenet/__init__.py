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


class InvertedResidualBlock(nn.Module):
    """This block use a deepth wise and point wise convolution to reduce computation cost. 
    First the input is upsample (if needed), then a deepth wise 3x3 conv and point wise conv are applied.

    ReLU6 is the default activation because it was found to be more robust when used with low-precision computation.

    Residual connections are only applied when input and output's dimensions matches (stride == 1).

    Args:
        in_features (int): [description]
        out_features (int): [description]
        activation (nn.Module, optional): [description]. Defaults to nn.ReLU6.
        downsampling (int, optional): [description]. Defaults to 1.
    """

    def __init__(self, in_features: int, out_features: int,  activation: nn.Module = nn.ReLU6, downsampling: int = 1, expansion: int = 6):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.expansion = expansion
        self.expanded_features = in_features * self.expansion

        weights = nn.Sequential()
        # we need to expandn the input only if expansion is greater than one
        if expansion > 1:
            weights.add_module('exp', ConvBnAct(in_features,  self.expanded_features,
                                                activation=activation, kernel_size=1))
        # add the depth wise and point wise conv
        weights.add_module('conv',
                           nn.Sequential(ConvBnAct(self.expanded_features, self.expanded_features,
                                                   conv=DeepthWiseConv2d,
                                                   activation=activation,
                                                   kernel_size=3,
                                                   stride=downsampling),
                                         Conv2dPad(self.expanded_features,
                                                   out_features, kernel_size=1),
                                         nn.BatchNorm2d(out_features))
                           )
        # do not apply residual when downsamping and when features are different
        # in mobilenet we do not use a shortcut
        self.block = ResidualAdd(
            weights) if downsampling == 1 and in_features == out_features else weights

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


MobileNetBasicBlock = InvertedResidualBlock


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
    Please refer to Table 2 in the original paper for an overview of the current architecture


    Args:
            in_channels (int, optional): [description]. Defaults to 3.
            blocks_sizes (List[int], optional): Number of features for each layer, called `c` in the paper. Defaults to [32, 16, 24, 32, 64, 96, 160, 320].
            depths (List[int], optional): Number of blocks at each layer, called `n` in the paper. Defaults to [1, 1, 2, 3, 4, 3, 3, 1].
            strides (List[int], optional): Number of stride for each layer, called `s` in the paper. Defaults to [2, 1, 2, 2, 2, 1, 2, 1].
            expansions (List[int], optional): Expansion for each block in each layer, called `t` in the paper. Defaults to [1, 6, 6, 6, 6, 6, 6].
            activation (nn.Module, optional): [description]. Defaults to nn.ReLU6.
            block (nn.Module, optional): [description]. Defaults to MobileNetBasicBlock.
    """

    def __init__(self, in_channels: int = 3, blocks_sizes: List[int] = [32, 16, 24, 32, 64, 96, 160, 320, 1280],
                 depths: List[int] = [1, 1, 2, 3, 4, 3, 3, 1],
                 strides: List[int] = [2, 1, 2, 2, 2, 1, 2, 1],
                 expansions: List[int] = [1, 6, 6, 6, 6, 6, 6],
                 activation: nn.Module = nn.ReLU6, block: nn.Module = MobileNetBasicBlock, *args, **kwargs):
        super().__init__()
        # TODO mostly copied from resnet, we should find a way to re use the resnet one!
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            ConvBnAct(in_channels, blocks_sizes[0], activation=activation,
                      kernel_size=3, stride=strides[0]),
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:-1]))

        self.blocks = nn.ModuleList([
            *[MobileNetLayer(in_channels,
                             out_channels, n=n, downsampling=s, activation=activation,
                             block=block, *args,  expansion=t, **kwargs)
              for (in_channels, out_channels), n, s, t in zip(self.in_out_block_sizes, depths[1:], strides[1:], expansions)]
        ])

        self.blocks.append(nn.Sequential(
            ConvBnAct(blocks_sizes[-2], blocks_sizes[-1], activation=nn.ReLU6, kernel_size=1, bias=False),
            # nn.AvgPool2d(kernel_size=7),
            # Conv2dPad(blocks_sizes[-1, blocks_sizes[0]], kernel_size=1),

            ))

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
