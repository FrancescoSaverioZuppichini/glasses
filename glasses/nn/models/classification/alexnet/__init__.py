from __future__ import annotations
from torch import nn
from torch import Tensor
from collections import OrderedDict
from typing import List
from functools import partial
from ..resnet import ReLUInPlace
from ....blocks import ConvAct
from ....models.VisionModule import VisionModule




AlexNetBasicBlock = ConvAct


class AlexNetGateBlock(nn.Module):
    """
    AlexNet gate, the head of the architecture, which decreases the resolution of the filters by means of stride and bigger kernels.
    """

    out_features: int = 192

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.block = nn.Sequential(
            OrderedDict(
                {
                    'conv1': nn.Conv2d(in_channels, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
                    'act1': ReLUInPlace(),
                    'pool1': nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                    'conv2': nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                    'act2': ReLUInPlace(),
                    'pool2': nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                }
            )
        )
        self.out_features = 192

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class AlexNetEncoder(nn.Module):
    """
    AlexNet encoder, composed by a gate which decreases the size of the filters by means of stride and bigger kernels, and simple convolutional layers.
    """

    def __init__(self, in_channels: int = 3, widths: List[int] = [384, 256, 256],
                 activation: nn.Module = ReLUInPlace, block: nn.Module = AlexNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.widths = widths

        self.gate = AlexNetGateBlock()

        self.in_out_block_sizes = list(zip(widths[:-1], widths[1:]))
        self.blocks = nn.ModuleList([
            block(self.gate.out_features,
                  widths[0], activation=activation, kernel_size=3),
            *[block(in_channels, out_channels, activation=activation, kernel_size=3)
              for (in_channels, out_channels) in self.in_out_block_sizes]
        ])

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        return x


class AlexNetDecoder(nn.Module):
    """
    This class represents the classifier of AlexNet. It converts the filters into 6x6 by means of the average pooling. Then, it maps the output to the
    correct class by means of fully connected layers. Dropout is used to decrease the overfitting.
    """
    filter_size: int = 6

    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((self.filter_size, self.filter_size))
        self.block = nn.Sequential(
            OrderedDict({
                'drop1': nn.Dropout(p=0.5),
                'fc1': nn.Linear(self.filter_size * self.filter_size * in_features, 4096),
                'act1': ReLUInPlace(),
                'drop2': nn.Dropout(p=0.5),
                'fc2': nn.Linear(4096, 4096),
                'act2': ReLUInPlace(),
                'fc3': nn.Linear(4096, n_classes)
            }))

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.block(x)
        return x


class AlexNet(VisionModule):
    """Implementation of AlexNet proposed in `ImageNet Classification with Deep Convolutional Neural Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_, 
    according to the `variation <https://pytorch.org/docs/stable/_modules/torchvision/models/alexnet.html>`_ implemented in torchvision.

    Create a default model

    Examples:
        >>> net = AlexNet()

    Customization

    You can easily customize this model

    Examples:
        >>> # change activation
        >>> AlexNet(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> AlexNet(n_classes=100)
        >>> # pass a different block
        >>> AlexNet(block=SENetBasicBlock)
        >>> # change the initial convolution
        >>> model = AlexNet()
        >>> model.encoder.gate.conv1 = nn.Conv2d(3, 64, kernel_size=5)


    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Default is 3.
        n_classes (int, optional): Number of classes. Default is 1000.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = AlexNetEncoder(in_channels, *args, **kwargs)
        self.decoder = AlexNetDecoder(self.encoder.widths[-1], n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
