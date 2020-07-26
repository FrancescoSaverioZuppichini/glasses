from __future__ import annotations
from torch import nn
from torch import Tensor
from collections import OrderedDict
from typing import List
from functools import partial


"""Implementations of AlexNet proposed in `ImageNet Classification with Deep Convolutional Neural Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks>`, 
according to the re-implementation in torchvision.models.
"""


ReLUInPlace = partial(nn.ReLU, inplace=True)


class AlexNet_BasicBlock(nn.Module):
    """Basic AlexNet block composed by one 3x3 conv. 


    Args:
        in_features (int): [description]
        out_features (int): [description]
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
        conv (nn.Module, optional): [description]. Defaults to nn.Conv2d.
    """

    def __init__(self, in_features: int, out_features: int,  activation: nn.Module = ReLUInPlace, conv: nn.Module = nn.Conv2d):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.block = nn.Sequential(
            OrderedDict(
                {
                    'conv': conv(in_features, out_features, kernel_size=3, padding=1, bias=False),
                    'act': activation()
                }
            ))
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x
    
    
    
class AlexNet_Encoder(nn.Module):
    """
    AlexNet encoder, composed by a gate which decreases the size of the filters by means of stride and bigger kernels, and simple convolutional layers.
    """

    def __init__(self, in_channels: int = 3, blocks_sizes: List[int] = [192, 384, 256, 256],
                 activation: nn.Module = ReLUInPlace, block: nn.Module = AlexNet_BasicBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            OrderedDict(
                {
                    'conv1': nn.Conv2d(in_channels, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
                    'act1': activation(),
                    'pool1': nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                    'conv2': nn.Conv2d(64, blocks_sizes[0], kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                    'act2': activation(),
                    'pool2': nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                }
            )
        )
        
    
        self.in_out_block_sizes = list(zip(blocks_sizes[:-1], blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            *[block(in_channels, out_channels, activation=activation)
              for (in_channels, out_channels) in self.in_out_block_sizes]
        ])
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        return x

    
    
class AlexNet_Decoder(nn.Module):
    """
    This class represents the classifier of AlexNet. It converts the filters into 6x6 by means of the average pooling. Then, it maps the output to the
    correct class by means of fully connected layers. Dropout is used to decrease the overfitting.
    """
    filter_size: int = 6
    def __init__(self, in_features: int, n_classes: int, widths: List[int] = [4096, 4096], 
                 activation: nn.Module = ReLUInPlace, dropout_probability: float = 0.5):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((self.filter_size, self.filter_size))
        self.fc = nn.Sequential(nn.Dropout(p=dropout_probability),
                                nn.Linear(self.filter_size * self.filter_size * in_features, widths[0]),
                                activation(),
                                nn.Dropout(p=dropout_probability),
                                nn.Linear(widths[0], widths[1]),
                                activation(),
                                nn.Linear(widths[1], n_classes))        

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
class AlexNet(nn.Module):
    """Implementation of AlexNet proposed in `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`, 
    according to the `variation <https://pytorch.org/docs/stable/_modules/torchvision/models/alexnet.html>` implemented in torchvision.

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
        self.encoder = AlexNet_Encoder(in_channels, *args, **kwargs)
        self.decoder = AlexNet_Decoder(self.encoder.blocks[-1].out_features, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x