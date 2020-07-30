from __future__ import annotations
from torch import nn
from torch import Tensor
from collections import OrderedDict
from typing import List
from functools import partial
from ..resnet import ReLUInPlace
from ....blocks import ConvAct


VGGBasicBlock = ConvAct


class VGGLayer(nn.Module):
    """ VGG layer.

    Args:
        nn ([type]): [description]
    """
    def __init__(self, in_channels: int, out_channels: int, block: nn.Module = VGGBasicBlock, n: int = 1, maxpool: nn.Module = nn.MaxPool2d, *args, **kwargs): 
        super().__init__()
        self.block = nn.Sequential(
            block(in_channels, out_channels, kernel_size=3, *args, **kwargs),
            *[block(out_channels,
                    out_channels, kernel_size=3, *args, **kwargs) for _ in range(n - 1)]
        )

        if maxpool is not None:
            self.block.add_module('maxpool', maxpool(kernel_size=2, stride=2))

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x

    
class VGGEncoder(nn.Module):
    """VGG encoder.
    """

    def __init__(self, in_channels: int = 3, blocks_sizes: List[int] = [64, 128, 256, 512, 512], depths: List[int] = [1, 1, 2, 2, 2],
                 activation: nn.Module = ReLUInPlace, block: nn.Module = VGGBasicBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes
        self.out_features = blocks_sizes[-1]        
        self.in_out_block_sizes = list(zip(blocks_sizes[:-1], blocks_sizes[1:]))

        self.blocks = nn.ModuleList([
            VGGLayer(in_channels, blocks_sizes[0], activation=activation, block=block, n=depths[0], *args, **kwargs),
            *[VGGLayer(in_channels, out_channels, activation=activation, block=block, n=n, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    
class VGGDecoder(nn.Module):
    """This class represents the classifier of VGG. It converts the filters into 6x6 by means of the average pooling. Then, it maps the output to the
    correct class by means of fully connected layers. Dropout is used to decrease the overfitting.
    """
    filter_size: int = 6
    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((7, 7))
        self.block = nn.Sequential(
            OrderedDict(
                {
                'decoder_linear1': nn.Linear(in_features * 7 * 7, 4096),
                'decoder_act1': nn.ReLU(True),
                'decoder_dropout1': nn.Dropout(),
                'decoder_linear2': nn.Linear(4096, 4096),
                'decoder_act2': nn.ReLU(True),
                'decoder_dropout2': nn.Dropout(),
                'decoder_linear3': nn.Linear(4096, n_classes)
                }
            )
        )


    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.block(x)
        return x
    
    
class VGG(nn.Module):
    """VGG.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = VGGEncoder(in_channels, *args, **kwargs)
        self.decoder = VGGDecoder(self.encoder.out_features, n_classes)
        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)