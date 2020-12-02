from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from collections import OrderedDict
from typing import List, Callable, Union
from functools import partial
from ....blocks import ConvBnAct
from glasses.utils.Storage import ForwardModuleStorage
from ..base import SegmentationModule
from ...base import Encoder
from ..unet import UNetEncoder


# DownBlock = UNetBasicBlock
# UpBlock = UNetBasicBlock

class FPNSmoothBlock(nn.Module):
    def __init__(self, in_features:int, out_features: int,  **kwargs):
        super().__init__()
        self.block = ConvBnAct(in_features, out_features, kernel_size=3, **kwargs)

    def forward(self, x: Tensor, target_size: Union[int, int]) -> Tensor:
        x = F.interpolate(x, size=target_size)
        x = self.block(x)
        return x

class UpLayer(nn.Module):
    """FPN up layer (right side). 

    Args:
        out_features (int): Number of input features
        out_features (int): Number of output features
        block (nn.Module, optional): Block used. Defaults to UpBlock.

    """

    def __init__(self, in_features: int, out_features: int,  block: nn.Module = ConvBnAct, upsample: bool = True, *args, **kwargs):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2) if upsample else nn.Identity()
        self.block = ConvBnAct(in_features, out_features, kernel_size=1, **kwargs)

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        out = self.up(x)
        if res is not None:
            res = self.block(res)
            out += res
        return out

class FPNDecoder(nn.Module):
    """
    UNet Decoder composed of several layer of upsampling layers aimed to decrease the features space and increase the resolution.
    """

    def __init__(self, start_features: int = 512, prediction_width: int = 128, pyramid_width: int = 256, lateral_widths: List[int] = None, *args, **kwargs):
        super().__init__()
        self.widths = [prediction_width] * len(lateral_widths)
        self.in_out_block_sizes = list(zip(lateral_widths, self.widths))

        self.middle = ConvBnAct(start_features, pyramid_width, kernel_size=1, **kwargs)
        self.layers = nn.ModuleList([
            UpLayer(lateral_features, pyramid_width, **kwargs)
            for lateral_features in lateral_widths
        ])

        self.smooth_layers = nn.ModuleList([
            FPNSmoothBlock(pyramid_width, prediction_width, **kwargs)
            for _ in lateral_widths
        ])

    def forward(self, x: Tensor, residuals: List[Tensor]) -> Tensor:
        x = self.middle(x)
        p_features = [x]
        for layer, res in zip(self.layers, residuals):
            x = layer(x, res)
            p_features.append(x)
        # we do not smooth the first p features to save computational cost
        target_size = p_features[-1].shape[2:4]
        p_features = p_features[::-1][1:]
        features = []
        for layer, p in zip(self.smooth_layers, p_features):
            x = layer(p, target_size)
            features.append(x)

        return features

class Merge(nn.Module):
    def __init__(self, policy: str = 'sum'):
        super().__init__()
        # assert policy in ['sum', 'cat'] "Only `sum` and `cat` policy are supported"
        self.policy = policy

    def forward(self, features):
        x = features
        if self.policy == 'sum':
            x = torch.sum(torch.stack(features, dim=1), dim=1)
        elif self.policy == 'cat':
            x = torch.cat(features, dim=1)
        return x
        


class FPN(SegmentationModule):
    """Implementation of FPN proposed in `Feature Pyramid Networks for Object Detection <https://arxiv.org/abs/1612.03144>`_

    Examples:

       Create a default model

        >>> FPN()

        You can easily customize your model

        >>> # change activation
        >>> FPN(activation=nn.SELU)
        >>> # change number of classes (default is 2 )
        >>> FPN(n_classes=2)
        >>> # change encoder
        >>> FPN = FPN(encoder=lambda *args, **kwargs: ResNet.resnet26(*args, **kwargs).encoder,)
        >>> FPN = FPN(encoder=lambda *args, **kwargs: EfficientNet.efficientnet_b2(*args, **kwargs).encoder,)
        >>> # change decoder
        >>> FPN(decoder=partial(FPNDecoder, widths=[256, 128, 64, 32, 16]))
        >>> # pass a different block to decoder
        >>> FPN(encoder=partial(FPNEncoder, block=SENetBasicBlock))
        >>> # all *Decoder class can be directly used
        >>> FPN = FPN(encoder=partial(ResNetEncoder, block=ResNetBottleneckBlock, depths=[2,2,2,2]))

    Args:

       in_channels (int, optional): [description]. Defaults to 1.
       n_classes (int, optional): [description]. Defaults to 2.
       encoder (Encoder, optional): [description]. Defaults to UNetEncoder.
       ecoder (nn.Module, optional): [description]. Defaults to UNetDecoder.
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 2,
                 encoder: Encoder = UNetEncoder,
                 decoder: nn.Module = FPNDecoder,
                 **kwargs):

        super().__init__(in_channels, n_classes, encoder, decoder, **kwargs)
        self.head = nn.Sequential(
            Merge(),
            nn.Conv2d(self.decoder.widths[-1], n_classes, kernel_size=1))
        # self.head = nn.Identity()
