from __future__ import annotations
import torch
from torch import nn
from torch import Tensor
from glasses.nn.blocks.residuals import ResidualAdd
from glasses.nn.blocks import Conv2dPad, ConvBnAct, BnActConv
from ..resnet import ResNetShorcut, ResNetLayer
from collections import OrderedDict
from typing import List
from functools import partial
from ..resnet import ResNetBottleneckBlock, ReLUInPlace, ResNetEncoder, ResNetShorcut, ResNetBottleneckPreActBlock, ResNetStemC
from glasses.nn.att import ChannelSE
from ....models.base import VisionModule, Encoder


FishNetShortCut = partial(BnActConv, kernel_size=1)


class FishNetChannelReductionShortcut(nn.Module):
    r"""Channel reduction output :math:`r(x)` is computed as follows:


    :math:`r(x)=\hat{x}=\left[\hat{x}(1), \hat{x}(2), \ldots, \hat{x}\left(c_{o u t}\right)\right], \quad \hat{x}(n)=\sum_{j=0}^{k} x(k \cdot n+j), n \in\left\{0,1, \ldots, c_{o u t}\right\}`

    Where :math:`k = \frac{c_{in}}{c_{ou}}` 

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
    """

    def __init__(self, in_features: int, out_features: int, *args, **kwargs):
        super().__init__()
        self.k = in_features // out_features

    def forward(self, x: Tensor) -> Tensor:
        depth, c, h, w = x.size()
        x_red = x.view(depth, c // self.k, self.k, h, w).sum(2)
        return x_red


FishNetBottleNeck = partial(
    ResNetBottleneckPreActBlock, shortcut=FishNetShortCut)


class FishNetBodyBlock(nn.Module):
    """FishNet body block, called the Up-sampling & Refinement block in the paper.

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        trans_features (int): [description]
        block (nn.Module, optional): [description]. Defaults to FishNetBottleNeck.
        depth (int, optional): [description]. Defaults to 1.
        trans_depth (int, optional): [description]. Defaults to 1.
    """

    def __init__(self, in_features: int, out_features: int, trans_features: int, block: nn.Module = FishNetBottleNeck, depth: int = 1, trans_depth: int = 1, *args, **kwargs):
        super().__init__()

        self.transfer = nn.Sequential(
            *[block(trans_features, trans_features) for _ in range(trans_depth)])

        self.block = nn.Sequential(
            block(in_features,  out_features,
                  shortcut=FishNetChannelReductionShortcut, *args, **kwargs),
            *[block(out_features, out_features, *args, **kwargs)
              for _ in range(depth-1)],
            nn.Upsample(scale_factor=2))

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        x = self.block(x)
        res = self.transfer(res)
        x = torch.cat([x, res], dim=1)
        return x


class FishNetHeadBlock(FishNetBodyBlock):
    """FishNet head block, called the Down-sampling & Refinement block in the paper.

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        trans_features (int): [description]
        block (nn.Module, optional): [description]. Defaults to FishNetBottleNeck.
        depth (int, optional): [description]. Defaults to 1.
        trans_depth (int, optional): [description]. Defaults to 1.
    """

    def __init__(self, in_features: int, out_features: int, trans_features: int, block: nn.Module = FishNetBottleNeck, depth: int = 1,  trans_depth: int = 1, *args, **kwargs):
        super().__init__(in_features, out_features,
                         trans_features, block, depth, trans_depth, *args, **kwargs)

        self.block = nn.Sequential(
            block(in_features,  out_features,
                  shortcut=ResNetShorcut, *args, **kwargs),
            *[block(out_features, out_features, *args, **kwargs)
              for _ in range(depth-1)],

            nn.MaxPool2d(kernel_size=2, stride=2))


class FishNetBrigde(nn.Module):
    def __init__(self, in_features: int, out_features: int, block: nn.Module = FishNetBottleNeck, depth: int = 1, activation: nn.Module = ReLUInPlace, *args, **kwargs):
        """A weird layer that 'bridges' the tail and the body of the model.

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            block (nn.Module, optional): [description]. Defaults to FishNetBottleNeck.
            depth (int, optional): [description]. Defaults to 1.
        """
        super().__init__()
        self.stem = nn.Sequential(
            BnActConv(in_features, in_features //
                      2, activation=activation, kernel_size=1, bias=False),
            BnActConv(in_features//2, in_features *
                      2, activation=activation, kernel_size=1),
        )

        self.block = nn.Sequential(FishNetBottleNeck(in_features*2, out_features, activation=activation),
                                   *[FishNetBottleNeck(out_features, out_features, activation=activation) for _ in range(depth - 1)])
        # very wrong SE implementation and application -> I have contacted the authors and he confirmed they got it wrong.
        self.att = nn.Sequential(nn.BatchNorm2d(in_features * 2),
                                 activation(),
                                 nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_features*2, in_features //
                                           16, kernel_size=1),
                                 activation(),
                                 nn.Conv2d(in_features//16,
                                           out_features, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        att = self.att(x)
        x = self.block(x)

        return (x * att) + att


class FishNetTailBlock(nn.Module):
    """FishNet Tail Block, simi

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        depth (int, optional): [description]. Defaults to 1.
        block (nn.Module, optional): [description]. Defaults to FishNetBottleNeck.
    """

    def __init__(self, in_features: int, out_features: int, depth: int = 1,
                 block: nn.Module = FishNetBottleNeck, *args, **kwargs):
        super().__init__()
        self.block = nn.Sequential(block(in_features, out_features, **kwargs),
                                   *[block(out_features, out_features, **kwargs)
                                     for _ in range(depth-1)],
                                   nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class FishNetHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, block: nn.Module = FishNetBodyBlock, depth: int = 1, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x

class FishNetEncoder(nn.Module):
    """
    FishNetEncoder encoder composed by a tail, body and head.

    The following image is taken from the paper and shows the architecture detail.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/FishNetEncoder.png?raw=true

    Args:
        in_channels (int, optional): [description]. Defaults to 3.
        start_features (int, optional): [description]. Defaults to 64.
        tail_depths (List[int], optional): [description]. Defaults to [1, 1, 1].
        body_depths (List[int], optional): [description]. Defaults to [1, 1, 1].
        body_trans_depths (List[int], optional): [description]. Defaults to [1, 1, 1].
        head_depths (List[int], optional): [description]. Defaults to [1, 1, 1].
        head_trans_depths (List[int], optional): [description]. Defaults to [1, 1, 1].
        bridge_depth (int, optional): [description]. Defaults to 1.
        block (nn.Module, optional): [description]. Defaults to FishNetBottleNeck.
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
    """

    def __init__(self, in_channels: int = 3, start_features: int = 64,
                 tail_depths: List[int] = [1, 1, 1],
                 body_depths: List[int] = [1, 1, 1],
                 body_trans_depths: List[int] = [1, 1, 1],
                 head_depths: List[int] = [1, 1, 1],
                 head_trans_depths: List[int] = [1, 1, 1],
                 bridge_depth: int = 1,
                 block: nn.Module = FishNetBottleNeck,
                 stem: nn.Module = ResNetStemC,
                 activation: nn.Module = ReLUInPlace,  *args, **kwargs):
        super().__init__()

        self.stem = stem(in_channels, start_features, activation)

        self.tail_widths, self.body_widths, self.head_widths = self.find_widths(
            start_features, len(tail_depths))

        self.tail = nn.ModuleList([
            FishNetTailBlock(in_features, out_features, depth=depth,
                             block=block, activation=activation, **kwargs)
            for (in_features, out_features), depth in zip(self.tail_widths, tail_depths)]
        )

        self.bridge = FishNetBrigde(
            self.tail_widths[-1][-1], self.body_widths[0][0], depth=bridge_depth, block=block, activation=activation)

        self.body = nn.ModuleList([])

        for i, (tail_w, (in_features, out_features), depth, trans_depth) in enumerate(zip(self.tail_widths[::-1], self.body_widths, body_depths, body_trans_depths)):
            self.body.append(FishNetBodyBlock(
                in_features, out_features, tail_w[0], depth=depth, trans_depth=trans_depth, block=block, activation=activation, dilation=2**i, padding=2**i))

        self.head = nn.ModuleList([])

        for body_w, (in_features, out_features), depth, trans_depth in zip(self.body_widths[::-1], self.head_widths, head_depths, head_trans_depths):
            self.head.append(FishNetHeadBlock(
                in_features, out_features, body_w[0], depth=depth, trans_depth=trans_depth, block=block, activation=activation))

    def forward(self, x):
        x = self.stem(x)
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
        # down
        residuals = residuals[::-1]
        for block, res in zip(self.head, residuals):
            x = block(x, res)

        return x

    @staticmethod
    def find_widths(start_features: int = 64, depth: int = 3) -> List[int]:
        """
        This code iteratively computes the correnct number of in and out features for each FishNet layer.

        Code copied from `Fishnet-PyTorch <https://github.com/zsef123/Fishnet-PyTorch>`_

        Args:
            start_features (int, optional): [description]. Defaults to 64.
            depth (int, optional): [description]. Defaults to 3.

        Returns:
            List[int]: [description]
        """
        # from
        depth = 3
        start_features = 64
        tail_channels = [(start_features, start_features*2)]
        for i in range(depth - 1):
            tail_channels.append(
                (tail_channels[-1][1], tail_channels[-1][1] * 2))

        in_c, transfer_c = tail_channels[-1][1], tail_channels[-2][1]
        body_channels = [
            (in_c, in_c), (in_c + transfer_c, (in_c + transfer_c)//2)]
        # First body module is not change feature map channel
        for i in range(1, depth-1):
            transfer_c = tail_channels[-i-2][1]
            in_c = body_channels[-1][1] + transfer_c
            body_channels.append((in_c, in_c//2))

        in_c = body_channels[-1][1] + tail_channels[0][0]
        head_channels = [(in_c, in_c)]
        for i in range(depth):
            transfer_c = body_channels[-i-1][0]
            in_c = head_channels[-1][1] + transfer_c
            head_channels.append((in_c, in_c))

        return tail_channels, body_channels, head_channels


class FishNetDecoder(nn.Sequential):
    """
    FishNet Decoder composed by 1x1 convs.
    """

    def __init__(self, in_features: int, n_classes: int, activation: nn.Module = nn.ReLU):
        super().__init__(
            nn.BatchNorm2d(in_features),
            activation(),
            nn.Conv2d(in_features, in_features//2, 1, bias=False),
            nn.BatchNorm2d(in_features//2),
            activation(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features // 2, n_classes, 1, bias=True),
            nn.Flatten()
        )


class FishNet(VisionModule):
    """Implementation of ResNet proposed in `FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction <https://arxiv.org/abs/1901.03495>`_

    Honestly, this model it is very weird and it has some mistakes in the paper that nobody ever cared to correct. It is a nice idea, but it could have been described better and definitly implemented better.
    The author's code is terrible, I have based mostly of my implemente on this amazing repo `Fishnet-PyTorch <https://github.com/zsef123/Fishnet-PyTorch>`_.

    The following image is taken from the paper and shows the architecture detail.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/FishNet.png?raw=true


    Create a default model

    Examples:
    
        >>> FishNet.fishnet99()
        >>> FishNet.fishnet150()


        You can easily customize your model

        >>> FishNet.fishnet99(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> FishNet.fishnet99(n_classes=100)
        >>> # pass a different block
        >>> block = lambda in_ch, out_ch, **kwargs: nn.Sequential(FishNetBottleNeck(in_ch, out_ch), SpatialSE(out_ch))
        >>> FishNet.fishnet99(block=block)

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = FishNetEncoder(in_channels, *args, **kwargs)
        self.head = FishNetDecoder(
            self.encoder.head_widths[-1][1], n_classes)

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.head(x)
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
        """Return a fishnet99 model

        Returns:
            FishNet: [description]
        """
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

    @classmethod
    def fishnet150(self, *args, **kwargs) -> FishNet:
        """Return a fishnet150 model

        Returns:
            FishNet: [description]
        """
        start_features = 64

        tail_depths = [2, 4, 8]
        bridge_depth = 4

        body_depths = [2, 2, 2]
        body_trans_depths = [2, 2, 2]

        head_depths = [2, 2, 4]
        head_trans_depths = [2, 2, 4]

        return FishNet(*args, start_features=start_features,
                       tail_depths=tail_depths,
                       bridge_depth=bridge_depth,
                       body_depths=body_depths,
                       body_trans_depths=body_trans_depths,
                       head_depths=head_depths,
                       head_trans_depths=head_trans_depths, **kwargs)
