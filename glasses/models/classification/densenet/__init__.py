from __future__ import annotations
import torch
from torch import nn
from torch import Tensor
from ..resnet import ResNetEncoder, ResNetHead, ResNet, ResNetStem
from collections import OrderedDict
from typing import List
from functools import partial
from ..resnet import ReLUInPlace
from glasses.nn.blocks.residuals import ResidualCat2d
from glasses.nn.blocks import Conv2dPad
from ....models.base import VisionModule

from glasses.utils.PretrainedWeightsProvider import Config, pretrained


class DenseNetBasicBlock(nn.Module):
    """Basic DenseNet block composed by one 3x3 convs with residual connection.
    The residual connection is perfomed by concatenate the input and the output.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNetBasicBlock.png?raw=true

    Args:
        out_features (int): Number of input features
        out_features (int): Number of output features
        conv (nn.Module, optional): [description]. Defaults to nn.Conv2d.
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
    """

    def __init__(self, in_features: int, out_features: int,  activation: nn.Module = ReLUInPlace, *args, **kwargs):
        super().__init__()
        self.block = nn.Sequential(OrderedDict({
            'bn': nn.BatchNorm2d(in_features),
            'act': activation(),
            'conv': Conv2dPad(in_features, out_features, kernel_size=3, *args, **kwargs)
        }))

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        return torch.cat([res, x], dim=1)


class DenseBottleNeckBlock(DenseNetBasicBlock):

    """Bottleneck block composed by two preactivated layer of convolution.
    The expensive 3x3 conv is computed after a cheap 1x1 conv donwsample the input resulting in less parameters.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNetBottleneckBlock.png?raw=true

    Args:
        out_features (int): Number of input features
        out_features (int): Number of output features
        conv (nn.Module, optional): [description]. Defaults to nn.Conv2d.
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
        expansion (int, optional): [description]. Defaults to 4.

    """

    def __init__(self, in_features: int, out_features: int,  activation: nn.Module = ReLUInPlace, expansion: int = 4, *args, **kwargs):
        super().__init__(in_features, out_features,  activation, *args, **kwargs)
        self.expansion = expansion
        self.expanded_features = out_features * self.expansion

        self.block = nn.Sequential(OrderedDict({
            'bn1': nn.BatchNorm2d(in_features),
            'act1': activation(),
            'conv1': Conv2dPad(in_features, self.expanded_features, kernel_size=1, bias=False, *args, **kwargs),
            'bn2': nn.BatchNorm2d(self.expanded_features),
            'act2': activation(),
            'conv2': Conv2dPad(self.expanded_features, out_features, kernel_size=3, bias=False, *args, **kwargs)
        }))


class TransitionBlock(nn.Module):
    """A transition block is used to downsample the output using 1x1 conv followed by 2x2 average pooling.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNetTransitionBlock.png?raw=true

    Args:
        out_features (int): Number of input features
        factor (int, optional): Reduction factor applied on the in_features. Defaults to 2
        conv (nn.Module, optional): [description]. Defaults to nn.Conv2d.
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
    """

    def __init__(self, in_features: int, factor: int = 2, activation: nn.Module = ReLUInPlace):
        super().__init__()
        self.block = nn.Sequential(
            OrderedDict(
                {
                    'bn': nn.BatchNorm2d(in_features),
                    'act': activation(),
                    'conv': Conv2dPad(in_features, in_features // factor,
                                      kernel_size=1, bias=False),
                    'pool': nn.AvgPool2d(kernel_size=2, stride=2)
                }
            ))

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class DenseNetLayer(nn.Module):
    """A DenseNet layer is composed by `n` `blocks` stacked together followed by a transition to downsample the output features.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNetLayer.png?raw=true

    Args:
        out_features (int): Number of input features
        grow_rate (int, optional): [description]. Defaults to 32.
        n (int, optional): [description]. Defaults to 4.
        block (nn.Module, optional): [description]. Defaults to DenseNetBasicBlock.
        transition_block (nn.Module, optional): A module applied after the block(s). Defaults to TransitionBlock.
    """

    def __init__(self, in_features: int, grow_rate: int = 32, n: int = 4,
                 block: nn.Module = DenseBottleNeckBlock, transition_block: nn.Module = TransitionBlock, *args, **kwargs):
        super().__init__()
        self.out_features = grow_rate * n + in_features
        self.block = nn.Sequential(
            # in each block, the number of features is equal to the input size + the outputs of all the previos layers (grow_rate * i)
            *[block(grow_rate * i + in_features, grow_rate, *args, **kwargs)
              for i in range(n)],
            # reduce the output features by a factor of 2
            transition_block(self.out_features, *args, **
                             kwargs) if transition_block else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class DenseNetEncoder(ResNetEncoder):
    """DenseNet encoder composed by multiple `DeseNetLayer` with increasing features size. The `.stem` is the same used in `ResNet`

    Args:
        in_channels (int, optional): [description]. Defaults to 3.
        start_features (int, optional): [description]. Defaults to 64.
        grow_rate (int, optional): [description]. Defaults to 32.
        depths (List[int], optional): [description]. Defaults to [4, 4, 4, 4].
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
        block (nn.Module, optional): [description]. Defaults to DenseNetBasicBlock.
    """

    def __init__(self, in_channels: int = 3, start_features: int = 64,  grow_rate: int = 32,
                 depths: List[int] = [4, 4, 4, 4],
                 activation: nn.Module = ReLUInPlace, block: nn.Module = DenseBottleNeckBlock, *args, **kwargs):
        super().__init__(in_channels)
        self.layers = nn.ModuleList([])
        self.widths = [start_features]
        # [REVIEW] I should decide if I want to have `start_features` or just widths
        self.stem = ResNetStem(in_channels, start_features, activation)
        in_features = start_features

        for deepth in depths[:-1]:
            self.layers.append(DenseNetLayer(
                in_features, grow_rate, deepth, block=block, *args, **kwargs))
            # in each layer the in_features are equal the features we have so far + the number of layer multiplied by the grow rate
            in_features += deepth * grow_rate
            in_features //= 2
            self.widths.append(in_features)

        self.widths.append(in_features + depths[-1] * grow_rate)

        self.layers.append(DenseNetLayer(
            in_features, grow_rate, depths[-1], block=block, *args,
            transition_block=lambda x: nn.Sequential(
                nn.BatchNorm2d(self.widths[-1]),
                activation()
            ), **kwargs))


class DenseNet(VisionModule):
    """Implementation of DenseNet proposed in `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_

    Create a default models

    Examples:
        >>> DenseNet.densenet121()
        >>> DenseNet.densenet161()
        >>> DenseNet.densenet169()
        >>> DenseNet.densenet201()

    You can easily customize your model

        >>> # change activation
        >>> DenseNet.densenet121(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> DenseNet.densenet121(n_classes=100)
        >>> # pass a different block
        >>> DenseNet.densenet121(block=...)
        >>> # change the initial convolution
        >>> model = DenseNet.densenet121()
        >>> model.encoder.gate.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        >>> # store each feature
        >>> x = torch.rand((1, 3, 224, 224))
        >>> model = DenseNet.densenet121()
        >>> # first call .features, this will activate the forward hooks and tells the model you'll like to get the features
        >>> model.encoder.features
        >>> model(torch.randn((1,3,224,224)))
        >>> # get the features from the encoder
        >>> features = model.encoder.features
        >>> print([x.shape for x in features])
        >>> # [torch.Size([1, 128, 28, 28]), torch.Size([1, 256, 14, 14]), torch.Size([1, 512, 7, 7]), torch.Size([1, 1024, 7, 7])]


    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    def __init__(self, in_channels: int = 3,  n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = DenseNetEncoder(in_channels, *args, **kwargs)
        self.head = ResNetHead(
            self.encoder.widths[-1], n_classes)
            
    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.head(x)
        return x

    @classmethod
    @pretrained()
    def densenet121(cls, *args, **kwargs) -> DenseNet:
        """Creates a densenet121 model. *Grow rate* is set to 32

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNet121.png?raw=true

        Returns:
            DenseNet: A densenet121 model
        """
        return DenseNet(*args, grow_rate=32, depths=[6, 12, 24, 16], **kwargs)

    @classmethod
    @pretrained()
    def densenet161(cls, *args, **kwargs) -> DenseNet:
        """Creates a densenet161 model. *Grow rate* is set to 48

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNet161.png?raw=true

        Returns:
            DenseNet: A densenet161 model
        """
        return DenseNet(*args, start_features=96, grow_rate=48, depths=[6, 12, 36, 24], **kwargs)

    @classmethod
    @pretrained()
    def densenet169(cls, *args, **kwargs) -> DenseNet:
        """Creates a densenet169 model. *Grow rate* is set to 32

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNet169.png?raw=true

        Returns:
            DenseNet: A densenet169 model
        """
        return DenseNet(*args, grow_rate=32, depths=[6, 12, 32, 32], **kwargs)

    @classmethod
    @pretrained()
    def densenet201(cls, *args, **kwargs) -> DenseNet:
        """Creates a densenet201 model. *Grow rate* is set to 32

         .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNet201.png?raw=true

         Returns:
             DenseNet: A densenet201 model
         """
        return DenseNet(*args, grow_rate=32, depths=[6, 12, 48, 32], **kwargs)
