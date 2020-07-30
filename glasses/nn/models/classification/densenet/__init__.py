from __future__ import annotations
import torch
from torch import nn
from torch import Tensor
from ..resnet import ResNetEncoder, ResnetDecoder, ResNet
from collections import OrderedDict
from typing import List
from functools import partial
from ..resnet import ReLUInPlace


class DenseNetBasicBlock(nn.Module):
    """Basic DenseNet block composed by one 3x3 convs with residual connection.
    The residual connection is perfomed by concatenate the input and the output.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNetBasicBlock.png?raw=true

    Args:
        in_features (int): [description]
        out_features (int): [description]
        conv (nn.Module, optional): [description]. Defaults to nn.Conv2d.
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
    """

    def __init__(self, in_features: int, out_features: int, conv: nn.Module = nn.Conv2d, activation: nn.Module = ReLUInPlace, *args, **kwargs):
        super().__init__()
        self.block = nn.Sequential(OrderedDict({
            'bn': nn.BatchNorm2d(in_features),
            'act': activation(),
            'conv': conv(in_features, out_features, kernel_size=3, padding=1, *args, **kwargs)
        }))

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x = torch.cat([res, x], dim=1)
        return x


class DenseBottleNeckBlock(DenseNetBasicBlock):
    expansion: int = 4

    """Bottleneck block composed by two preactivated layer of convolution.
    The expensive 3x3 conv is computed after a cheap 1x1 conv donwsample the input resulting in less parameters.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNetBottleneckBlock.png?raw=true

    Args:
        in_features (int): [description]
        out_features (int): [description]
        conv (nn.Module, optional): [description]. Defaults to nn.Conv2d.
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
        expansion (int, optional): [description]. Defaults to 4.

    """

    def __init__(self, in_features: int, out_features: int, conv: nn.Module = nn.Conv2d, activation: nn.Module = ReLUInPlace, expansion: int = 4, *args, **kwargs):
        super().__init__(in_features, out_features, conv, activation, *args, **kwargs)
        self.expansion = expansion
        self.expanded_features = out_features * self.expansion

        self.block = nn.Sequential(OrderedDict({
            'bn1': nn.BatchNorm2d(in_features),
            'act1': activation(),
            'conv1': conv(in_features, self.expanded_features, kernel_size=1, bias=False, *args, **kwargs),
            'bn2': nn.BatchNorm2d(self.expanded_features),
            'act2': activation(),
            'conv2': conv(self.expanded_features, out_features, kernel_size=3, padding=1,  bias=False, *args, **kwargs)
        }))


class TransitionBlock(nn.Module):
    """A transition block is used to downsample the output using 1x1 conv followed by 2x2 average pooling. The output's features are further reduced by half.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNetTransitionBlock.png.png?raw=true

    Args:
        in_features (int): [description]
        factor (int, optional): Reduction factor. Defaults to 2.
        conv (nn.Module, optional): [description]. Defaults to nn.Conv2d.
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
    """

    def __init__(self, in_features: int, factor: int = 2, conv: nn.Module = nn.Conv2d, activation: nn.Module = ReLUInPlace):
        super().__init__()
        self.block = nn.Sequential(
            OrderedDict(
                {
                    'bn': nn.BatchNorm2d(in_features),
                    'act': activation(),
                    'conv': conv(in_features, in_features // factor,
                                 kernel_size=1, bias=False),
                    'pool': nn.AvgPool2d(kernel_size=2, stride=2)
                }
            ))

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class DenseNetLayer(nn.Module):
    """A DenseNet layer is composed by `n` `blocks` stacked together followed by a transition to downsample the output features.
    To disable the transition block simply pass `None`.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNetLayer.png.png?raw=true

    Args:
        in_features (int): [description]
        grow_rate (int, optional): [description]. Defaults to 32.
        n (int, optional): [description]. Defaults to 4.
        block (nn.Module, optional): [description]. Defaults to DenseNetBasicBlock.
        transition_block (nn.Module, optional): [description]. Defaults to TransitionBlock.
    """

    def __init__(self, in_features: int, grow_rate: int = 32, n: int = 4, block: nn.Module = DenseBottleNeckBlock, transition_block: nn.Module = TransitionBlock, *args, **kwargs):
        super().__init__()
        self.out_features = grow_rate * n + in_features
        self.block = nn.Sequential(
            *[block(grow_rate * i + in_features, grow_rate, *args, **kwargs)
              for i in range(n)],
            # reduce the output's features
            transition_block(self.out_features, *args, **
                             kwargs) if transition_block else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class DenseNetEncoder(ResNetEncoder):

    """The encoder composed by multiple `DeseNetLayer` with increasing features size. The `.gate` is the same used in `ResNet`

    Args:
        in_channels (int, optional): [description]. Defaults to 3.
        start_features (int, optional): Initial gate features size. Defaults to 64.
        grow_rate (int, optional): Number of features in each conv block. Defaults to 32.
        depths (List[int], optional): List of layer's depth, each number is the number of blocks in the i-th layer. Defaults to [4, 4, 4, 4].
        activation (nn.Module, optional): Activation function used. Defaults to ReLUInPlace.
        block (nn.Module, optional): Block used. Defaults to DenseNetBasicBlock.
    """

    def __init__(self, in_channels: int = 3, start_features: int = 64,  grow_rate: int = 32,
                 depths: List[int] = [4, 4, 4, 4],
                 activation: nn.Module = ReLUInPlace, block: nn.Module = DenseBottleNeckBlock, *args, **kwargs):
        super().__init__(in_channels, [64])

        self.blocks = nn.ModuleList([])

        in_features = start_features

        for deepth in depths[:-1]:
            self.blocks.append(DenseNetLayer(
                in_features, grow_rate, deepth, block=block, *args, **kwargs))
            # in each layer the in_features are equal the features we have so far + the number of layer multiplied by the grow rate
            in_features += deepth * grow_rate
            in_features //= 2
        # last layer does not have a transiction block
        self.blocks.append(DenseNetLayer(
            in_features, grow_rate, depths[-1], block=block, transition_block=None, *args, **kwargs))
        self.out_features = in_features + depths[-1] * grow_rate
        # TODO should the bast `bn` be in the `Encoder?`
        self.bn = nn.BatchNorm2d(self.out_features)

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        x = self.bn(x)
        return x


class DenseNet(nn.Module):
    """Implementations of DenseNet proposed in `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_

    Create a default model

    Examples:
        >>> DenseNet.densenet121()
        >>> DenseNet.densenet161()
        >>> DenseNet.densenet169()
        >>> DenseNet.densenet201()

    Customization

    You can easily customize your densenet

    Examples:
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
        >>> features = []
        >>> x = model.encoder.gate(x)
        >>> for block in model.encoder.blocks:
            >>> x = block(x)
            >>> features.append(x)
        >>> print([x.shape for x in features])
        # [torch.Size([1, 128, 28, 28]), torch.Size([1, 256, 14, 14]), torch.Size([1, 512, 7, 7]), torch.Size([1, 1024, 7, 7])]

        >>>

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    def __init__(self, in_channels: int = 3,  n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = DenseNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(
            self.encoder.out_features, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @classmethod
    def densenet121(cls, *args, **kwargs) -> DenseNet:
        """Creates a densenet121 model. *Grow rate* is set to 32

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNet121.png?raw=true

        Returns:
            DenseNet: A densenet121 model
        """
        return DenseNet(*args, grow_rate=32, depths=[6, 12, 24, 16], **kwargs)

    @classmethod
    def densenet161(cls, *args, **kwargs) -> DenseNet:
        """Creates a densenet161 model. *Grow rate* is set to 48

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNet161.png?raw=true

        Returns:
            DenseNet: A densenet161 model
        """
        return DenseNet(*args, grow_rate=48, depths=[6, 12, 36, 24], **kwargs)

    @classmethod
    def densenet169(cls, *args, **kwargs) -> DenseNet:
        """Creates a densenet169 model. *Grow rate* is set to 32

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNet169.png?raw=true

        Returns:
            DenseNet: A densenet169 model
        """
        return DenseNet(*args, grow_rate=32, depths=[6, 12, 32, 32], **kwargs)

    @classmethod
    def densenet201(cls, *args, **kwargs) -> DenseNet:
        """Creates a densenet201 model. *Grow rate* is set to 32

         .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DenseNet201.png?raw=true

         Returns:
             DenseNet: A densenet201 model
         """
        return DenseNet(*args, grow_rate=32, depths=[6, 12, 48, 32], **kwargs)
