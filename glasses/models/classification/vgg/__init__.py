from __future__ import annotations
from torch import nn
from torch import Tensor
from collections import OrderedDict
from typing import List
from functools import partial
from ..resnet import ReLUInPlace
from glasses.nn.blocks import ConvAct, ConvBnAct
from glasses.utils.PretrainedWeightsProvider import Config, pretrained
from ....models.base import VisionModule, Encoder


"""Implementations of VGG proposed in `Very Deep Convolutional Networks For Large-Scale Image Recognition <https://arxiv.org/pdf/1409.1556.pdf>`_
"""

VGGBasicBlock = partial(ConvAct, kernel_size=3, bias=True)


class VGGLayer(nn.Module):
    """ This class implements a VGG layer, which is composed by a number of blocks (default is VGGBasicBlock, which is a simple 
    convolution-activation transformation) eventually followed by maxpooling.

    Args:
        in_channels (int): [description]
        out_channels (int): [description]
        block (nn.Module, optional): [description]. Defaults to VGGBasicBlock.
        n (int, optional): [description]. Defaults to 1.
        maxpool (nn.Module, optional): [description]. Defaults to nn.MaxPool2d.
    """

    def __init__(self, in_features: int, out_features: int, block: nn.Module = VGGBasicBlock, n: int = 1, maxpool: nn.Module = nn.MaxPool2d, *args, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            block(in_features, out_features, *args, **kwargs),
            *[block(out_features,
                    out_features, *args, **kwargs) for _ in range(n - 1)]
        )

        self.block.add_module('maxpool', maxpool(kernel_size=2, stride=2))

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class VGGEncoder(nn.Module):
    """VGG encoder, composed by default by a sequence of VGGLayer modules with an increasing number of output features.

    Args:
        in_channels (int, optional): [description]. Defaults to 3.
        widths (List[int], optional): [description]. Defaults to [64, 128, 256, 512, 512].
        depths (List[int], optional): [description]. Defaults to [1, 1, 2, 2, 2].
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
        block (nn.Module, optional): [description]. Defaults to VGGBasicBlock.
    """

    def __init__(self, in_channels: int = 3, widths: List[int] = [64, 128, 256, 512, 512], depths: List[int] = [1, 1, 2, 2, 2],
                 activation: nn.Module = ReLUInPlace, block: nn.Module = VGGBasicBlock, *args, **kwargs):

        super().__init__()

        self.widths = widths
        self.out_features = widths[-1]
        self.in_out_block_sizes = list(
            zip(widths[:-1], widths[1:]))

        self.stem = nn.Identity()
        self.layers = nn.ModuleList([
            VGGLayer(in_channels, widths[0], activation=activation,
                     block=block, n=depths[0], *args, **kwargs),
            *[VGGLayer(in_channels, out_channels, activation=activation, block=block, n=n, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.layers:
            x = block(x)
        return x


class VGGHead(nn.Sequential):
    """This class represents the classifier of VGG. It converts the filters into 6x6 by means of the average pooling. Then, it maps the output to the
    correct class by means of fully connected layers. Dropout is used to decrease the overfitting.

        Args:
        out_features (int): Number of input features
        n_classes (int): [description]
    """

    def __init__(self, in_features: int, n_classes: int):
        super().__init__(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(in_features * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_classes)
        )


class VGG(VisionModule):
    """Implementation of VGG proposed in `Very Deep Convolutional Networks For Large-Scale Image Recognition <https://arxiv.org/pdf/1409.1556.pdf>`_

    Create a default model

    Examples:
        >>> VGG.vgg11()
        >>> VGG.vgg13()
        >>> VGG.vgg16()
        >>> VGG.vgg19()
        >>> VGG.vgg11_bn()
        >>> VGG.vgg13_bn()
        >>> VGG.vgg16_bn()
        >>> VGG.vgg19_bn()

    Please be aware that the `bn` models uses BatchNorm but they are very old and people back then don't know the bias is superfluous 
    in a conv followed by a batchnorm.

    Customization

    You can easily create your custom VGG-like model

    Examples:
        >>> # change activation
        >>> VGG.vgg11(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> VGG.vgg11(n_classes=100)
        >>> # pass a different block
        >>> from nn.models.classification.senet import SENetBasicBlock
        >>> VGG.vgg11(block=SENetBasicBlock)
        >>> # store the features tensor after every block
        >>> x = torch.rand((1, 3, 224, 224))
        >>> model = VGG.vgg11()
        >>> features = []
        >>> for block in model.encoder.layers:
            >>> x = block(x)
            >>> features.append(x)
        >>> print([x.shape for x in features])
        >>> # [torch.Size([1, 64, 112, 112]), torch.Size([1, 128, 56, 56]), torch.Size([1, 256, 28, 28]), torch.Size([1, 512, 14, 14]), torch.Size([1, 512, 7, 7])]

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = VGGEncoder(in_channels, *args, **kwargs)
        self.head = VGGHead(self.encoder.out_features, n_classes)
        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.head(x)
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @classmethod
    @pretrained()
    def vgg11(cls, *args, **kwargs) -> VGG:
        """Creates a vgg11 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/VGG11.png?raw=true

        Returns:
            VGG: A vgg11 model
        """
        return VGG(*args, **kwargs)

    @classmethod
    @pretrained()
    def vgg13(cls, *args, **kwargs) -> VGG:
        """Creates a vgg13 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/VGG13.png?raw=true

        Returns:
            VGG: A vgg13 model
        """
        return VGG(*args, depths=[2, 2, 2, 2, 2], **kwargs)

    @classmethod
    @pretrained()
    def vgg16(cls, *args, **kwargs) -> VGG:
        """Creates a vgg16 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/VGG16.png?raw=true

        Returns:
            VGG: A vgg16 model
        """
        return VGG(*args, depths=[2, 2, 3, 3, 3], **kwargs)

    @classmethod
    @pretrained()
    def vgg19(cls, *args, **kwargs) -> VGG:
        """Creates a vgg19 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/VGG19.png?raw=true

        Returns:
            VGG: A vgg19 model
        """
        return VGG(*args, depths=[2, 2, 4, 4, 4], **kwargs)

    @classmethod
    @pretrained()
    def vgg11_bn(cls, *args, **kwargs) -> VGG:
        """Creates a vgg11 model with batchnorm

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/VGG13.png?raw=true

        Returns:
            VGG: A vgg13 model
        """
        return VGG(*args, block=ConvBnAct,  kernel_size=3, bias=True, **kwargs)

    @classmethod
    @pretrained()
    def vgg13_bn(cls, *args, **kwargs) -> VGG:
        """Creates a vgg13 model with batchnorm

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/VGG13.png?raw=true

        Returns:
            VGG: A vgg13 model
        """
        return VGG(*args, block=ConvBnAct, depths=[2, 2, 2, 2, 2], kernel_size=3, bias=True, **kwargs)

    @classmethod
    @pretrained()
    def vgg16_bn(cls, *args, **kwargs) -> VGG:
        """Creates a vgg16 model with batchnorm

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/VGG16.png?raw=true

        Returns:
            VGG: A vgg16 model
        """
        return VGG(*args, block=ConvBnAct, depths=[2, 2, 3, 3, 3], kernel_size=3, bias=True, **kwargs)

    @classmethod
    @pretrained()
    def vgg19_bn(cls, *args, **kwargs) -> VGG:
        """Creates a vgg19 model with batchnorm

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/VGG19.png?raw=true

        Returns:
            VGG: A vgg19 model
        """
        return VGG(*args,  block=ConvBnAct, depths=[2, 2, 4, 4, 4], kernel_size=3, bias=True, **kwargs)
