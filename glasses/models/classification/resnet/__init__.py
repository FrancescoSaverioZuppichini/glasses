from __future__ import annotations
from torch import nn
from torch import Tensor
from glasses.nn.blocks.residuals import ResidualAdd
from glasses.nn.blocks import Conv2dPad, BnActConv, ConvBnAct
from collections import OrderedDict
from typing import List
from functools import partial
from glasses.utils.PretrainedWeightsProvider import Config, pretrained
from ....models.base import Encoder, VisionModule

"""Implementation of ResNet proposed in `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`
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
        self.conv = Conv2dPad(in_features, out_features,
                              kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetShorcutD(nn.Sequential):
    """Shorcut function proposed in `Bag of Tricks for Image Classification with Convolutional Neural Networks <https://arxiv.org/pdf/1812.01187.pdf>`_

    It applies average pool instead of `stride=2` in the convolution layer

    Args:
        in_features (int): features (channels) of the input
        out_features (int): features (channels) of the desidered output
    """

    def __init__(self, in_features: int, out_features: int, stride: int = 2):
        super().__init__(OrderedDict({
            'pool': nn.AvgPool2d((2, 2)) if stride == 2 else nn.Identity(),
            'conv': Conv2dPad(in_features, out_features,
                              kernel_size=1,  bias=False),
            'bn': nn.BatchNorm2d(out_features)
        }))


class ResNetBasicBlock(nn.Module):
    """Basic ResNet block composed by two 3x3 convs with residual connection.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNetBasicBlock.png?raw=true

    *The residual connection is showed as a black line*

    The output of the layer is defined as:

    :math:`x' = F(x) + x`

    Args:
        out_features (int): Number of input features
        out_features (int): Number of output features
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
        stride (int, optional): [description]. Defaults to 1.
        conv (nn.Module, optional): [description]. Defaults to nn.Conv2d.
    """

    def __init__(self, in_features: int, out_features: int,
                 activation: nn.Module = ReLUInPlace,
                 stride: int = 1, shortcut: nn.Module = ResNetShorcut, **kwargs):
        super().__init__()
        self.should_apply_shortcut = in_features != out_features or stride != 1

        self.block = nn.Sequential(
            OrderedDict(
                {
                    'conv1': Conv2dPad(in_features, out_features, kernel_size=3, stride=stride, bias=False,  **kwargs),
                    'bn1': nn.BatchNorm2d(out_features),
                    'act1': activation(),
                    'conv2': Conv2dPad(out_features, out_features, kernel_size=3, bias=False),
                    'bn2': nn.BatchNorm2d(out_features),
                }
            ))
        self.shortcut = shortcut(
            in_features, out_features, stride=stride) if self.should_apply_shortcut else nn.Identity()

        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        res = self.shortcut(res)
        x += res
        x = self.act(x)
        return x


class ResNetBottleneckBlock(ResNetBasicBlock):
    """ResNet Bottleneck block based on the torchvision implementation.

    Even if the paper says that the first conv1x1 compresses the features. We followed the original implementation that expands the `out_features` by a factor equal to `reduction`.

    The stride is applied into the 3x3 conv, `this improves https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch`

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNetBottleNeckBlock.png?raw=true

    *The residual connection is showed as a black line*

    Args:
        out_features (int): Number of input features
        out_features (int): Number of output features
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
        stride (int, optional): [description]. Defaults to 1.
        conv (nn.Module, optional): [description]. Defaults to nn.Conv2d.
        reduction (int, optional): [description]. Defaults to 4.
    """

    def __init__(self, in_features: int, out_features: int, features: int = None, activation: nn.Module = ReLUInPlace, reduction: int = 4, stride=1, shortcut=ResNetShorcut, **kwargs):
        super().__init__(in_features, out_features, activation, stride, shortcut=shortcut)
        print
        self.features = out_features // reduction if features is None else features
        self.block = nn.Sequential(
            ConvBnAct(in_features, self.features,
                      activation=activation, kernel_size=1),
            ConvBnAct(self.features, self.features, activation=activation,
                      kernel_size=3, stride=stride, **kwargs),
            ConvBnAct(self.features, out_features,
                      activation=None, kernel_size=1),
        )


class ResNetBasicPreActBlock(ResNetBasicBlock):
    reduction: int = 1
    """Pre activation ResNet basic block proposed in `Identity Mappings in Deep Residual Networks <https://arxiv.org/pdf/1603.05027.pdf>`_

    Args:
        out_features (int): Number of input features
        out_features (int): Number of ouimport inspect
    """

    def __init__(self, in_features: int, out_features: int, activation: nn.Module = ReLUInPlace, stride: int = 1, shortcut: nn.Module = ResNetShorcut, **kwargs):
        super().__init__(in_features, out_features, activation,
                         stride=stride, shortcut=shortcut, **kwargs)
        self.block = nn.Sequential(
            OrderedDict(
                {
                    'bn1': nn.BatchNorm2d(in_features),
                    'act1': activation(),
                    'conv1': Conv2dPad(in_features, out_features, kernel_size=3, bias=False, stride=stride, **kwargs),
                    'bn2': nn.BatchNorm2d(out_features),
                    'act2': activation(),
                    'conv2': Conv2dPad(out_features, out_features, kernel_size=3, bias=False),
                }
            ))

        self.act = nn.Identity()


class ResNetBottleneckPreActBlock(ResNetBottleneckBlock):
    reduction: int = 4

    """Pre activation ResNet bottleneck block proposed in `Identity Mappings in Deep Residual Networks <https://arxiv.org/pdf/1603.05027.pdf>`_

    Args:
        out_features (int): Number of input features
        out_features (int): Number of output features
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
        stride (int, optional): [description]. Defaults to 1.
        conv (nn.Module, optional): [description]. Defaults to nn.Conv2d.
    """

    def __init__(self, in_features: int, out_features: int,  features: int = None, activation: nn.Module = ReLUInPlace, reduction: int = 4, stride=1, shortcut=ResNetShorcut, **kwargs):
        super().__init__(in_features, out_features, features,
                         activation, stride=stride, shortcut=shortcut, **kwargs)
        # TODO I am not sure it is correct
        features = out_features // reduction
        self.block = nn.Sequential(
            BnActConv(in_features, self.features, activation=activation,
                      kernel_size=1, bias=False),
            BnActConv(self.features, self.features, activation=activation, bias=False,
                      kernel_size=3, stride=stride, **kwargs),
            BnActConv(self.features, out_features,
                      activation=activation, bias=False, kernel_size=1)
        )

        self.act = nn.Identity()


class ResNetLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, block: nn.Module = ResNetBasicBlock, depth: int = 1, stride: int = 2, *args, **kwargs):
        super().__init__()
        # 'We perform stride directly by convolutional layers that have a stride of 2.'
        self.block = nn.Sequential(
            block(in_features, out_features,
                  stride=stride,  **kwargs),
            *[block(out_features,
                    out_features, **kwargs) for _ in range(depth - 1)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class ResNetStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int,  activation: nn.Module = ReLUInPlace):
        super().__init__(
            ConvBnAct(in_features, out_features,
                      activation=activation, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


class ResNetStemC(nn.Sequential):
    """
    Modified stem proposed in `Bag of Tricks for Image Classification with Convolutional Neural Networks <https://arxiv.org/pdf/1812.01187.pdf>`_

    The observation is that the computational cost of a convolution is quadratic to the kernel width or height. A 7 × 7 convolution is 5.4
    times more expensive than a 3 × 3 convolution. So this tweak replacing the 7 × 7 convolution in the input stem with three conservative 3 × 3 convolution
    """

    def __init__(self, in_features: int, out_features: int,  activation: nn.Module = ReLUInPlace):
        super().__init__(
            ConvBnAct(in_features, out_features // 2,
                      activation=activation, kernel_size=3, stride=2),
            ConvBnAct(out_features // 2, out_features // 2,
                      activation=activation, kernel_size=3),
            ConvBnAct(out_features // 2,
                      out_features, activation=activation, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


class ResNetEncoder(Encoder):
    """
    ResNet encoder composed by increasing different layers with increasing features.

    Args:
        in_channels (int, optional): [description]. Defaults to 3.
        start_features (int, optional): [description]. Defaults to 64.
        widths (List[int], optional): [description]. Defaults to [64, 128, 256, 512].
        depths (List[int], optional): [description]. Defaults to [2, 2, 2, 2].
        activation (nn.Module, optional): [description]. Defaults to ReLUInPlace.
        block (nn.Module, optional): Block used, there are several ones such as `ResNetBasicBlock` and `ResNetBottleneckBlock` . Defaults to ResNetBasicBlock.
        stem (nn.Module, optional): Stem used. Defaults to ResNetStem.
    """

    def __init__(self, in_channels: int = 3, start_features: int = 64,  widths: List[int] = [64, 128, 256, 512], depths: List[int] = [2, 2, 2, 2],
                 activation: nn.Module = ReLUInPlace, block: nn.Module = ResNetBasicBlock, stem: nn.Module = ResNetStem, downsample_first: bool = False, **kwargs):

        super().__init__()
        self.widths = widths
        self.start_features = start_features
        self.in_out_widths = list(zip(widths, widths[1:]))
        self.stem = stem(in_channels, start_features, activation)

        self.layers = nn.ModuleList([
                ResNetLayer(start_features, widths[0], depth=depths[0], activation=activation,
                            block=block, stride=2 if downsample_first else 1, **kwargs),
            *[ResNetLayer(in_features,
                          out_features, depth=n, activation=activation,
                          block=block,  **kwargs)
              for (in_features, out_features), n in zip(self.in_out_widths, depths[1:])]
        ])

    def forward(self, x):
        x = self.stem(x)
        for block in self.layers:
            x = block(x)
        return x

    @property
    def stages(self):
        return [self.stem[-2],
                *self.layers[:-1]]

    @property
    def features_widths(self):
        return [self.start_features, *self.widths[:-1]]


class ResNetHead(nn.Sequential):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.add_module('pool', nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module('flat', nn.Flatten())
        self.add_module('fc', nn.Linear(in_features, n_classes))


class ResNet(VisionModule):
    """Implementation of ResNet proposed in `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_

    Examples:

        Default models

        >>> ResNet.resnet18()
        >>> ResNet.resnet26()
        >>> ResNet.resnet34()
        >>> ResNet.resnet50()
        >>> ResNet.resnet101()
        >>> ResNet.resnet152()
        >>> ResNet.resnet200()

        Variants (d) proposed in `Bag of Tricks for Image Classification with Convolutional Neural Networks <https://arxiv.org/pdf/1812.01187.pdf>`_

        >>> ResNet.resnet26d()
        >>> ResNet.resnet34d()
        >>> ResNet.resnet50d()
        >>> # You can construct your own one by chaning `stem` and `block`
        >>> resnet101d = ResNet.resnet101(stem=ResNetStemC, block=partial(ResNetBottleneckBlock, shortcut=ResNetShorcutD))

        You can easily customize your model

        >>> # change activation
        >>> ResNet.resnet18(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> ResNet.resnet18(n_classes=100)
        >>> # pass a different block
        >>> ResNet.resnet18(block=SENetBasicBlock)
        >>> # change the steam
        >>> model = ResNet.resnet18(stem=ResNetStemC)
        >>> change shortcut
        >>> model = ResNet.resnet18(block=partial(ResNetBasicBlock, shortcut=ResNetShorcutD))
        >>> # store each feature
        >>> x = torch.rand((1, 3, 224, 224))
        >>> # get features
        >>> model = ResNet.resnet18()
        >>> # first call .features, this will activate the forward hooks and tells the model you'll like to get the features
        >>> model.encoder.features
        >>> model(torch.randn((1,3,224,224)))
        >>> # get the features from the encoder
        >>> features = model.encoder.features
        >>> print([x.shape for x in features])
        >>> #[torch.Size([1, 64, 112, 112]), torch.Size([1, 64, 56, 56]), torch.Size([1, 128, 28, 28]), torch.Size([1, 256, 14, 14])]

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.head = ResNetHead(
            self.encoder.widths[-1], n_classes)

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.head(x)
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @classmethod
    @pretrained()
    def resnet18(cls, *args,  block=ResNetBasicBlock, **kwargs) -> ResNet:
        """Creates a resnet18 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNet18.png?raw=true

        Returns:
            ResNet: A resnet18 model
        """
        model = cls(*args, **kwargs, block=block, depths=[2, 2, 2, 2])

        return model

    @classmethod
    @pretrained()
    def resnet26(cls, *args,  block=ResNetBottleneckBlock, **kwargs) -> ResNet:
        """Creates a resnet26 model

        Returns:
            ResNet: A resnet26 model
        """
        model = cls(*args, **kwargs, block=block,
                    widths=[256, 512, 1024, 2048], depths=[2, 2, 2, 2])

        return model

    @classmethod
    @pretrained()
    def resnet26d(cls, *args,  block=ResNetBottleneckBlock, **kwargs) -> ResNet:
        """Creates a resnet26d model

        Returns:
            ResNet: A resnet26d model
        """
        model = cls(*args, **kwargs, stem=ResNetStemC, block=partial(block, shortcut=ResNetShorcutD),
                    widths=[256, 512, 1024, 2048], depths=[2, 2, 2, 2])

        return model

    @classmethod
    @pretrained()
    def resnet34(cls, *args,  block=ResNetBasicBlock, **kwargs) -> ResNet:
        """Creates a resnet34 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNet34.png?raw=true

        Returns:
            ResNet: A resnet34 model
        """
        return cls(*args, **kwargs, block=block, depths=[3, 4, 6, 3])

    @classmethod
    @pretrained()
    def resnet34d(cls, *args,  block=ResNetBasicBlock, **kwargs) -> ResNet:
        """Creates a resnet34 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNet34.png?raw=true

        Returns:
            ResNet: A resnet34 model
        """
        model = cls(*args, **kwargs, stem=ResNetStemC, block=partial(block, shortcut=ResNetShorcutD),
                     depths=[3, 4, 6, 3])
        return model

    @classmethod
    @pretrained()
    def resnet50(cls, *args, block=ResNetBottleneckBlock, **kwargs) -> ResNet:
        """Creates a resnet50 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNet50.png?raw=true

        Returns:
            ResNet: A resnet50 model
        """
        return cls(*args, **kwargs, block=block, widths=[256, 512, 1024, 2048], depths=[3, 4, 6, 3])

    @classmethod
    @pretrained()
    def resnet50d(cls, *args, block=ResNetBottleneckBlock, **kwargs) -> ResNet:
        """Creates a resnet50d model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNet50.png?raw=true

        Returns:
            ResNet: A resnet50d model
        """
        return cls(*args, **kwargs,  stem=ResNetStemC, block=partial(block, shortcut=ResNetShorcutD), widths=[256, 512, 1024, 2048], depths=[3, 4, 6, 3])

    @classmethod
    @pretrained()
    def resnet101(cls, *args, block=ResNetBottleneckBlock, **kwargs) -> ResNet:
        """Creates a resnet101 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNet101.png?raw=true

        Returns:
            ResNet: A resnet101 model
        """
        return cls(*args, **kwargs, block=block,  widths=[256, 512, 1024, 2048], depths=[3, 4, 23, 3])

    @classmethod
    @pretrained()
    def resnet152(cls, *args, block=ResNetBottleneckBlock, **kwargs) -> ResNet:
        """Creates a resnet152 model

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/resnet/ResNet152.png?raw=true

        Returns:
            ResNet: A resnet152 model
        """
        return cls(*args, **kwargs, block=block,  widths=[256, 512, 1024, 2048], depths=[3, 8, 36, 3])

    @classmethod
    def resnet200(cls, *args, block=ResNetBottleneckBlock, **kwargs):
        """Creates a resnet200 model

        Returns:
            ResNet: A resnet200 model
        """
        return cls(*args, **kwargs, block=block,  widths=[256, 512, 1024, 2048], depths=[3, 24, 36, 3])
