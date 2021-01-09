from __future__ import annotations
import torch
import numpy as np
from torch import nn
from torch import Tensor
from glasses.nn.blocks.residuals import ResidualAdd
from collections import OrderedDict
from typing import List, Union, Dict
from functools import partial
from glasses.nn.blocks import Conv2dPad, ConvBnAct
from glasses.nn.att import ChannelSE
from ....models.utils.scaler import CompoundScaler
from glasses.utils.PretrainedWeightsProvider import Config
from ....models.base import VisionModule, Encoder
from ..resnet import ResNetLayer
from glasses.utils.PretrainedWeightsProvider import Config, pretrained


class InvertedResidualBlock(nn.Module):
    """Inverted residual block proposed originally for MobileNetV2. 

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/EfficientNetBasicBlock.png?raw=true


    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        stride (int, optional): Stide used in the depth convolution. Defaults to 1.
        expansion (int, optional): The expansion ratio applied. Defaults to 6.
        activation (nn.Module, optional): The activation funtion used. Defaults to nn.SiLU.
        drop_rate (float, optional): If > 0, add a  nn.Dropout2d at the end of the block. Defaults to 0.2.
        se (bool, optional): If True, add a ChannelSE module after the depth convolution. Defaults to True.
        kernel_size (int, optional): [description]. Defaults to 3.
    """

    def __init__(self, in_features: int, out_features: int, stride: int = 1, expansion: int = 6, activation: nn.Module = nn.SiLU, drop_rate: float = 0.2, se: bool = True, kernel_size: int = 3, **kwargs):
        super().__init__()

        reduced_features = in_features // 4
        expanded_features = in_features * expansion
        # do not apply residual when downsamping and when features are different
        # in mobilenet we do not use a shortcut
        self.should_apply_residual = stride == 1 and in_features == out_features
        self.block = nn.Sequential(
            OrderedDict({
                'exp': ConvBnAct(in_features, expanded_features,  activation=activation, kernel_size=1) if expansion > 1 else nn.Identity(),
                'depth':  ConvBnAct(expanded_features, expanded_features,
                                    activation=activation,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    groups=expanded_features,
                                    **kwargs),
                # apply se after depth-wise
                'att':  ChannelSE(expanded_features,
                                  reduced_features=reduced_features, activation=activation) if se else nn.Identity(),
                'point': nn.Sequential(ConvBnAct(expanded_features,
                                                 out_features, kernel_size=1, activation=None)),

                'drop': nn.Dropout2d(drop_rate) if self.should_apply_residual and drop_rate > 0 else nn.Identity()
            })
        )

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.should_apply_residual:
            x += res
        return x


EfficientNetBasicBlock = InvertedResidualBlock
EfficientNetStem = ConvBnAct
EfficientNetLayer = partial(ResNetLayer, block=EfficientNetBasicBlock)


class EfficientNetEncoder(Encoder):
    """
    EfficientNet encoder composed by multiple different layers with increasing features.

    Be awere that `widths` and `strides` also includes the width and stride for the steam in the first position.

    Args:
        in_channels (int, optional): [description]. Defaults to 3.
        widths (List[int], optional): [description]. Defaults to [32, 16, 24, 40, 80, 112, 192, 320, 1280].
        depths (List[int], optional): [description]. Defaults to [1, 2, 2, 3, 3, 4, 1].
        strides (List[int], optional): [description]. Defaults to [2, 1, 2, 2, 2, 1, 2, 1].
        expansions (List[int], optional): [description]. Defaults to [1, 6, 6, 6, 6, 6, 6].
        kernel_sizes (List[int], optional): [description]. Defaults to [3, 3, 5, 3, 5, 5, 3].
        se (List[bool], optional): [description]. Defaults to [True, True, True, True, True, True, True].
        drop_rate (float, optional): [description]. Defaults to 0.2.
        activation (nn.Module, optional): [description]. Defaults to nn.SiLU.
    """

    def __init__(self, in_channels: int = 3,
                 widths: List[int] = [32, 16, 24, 40, 80, 112, 192, 320, 1280],
                 depths: List[int] = [1, 2, 2, 3, 3, 4, 1],
                 strides: List[int] = [2, 1, 2, 2, 2, 1, 2, 1],
                 expansions: List[int] = [1, 6, 6, 6, 6, 6, 6],
                 kernel_sizes: List[int] = [3, 3, 5, 3, 5, 5, 3],
                 se: List[bool] = [True, True, True, True, True, True, True],
                 drop_rate: float = 0.2,
                 stem: nn.Module = EfficientNetStem,
                 activation: nn.Module = partial(nn.SiLU, inplace=True), **kwargs):
        super().__init__()

        self.widths, self.depths = widths, depths
        self.strides, self.expansions, self.kernel_sizes = strides, expansions, kernel_sizes
        self.stem = stem(
            in_channels, widths[0],  activation=activation, kernel_size=3, stride=strides[0])
        strides = strides[1:]
        self.in_out_block_sizes = list(zip(widths, widths[1:-1]))

        self.layers = nn.ModuleList([
            *[EfficientNetLayer(in_features,
                                out_features,
                                depth=n,
                                stride=s,
                                expansion=t,
                                kernel_size=k,
                                se=se,
                                drop_rate=drop_rate,
                                activation=activation, **kwargs)
              for (in_features, out_features), n, s, t, k, se
                in zip(self.in_out_block_sizes, depths, strides, expansions, kernel_sizes, se)]
        ])

        self.layers.append(ConvBnAct(self.widths[-2], self.widths[-1],
                                     activation=activation, kernel_size=1))

    def forward(self, x):
        x = self.stem(x)
        for block in self.layers:
            x = block(x)
        return x

    @property
    def stages(self):
        # find the layers where the input is // 2 
        # skip first stride because it is for the stem!
        # skip the last layer because it is just a conv-bn-act
        # and we haven't a stride for it
        layers = np.array(self.layers[:-1])[np.array(self.strides[1:]) == 2].tolist()[:-1]
            
        return [self.stem[-1],
               *layers]

    @property
    def features_widths(self):
        # skip the last layer because it is just a conv-bn-act
        # and we haven't a stride for it
        widths = np.array(self.widths[:-1])[np.array(self.strides) == 2].tolist()
        # we also have to remove the last one, because it is the spatial size of the network output 
        return widths[:-1]


class EfficientNetHead(nn.Sequential):
    """
    This class represents the head of EfficientNet. It performs a global pooling, dropout and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(self, in_features: int, n_classes: int, drop_rate: float = 0.2):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout2d(drop_rate)
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x


class EfficientNet(VisionModule):
    """Implementation of EfficientNet proposed in `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/EfficientNet.png?raw=true

    The basic architecture is similar to MobileNetV2 as was computed by using  `Progressive Neural Architecture Search <https://arxiv.org/abs/1905.11946>`_ .

    The following table shows the basic architecture (EfficientNet-efficientnet_b0):

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/EfficientNetModelsTable.jpeg?raw=true

    Then, the architecture is scaled up from `-efficientnet_b0` to `-efficientnet_b7` using compound scaling.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/EfficientNetScaling.jpg?raw=true

    Examples:

        Default models

        >>> EfficientNet.efficientnet_b0()
        >>> EfficientNet.efficientnet_b1()
        >>> EfficientNet.efficientnet_b2()
        >>> EfficientNet.efficientnet_b3()
        >>> EfficientNet.efficientnet_b4()
        >>> EfficientNet.efficientnet_b5()
        >>> EfficientNet.efficientnet_b6()
        >>> EfficientNet.efficientnet_b7()
        >>> EfficientNet.efficientnet_b8()
        >>> EfficientNet.efficientnet_l2()

    You can easily customize your model

        >>> EfficientNet.efficientnet_b0(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> EfficientNet.efficientnet_b0(n_classes=100)
        >>> # pass a different block
        >>> EfficientNet.efficientnet_b0(block=...)
        >>> # store each feature
        >>> x = torch.rand((1, 3, 224, 224))
        >>> model = EfficientNet.efficientnet_b0()
        >>> # first call .features, this will activate the forward hooks and tells the model you'll like to get the features
        >>> model.encoder.features
        >>> model(torch.randn((1,3,224,224)))
        >>> # get the features from the encoder
        >>> features = model.encoder.features
        >>> print([x.shape for x in features])
        >>> # [torch.Size([1, 32, 112, 112]), torch.Size([1, 24, 56, 56]), torch.Size([1, 40, 28, 28]), torch.Size([1, 80, 14, 14])]

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """
    models_config = {
        # name : width_factor, depth_factor, dropout_rate
        'efficientnet_b0': (1.0, 1.0, 0.2),
        'efficientnet_b1': (1.0, 1.1, 0.2),
        'efficientnet_b2': (1.1, 1.2, 0.3),
        'efficientnet_b3': (1.2, 1.4, 0.3),
        'efficientnet_b4': (1.4, 1.8, 0.4),
        'efficientnet_b5': (1.6, 2.2, 0.4),
        'efficientnet_b6': (1.8, 2.6, 0.5),
        'efficientnet_b7': (2.0, 3.1, 0.5),
        'efficientnet_b8': (2.2, 3.6, 0.5),
        'efficientnet_l2': (4.3, 5.3, 0.5),
    }

    default_depths: List[int] = [1, 2, 2, 3, 3, 4, 1]
    default_widths: List[int] = [
        32, 16, 24, 40, 80, 112, 192, 320, 1280]

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = EfficientNetEncoder(in_channels, *args, **kwargs)
        self.head = EfficientNetHead(
            self.encoder.widths[-1], n_classes, drop_rate=kwargs['drop_rate'])

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.head(x)
        return x

    def initialize(self):
        # initialization copied from MobileNetV2
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @classmethod
    def from_config(cls, config, key, *args, **kwargs) -> EfficientNet:
        width_factor, depth_factor, drop_rate = config[key]
        widths, depths = CompoundScaler()(width_factor, depth_factor,
                                          cls.default_widths, cls.default_depths)
        return cls(*args, depths=depths, widths=widths, drop_rate=drop_rate, **kwargs,)

    @classmethod
    @pretrained()
    def efficientnet_b0(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b0', *args, **kwargs)

    @classmethod
    @pretrained()
    def efficientnet_b1(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b1', *args, **kwargs)

    @classmethod
    @pretrained()
    def efficientnet_b2(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b2', *args, **kwargs)

    @classmethod
    @pretrained()
    def efficientnet_b3(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b3', *args, **kwargs)

    @classmethod
    @pretrained()
    def efficientnet_b4(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b4', *args, **kwargs)

    @classmethod
    def efficientnet_b5(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b5', *args, **kwargs)

    @classmethod
    def efficientnet_b6(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b6', *args, **kwargs)

    @classmethod
    def efficientnet_b7(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b7', *args, **kwargs)

    @classmethod
    def efficientnet_b8(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b8', *args, **kwargs)

    @classmethod
    def efficientnet_l2(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_l2',  *args, **kwargs)


class EfficientNetLite(EfficientNet):
    """Implementations of EfficientNetLite proposed in `Higher accuracy on vision models with EfficientNet-Lite <https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html>`_

    Main differences from the EfficientNet implementation are:


    - Removed squeeze-and-excitation networks since they are not well supported
    - Replaced all swish activations with RELU6, which significantly improved the quality of post-training quantization (explained later)
    - Fixed the stem and head while scaling models up in order to reduce the size and computations of scaled models

    Examples:
        Create a default model

        >>> EfficientNetLite.efficientnet_lite0()
        >>> EfficientNetLite.efficientnet_lite1()
        >>> EfficientNetLite.efficientnet_lite2()
        >>> EfficientNetLite.efficientnet_lite3()
        >>> EfficientNetLite.efficientnet_lite4()


    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    models_config = {
        # name : width_factor, depth_factor, dropout_rate
        'efficientnet_lite0': (1.0, 1.0, 0.2),
        'efficientnet_lite1': (1.0, 1.1, 0.2),
        'efficientnet_lite2': (1.1, 1.2, 0.3),
        'efficientnet_lite3': (1.2, 1.4, 0.3),
        'efficientnet_lite4': (1.4, 1.8, 0.3),
    }

    @classmethod
    def from_config(cls, config, key, *args, **kwargs) -> EfficientNet:
        width_factor, depth_factor, drop_rate = config[key]
        widths, depths = CompoundScaler()(width_factor, depth_factor,
                                          cls.default_widths, cls.default_depths)
        # in lite models the steam and head width are not scaled
        widths[0] = cls.default_widths[0]
        widths[-1] = cls.default_widths[-1]

        depths[0] = cls.default_depths[0]
        depths[-1] = cls.default_depths[-1]
        # and se is disabled are not well supported for some mobile accelerators.
        # ot sure why since there are just convolutions.
        se = [False] * len(widths)
        # all swish function are replaced with ReLU6 for easier post-quantization
        return cls(*args, depths=depths,
                   widths=widths,
                   se=se,
                   drop_rate=drop_rate,
                   activation=partial(nn.ReLU6, inplace=True),  **kwargs)

    @classmethod
    def efficientnet_lite0(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_lite0', *args, **kwargs)

    @classmethod
    def efficientnet_lite1(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_lite1', *args, **kwargs)

    @classmethod
    def efficientnet_lite2(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_lite2', *args, **kwargs)

    @classmethod
    def efficientnet_lite3(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_lite3', *args, **kwargs)

    @classmethod
    def efficientnet_lite4(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_lite4')
