from __future__ import annotations
import torch
from torch import nn
from torch import Tensor
from ....blocks.residuals import ResidualAdd
from collections import OrderedDict
from typing import List, Union
from functools import partial
from ..mobilenet import InvertedResidualBlock, DepthWiseConv2d, MobileNetEncoder, MobileNetDecoder
from ....blocks import Conv2dPad, ConvBnAct
from ..se import ChannelSE
from ....models.utils.scaler import CompoundScaler
from ....activation import Swish
from glasses.utils.PretrainedWeightsProvider import Config
from ....models.VisionModule import VisionModule


from glasses.utils.PretrainedWeightsProvider import Config, pretrained

class EfficientNetBasicBlock(InvertedResidualBlock):
    """EfficientNet basic block. It is an inverted residual block from `MobileNetV2` but with `ChannelSE` after the depth-wise conv. 
    Residual connections are applied when there the input and output features number are the same.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/EfficientNetBasicBlock.png?raw=true


    Args:
        in_features (int): [description]
        activation (nn.Module, optional): [description]. Defaults to Swish.
        drop_rate (float, optional): [description]. Defaults to 0.2.
    """
    def __init__(self, in_features: int, out_features: int, activation: nn.Module = Swish, drop_rate: float =0.2, **kwargs):
        super().__init__(in_features, out_features, activation=activation, **kwargs)
        reduced_features = in_features // 4
        se = ChannelSE(self.expanded_features,
                       reduced_features=reduced_features, activation=activation)
        # squeeze and excitation is applied after the depth wise conv
        self.block.block.point = nn.Sequential(
            se,
            self.block.block.point
        )
        if self.should_apply_residual:
            self.block.block.add_module('drop', nn.Dropout2d(drop_rate))


class EfficientNetLayer(nn.Module):
    """EfficientNet layer composed by `block` stacked one after the other. The first block will downsample the input

    Args:
        in_features (int): [description]
        out_features (int): [description]
        block (nn.Module, optional): [description]. Defaults to EfficientNetBasicBlock.
        depth (int, optional): [description]. Defaults to 1.
        stride (int, optional): [description]. Defaults to 2.
    """
    def __init__(self, in_features: int, out_features: int, block: nn.Module = EfficientNetBasicBlock,
                 depth: int = 1, stride: int = 2,  **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            block(in_features, out_features,**kwargs,
                  stride=stride ),
            *[block(out_features,
                    out_features, **kwargs) for _ in range(depth - 1)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet encoder composed by increasing different layers with increasing features.

    Args:
        in_channels (int, optional): [description]. Defaults to 3.
        widths (List[int], optional): [description]. Defaults to [ 32, 16, 24, 40, 80, 112, 192, 320, 1280].
        depths (List[int], optional): [description]. Defaults to [1, 2, 2, 3, 3, 4, 1].
        strides (List[int], optional): [description]. Defaults to [1, 2, 2, 2, 2, 1, 2].
        expansions (List[int], optional): [description]. Defaults to [1, 6, 6, 6, 6, 6, 6].
        kernels_sizes (List[int], optional): [description]. Defaults to [3, 3, 5, 3, 5, 5, 3].
        activation (nn.Module, optional): [description]. Defaults to Swish.
    """

    def __init__(self, in_channels: int = 3,
                 widths: List[int] = [
                     32, 16, 24, 40, 80, 112, 192, 320, 1280],
                 depths: List[int] = [1, 2, 2, 3, 3, 4, 1],
                 strides: List[int] = [1, 2, 2, 2, 1, 2, 1],
                 expansions: List[int] = [1, 6, 6, 6, 6, 6, 6],
                 kernels_sizes: List[int] = [3, 3, 5, 3, 5, 5, 3],
                 activation: nn.Module = Swish, **kwargs):
        super().__init__()

        self.widths, self.depths = widths, depths
        self.gate = ConvBnAct(
            in_channels, self.widths[0],  activation=activation, kernel_size=3, stride=2)

        self.in_out_block_sizes = list(zip(widths, widths[1:-1]))

        self.blocks = nn.ModuleList([
            *[EfficientNetLayer(in_channels,
                                out_channels,  depth=n, stride=s,  expansion=t, kernel_size=k, activation=activation, **kwargs)
              for (in_channels, out_channels), n, s, t, k
                in zip(self.in_out_block_sizes, depths, strides, expansions, kernels_sizes)]
        ])

        self.blocks.append(
            ConvBnAct(self.widths[-2], self.widths[-1],
                      activation=activation, kernel_size=1),
        )

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class EfficientNet(VisionModule):
    """Implementations of EfficientNet proposed in `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_
    
    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/EfficientNet.png?raw=true

    The basic architecture is similar to MobileNetV2 as was computed by using  `Progressive Neural Architecture Search <https://arxiv.org/abs/1905.11946>`_ . 
    
    The following table shows the basic architecture (EfficientNet-efficientnet_b0):

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/EfficientNetModelsTable.jpeg?raw=true

    Then, the architecture is scaled up from `-efficientnet_b0` to `-efficientnet_b7` using compound scaling. 

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/EfficientNetScaling.jpg?raw=true

    Create a default model

    Examples:
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


    Customization

    You can easily customize your model
    
    Examples:

        >>> EfficientNet.efficientnet_b0(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> EfficientNet.efficientnet_b0(n_classes=100)
        >>> # pass a different block
        >>> EfficientNet.efficientnet_b0(block=...)
        >>> # change the initial convolution
        >>> model = EfficientNet.efficientnet_b0()
        >>> model.encoder.gate.conv = nn.Conv2d(3, 32, kernel_size=7)
        >>> # store each feature
        >>> x = torch.rand((1, 3, 224, 224))
        >>> model = EfficientNet.efficientnet_b0()
        >>> features = []
        >>> x = model.encoder.gate(x)
        >>> for block in model.encoder.blocks:
        >>>     x = block(x)
        >>>     features.append(x)
        >>> print([x.shape for x in features])
        >>> # [torch.Size([1, 16, 112, 112]), torch.Size([1, 24, 56, 56]), torch.Size([1, 40, 28, 28]), torch.Size([1, 80, 14, 14]), torch.Size([1, 112, 7, 7]), torch.Size([1, 192, 7, 7]), torch.Size([1, 320, 4, 4]), torch.Size([1, 1280, 4, 4])]

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    configs = {
        'efficientnet_b0':  Config(resize=224, input_size=224, interpolation='bicubic'),
        'efficientnet_b1':  Config(resize=240, input_size=240, interpolation='bicubic'),
        'efficientnet_b2':  Config(resize=260, input_size=260, interpolation='bicubic'),
        'efficientnet_b3':  Config(resize=300, input_size=300, interpolation='bicubic'),
        'efficientnet_b4':  Config(resize=380, input_size=380, interpolation='bicubic'),
        'efficientnet_b5':  Config(resize=456, input_size=456, interpolation='bicubic'),
        'efficientnet_b6':  Config(resize=528, input_size=528, interpolation='bicubic'),
        'efficientnet_b7':  Config(resize=600, input_size=600, interpolation='bicubic'),
        'efficientnet_b8':  Config(resize=672, input_size=672, interpolation='bicubic'),
        'efficientnet_l2':  Config(resize=800, input_size=800, interpolation='bicubic')
    }


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
        self.decoder = MobileNetDecoder(
            self.encoder.widths[-1], n_classes, drop_rate=kwargs['drop_rate'])

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
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
        widths, depths = CompoundScaler()(width_factor, depth_factor,  cls.default_widths, cls.default_depths)
        return EfficientNet(*args, **kwargs, depths=depths, widths=widths, drop_rate=drop_rate)

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
        return cls.from_config(cls.models_config, 'efficientnet_b2',*args, **kwargs)


    @classmethod
    @pretrained()
    def efficientnet_b3(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b3',*args, **kwargs)


    @classmethod
    def efficientnet_b4(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b4',*args, **kwargs)


    @classmethod
    def efficientnet_b5(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b5',*args, **kwargs)


    @classmethod
    def efficientnet_b6(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b6',*args, **kwargs)

    @classmethod
    def efficientnet_b7(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b7',*args, **kwargs)

    @classmethod
    def  efficientnet_b8(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_b8',*args, **kwargs)


    @classmethod
    def  efficientnet_l2(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.models_config, 'efficientnet_l2')
