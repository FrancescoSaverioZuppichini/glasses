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
from ....utils.scaler import CompoundScaler

class Swish(nn.Module):
    """Swish function
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class EfficientNetBasicBlock(InvertedResidualBlock):
    """EfficientNet basic block. It is an inverted residual block from `MobileNetV2` but with `ChannelSE` after the depth-wise conv. 
    Residual connections are applied when there the input and output features number are the same.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/EfficientNetBasicBlock.png?raw=true


    Args:
        in_features (int): [description]
        activation (nn.Module, optional): [description]. Defaults to Swish.
        drop_rate (float, optional): [description]. Defaults to 0.2.
    """
    def __init__(self, in_features: int, *args, activation: nn.Module = Swish, drop_rate=0.2, **kwargs):
        super().__init__(in_features, *args, activation=activation, **kwargs)
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
        downsampling (int, optional): [description]. Defaults to 2.
    """
    def __init__(self, in_features: int, out_features: int, block: nn.Module = EfficientNetBasicBlock,
                 depth: int = 1, downsampling: int = 2, *args, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            block(in_features, out_features, *args,
                  downsampling=downsampling,  **kwargs),
            *[block(out_features,
                    out_features, *args, **kwargs) for _ in range(depth - 1)]
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
                 strides: List[int] = [1, 2, 2, 2, 2, 1, 2],
                 expansions: List[int] = [1, 6, 6, 6, 6, 6, 6],
                 kernels_sizes: List[int] = [3, 3, 5, 3, 5, 5, 3],
                 activation: nn.Module = Swish, *args, **kwargs):
        super().__init__()

        self.widths, self.depths = widths, depths
        self.gate = ConvBnAct(
            in_channels, self.widths[0],  activation=activation, kernel_size=3, stride=2, bias=False)

        self.in_out_block_sizes = list(zip(widths, widths[1:-1]))

        self.blocks = nn.ModuleList([
            *[EfficientNetLayer(in_channels,
                                out_channels, *args, depth=n, downsampling=s,  expansion=t, kernel_size=k, activation=activation, **kwargs)
              for (in_channels, out_channels), n, s, t, k
                in zip(self.in_out_block_sizes, depths, strides, expansions, kernels_sizes)]
        ])

        self.blocks.append(
            ConvBnAct(self.widths[-2], self.widths[-1],
                      activation=activation, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class EfficientNet(nn.Module):
    """Implementations of EfficientNet proposed in `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
 <https://arxiv.org/abs/1905.11946>`_
    
    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/EfficientNet.png?raw=true

    The basic architecture is similar to MobileNetV2 as was computed by using  `Progressive Neural Architecture Search <https://arxiv.org/abs/1905.11946>`_ . 
    
    The following table shows the basic architecture (EfficientNet-b0):

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/EfficientNetModelsTable.jpeg?raw=true

    Then, the architecture is scaled up from `-b0` to `-b7` using compound scaling. 

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/EfficientNetScaling.jpg?raw=true

    Create a default model

    Examples:
        >>> EfficientNet.b0()
        >>> EfficientNet.b1()
        >>> EfficientNet.b2()
        >>> EfficientNet.b3()
        >>> EfficientNet.b4()
        >>> EfficientNet.b5()
        >>> EfficientNet.b6()
        >>> EfficientNet.b7()


    Customization

    You can easily customize your model
    
    Examples:

        >>> EfficientNet.b0(activation = nn.SELU)
        >>> # change number of classes (default is 1000 )
        >>> EfficientNet.b0(n_classes=100)
        >>> # pass a different block
        >>> EfficientNet.b0(block=...)
        >>> # change the initial convolution
        >>> model = EfficientNet.b0()
        >>> model.encoder.gate.conv = nn.Conv2d(3, 32, kernel_size=7)
        >>> # store each feature
        >>> x = torch.rand((1, 3, 224, 224))
        >>> model = EfficientNet.b0()
        >>> features = []
        >>> x = model.encoder.gate(x)
        >>> for block in model.encoder.blocks:
        >>>     x = block(x)
        >>>     features.append(x)
        >>> print([x.shape for x in features])
        >>> EfficientNet.b0(activation = nn.SELU)
        >>> # [torch.Size([1, 16, 112, 112]), torch.Size([1, 24, 56, 56]), torch.Size([1, 40, 28, 28]), torch.Size([1, 80, 14, 14]), torch.Size([1, 112, 7, 7]), torch.Size([1, 192, 7, 7]), torch.Size([1, 320, 4, 4]), torch.Size([1, 1280, 4, 4])]
​
        You can access the network configuration (used for scaling) by

        >>> EfficientNet.config
        >>> # output:
        >>> #  {'b0': (1.0, 1.0, 224, 0.2),
        >>> #   'b1': (1.0, 1.1, 240, 0.2),
        >>> #   'b2': (1.1, 1.2, 260, 0.3),
        >>> #   'b3': (1.2, 1.4, 300, 0.3),
        >>> #   'b4': (1.4, 1.8, 380, 0.4),
        >>> #   'b5': (1.6, 2.2, 456, 0.4),
        >>> #   'b6': (1.8, 2.6, 528, 0.5),
        >>> #   'b7': (2.0, 3.1, 600, 0.5)}


    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    config = {
        # name : width_factor, depth_factor, input_size, dropout_rate
        'b0': (1.0, 1.0, 224, 0.2),
        'b1': (1.0, 1.1, 240, 0.2),
        'b2': (1.1, 1.2, 260, 0.3),
        'b3': (1.2, 1.4, 300, 0.3),
        'b4': (1.4, 1.8, 380, 0.4),
        'b5': (1.6, 2.2, 456, 0.4),
        'b6': (1.8, 2.6, 528, 0.5),
        'b7': (2.0, 3.1, 600, 0.5),
    }

    default_depths: List[int] = [1, 2, 2, 3, 3, 4, 1]
    default_widths: List[int] = [
        32, 16, 24, 40, 80, 112, 192, 320, 1280]

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = EfficientNetEncoder(in_channels, *args, **kwargs)
        self.decoder = MobileNetDecoder(
            self.encoder.widths[-1], n_classes)

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
        width_factor, depth_factor, _, drop_rate = config[key]
        widths, depths = CompoundScaler()(width_factor, depth_factor,  cls.default_widths, cls.default_depths)
        return EfficientNet(*args, **kwargs, depths=depths, widths=widths, drop_rate=drop_rate)

    @classmethod
    def b0(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.config, 'b0', *args, **kwargs)
    
    @classmethod
    def b1(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.config, 'b1', *args, **kwargs)


    @classmethod
    def b2(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.config, 'b2',*args, **kwargs)


    @classmethod
    def b3(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.config, 'b3',*args, **kwargs)


    @classmethod
    def b4(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.config, 'b4',*args, **kwargs)


    @classmethod
    def b5(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.config, 'b5',*args, **kwargs)


    @classmethod
    def b6(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.config, 'b6',*args, **kwargs)

    @classmethod
    def b7(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.config, 'b7',*args, **kwargs)

    @classmethod
    def b8(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.config, 'b8',*args, **kwargs)


    @classmethod
    def l2(cls, *args, **kwargs) -> EfficientNet:
        return cls.from_config(cls.config, 'l2')
