import torch
from torch import nn
from torch import Tensor
from collections import OrderedDict
from functools import partial
import math

ReLUInPlace = partial(nn.ReLU, inplace=True)


class SpatialSE(nn.Module):
    """Implementation of Squeeze and Excitation Module proposed in `Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_
    The idea is to apply a learned an channel-wise attention.

    It squeezes spatially and excitates channel-wise.

    The authors reported a bigger performance increase where the number of features are higher.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/se.png?raw=true

    Further visualisation from `Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks <https://arxiv.org/abs/1803.02579>`_

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/SpatialSE.png?raw=true


    Examples:

        Add `SpatialSE` to your own model is very simple.

        >>> nn.Sequential(
        >>>    nn.Conv2d(32, 64, kernel_size=3),
        >>>    SpatialSE(64, reduction=4)
        >>>    nn.ReLU(),
        >>> )

        You can also direcly specify the number of features inside the module

        >>> nn.Sequential(
        >>>    nn.Conv2d(32, 64, kernel_size=3),
        >>>    SpatialSE(64, reduced_features=10)
        >>>    nn.ReLU(),
        >>> )

        The following example shows a more advance scenarion where we add Squeeze ad Excitation to a `ResNetBasicBlock`.

        >>> class SENetBasicBlock(ResNetBasicBlock):
        >>>     def __init__(self, in_features: int, out_features: int, reduction: int =16, *args, **kwargs):
        >>>        super().__init__(in_features, out_features, *args, **kwargs)
        >>>        # add se to the `.block`
        >>>        self.block.add_module('se', SpatialSE(out_features))


        Creating the original seresnet50

        >>> from glasses.nn.models.classification.resnet import ResNet, ResNetBottleneckBlock
        >>> from glasses.nn.att import EfficientChannelAtt, WithAtt
        >>> se_resnet50 = ResNet.resnet50(block=WithAtt(ResNetBottleneckBlock, att=SpatialSE))
        >>> se_resnet50.summary()
    Args:
        features (int): Number of features
        reduction (int, optional): Reduction ratio used to downsample the input. Defaults to 16.
        reduced_features (int, optional): If passed, use it instead of calculating the reduced features using `reduction`. Defaults to None.
    """

    def __init__(self, features: int, reduction: int = 16, reduced_features: int = None, activation: nn.Module = ReLUInPlace):
        super().__init__()
        self.reduced_features = features // reduction if reduced_features is None else reduced_features

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(OrderedDict(
            {
                'fc1': nn.Linear(features, self.reduced_features, bias=False),
                'act1': activation(),
                'fc2': nn.Linear(self.reduced_features, features, bias=False),
                'act2': nn.Sigmoid()
            }
        ))

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y has shape [B, C]
        y = self.att(y)
        # resphape to [B, C, 1, 1]  to match the space dims of x
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelSE(SpatialSE):
    """Modified implementation of Squeeze and Excitation Module proposed in `Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks <https://arxiv.org/abs/1803.02579>`_

    It squeezes channel-wise and excitates spatially.


    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/ChannelSE.png?raw=true

    Examples:
        Add `ChannelSE` to your own model is very simple.

        >>> nn.Sequential(
        >>>    nn.Conv2d(32, 64, kernel_size=3),
        >>>    ChannelSE(64, reduction=4)
        >>>    nn.ReLU(),
        >>> )

        You can also direcly specify the number of features inside the module

        >>> nn.Sequential(
        >>>    nn.Conv2d(32, 64, kernel_size=3),
        >>>    ChannelSE(64, reduced_features=10)
        >>>    nn.ReLU(),
        >>> )


        Creating the cseresnet50

        >>> from glasses.nn.models.classification.resnet import ResNet, ResNetBottleneckBlock
        >>> from glasses.nn.att import EfficientChannelAtt, WithAtt
        >>> se_resnet50 = ResNet.resnet50(block=WithAtt(ResNetBottleneckBlock, att=ChannelSE))
        >>> se_resnet50.summary()

    Args:
        features (int): Number of features
        reduction (int, optional): Reduction ratio used to downsample the input. Defaults to 16.
        reduced_features (int, optional): If passed, use it instead of calculating the reduced features using `reduction`. Defaults to None.
    """

    def __init__(self, features: int,  *args, activation: nn.Module = ReLUInPlace, **kwargs):
        super().__init__(features, *args, activation=activation, **kwargs)
        self.att = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(features, self.reduced_features, kernel_size=1),
            'act1': activation(),
            'conv2': nn.Conv2d(self.reduced_features, features, kernel_size=1),
            'act2': nn.Sigmoid()
        })
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x)
        y = self.att(y)

        return x * y


class SpatialChannelSE(nn.Module):
    """Implementation of Spatial and Channel Squeeze and Excitation Module proposed in `Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks <https://arxiv.org/abs/1803.02579>`_

    This module combines booth Spatial and Channel Squeeze and Excitation

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/SpatialAndChannelSE.png?raw=true

    Examples:
        Add `SpatialChannelSE` to your own model is very simple.

        >>> nn.Sequential(
        >>>    nn.Conv2d(32, 64, kernel_size=3),
        >>>    SpatialChannelSE(64, reduction=4)
        >>>    nn.ReLU(),
        >>> )

        You can also direcly specify the number of features inside the module

        >>> nn.Sequential(
        >>>    nn.Conv2d(32, 64, kernel_size=3),
        >>>    SpatialChannelSE(64, reduced_features=10)
        >>>    nn.ReLU(),
        >>> )

        Creating scseresnet50

        >>> from glasses.nn.models.classification.resnet import ResNet, ResNetBottleneckBlock
        >>> from glasses.nn.att import EfficientChannelAtt, WithAtt
        >>> se_resnet50 = ResNet.resnet50(block=WithAtt(ResNetBottleneckBlock, att=SpatialChannelSE))
        >>> se_resnet50.summary()


    Args:
        features (int): Number of features
        reduction (int, optional): Reduction ratio used to downsample the input. Defaults to 16.
        reduced_features (int, optional): If passed, use it instead of calculating the reduced features using `reduction`. Defaults to None.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.spatial_se = SpatialSE(*args, **kwargs)
        self.channel_se = ChannelSE(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        s_se = self.spatial_se(x)
        c_se = self.channel_se(x)

        return x * (s_se + c_se)


class EfficientChannelAtt(nn.Module):
    """Implementation of Efficient Channel Attention proposed in `ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks <https://arxiv.org/pdf/1910.03151.pdf>`_

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/EfficientChannelAtt.png?raw=true

    Examples:

        >>> # create ecaresnet50
        >>> from glasses.nn.models.classification.resnet import ResNet, ResNetBottleneckBlock
        >>> from glasses.nn.att import EfficientChannelAtt, WithAtt
        >>> eca_resnet50 = ResNet.resnet50(block=WithAtt(ResNetBottleneckBlock, att=EfficientChannelAtt))
        >>> eca_resnet50.summary()

    Args:
        features (int, optional): Number of features features. Defaults to None.
        kernel_size (int, optional): [description]. Defaults to 3.
        gamma (int, optional): [description]. Defaults to 2.
        beta (int, optional): [description]. Defaults to 1.
    """

    def __init__(self, features: int, kernel_size: int = 3, gamma: int = 2, beta: int = 1):

        super().__init__()
        assert kernel_size % 2 == 1

        t = int(abs(math.log(features, 2) + beta) / gamma)
        k = t if t % 2 else t + 1

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y = self.pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class WithAtt:
    """Utility class that adds an attention module after `.block`. 

    :Usage:

        >>> WithAtt(ResNetBottleneckBlock, att=SpatialSE)
        >>> WithAtt(ResNetBottleneckBlock, att=EfficientChannelAtt)
        >>> from functools import partial 
        >>> WithAtt(ResNetBottleneckBlock, att=partial(SpatialSE, reduction=8))
    """

    def __init__(self, block: nn.Module, att: nn.Module = SpatialSE):
        self.block = block
        self.att = att

    def __call__(self, in_features: int, out_features: int, *args, **kwargs) -> nn.Module:
        b = self.block(in_features, out_features, *args, **kwargs)
        b.block.add_module('se', self.att(out_features))
        return b
