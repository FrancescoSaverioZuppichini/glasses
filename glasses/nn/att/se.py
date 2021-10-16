from torch import nn
from torch import Tensor
from collections import OrderedDict
from functools import partial
from einops.layers.torch import Reduce, Rearrange

ReLUInPlace = partial(nn.ReLU, inplace=True)


class SpatialSE(nn.Module):
    """Modernized Implementation of Squeeze and Excitation Module proposed in `Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_ with single kernel convolution instead of fully connected layers.

    See `LegacySpatialSE` for usage.
    """

    def __init__(
        self,
        features: int,
        reduction: int = 16,
        reduced_features: int = None,
        activation: nn.Module = ReLUInPlace,
    ):
        super().__init__()
        self.reduced_features = (
            features // reduction if reduced_features is None else reduced_features
        )

        self.pool = Reduce("b c h w -> b c 1 1", reduction="mean")
        self.att = nn.Sequential(
            OrderedDict(
                {
                    "fc1": nn.Conv2d(
                        features, self.reduced_features, kernel_size=1, bias=False
                    ),
                    "act1": activation(),
                    "fc2": nn.Conv2d(
                        self.reduced_features, features, kernel_size=1, bias=False
                    ),
                    "act2": nn.Sigmoid(),
                    "proj": nn.Identity(),
                }
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.pool(x)
        y = self.att(y)
        return x * y


class LegacySpatialSE(SpatialSE):
    """Implementation of Squeeze and Excitation Module proposed in `Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_

    The idea is to apply a learned weight to rescale the channels.

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
        >>> from glasses.nn.att import ECA, WithAtt
        >>> se_resnet50 = ResNet.resnet50(block=WithAtt(ResNetBottleneckBlock, att=SpatialSE))
        >>> se_resnet50.summary()
    Args:
        features (int): Number of features
        reduction (int, optional): Reduction ratio used to downsample the input. Defaults to 16.
        reduced_features (int, optional): If passed, use it instead of calculating the reduced features using `reduction`. Defaults to None.
    """

    def __init__(
        self,
        features: int,
        *args,
        activation: nn.Module = ReLUInPlace,
        **kwargs,
    ):

        super().__init__(features, *args, **kwargs)
        self.pool = Reduce("b c h w -> b c", reduction="mean")
        self.att = nn.Sequential(
            OrderedDict(
                {
                    "fc1": nn.Linear(features, self.reduced_features, bias=False),
                    "act1": activation(),
                    "fc2": nn.Linear(self.reduced_features, features, bias=False),
                    "act2": nn.Sigmoid(),
                    "proj": Rearrange("b c -> b c 1 1"),
                }
            )
        )


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
        >>> from glasses.nn.att import ECA, WithAtt
        >>> se_resnet50 = ResNet.resnet50(block=WithAtt(ResNetBottleneckBlock, att=ChannelSE))
        >>> se_resnet50.summary()

    Args:
        features (int): Number of features
        reduction (int, optional): Reduction ratio used to downsample the input. Defaults to 16.
        reduced_features (int, optional): If passed, use it instead of calculating the reduced features using `reduction`. Defaults to None.
    """

    def __init__(
        self, features: int, *args, activation: nn.Module = ReLUInPlace, **kwargs
    ):
        super().__init__(features, *args, activation=activation, **kwargs)
        self.pool = Reduce("b c h w -> b 1 h w", reduction="mean")
        self.att = nn.Sequential(
            OrderedDict(
                {
                    "conv1": nn.Conv2d(1, self.reduced_features, kernel_size=1),
                    "act1": activation(),
                    "conv2": nn.Conv2d(self.reduced_features, 1, kernel_size=1),
                    "act2": nn.Sigmoid(),
                    "proj": nn.Identity(),
                }
            )
        )


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
        >>> from glasses.nn.att import ECA, WithAtt
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
