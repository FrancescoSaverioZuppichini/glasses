from __future__ import annotations
from torch import nn
from torch import Tensor
from collections import OrderedDict
from functools import partial
from glasses.nn.blocks import ConvBnAct
from glasses.nn.att import ChannelSE
from glasses.nn import StochasticDepth
from ..efficientnet import (
    EfficientNet,
    InvertedResidualBlock,
    EfficientNetEncoder,
)


class FusedInvertedResidualBlock(nn.Module):
    """Inverted residual block proposed originally for EfficientNetV2.


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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int = 1,
        expansion: int = 6,
        activation: nn.Module = nn.SiLU,
        drop_rate: float = 0.2,
        se: bool = True,
        kernel_size: int = 3,
        **kwargs
    ):
        super().__init__()

        expanded_features = in_features * expansion
        # do not apply residual when downsamping and when features are different
        # in mobilenet we do not use a shortcut
        self.should_apply_residual = stride == 1 and in_features == out_features
        self.block = nn.Sequential(
            OrderedDict(
                {
                    "fused": ConvBnAct(
                        in_features,
                        expanded_features,
                        activation=activation,
                        kernel_size=kernel_size,
                        stride=stride,
                        # groups=in_features,
                        **kwargs,
                    ),
                    # apply se after depth-wise
                    "att": ChannelSE(
                        expanded_features,
                        reduced_features=in_features // 4,
                        activation=activation,
                    )
                    if se
                    else nn.Identity(),
                    "point": nn.Sequential(
                        ConvBnAct(
                            expanded_features,
                            out_features,
                            kernel_size=1,
                            activation=None,
                        )
                    ),
                    "drop": StochasticDepth(drop_rate)
                    if self.should_apply_residual and drop_rate > 0
                    else nn.Identity(),
                }
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.should_apply_residual:
            x += res
        return x


EfficientNetV2BasicBlock = FusedInvertedResidualBlock
EfficientNetV2Encoder = partial(
    EfficientNetEncoder,
    widths=[24, 24, 48, 64, 128, 160, 272, 1792],
    depths=[2, 4, 4, 6, 9, 15],
    strides=[2, 1, 2, 2, 2, 1, 2, 1],
    expansions=[1, 4, 4, 4, 6, 6],
    kernel_sizes=[3, 3, 3, 3, 3, 3, 3],
    se=[False, False, False, True, True, True],
    drop_rate=0.25,
    blocks=[
        FusedInvertedResidualBlock,
        FusedInvertedResidualBlock,
        FusedInvertedResidualBlock,
        InvertedResidualBlock,
        InvertedResidualBlock,
        InvertedResidualBlock,
    ],
)


class EfficientNetV2(EfficientNet):
    """Implementation of EfficientNet proposed in `EfficientNetV2: Smaller Models and Faster Training
    https://arxiv.org/abs/2104.00298>`_



       Args:
           in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
           n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    def __init__(self, encoder: nn.Module = EfficientNetV2Encoder, **kwargs):
        super().__init__(encoder, **kwargs)
