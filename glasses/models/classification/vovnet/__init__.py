from __future__ import annotations
import torch
from torch import nn
from torch.functional import Tensor

from glasses.models.base import Encoder
from ..resnet import ReLUInPlace, ResNet, ResNetEncoder, ResNetStem3x3
from glasses.nn.blocks import Conv3x3BnAct, ConvBnAct
from typing import List
from functools import partial
from glasses.nn.blocks.residuals import ResidualAdd
from glasses.nn.att import EffectiveSE


class VoVNetBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        stage_features: int = 64,
        repeat: int = 5,
        block: nn.Module = Conv3x3BnAct,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            block(in_features, stage_features),
            *[block(stage_features, stage_features) for _ in range(repeat - 1)],
        )
        self.aggregate = ConvBnAct(
            in_features + (stage_features * repeat), out_features, kernel_size=1
        )

    def forward(self, x: Tensor) -> Tensor:
        features = [x]
        for block in self.blocks:
            x = block(x)
            features.append(x)
        x = torch.cat(features, dim=1)
        x = self.aggregate(x)
        return x


class VoVNetV2Block(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, *args, **kwargs):

        super().__init__(
            ResidualAdd(
                VoVNetBlock(in_features, out_features, *args, **kwargs),
                EffectiveSE(out_features),
            )
        )


class VoVNetLayer(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth: int,
        pool: nn.Module = nn.MaxPool2d,
        block: nn.Module = VoVNetBlock,
        **kwargs
    ):
        super().__init__(
            block(in_features, out_features, **kwargs),
            *[
                block(out_features, out_features, use_residual=True, **kwargs)
                for _ in range(depth - 1)
            ],
            pool(kernel_size=3, stride=2),
        )


VoVNetStem = partial(ResNetStem3x3, widths=[64, 64])


class VoVEncoder(Encoder):
    def __init__(
        self,
        in_channels: int = 3,
        start_features: int = 128,
        widths: List[int] = [256, 512, 768, 1024],
        depths: List[int] = [1, 1, 2, 2],
        stages_widths: List[int] = [128, 160, 192, 224],
        activation: nn.Module = ReLUInPlace,
        block: nn.Module = VoVNetBlock,
        stem: nn.Module = VoVNetStem,
        **kwargs
    ):
        super().__init__()
        self.widths = widths
        self.start_features = start_features
        self.in_out_widths = list(zip(widths, widths[1:]))
        self.stem = stem(in_channels, start_features, activation=activation)

        self.layers = nn.ModuleList(
            [
                VoVNetLayer(
                    start_features,
                    widths[0],
                    stage_features=stages_widths[0],
                    depth=depths[0],
                    block=block,
                    **kwargs,
                ),
                *[
                    VoVNetLayer(
                        in_features,
                        out_features,
                        stage_features=stage_features,
                        depth=depth,
                        block=block,
                        **kwargs,
                    )
                    for (in_features, out_features), depth, stage_features in zip(
                        self.in_out_widths, depths[1:], stages_widths[1:]
                    )
                ],
            ]
        )

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def stages(self):
        return [self.stem[-2], *self.layers[:-1]]

    @property
    def features_widths(self):
        return [self.start_features, *self.widths[:-1]]


class VoVNet(ResNet):
    """VoVNet proposed in
       `An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection
    <https://arxiv.org/abs/1904.09730>`_

    """

    def __init__(self, encoder: nn.Module = VoVEncoder, **kwargs):
        super().__init__(encoder, **kwargs)

    @classmethod
    def vovnet27s(cls, *args, **kwargs) -> VoVNet:
        """Creates a vovnet 27 model

        Returns:
            VoVNet: A vovnet 27 model
        """
        model = cls(
            *args,
            widths=[128, 256, 384, 512],
            stages_widths=[64, 80, 96, 112],
            depths=[1, 1, 1, 1],
            **kwargs,
        )

        return model

    @classmethod
    def vovnet39(cls, *args, **kwargs) -> VoVNet:
        """Creates a vovnet39 model

        Returns:
            VoVNet: A vovnet39 model
        """
        model = cls(*args, **kwargs)

        return model

    @classmethod
    def vovnet57(cls, *args, **kwargs) -> VoVNet:
        """Creates a vovnet57 model

        Returns:
            VoVNet: A vovnet57 model
        """
        model = cls(*args, depths=[1, 1, 4, 3], **kwargs)

        return model
