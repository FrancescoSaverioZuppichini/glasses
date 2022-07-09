import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Reduce


class CBAMChannelAtt(nn.Module):
    def __init__(
        self,
        features: int,
        reduction: int = 16,
        reduced_features: int = None,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.reduced_features = (
            features // reduction if reduced_features is None else reduced_features
        )
        self.avg_pool = Reduce("b c h w -> b c 1 1", reduction="mean")
        self.max_pool = Reduce("b c h w -> b c 1 1", reduction="max")
        self.att = nn.Sequential(
            nn.Conv2d(features, self.reduced_features, kernel_size=1),
            activation(),
            nn.Conv2d(self.reduced_features, features, kernel_size=1),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y_avg = self.att(self.avg_pool(x))
        y_max = self.att(self.max_pool(x))
        y = self.gate(y_avg + y_max)
        return x * y


class CBAMSpatialAtt(nn.Module):
    def __init__(
        self,
        kernel_size: int = 7,
    ):
        super().__init__()

        self.avg_pool = Reduce("b c h w -> b 1 h w", reduction="mean")
        self.max_pool = Reduce("b c h w -> b 1 h w", reduction="max")
        self.att = nn.Sequential(
            nn.Conv2d(
                2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
            )
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y = torch.cat([y_avg, y_max], dim=1)
        y = self.gate(self.att(y))
        return x * y


class CBAM(nn.Module):
    """Implementation of Convolutional Block Attention Module proposed in `CBAM: Convolutional Block Attention Module <https://arxiv.org/abs/1807.06521>`_

    .. image::

    Examples:

        >>> # create cbamresnet50
        >>> from glasses.models.classification.resnet import ResNet, ResNetBottleneckBlock
        >>> from glasses.nn.att import CBAM, WithAtt
        >>> cbam_resnet50 = ResNet.resnet50(block=WithAtt(ResNetBottleneckBlock, att=CBAM))
        >>> cbam_resnet50.summary()

    Args:
        features (int, optional): Number of features features. Defaults to None.
        reduction (int, optional): Reduction ratio used to downsample the input. Defaults to 16.
        reduced_features:  If passed, use it instead of calculating the reduced features using `reduction`. Defaults to None.
        kernel_size (int, optional): kernel_size of the Conv2d to produce the 2D spatial attention map. Defaults to 7.
    """

    def __init__(
        self,
        features: int,
        reduction: int = 16,
        reduced_features: int = None,
        activation: nn.Module = nn.ReLU,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.channel_att = CBAMChannelAtt(
            features, reduction, reduced_features, activation
        )
        self.spatial_att = CBAMSpatialAtt(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
