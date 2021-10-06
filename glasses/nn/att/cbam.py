import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce

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
