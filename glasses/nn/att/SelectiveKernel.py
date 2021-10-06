import torch
import torch.nn as nn
# from .utils import make_divisible
from einops.layers.torch import Rearrange, Reduce

def _kernel_valid(k):
    if isinstance(k, (list, tuple)):
        for ki in k:
            return _kernel_valid(ki)
    assert k >=3 and k % 2

class SelectiveKernelAtt(nn.Module):
    def __init__(
        self,
        features: int,
        num_paths: int = 2,
        mid_features: int = 32,
        act_layer: nn.Module = nn.ReLU,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        self.num_paths = num_paths
        self.sum = Reduce("b n c h w -> b c h w", reduction="sum")
        self.avg_pool = Reduce("b c h w -> b c 1 1", reduction="mean")
        self.att = nn.Sequential(
            nn.Conv2d(features, mid_features, kernel_size=1, bias=False),
            norm_layer(mid_features),
            act_layer(inplace=True),
            nn.Conv2d(mid_features, features * num_paths, kernel_size=1, bias=False),
            Rearrange('b (n c) h w -> b n c h w', n=num_paths, c=features),
            nn.Softmax(dim=1),
        )

    
    def forward(self, x):
        assert x.shape[1] == self.num_paths
        x = self.sum(x)
        x = self.avg_pool(x)
        x = self.att(x)
        return x