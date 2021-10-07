import torch
import torch.nn as nn
from typing import Union, List

from glasses.nn.att.utils import make_divisible
from ..blocks import ConvBnAct
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
        self.att = nn.Sequential(
            Reduce("b n c h w -> b c h w", reduction="sum"),
            Reduce("b c h w -> b c 1 1", reduction="mean"),
            nn.Conv2d(features, mid_features, kernel_size=1, bias=False),
            norm_layer(mid_features),
            act_layer(inplace=True),
            nn.Conv2d(mid_features, features * num_paths, kernel_size=1, bias=False),
            Rearrange('b (n c) h w -> b n c h w', n=num_paths, c=features),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        assert x.shape[1] == self.num_paths
        x = self.att(x)
        return x


class SelectiveKernel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        kernel_size: Union[List, int] = None,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        reduction: int = 16,
        reduction_divisor: int = 8,
        reduced_features: int = None,
        keep_3x3: bool = True,
        activation: nn.Module = nn.ReLU,
        normalization: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        out_features = out_features or in_features
        kernel_size = kernel_size or [3, 5]
        _kernel_valid(kernel_size)
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2
        if keep_3x3:
            dilation = [dilation * (k - 1) // 2 for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)
        self.num_paths = len(kernel_size)
        self.in_features = in_features
        self.out_features = out_features,
        groups = min(out_features, groups)
        
        self.paths = nn.ModuleList([
            ConvBnAct(in_features = in_features, 
                      out_features = out_features, 
                      activation = activation, 
                      normalization=normalization,
                      mode = "same",
                      stride=stride,
                      kernel_size=k, 
                      dilation=d)
            for k, d in zip(kernel_size, dilation)
        ])

        attn_features = reduced_features or make_divisible(out_features // reduction, divisor=reduction_divisor)
        self.attn = SelectiveKernelAtt(out_features, self.num_paths, attn_features)
    
    def forward(self, x):
        x_paths = [op(x) for op in self.paths]
        x = torch.stack(x_paths, dim=1)
        x_attn = self.attn(x)
        x = x * x_attn
        return torch.sum(x, dim=1)


        
