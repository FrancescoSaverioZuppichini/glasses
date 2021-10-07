from torch import nn
import torch
from glasses.nn.blocks import ConvBnAct, Conv3x3BnAct
from glasses.nn.att.utils import make_divisible
from einops.layers.torch import Rearrange, Reduce
from typing import Union, List

class SKAtt(nn.Module):
    def __init__(self, 
                 in_features: int,
                 out_features: int = None,
                 kernel_size: Union[List, int] = [3, 5],
                 stride: int = 1,
                 groups: int = 1, 
                 reduction: int = 16,
                 reduction_divisor: int = 8,
                 reduced_features: int = None,
                 keep_3x3: bool = True,
                 ):
        super().__init__()
        out_features = out_features or in_features
        mid_features = reduced_features or make_divisible(out_features // reduction, divisor=reduction_divisor)
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2
        if keep_3x3:
            dilation = [1 * (k - 1) // 2 for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [1 * (k - 1) // 2 for k in kernel_size]
        groups = min(out_features, groups)
        self.num_paths = len(kernel_size)

        self.split = nn.ModuleList([
            ConvBnAct(in_features = in_features,
                      out_features = out_features,
                      mode = "same",
                      stride = stride,
                      kernel_size = k,
                      dilation = d, 
                      padding = k // 2,
                      groups = groups)
            for k, d in zip(kernel_size, dilation)
        ])

        self.fuse = nn.Sequential(
            Reduce('b s c h w -> b c h w', reduction='sum', s=len(self.split)), 
            Reduce('b c h w -> b c 1 1', reduction='mean'),
            ConvBnAct(in_features, mid_features, kernel_size=1, bias=False)
        )

        self.select = nn.Sequential(
            nn.Conv2d(mid_features, len(self.split) * in_features, kernel_size=1, bias=False),
            Rearrange('b (s c) h w -> b s c h w', s=len(self.split), c=in_features),
            nn.Softmax(dim=1),
            Reduce('b s c h w -> b c h w', reduction='sum', s=len(self.split))
        )

    def forward(self, x):
        splitted = [path(x) for path in self.split]
        splitted = torch.stack(splitted, dim=1)
        x_attn = self.fuse(splitted)
        x_attn = self.select(x_attn)
        x = x * x_attn
        return x
