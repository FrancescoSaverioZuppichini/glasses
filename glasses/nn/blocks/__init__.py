
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from typing import Callable
from torch import Tensor
from enum import Enum


class Lambda(nn.Module):
    """[summary]

    Args:
        lambd (Callable[Tensor]): A function that does something

    Examples:
        >>> add_two = Lambda(lambd x: x + 2)
        >>> add_two(Tensor([0])) // 2
    """

    def __init__(self, lambd: Callable[[Tensor], Tensor]):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        return self.lambd(x)


class Conv2dPad(nn.Conv2d):
    """
    Replacement for nn.Conv2d with padding by default.

    Examples:
        >>> conv = Conv2dPad(1, 5, kernel_size=3)
        >>> x = torch.rand((1, 1, 5, 5))
        >>> print(conv(x).shape) 
        [1,1,5,5]
    """
    MODES = ['auto']

    def __init__(self, *args, mode: str = 'auto', **kwargs):
        super().__init__(*args, **kwargs)
        if mode == 'auto':
            # dynamic add padding based on the kernel_size
            self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


class ConvBnAct(nn.Sequential):
    """Utility module that stacks one convolution layer, a normalization layer and an activation function.

    Example:
        >>> ConvBnAct(32, 64, kernel_size=3)
            ConvBnAct(
                (conv): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): ReLU()
            )

        >>> ConvBnAct(32, 64, kernel_size=3, normalization = None )
            ConvBnAct(
                (conv): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (act): ReLU()
            )

        >>> ConvBnAct(32, 64, kernel_size=3, activation = None )
            ConvBnAct(
                (conv): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )

    We also provide additional modules built on top of this one: `ConvBn`, `ConvAct`, `Conv3x3BnAct`
    Args:
            in_features (int): [description]
            out_features (int): [description]
            conv (nn.Module, optional): Convolution layer. Defaults to Conv2dPad.
            normalization (nn.Module, optional): Normalization layer. Defaults to nn.BatchNorm2d.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU.
    """
    def __init__(self, in_features: int, out_features: int, conv: nn.Module = Conv2dPad, normalization: nn.Module = nn.BatchNorm2d, activation: nn.Module = nn.ReLU, *args, **kwargs):

        super().__init__()
        self.add_module('conv', conv(
            in_features, out_features, *args, **kwargs))
        if normalization:
            self.add_module('bn', normalization(out_features))
        if activation:
            self.add_module('act', activation())


ConvBn = partial(ConvBnAct, activation=None)
ConvAct = partial(ConvBnAct, normalization=None)
Conv3x3BnAct = partial(ConvBnAct, kernel_size=3)
