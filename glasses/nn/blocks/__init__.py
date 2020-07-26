 
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from typing import Callable
from torch import Tensor
from enum import Enum


ReLUInPlace = partial(nn.ReLU, inplace=True)


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


class ConvAct(nn.Module):
    """Basic block composed by a convolution with adaptive padding followed by an activation function. 

    Args:
        in_features (int): [description]
        out_features (int): [description]
        activation (nn.Module, optional): [description]. Default is ReLUInPlace.
        conv (nn.Module, optional): [description]. Default is Conv2dPad.
        kernel_size (int): [description]. Default is 3.
    """

    def __init__(self, in_features: int, out_features: int,  activation: nn.Module = ReLUInPlace, conv: nn.Module = Conv2dPad, kernel_size: int = 3, **kwargs):
        super().__init__()
        self.conv = conv(in_features, out_features, kernel_size=kernel_size, **kwargs)
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.conv(x))
        return x