 
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
