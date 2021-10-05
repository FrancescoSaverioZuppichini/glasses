import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable
from functools import partial


class ResidualAdd(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        shortcut: nn.Module = nn.Identity(),
        in_place: bool = True,
    ):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        self.in_place = in_place

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        res = self.shortcut(res)
        if self.in_place:
            x += res
        else:
            x = x + res
        return res
