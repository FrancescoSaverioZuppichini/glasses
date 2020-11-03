import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable
from functools import partial


class Residual(nn.Module):
    """It applies residual connection to a `nn.Module` where the output becomes

    :math:`y = F(x) + x`

    Examples:
        >>> block = nn.Identity() // does nothing
        >>> res = Residual(block, res_func=lambda x, res: x + res)
        >>> res(x) // tensor([2])

        .. image:: https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/Residual.png?raw=true

        You can also pass a `shortcut` function

        >>> res = Residual(block, res_func=lambda x, res: x + res, shortcut=lambda x: x * 2)
        >>> res(x) // tensor([3])

        .. image:: https://github.com/FrancescoSaverioZuppichini/torchlego/blob/develop/doc/images/Residual_shorcut.png?raw=true


    """

    def __init__(self, block: nn.Module,
                 res_func: Callable[[Tensor], Tensor] = None,
                 shortcut: nn.Module = None, *args, **kwargs):
        """

        Args:
            block (nn.Module): A Pytorch module
            res_func (Callable[[Tensor], Tensor], optional): The residual function. Defaults to None.
            shortcut (nn.Module, optional): A function applied before the input is passed to `block`. Defaults to None.
        """
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        self.res_func = res_func

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.shortcut is not None:
            res = self.shortcut(res)
        if self.res_func is not None:
            x = self.res_func(x, res)
        return x


def add(x: Tensor, res: Tensor) -> Tensor:
    return x.add_(res)

class ResidualAdd(Residual):
    def __init__(self, *args, **kwags):
        super().__init__(*args, res_func=add, **kwags)

# ResidualAdd = partial(Residual, res_func=add)
ResidualCat = partial(Residual, res_func=lambda x, res: torch.cat([x, res]))
ResidualCat2d = partial(Residual, res_func=lambda x, res: torch.cat([x, res], dim=1))


class InputForward(nn.Module):
    """
    This module passes the input to multiple modules and applies a aggregation function on the result.

    .. image:: https://raw.githubusercontent.com/FrancescoSaverioZuppichini/torchlego/develop/doc/images/InputForward.png
    """

    def __init__(self, blocks: nn.Module, aggr_func: Callable[[Tensor], Tensor]):
        super().__init__()
        self.layers = blocks
        self.aggr_func = aggr_func

    def forward(self, x: Tensor) -> Tensor:
        out = None
        for block in self.layers:
            block_out = block(x)
            out = block_out if out is None else self.aggr_func(
                [block_out, out])
        return out


Cat = partial(InputForward, aggr_func=lambda x: torch.cat(x, dim=0))
Cat2d = partial(InputForward, aggr_func=lambda x: torch.cat(x, dim=1))

"""Pass the input to multiple modules and concatenates the output, for 1D input you can use `Cat`, while for 2D inputs, such as images, you can use `Cat2d`.

.. image:: https://raw.githubusercontent.com/FrancescoSaverioZuppichini/torchlego/develop/doc/images/Cat.png

Examples:

    >>> blocks = nn.ModuleList([nn.Conv2d(32, 64, kernel_size=3), nn.Conv2d(32, 64, kernel_size=3)])
    >>> x = torch.rand(1, 32, 48, 48)
    >>> Cat2d(blocks)(x).shape 
    # torch.Size([1, 128, 46, 46])
"""
