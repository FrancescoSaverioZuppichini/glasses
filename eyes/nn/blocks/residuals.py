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
        if self.shortcut is not None:
            res = self.shortcut(res)
        x = self.block(x)
        if self.res_func is not None:
            x = self.res_func(x, res)
        return x


ResidualAdd = partial(Residual, res_func=lambda x, res: x + res)
ResidualCat = partial(Residual, res_func=lambda x, res: torch.cat([x, res]))
ResidualCat2d = partial(ResidualCat, res_func=lambda x,
                        res: torch.cat([x, res], dim=1))
