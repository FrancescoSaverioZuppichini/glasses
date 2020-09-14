import torch
import torch.nn as nn


class Swish(nn.Module):
    r"""Swish function proposed in `SWISH: A SELF-GATED ACTIVATION FUNCTION <https://arxiv.org/pdf/1710.05941v1.pdf>`_ 

    .. math::
    
        {\displaystyle \operatorname {swish} (x):=x\times \operatorname {sigmoid} (\beta x)={\frac {x}{1+e^{-\beta x}}}.}

    :Usage:

        >>> swish = Swish()
        >>> x = torch.ones(10)
        >>> swish(x)

    Args:
        inplace (bool, optional): Perform the computation in-place. Defaults to False.
    """

    def __init__(self, inplace=False):

        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sig = torch.sigmoid(x)
        out = x.mul_(sig)  if self.inplace else x * sig
        return out