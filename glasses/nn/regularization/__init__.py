import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor


class DropBlock(nn.Module):
    def __init__(self, block_size: int = 7, p: float = 0.5):
        """Implementation of Drop Block proposed in `DropBlock: A regularization method for convolutional networks <https://arxiv.org/abs/1810.12890>`_

        Similar to dropout but it maskes clusters of close pixels. The following image shows normal Dropout (b) and Drop Block (c)

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DropBlock.jpg?raw=true

        The following picture shows the effect of DropBlock on an input image

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DropBlockGrogu.png

        Args:
            block_size (int, optional): Dimension of the pixel cluster. Defaults to 7.
            p (float, optional): probability, the bigger the mode clusters. Defaults to 0.5.
        """
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        """Compute gamma, eq (1) in the paper

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: gamma
        """
        return (
            self.p
            * x.shape[-1] ** 2
            / (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            gamma = self.calculate_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x

    def __repr__(self):
        return f"DropBlock(p={self.p})"


class StochasticDepth(nn.Module):
    """Implementation of Stochastic Depth proposed in `Deep Networks with Stochastic Depth <https://arxiv.org/abs/1603.09382>`_.

    The main idea is to skip one layer completely.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            probs = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.p
            x = torch.div(x, self.p) * probs
        return x

    def __repr__(self):
        return f"StochasticDepth(p={self.p})"
