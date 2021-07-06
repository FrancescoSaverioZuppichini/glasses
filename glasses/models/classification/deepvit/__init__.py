from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from ..vit import MultiHeadAttention, ViT
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class ReMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        emb_size: int = 768,
        num_heads: int = 12,
        att_drop_p: float = 0.0,
        projection_drop_p: float = 0.2,
        qkv_bias: bool = False,
    ):
        r"""

        Implementation of multi head attention proposed in `DeepViT: Towards Deeper Vision Transformer <https://arxiv.org/abs/2103.11886>`_

        The Attention is computed by:

        .. math::

            \begin{equation}
            \operatorname{Re}-\operatorname{Attention}(Q, K, V)=\operatorname{Norm}\left(\Theta^{\top}\left(\operatorname{Softmax}\left(\frac{Q K^{\top}}{\sqrt{d}}\right)\right)\right) V
            \end{equation}

        We used multi head attention

        .. math::

            \begin{equation}
            \begin{aligned}
            \operatorname{MultiHead}(Q, K, V) &=\text { Concat }\left(\text { head }_{1}, \ldots, \text { head }_{\mathrm{h}}\right) W^{O} \\
            \text { where head }_{\mathrm{i}} &=\operatorname{Attention}\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right)
            \end{aligned}
            \end{equation}

        Args:
            emb_size (int, optional):  Embedding dimensions Defaults to 768.
            num_heads (int, optional): Number of heads. Defaults to 12.
            att_drop_p (float, optional): Attention dropout probability. Defaults to 0..
            projection_drop_p (float, optional): Projection dropout probability. Defaults to 0..
            qkv_bias (bool, optional): If yes, apply bias to the qkv projection matrix. Defaults to False.

        """
        super().__init__(emb_size,
                         num_heads,
                         att_drop_p,
                         projection_drop_p,
                         qkv_bias)

        self.theta = nn.Parameter(torch.randn(num_heads, num_heads))
        self.re_norm = nn.Sequential(
            # we need to normalize each head
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(self.num_heads),
            Rearrange('b i j h -> b h i j'),
        )

    def re_attend(self, x: Tensor) -> Tensor:
        # [B, H, EMB, EMB] @ [H, H]
        out = torch.einsum('bhij, hg -> bgij', x, self.theta)
        out = self.re_norm(out)
        return out

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        attended = self.attend(x, mask)
        re_attended = self.re_attend(attended)
        out = rearrange(re_attended, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


DeepVit = partial(ViT, attention=ReMultiHeadAttention)
"""
Implementation of  DeepViT proposed in `DeepViT: Towards Deeper Vision Transformer <https://arxiv.org/abs/2103.11886>`
"""