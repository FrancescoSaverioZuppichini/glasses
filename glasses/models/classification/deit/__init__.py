from __future__ import annotations
import torch
from torch import nn
from torch import Tensor
from glasses.nn.blocks.residuals import ResidualAdd
from glasses.nn.blocks import Lambda
from collections import OrderedDict
from glasses.utils.PretrainedWeightsProvider import pretrained
from ....models.base import Encoder, VisionModule
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from typing import Any, List
from ..vit import ViT, ViTTokens
from glasses.utils.PretrainedWeightsProvider import pretrained


class DeiTTokens(ViTTokens):
    def __init__(self, emb_size: int):
        """Tokens for DeiT, it contains the `cls` token present in ViT plus a special token, `dist`, used for distillation.

        Args:
            emb_size (int):  Embedding dimensions
        """
        super().__init__(emb_size)
        self.dist = nn.Parameter(torch.randn(1, 1, emb_size))


class DeiTClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        """DeiT classification head, it relies on two heads using the `cls` and the`dist` token respectively. 
        At test time, the prediction is made by avering the results from the two, while during training both predictions are returned.
        
        Args:
            emb_size (int, optional):  Embedding dimensions Defaults to 768.
            n_classes (int, optional): [description]. Defaults to 1000.
        """
        super().__init__()

        self.head = nn.Linear(emb_size, n_classes)
        self.dist_head = nn.Linear(emb_size, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x, x_dist = x[:, 0], x[:, 1]
        x_head = self.head(x)
        x_dist_head = self.dist_head(x_dist)
        
        if self.training:
            x = x_head, x_dist_head
        else:
            x = (x_head + x_dist_head) / 2
        return x


class DeiT(ViT):
    """Implementation of DeiT proposed in `Training data-efficient image transformers & distillation through attention <https://arxiv.org/pdf/2010.11929.pdf>`_

    An attention based distillation is proposed where a new token is added to the model, the `dist` token.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/DeiT.png?raw=true

    Examples:

        Default models

        >>> DeiT.deit_tiny_patch16_224()
        >>> DeiT.deit_small_patch16_224()
        >>> DeiT.deit_base_patch16_224()
        >>> DeiT.deit_base_patch16_384()


    Args:  
        ViT ([type]): [description]
    """
    def __init__(self, *args, tokens: nn.Module = DeiTTokens, emb_size: int = 768, n_classes: int = 1000, **kwargs):
        super().__init__(*args, tokens=tokens, emb_size=emb_size, **kwargs)
        self.head = DeiTClassificationHead(emb_size, n_classes)

    @classmethod
    @pretrained()
    def deit_tiny_patch16_224(cls, **kwargs):
        model = cls(patch_size=16, emb_size=192, depth=12,
                    num_heads=3, qkv_bias=True, **kwargs)

        return model

    @classmethod
    @pretrained()
    def deit_small_patch16_224(cls, **kwargs):
        model = cls(patch_size=16, emb_size=384, depth=12,
                    num_heads=6, qkv_bias=True, **kwargs)

        return model

    @classmethod
    def deit_base_patch16_224(cls, **kwargs):
        model = cls(patch_size=16, emb_size=768, depth=12,
                    num_heads=12, qkv_bias=True, **kwargs)
        return model

    @classmethod
    def deit_base_patch16_384(cls, **kwargs):
        model = cls(img_size=384, patch_size=16, emb_size=768, depth=12,
                    num_heads=12, qkv_bias=True,  **kwargs)
        return model
