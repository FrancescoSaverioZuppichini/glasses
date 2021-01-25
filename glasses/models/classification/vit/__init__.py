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
from typing import List


class ViTTokens(nn.Module):
    def __init__(self, emb_size: int):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x: Tensor) -> List[Tensor]:
        b = x.shape[0]
        tokens = []
        for token in self.parameters():
            # for each token repeat itself over the batch dimension
            tokens.append(repeat(token, '() n e -> b n e', b=b))
        return tokens

    def __len__(self):
        return len(list(self.parameters()))

    # def __repr__(self):
    #     buffer = ""
    #     for name, token in self.named_parameters():
    #         buffer += f"({name}): {list(token.data.shape)}\n"
    #     return buffer


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224, tokens: nn.Module = ViTTokens):
        """

        Patch Embedding layer used in ViT. In order to work with transformers, this layer decompose 
        the input in multiple patches, add class token parameter and a position encoding (both learnable) and flat them.

        The following image from the authors shows the architecture.


        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/ViTPatchesPositionEmbedding.png?raw=true

        Example:

            Change the tokens

            >>> class MyTokens(ViTTokens):
            >>>     def __init__(self, emb_size: int):
            >>>         super().__init__(emb_size)
            >>>         self.my_new_token = nn.Parameter(torch.randn(1, 1, emb_size))
            >>> PatchEmbedding(tokens=MyTokens)

        Args:
            in_channels (int, optional): Number of input's channels. Defaults to 3.
            patch_size (int, optional): Size of the each patch. Defaults to 16.
            emb_size (int, optional):  Embedding dimensions Defaults to 768.
            img_size (int, optional): Size of the input image, this is needed to calculate the final number of patches. Defaults to 224.
            tokens (nn.Module, optional): A module that contains the tokens as his parameters. Defaults to ViTTokens.

        """
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size,
                      kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.tokens = tokens(emb_size)
        self.positions = nn.Parameter(torch.randn(
            (img_size // patch_size) ** 2 + len(self.tokens), emb_size))

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        # get the tokens
        tokens = self.tokens(x)
        # prepend the tokens to the input
        x = torch.cat([*tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 12, att_drop_p: float = 0., projection_drop_p: float = 0., qkv_bias: bool = False):
        """
        Classic multi head attention proposed in `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

        Args:
            emb_size (int, optional):  Embedding dimensions Defaults to 768.
            num_heads (int, optional): Number of heads. Defaults to 12.
            att_drop_p (float, optional): Attention dropout probability. Defaults to 0..
            projection_drop_p (float, optional): Projection dropout probability. Defaults to 0..
            qkv_bias (bool, optional): If yes, apply bias to the qkv projection matrix. Defaults to False.

        """
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3, bias=qkv_bias)
        self.att_drop = nn.Dropout(att_drop_p)
        self.projection = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.Dropout(projection_drop_p)
        )

        self.scaling = (self.emb_size // num_heads) ** -0.5

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(
            self.qkv(x), "b n (qkv h d) -> (qkv) b h n d", h=self.num_heads, qkv=3)

        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk',
                              queries, keys) * self.scaling
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        att = F.softmax(energy, dim=-1)
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

# TODO move it to blocks


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x += out
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0., activation: nn.Module = nn.GELU):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            activation(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(drop_p)
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 activation: nn.Module = nn.GELU,
                 ** kwargs):
        """
        Transformer Encoder block proposed in `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

        The following image from the authors shows the architecture.


        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/ViTTransformerBlock.png?raw=true

        Args:
            emb_size (int, optional):  Embedding dimensions Defaults to 768.
            forward_expansion (int, optional): [description]. Defaults to 4.
            forward_drop_p (float, optional): [description]. Defaults to 0..
        """
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs)
            )),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p, activation=activation)),
            )
        )


class TransformerEncoder(Encoder):
    def __init__(self, depth: int = 12, emb_size: int = 786, block: nn.Module = TransformerEncoderBlock,  **kwargs):
        """
        Transformer Encoder proposed in `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

        .. warning::
            Even if `TransformerEncoder` uses the `Encoder` APIs you won't be able to use it with `segmentation` models
            since they will expect 3-D tensors as inputs. 

        Args:
            depth (int, optional): Number of transformer's blocks. Defaults to 12.
            block ( nn.Module, optional): Block used inside the transformer encoder. Defaults to TransformerEncoderBlock.
            emb_size (int, optional):  Embedding dimensions Defaults to 768.

        """
        super().__init__()
        self.widths = [emb_size] * depth
        self.layers = nn.ModuleList([block(emb_size, **kwargs)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_size)
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class ViTClassificationHead(nn.Sequential):
    POLICIES = ['token', 'mean']

    def __init__(self, emb_size: int = 768, n_classes: int = 1000, policy: str = 'token'):
        """
        ViT Classification Head

        Args:
            emb_size (int, optional):  Embedding dimensions Defaults to 768.
            n_classes (int, optional): [description]. Defaults to 1000.
            policy (str, optional): Pooling policy, can be token or mean. Defaults to 'token'.
        """

        assert policy in self.POLICIES, f"Only policies {','.join(self.POLICIES)} are supported"

        super().__init__(OrderedDict({
            'pool': Reduce('b n e -> b e', reduction='mean') if policy == 'mean' else Lambda(lambda x: x[:, 0]),
            'fc': nn.Linear(emb_size, n_classes)
        }))


class ViT(nn.Sequential, VisionModule):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 224,
                 tokens: nn.Module = ViTTokens,
                 depth: int = 12,
                 n_classes: int = 1000,
                 **kwargs):
        """
        Implementation of Vision Transformer (ViT) proposed in `An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale <https://arxiv.org/pdf/2010.11929.pdf>`_

        The following image from the authors shows the architecture.

        .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/ViT.png?raw=true

        Examples:

            Default models

            >>> ViT.vit_small_patch16_224()
            >>> ViT.vit_base_patch16_224()
            >>> ViT.vit_base_patch16_384()
            >>> ViT.vit_base_patch32_384()
            >>> ViT.vit_huge_patch16_224()
            >>> ViT.vit_huge_patch32_384()
            >>> ViT.vit_large_patch16_224()
            >>> ViT.vit_large_patch16_384()
            >>> ViT.vit_large_patch32_384()

            You can easily customize your model


            >>> # change activation
            >>> ViT.vit_base_patch16_224(activation = nn.SELU)
            >>> # change number of classes (default is 1000 )
            >>> ViT.vit_base_patch16_224(n_classes=100)
            >>> # pass a different block, default is TransformerEncoderBlock
            >>> ViT.vit_base_patch16_224(block=MyCoolTransformerBlock)
            >>> # get features
            >>> model = ViT.vit_base_patch16_224
            >>> # first call .features, this will activate the forward hooks and tells the model you'll like to get the features
            >>> model.encoder.features
            >>> model(torch.randn((1,3,224,224)))
            >>> # get the features from the encoder
            >>> features = model.encoder.features
            >>> print([x.shape for x in features])
            >>> #[[torch.Size([1, 197, 768]),  torch.Size([1, 197, 768]), ...]
            >>> # change the tokens, you have to subclass ViTTokens
            >>> class MyTokens(ViTTokens):
            >>>     def __init__(self, emb_size: int):
            >>>         super().__init__(emb_size)
            >>>         self.my_new_token = nn.Parameter(torch.randn(1, 1, emb_size))
            >>> ViT(tokens=MyTokens)

        Args:
            in_channels (int, optional): [description]. Defaults to 3.
            patch_size (int, optional): [description]. Defaults to 16.
            emb_size (int, optional):  Embedding dimensions Defaults to 768.
            img_size (int, optional): [description]. Defaults to 224.
            tokens (nn.Module, optional): A module that contains the tokens as his parameters. Defaults to ViTTokens.
            depth (int, optional): [description]. Defaults to 12.
            n_classes (int, optional): [description]. Defaults to 1000.
        """
        super().__init__(OrderedDict({
            'embedding': PatchEmbedding(in_channels, patch_size, emb_size, img_size, tokens),
            'encoder': TransformerEncoder(depth, emb_size, **kwargs),
            'head': ViTClassificationHead(emb_size, n_classes)
        }))

    @classmethod
    def vit_small_patch16_224(cls, **kwargs):
        return cls(depth=8, num_heads=8, forward_expansion=3, **kwargs)

    @classmethod
    def vit_base_patch16_224(cls,  **kwargs):
        return cls(depth=12, num_heads=12, forward_expansion=4, qkv_bias=True, **kwargs)

    @classmethod
    def vit_base_patch16_384(cls, **kwargs):
        return cls.vit_base_patch16_224(img_size=384, **kwargs)

    @classmethod
    def vit_base_patch32_384(cls, **kwargs):
        return cls.vit_base_patch16_384(patch_size=32, **kwargs)

    @classmethod
    def vit_large_patch16_224(cls, **kwargs):
        return cls(emb_size=1024, depth=24, num_heads=16, qkv_bias=True, **kwargs)

    @classmethod
    def vit_large_patch16_384(cls, **kwargs):
        return cls.vit_large_patch16_224(img_size=384, **kwargs)

    @classmethod
    def vit_large_patch32_384(cls, **kwargs):
        return cls.vit_large_patch16_384(patch_size=32, **kwargs)

    @classmethod
    def vit_huge_patch16_224(cls, **kwargs):
        return cls(emb_size=1280, depth=32, num_heads=16, **kwargs)

    @classmethod
    def vit_huge_patch32_384(cls, **kwargs):
        return cls.vit_huge_patch16_224(patch_size=32, img_size=384, **kwargs)
