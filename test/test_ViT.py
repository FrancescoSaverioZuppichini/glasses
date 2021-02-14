import torch
from torch import nn
import pytest
from glasses.models.classification.vit import ViT, ViTClassificationHead
from functools import partial


def test_head():
    x = torch.rand((1, 10, 32))

    head = ViTClassificationHead(emb_size = 32, policy = 'mean')
    head(x)
    head = ViTClassificationHead(emb_size = 32, policy = 'token')
    head(x)

    with pytest.raises(AssertionError):
        head = ViTClassificationHead(emb_size = 32, policy = 'trolololo')

def test_features():
    x = torch.rand((1, 3, 224, 224))
    model = ViT.vit_base_patch16_224(emb_size=24)
    model.encoder.features
    model(x)
    features = model.encoder.features

    assert len(features) == len(model.encoder.layers) - 1

def test_vit():
    x = torch.rand((1, 3, 224, 224))
    model = ViT.vit_small_patch16_224().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = ViT.vit_base_patch16_224(n_classes = 100).eval()
    pred = model(x)
    assert pred.shape[-1] == 100

    x = torch.rand((1, 3, 384, 384))
    model = ViT.vit_base_patch16_384().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    x = torch.rand((1, 3, 384, 384))
    model = ViT.vit_base_patch32_384().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    x = torch.rand((1, 3, 224, 224))
    model = ViT.vit_large_patch16_224().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    x = torch.rand((1, 3, 384, 384))
    model = ViT.vit_large_patch16_384().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    x = torch.rand((1, 3, 384, 384))
    model = ViT.vit_large_patch32_384().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000
    # cannot test huge models, too big
