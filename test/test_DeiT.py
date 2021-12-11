import torch
from torch import nn
import pytest
from glasses.models.classification.deit import DeiT, DeiTClassificationHead
from functools import partial


def test_head():
    x = torch.rand((1, 10, 32))

    head = DeiTClassificationHead(emb_size = 32)
    cls_out, dist_out = head(x)

    assert cls_out.shape == dist_out.shape

    head = DeiTClassificationHead(emb_size = 32).eval()
    out  = head(x)

    assert out.shape == cls_out.shape

def test_features():
    x = torch.rand((1, 3, 224, 224))
    model = DeiT.deit_base_patch16_224()
    model.encoder.features
    model(x)
    features = model.encoder.features

    assert len(features) == len(model.encoder.layers) - 1

def test_vit():
    x = torch.rand((1, 3, 224, 224))
    model = DeiT.deit_tiny_patch16_224().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = DeiT.deit_small_patch16_224(n_classes = 100).eval()
    pred = model(x)
    assert pred.shape[-1] == 100

    x = torch.rand((1, 3, 224, 224))
    model = DeiT.deit_base_patch16_224().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    x = torch.rand((1, 3, 384, 384))
    model = DeiT.deit_base_patch16_384().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

