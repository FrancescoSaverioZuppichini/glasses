import torch
from glasses.models.classification import DenseNet


def test_DenseNet():
    x = torch.rand(1, 3, 224, 224)
    model = DenseNet().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000


def test_densenet121():
    x = torch.rand(1, 3, 224, 224)
    model = DenseNet().densenet121()
    pred = model(x)
    assert pred.shape[-1] == 1000


def test_densenet161():
    x = torch.rand(1, 3, 224, 224)
    model = DenseNet().densenet161()
    pred = model(x)
    assert pred.shape[-1] == 1000


def test_densenet169():
    x = torch.rand(1, 3, 224, 224)
    model = DenseNet().densenet169()
    pred = model(x)
    assert pred.shape[-1] == 1000


def test_densenet201():
    x = torch.rand(1, 3, 224, 224)
    model = DenseNet().densenet201()
    pred = model(x)
    assert pred.shape[-1] == 1000
