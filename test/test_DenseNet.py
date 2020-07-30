import torch
from glasses.nn.models.classification.densenet import DenseNet


def test_alexnet():
    x = torch.rand(1, 3,224,224)
    model = DenseNet().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000
