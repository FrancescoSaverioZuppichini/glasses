import torch
from glasses.models.classification import AlexNet


def test_alexnet():
    x = torch.rand(1, 3,224,224)
    model = AlexNet().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000
