import torch
from glasses.nn.models.classification.mobilenet import MobileNet


def test_alexnet():
    x = torch.rand(1, 3,224,224)
    model = MobileNet().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000
