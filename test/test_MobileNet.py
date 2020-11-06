import torch
from glasses.nn.models.classification.mobilenet import MobileNet


def test_alexnet():
    x = torch.rand(1, 3,224,224)
    model = MobileNet.mobilenet_v2().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000
