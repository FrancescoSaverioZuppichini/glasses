import torch
from eyes.nn.models.classification.resnet import ResNet


def test_resnet():
    x = torch.rand(1, 3,224,224)
    model = ResNet.resnet18().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = ResNet.resnet34().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = ResNet.resnet50().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000
