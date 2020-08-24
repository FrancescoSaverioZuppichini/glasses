import torch
from glasses.nn.models.classification.senet import SEResNet


def test_seresnet():
    x = torch.rand(1, 3, 224, 224)
    model = SEResNet.resnet18().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = SEResNet.resnet34().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = SEResNet.resnet50().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = SEResNet.resnet101().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = SEResNet.resnet152().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000
