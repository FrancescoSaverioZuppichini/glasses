import torch
from glasses.nn.models.classification.resnet import ResNet, ResNetBasicPreActBlock, ResNetBottleneckPreActBlock

def test_resnet():
    x = torch.rand(1, 3, 224, 224)
    model = ResNet.resnet18().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = ResNet.resnet34().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = ResNet.resnet50().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = ResNet.resnet101().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = ResNet.resnet152().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000


    model = ResNet.resnet34(block=ResNetBasicPreActBlock).eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = ResNet.resnet34(block=ResNetBottleneckPreActBlock).eval()
    pred = model(x)
    assert pred.shape[-1] == 1000