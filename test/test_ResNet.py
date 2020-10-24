import torch
from glasses.nn.models.classification.resnet import ResNet, ResNetBasicPreActBlock, ResNetBottleneckPreActBlock
from glasses.nn.models.classification.resnetxt import ResNetXt
from glasses.nn.models.classification.wide_resnet import WideResNet, WideResNetBottleNeckBlock
def test_resnet():
    x = torch.rand(1, 3, 224, 224)
    model = ResNet.resnet18().eval()
    # model.summary(device=torch.device('cpu'))

    pred = model(x)
    assert pred.shape[-1] == 1000
    
    model = ResNet.resnet26().eval()
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

    model = ResNet.resnet200().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000


    model = ResNet.resnet34(block=ResNetBasicPreActBlock).eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = ResNet.resnet34(block=ResNetBottleneckPreActBlock).eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

def test_resnetxt():
    x = torch.rand(1, 3, 224, 224)
    model = ResNetXt.resnext50_32x4d().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = ResNetXt.resnext101_32x8d().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

def test_wide_resnet():
    x = torch.rand(1, 3, 224, 224)
    model = WideResNet.wide_resnet50_2().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = WideResNet.wide_resnet101_2().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000


    block = WideResNetBottleNeckBlock(32, 256, width_factor=2)

    assert block.block.conv2.in_channels ==  128

def test_resnet_pretrain():
    pass