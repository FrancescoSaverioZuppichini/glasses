from functools import partial

import torch
from glasses.models.classification.resnet import *
from glasses.models.classification.resnetxt import ResNetXt
from glasses.models.classification.wide_resnet import (
    WideResNet, WideResNetBottleNeckBlock)


def test_resnet():
    with torch.no_grad():

        x = torch.rand(1, 3, 224, 224)
        model = ResNet.resnet18().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
        # model.summary(device=torch.device('cpu'))
        model = ResNet.resnet18(stem=ResNetStemC)
        pred = model(x)
        assert pred.shape[-1] == 1000
        model = ResNet.resnet18(block=partial(ResNetBasicBlock, shortcut=ResNetShorcutD))
        pred = model(x)
        assert pred.shape[-1] == 1000
        pred = model(x)
        assert pred.shape[-1] == 1000
        
        model = ResNet.resnet26().eval()
        model.encoder.features
        pred = model(x)
        features = model.encoder.features
        f_widths = [f.shape[1] for f in features]
        for w, f_w in zip(model.encoder.features_widths, f_widths):
            assert w == f_w
        assert pred.shape[-1] == 1000

        model = ResNet.resnet26d().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000

        model = ResNet.resnet34().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000

        model = ResNet.resnet50().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000

        model = ResNet.resnet50d().eval()
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

    model = ResNetXt.resnext101_32x16d().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000
    # too big
    # model = ResNetXt.resnext101_32x32d().eval()
    # pred = model(x)
    # assert pred.shape[-1] == 1000

    # too big
    # model = ResNetXt.resnext101_32x48d().eval()
    # pred = model(x)
    # assert pred.shape[-1] == 1000

def test_wide_resnet():
    x = torch.rand(1, 3, 224, 224)
    model = WideResNet.wide_resnet50_2().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = WideResNet.wide_resnet101_2().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000


    block = WideResNetBottleNeckBlock(32, 256, width_factor=2)

    assert block.block[1].conv.in_channels ==  128

def test_resnet_pretrain():
    pass
