from glasses.nn.models.classification.efficientnet import EfficientNet
import torch
import torch.nn as nn

def test_EfficientNet():
    model = EfficientNet.efficientnet_b0()
    model.encoder.gate.conv = nn.Conv2d(3, 32, kernel_size=7)
    # store each feature
    x = torch.rand((1, 3, 224, 224))
    model = EfficientNet.efficientnet_b0()
    features = []
    x = model.encoder.gate(x)
    for block in model.encoder.blocks:
        x = block(x)
        features.append(x)

    assert len(features) == len(model.default_depths) + 1 #count for the last one also

def test_EfficientNetb1():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b1()
    pred = model(x)
    assert pred.shape[-1] == 1000

def test_EfficientNetb2():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b2()
    pred = model(x)
    assert pred.shape[-1] == 1000

    
def test_EfficientNetb3():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b3()
    pred = model(x)
    assert pred.shape[-1] == 1000

def test_EfficientNetb3():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b3()
    pred = model(x)
    assert pred.shape[-1] == 1000

def test_EfficientNetb4():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b4()
    pred = model(x)
    assert pred.shape[-1] == 1000

def test_EfficientNetb5():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b5()
    pred = model(x)
    assert pred.shape[-1] == 1000

def test_EfficientNetb6():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b6()
    pred = model(x)
    assert pred.shape[-1] == 1000

def test_EfficientNetb7():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b7()
    pred = model(x)
    assert pred.shape[-1] == 1000

