import torch
import torch.nn as nn
from glasses.models.classification import EfficientNet, EfficientNetLite


def test_EfficientNet():
    x = torch.rand((1, 3, 224, 224))
    model = EfficientNet.efficientnet_b0().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000
    

def test_EfficientNetb1():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b1().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

def test_EfficientNetb2():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b2().eval()
    model.encoder.features
    pred = model(x)
    # test also features
    features = model.encoder.features
    f_widths = [f.shape[1] for f in features]
    for w, f_w in zip(model.encoder.features_widths, f_widths):
        assert w == f_w

    assert pred.shape[-1] == 1000

def test_EfficientNetb3():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b3().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

def test_EfficientNetb4():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b4().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

def test_EfficientNetb5():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b5().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

def test_EfficientNetb6():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b6().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

def test_EfficientNetb7():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b7().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000


def test_EfficientNetb8():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNet.efficientnet_b8().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

# too big
# def test_EfficientNetl2():
#     x = torch.rand(1, 3, 224, 224)
#     model = EfficientNet.efficientnet_l2()
#     pred = model(x)
#     assert pred.shape[-1] == 1000


def test_EfficientNetLite0():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNetLite.efficientnet_lite0()
    pred = model(x)
    assert pred.shape[-1] == 1000



def test_EfficientNetLite1():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNetLite.efficientnet_lite1()
    pred = model(x)
    assert pred.shape[-1] == 1000


def test_EfficientNetLite2():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNetLite.efficientnet_lite2()
    pred = model(x)
    assert pred.shape[-1] == 1000


def test_EfficientNetLite3():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNetLite.efficientnet_lite3()
    pred = model(x)
    assert pred.shape[-1] == 1000

def test_EfficientNetLite4():
    x = torch.rand(1, 3, 224, 224)
    model = EfficientNetLite.efficientnet_lite4()
    pred = model(x)
    assert pred.shape[-1] == 1000
