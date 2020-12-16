import torch
from glasses.nn.models.classification.regnet import *
from glasses.nn.models.classification.resnet import ResNetStemC, ResNetShorcutD

from functools import partial


def test_regnet():
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        model = RegNet.regnetx_002()
        pred = model(x)
        assert pred.shape[-1] == 1000
        n_classes = 10
        model = RegNet.regnetx_002(n_classes=n_classes).eval()
        pred = model(x)
        assert pred.shape[-1] == n_classes
        model = RegNet.regnetx_002(block=RegNetYBotteneckBlock)
        pred = model(x)
        assert pred.shape[-1] == 1000
        # change the steam
        model = RegNet.regnetx_002(stem=ResNetStemC)
        pred = model(x)
        assert pred.shape[-1] == 1000
        model = RegNet.regnetx_002(block=partial(RegNetYBotteneckBlock, shortcut=ResNetShorcutD))
        pred = model(x)
        assert pred.shape[-1] == 1000
        model = RegNet.regnetx_004().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
        model = RegNet.regnetx_006().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
        model = RegNet.regnetx_008().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
        model = RegNet.regnety_002().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
        model = RegNet.regnety_004().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
        model = RegNet.regnety_006().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000