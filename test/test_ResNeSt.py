from functools import partial

import torch
from glasses.models.classification.resnest import *


def test_resnest():
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        model = ResNeSt.resnest14d().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
        n_classes = 10
        model = ResNeSt.resnest14d(n_classes=n_classes).eval()
        pred = model(x)
        assert pred.shape[-1] == n_classes
        model = ResNeSt.resnest26d().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
        model = ResNeSt.resnest50d().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
        model = ResNeSt.resnest50d_1s4x24d().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
        model = ResNeSt.resnest50d_4s2x40d().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
        model = ResNeSt.resnest101e().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
        model = ResNeSt.resnest50d_fast().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
        # too big!
        # model = ResNetSt.resnest200e().eval()
        # pred = model(x)
        # assert pred.shape[-1] == 1000
        # model = ResNetSt.resnest269e().eval()
        # pred = model(x)
        # assert pred.shape[-1] == 1000
 

