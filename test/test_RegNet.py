from functools import partial

import torch
from glasses.models.classification.regnet import *
from glasses.models.classification.resnet import ResNetShorcutD, ResNetStemC


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
        model = RegNet.regnetx_016().eval()
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
        model = RegNet.regnety_008().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
        model = RegNet.regnety_016().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000

def test_regnet_scaler():
      depths, widths, groups_width = RegNetScaler()(w_0 = 24, w_a = 24.48, w_m = 2.54, group_w = 16, depth = 22 )
      assert depths == [1, 2, 7, 12]
      assert widths == [32, 64, 160, 384]
      assert groups_width == 16
