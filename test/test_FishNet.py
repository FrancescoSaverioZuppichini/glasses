import torch
import torch.nn as nn
from glasses.models.classification.fishnet import FishNet, FishNetBottleNeck
from glasses.nn.att import SpatialSE
from torchsummary import summary


def test_fishnet():
    device = torch.device('cpu')
    x = torch.rand(1, 3,224,224)

    model = FishNet().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000
    # test fishnet99
    model = FishNet.fishnet99().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    # n_params, _ = summary(model.to(device), (3, 224, 224), device=device)
    # # we know the correct number of paramters of fishnet
    # assert n_params.item() == 16628904

    # test fishnet150
    model = FishNet.fishnet150().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    # n_params, _ = summary(model.to(device), (3, 224, 224), device=device)
    # # we know the correct number of paramters of fishnet
    # assert n_params.item() == 24959400

    block = lambda in_ch, out_ch, **kwargs: nn.Sequential(FishNetBottleNeck(in_ch, out_ch), SpatialSE(out_ch))
    model = FishNet.fishnet99(block=block)  
    pred = model(x)
    assert pred.shape[-1] == 1000
    # summary(model.to(device), (3, 224, 224))
