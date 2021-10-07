import torch
from glasses.nn.att import ChannelSE, ECA, SpatialChannelSE, SpatialSE, CBAM, SelectiveKernel


def test_att():
    x = torch.rand(2, 48, 8, 8)

    se = SpatialSE(x.shape[1])
    res = se(x)

    assert res.shape == x.shape

    se = SpatialSE(x.shape[1], reduced_features=10)

    assert se.att.fc1.out_features == 10

    se = ChannelSE(x.shape[1])
    res = se(x)

    assert res.shape == x.shape

    se = SpatialChannelSE(x.shape[1])
    res = se(x)

    assert res.shape == x.shape

    eca = ECA(x.shape[1])

    res = eca(x)
    assert res.shape == x.shape

    cbam = CBAM(x.shape[1])

    res = cbam(x)
    assert res.shape == x.shape

    sk = SelectiveKernel(x.shape[1], kernel_size=[3, 5])
    
    res = sk(x)
    assert res.shape == x.shape
    