import torch
from glasses.nn.att import (
    ChannelSE,
    ECA,
    SpatialChannelSE,
    SpatialSE,
    CBAM,
    LegacySpatialSE,
)


def test_att():
    x = torch.rand(1, 48, 8, 8)

    se = LegacySpatialSE(x.shape[1])
    res = se(x)

    assert res.shape == x.shape

    se = LegacySpatialSE(x.shape[1], reduced_features=10)

    assert se.att.fc1.out_features == 10

    x = torch.rand(1, 48, 8, 8)

    se = SpatialSE(x.shape[1])
    res = se(x)

    assert res.shape == x.shape

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
