from glasses.nn.regularization import DropBlock
import torch
from torch import Tensor
from torch import nn
from glasses.nn.blocks import Conv2dPad, ConvBnAct, ConvNormRegAct, Lambda
import pytest
from torchvision.ops import StochasticDepth


def test_Conv2dPad():
    x = torch.rand((1, 1, 5, 5))
    block = Conv2dPad(1, 5, kernel_size=3)
    res = block(x)
    assert x.shape[-1] == res.shape[-1]
    assert x.shape[-2] == res.shape[-2]
    # no padding
    block = Conv2dPad(1, 5, kernel_size=3, mode=None)
    res = block(x)
    assert x.shape[-1] != res.shape[-1]
    assert x.shape[-2] != res.shape[-2]
    assert res.shape[-1] == 3
    assert res.shape[-2] == 3


def test_Lambda():
    add_two = Lambda(lambda x: x + 2)
    x = add_two(Tensor([0]))
    assert x == 2


def test_ConvBnAct():
    conv = ConvBnAct(32, 64, kernel_size=3)
    assert conv.conv != None
    assert conv.norm != None
    assert conv.act != None

    assert type(conv.conv) is Conv2dPad
    assert type(conv.norm) is nn.BatchNorm2d
    assert type(conv.act) is nn.ReLU

    conv = ConvBnAct(32, 64, kernel_size=3, activation=None)
    assert type(conv.conv) is Conv2dPad
    assert type(conv.norm) is nn.BatchNorm2d
    with pytest.raises(AttributeError):
        conv.act

    conv = ConvBnAct(32, 64, kernel_size=3, normalization=None)
    assert type(conv.conv) is Conv2dPad
    assert type(conv.act) is nn.ReLU
    with pytest.raises(AttributeError):
        conv.norm

    conv = ConvBnAct(
        32,
        64,
        kernel_size=3,
        conv=nn.Conv2d,
        activation=nn.SELU,
        normalization=nn.Identity,
    )
    assert type(conv.conv) is nn.Conv2d
    assert type(conv.act) is nn.SELU
    assert type(conv.norm) is nn.Identity


def test_ConvNormRegAct():
    conv = ConvNormRegAct(32, 64, kernel_size=3)
    assert conv.conv != None
    assert conv.norm != None
    assert conv.reg != None
    assert conv.act != None

    assert type(conv.conv) is Conv2dPad
    assert type(conv.norm) is nn.BatchNorm2d
    assert type(conv.reg) is StochasticDepth
    assert type(conv.act) is nn.ReLU

    conv = ConvNormRegAct(32, 64, kernel_size=3, activation=None)
    assert type(conv.conv) is Conv2dPad
    assert type(conv.norm) is nn.BatchNorm2d
    with pytest.raises(AttributeError):
        conv.act

    conv = ConvNormRegAct(32, 64, kernel_size=3, normalization=None)
    assert type(conv.conv) is Conv2dPad
    assert type(conv.act) is nn.ReLU
    with pytest.raises(AttributeError):
        conv.norm

    conv = ConvNormRegAct(32, 64, kernel_size=3, regularization=None)
    assert type(conv.conv) is Conv2dPad
    assert type(conv.act) is nn.ReLU
    assert type(conv.norm) is nn.BatchNorm2d

    with pytest.raises(AttributeError):
        conv.reg

    conv = ConvNormRegAct(
        32,
        64,
        kernel_size=3,
        conv=nn.Conv2d,
        activation=nn.SELU,
        normalization=nn.Identity,
    )
    assert type(conv.conv) is nn.Conv2d
    assert type(conv.act) is nn.SELU
    assert type(conv.norm) is nn.Identity

    conv = ConvNormRegAct(32, 64, kernel_size=3, p=0.9)
    assert conv.reg.p == 0.9
