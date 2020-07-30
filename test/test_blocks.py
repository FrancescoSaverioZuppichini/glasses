import torch
from torch import Tensor
from torch import nn
from glasses.nn.blocks import Conv2dPad, ConvAct, Lambda

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
