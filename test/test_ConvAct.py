import torch
from glasses.nn.blocks import ConvAct

def test_convact():
    x = torch.rand((1, 1, 5, 5))
    block = ConvAct(1, 5, kernel_size=3)
    pred = block(x)
    assert x.shape[-1] == pred.shape[-1]
    assert x.shape[-2] == pred.shape[-2]
    # no padding
    block = ConvAct(1, 5, kernel_size=3, mode=None)
    pred = block(x)
    assert x.shape[-1] != pred.shape[-1]
    assert x.shape[-2] != pred.shape[-2]
    assert pred.shape[-1] == 3