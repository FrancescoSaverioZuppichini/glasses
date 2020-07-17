import torch
from eyes.utils import Conv2dPad

def test_conv2dpad():
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