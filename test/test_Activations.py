import torch
from glasses.nn.activation import Swish


def test_Swish():
    x = torch.ones(10)
    swish = Swish()
    out = swish(x)

    assert not torch.equal(x, out)

    swish = Swish(inplace=True)
    out = swish(x)

    assert torch.equal(x, out)