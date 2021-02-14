import torch
from glasses.nn.regularization import DropBlock

def test_drop_block():
    drop = DropBlock()
    x = torch.ones((1,3,28,28))
    x_drop = drop(x)

    assert not torch.equal(x, x_drop)
    assert drop.training

    drop = drop.eval()
    x_drop = drop(x)
    assert torch.equal(x, x_drop)
    assert not drop.training

    assert drop.__repr__() == 'DropBlock(p=0.5)'