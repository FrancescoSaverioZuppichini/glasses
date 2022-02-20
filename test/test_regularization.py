import torch
from glasses.nn.regularization import DropBlock, DropPath


def test_drop_block():
    drop = DropBlock()
    x = torch.ones((1, 3, 28, 28))
    x_drop = drop(x)

    assert not torch.equal(x, x_drop)
    assert drop.training

    drop = drop.eval()
    x_drop = drop(x)
    assert torch.equal(x, x_drop)
    assert not drop.training

    assert drop.__repr__() == "DropBlock(p=0.5)"


def test_drop_path():
    stoc = DropPath()
    assert stoc.__repr__() == "DropPath(p=0.5)"

    x = torch.ones((2, 3, 28, 28))
    stoc = DropPath(p=1)
    out = stoc(x)

    assert out.sum() > 0

    stoc = DropPath(p=10e-6)
    out = stoc(x)
    assert out.sum() == 0
