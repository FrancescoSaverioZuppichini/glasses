import unittest
import torch
import torch.nn as nn
from glasses.nn.blocks.residuals import ResidualAdd, ResidualCat, Cat
from glasses.nn.blocks import Lambda

def test_add():
    x = torch.tensor(1)
    add_one = lambda x: x + 1
    adder = ResidualAdd(nn.Identity())
    # 1 + 1
    assert adder(x) == 2
    x = torch.tensor(1)
    adder = ResidualAdd(nn.Identity(), shortcut=add_one)
    # 1 + 1 + 1
    assert adder(x) == 3

def test_concat():
    x = torch.tensor([1])
    catter = ResidualCat(nn.Identity())
    assert catter(x).sum() == 2


def test_cat():
    x = torch.tensor([1])
    catter = Cat(nn.ModuleList([nn.Identity(), nn.Identity()]))
    print(catter(x))
    assert catter(x).sum() == 2