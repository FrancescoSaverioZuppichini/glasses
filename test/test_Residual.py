import torch
import torch.nn as nn
from glasses.nn.blocks.residuals import ResidualAdd
from glasses.nn.blocks import Lambda


def test_add():
    x = torch.tensor(1)
    adder = ResidualAdd(nn.Identity())
    # 1 + 1
    assert adder(x) == 2
