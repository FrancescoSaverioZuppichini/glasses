import torch
import torch.nn as nn
from glasses.nn.models.classification.se import SEModule

def test_tracker():
    x = torch.rand(1,48,8,8)
    se = SEModule(x.shape[1])
    res = se(x)

    assert res.shape == x.shape
    
