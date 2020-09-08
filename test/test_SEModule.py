import torch
import torch.nn as nn
from glasses.nn.models.classification.se import SCSEModule, CSEModule, SSEModule

def test_tracker():
    x = torch.rand(1,48,8,8)

    se = SSEModule(x.shape[1])
    res = se(x)

    assert res.shape == x.shape

    se = SSEModule(x.shape[1], reduced_features=10)

    assert se.att.fc1.out_features == 10

    se = CSEModule(x.shape[1])
    res = se(x)

    assert res.shape == x.shape

    se = SCSEModule(x.shape[1])
    res = se(x)

    assert res.shape == x.shape
 
    
