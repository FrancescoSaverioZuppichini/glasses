import torch
import torch.nn as nn
from glasses.utils.Tracker import Tracker

def test_tracker():
    x = torch.rand(64, 1)
    model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64,10), nn.ReLU())
    tr = Tracker(model)
    tr(x)

    assert len(tr.traced) == 4
    assert len(tr.parametrized) == 2
    
