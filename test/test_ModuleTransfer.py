import torch
import torch.nn as nn
from glasses.utils.ModuleTransfer import ModuleTransfer
from glasses.utils.Tracker import Tracker
import pytest

def test_ModuleTransfer():
    model_a = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64,10), nn.ReLU())
    def block(in_features, out_features):
        return nn.Sequential(nn.Linear(in_features, out_features),
                        nn.ReLU())
    model_b = nn.Sequential(block(1,64), block(64,10))
    # model_a and model_b are the same thing but defined in two different ways
    x = torch.ones(1, 1)
    trans = ModuleTransfer(src=model_a, dest=model_b, verbose=1)
    trans(x)
    
    src_traced = Tracker(model_a)(x).parametrized
    dest_traced = Tracker(model_b)(x).parametrized

    for dest_m, src_m in zip(dest_traced, src_traced):
        for key in src_m.state_dict().keys():
            for a, b in zip(src_m.state_dict()[key], dest_m.state_dict()[key]):
                assert torch.equal(a, b)

    # let's do it again with a wrong src
    model_a = nn.Sequential(nn.Linear(1, 64))
    trans = ModuleTransfer(src=model_a, dest=model_b, verbose=1)
    with pytest.raises(Exception):
        trans(x)
