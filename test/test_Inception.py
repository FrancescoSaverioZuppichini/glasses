import torch
from glasses.nn.models.classification.inception import Inception


def test_Inception():
    x = torch.rand(1, 3, 224, 224)
    model = Inception().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000
