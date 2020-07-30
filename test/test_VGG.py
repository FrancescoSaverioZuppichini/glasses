import torch
from glasses.nn.models.classification.vgg import VGG


def test_vgg():
    x = torch.rand(1, 3, 224,224)
    model = VGG().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000
