import torch
import torch.nn as nn
from glasses.interpretability import GradCam
from glasses.models.base import Freezable, Interpretable


def test_Freezable():
    class MyModel(nn.Sequential, Freezable, Interpretable):
        def __init__(self):
            super().__init__(nn.Conv2d(3, 32, kernel_size=3, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                             nn.AdaptiveAvgPool2d((1, 1)),
                             nn.Flatten(), nn.Linear(32, 10))

    model = MyModel()

    model.freeze()

    for param in model.parameters():
        assert not param.requires_grad

    model.unfreeze()

    for param in model.parameters():
        assert param.requires_grad

    model.freeze(model[0])

    for param in model[0].parameters():
        assert not param.requires_grad

    for param in model[1].parameters():
        assert param.requires_grad

    x = torch.randn((1, 3, 224, 224))
    model.interpret(x, using=GradCam())
