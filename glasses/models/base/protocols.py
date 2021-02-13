import torch
from glasses.interpretability import Interpretability
from torch import nn


class Freezable:
    """
    A protocol that allows to freeze and unfreeze weights of the class that uses it

    :Example:

    >>> model = ResNet.resnet18()
    >>> Freezable.set_requires_grad(model.encoder)
    >>> class MyModel(nn.Sequential, Freezable):
    >>>    def __init__(self):
    >>>       super().__init__(nn.Conv2d(3, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU())
    >>> model = MyModel()
    >>> model.freeze()
    >>> model.unfreeze()
    >>> # freeze only one specific layer
    >>> model.freeze(model[0])
    """

    @staticmethod
    def set_requires_grad(module, to: bool = False):
        for param in module.parameters():
            param.requires_grad = to

    def freeze(self, who: nn.Module = None):
        who = self if who is None else who
        self.set_requires_grad(who, to=False)

    def unfreeze(self, who: nn.Module = None):
        who = self if who is None else who
        self.set_requires_grad(who, to=True)


class Interpretable:
    """Protocol that allows the clas that subclass it to interpret an input using and instance of `Interpretability`
    """

    def interpret(self, x : torch.Tensor, using: Interpretability(), *args, **kwargs):
        return using(x, self, *args, **kwargs)
