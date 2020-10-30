import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import nn
from .utils import convert_to_grayscale
from torch.nn import ReLU
from torch.autograd import Variable
from torchvision.transforms import *
from glasses.utils.Storage import ForwardModuleStorage, BackwardModuleStorage
from .utils import find_first_layer
from .Interpretability import Interpretability


class SaliencyMapResult:
    def __init__(self, saliency_map: torch.Tensor):
        self.saliency_map = saliency_map

    def show(self) -> plt.figure:
        fig = plt.figure()

        plt.imshow(self.saliency_map.squeeze())

        return fig


class SaliencyMap(Interpretability):
    """Implementation of `Deep Inside Convolutional Networks: Visualising Image Classification Models
    and Saliency Maps <https://arxiv.org/abs/1312.6034>`_
    """

    def guide(self, module):
        def guide_relu(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0.0),)

        for module in module.modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(guide_relu)

    def __call__(self, x: torch.Tensor, module: nn.Module, layer: nn.Module = None, target: int = None, guide: bool = True) -> SaliencyMapResult:

        layer = find_first_layer(
            x, module, nn.Conv2d) if layer is None else layer
        gradients_storage = BackwardModuleStorage([layer])

        if guide:
            self.guide(module)

        x = Variable(x, requires_grad=True)

        out = module(x)

        if target is None:
            target = torch.argmax(torch.softmax(out, dim=1))

        ctx = torch.zeros(out.size())
        ctx[0][int(target)] = 1

        module.zero_grad()

        out.backward(gradient=ctx)

        grads = gradients_storage[layer][0][0]

        saliency_map = grads.data.cpu().numpy()[0]

        saliency_map = convert_to_grayscale(saliency_map)
        saliency_map = torch.from_numpy(saliency_map)

        return SaliencyMapResult(saliency_map)
