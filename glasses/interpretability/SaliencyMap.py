import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from glasses.utils.Storage import BackwardModuleStorage, ForwardModuleStorage
from torch import nn
from torch.autograd import Variable
from torch.nn import ReLU
from torchvision.transforms import *

from .Interpretability import Interpretability
from .utils import convert_to_grayscale, find_first_layer


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

    def __call__(self, x: torch.Tensor, module: nn.Module, layer: nn.Module = None, ctx: torch.Tensor = None, target: int = None, guide: bool = True) -> SaliencyMapResult:
        """Run SaliencyMap on the input given a model

        Args:
            x (torch.Tensor): Input tensor, e.g. an image
            module (nn.Module): Model
            layer (nn.Module, optional): The layer we wish to interpreter, if `None` then the last conv layer will be used. Defaults to None.
            target (int, optional): The target tensor, if `None` the model output (after softmax and argmax) wil be used. Defaults to None.
            ctx (torch.Tensor, optional): The tensor w.r we derive, if `None` we will use the one-hot encoding of the target. Defaults to None.

        Returns:
            SaliencyMapResult: The result of the saliency map, you can call `.show` to see it.
        """
        layer = find_first_layer(
            x, module, nn.Conv2d) if layer is None else layer
        gradients_storage = BackwardModuleStorage([layer])
        if guide:
            self.guide(module)

        x = Variable(x, requires_grad=True)

        out = module(x)

        if target is None:
            target = torch.argmax(torch.softmax(out, dim=1))

        if ctx is None:
            ctx = torch.zeros(out.size())
            ctx[0][int(target)] = 1

        out.backward(gradient=ctx)

        grads = gradients_storage[layer][0]

        saliency_map = grads.data.cpu().numpy()[0]
        saliency_map = convert_to_grayscale(saliency_map)
        saliency_map = torch.from_numpy(saliency_map)

        return SaliencyMapResult(saliency_map)
