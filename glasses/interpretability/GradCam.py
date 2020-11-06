import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from torch.nn import ReLU
from torch.autograd import Variable
from torch.nn import AvgPool2d, Conv2d, Linear, ReLU, MaxPool2d, BatchNorm2d
from .Interpretability import Interpretability
from typing import Type, Callable
from glasses.utils.Storage import ForwardModuleStorage, BackwardModuleStorage
from .utils import tensor2cam, image2cam, find_last_layer


class GradCamResult:
    def __init__(self, img: torch.Tensor,
                 cam: torch.Tensor,
                 postpreocessing: Callable[[torch.Tensor], torch.Tensor]):
        self.img = img
        self.cam = cam
        self.postpreocessing = postpreocessing

    def show(self) -> plt.figure:
        img = self.img
        if self.postpreocessing is not None:
            img = self.postpreocessing(self.img)
        cam_on_img = tensor2cam(img.squeeze(0), self.cam)

        fig = plt.figure()

        plt.imshow(cam_on_img)

        return fig


class GradCam(Interpretability):
    """Implementation of GradCam proposed in `Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization <https://arxiv.org/abs/1610.02391>`_
    """
    def __call__(self, x: torch.Tensor, module: nn.Module, layer: nn.Module = None, target: int = None,
                 postprocessing: Callable[[torch.Tensor], torch.Tensor] = None) -> GradCamResult:
        layer = find_last_layer(
            x, module, nn.Conv2d) if layer is None else layer
        # register forward and backward storages
        weights_storage = ForwardModuleStorage(module, [layer])
        gradients_storage = BackwardModuleStorage([layer])

        x = Variable(x, requires_grad=True)

        out = module(x)

        if target is None:
            target = torch.argmax(torch.softmax(out, dim=1))

        ctx = torch.zeros(out.size())
        ctx[0][int(target)] = 1

        out.backward(gradient=ctx, retain_graph=True)
        # get back the weights and the gradients
        weights = weights_storage[layer][0]
        grads = gradients_storage[layer][0][0]
        # compute grad cam
        avg_channel_grad = F.adaptive_avg_pool2d(grads.data, 1)
        cam = F.relu(torch.sum(weights * avg_channel_grad, dim=1)).squeeze(0)
        return GradCamResult(x.detach(), cam, postprocessing)
