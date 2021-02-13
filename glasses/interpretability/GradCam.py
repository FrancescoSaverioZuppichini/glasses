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

    def __call__(self, x: torch.Tensor, module: nn.Module, layer: nn.Module = None, target: int = None, ctx: torch.Tensor = None, postprocessing: Callable[[torch.Tensor], torch.Tensor] = None) -> GradCamResult:
        """Run GradCam on the input given a model

        Args:
            x (torch.Tensor): Input tensor, e.g. an image
            module (nn.Module): Model
            layer (nn.Module, optional): The layer we wish to interpreter, if `None` then the last conv layer will be used. Defaults to None.
            target (int, optional): The target tensor, if `None` the model output (after softmax and argmax) wil be used. Defaults to None.
            ctx (torch.Tensor, optional): The tensor w.r we derive, if `None` we will use the one-hot encoding of the target. Defaults to None.
            postprocessing (Callable[[torch.Tensor], torch.Tensor], optional): A function used to post process the output, e.g. de-normalize. Defaults to None.

        Returns:
            GradCamResult: The result of the gradcam, you can call `.show` to see it.
        """
        layer = find_last_layer(
            x, module, nn.Conv2d) if layer is None else layer
        # register forward and backward storages
        weights_storage = ForwardModuleStorage(module, [layer])
        gradients_storage = BackwardModuleStorage([layer])

        x = Variable(x, requires_grad=True)

        out = module(x)

        if target is None:
            target = torch.argmax(torch.softmax(out, dim=1))

        if ctx is None:
            ctx = torch.zeros(out.size())
            ctx[0][int(target)] = 1
        
        out.backward(gradient=ctx)
        # get back the weights and the gradients
        weights = weights_storage[layer]
        grads = gradients_storage[layer][0]
        # compute grad cam
        avg_channel_grad = F.adaptive_avg_pool2d(grads.data, 1)
        cam = F.relu(torch.sum(weights * avg_channel_grad, dim=1)).squeeze(0)
        return GradCamResult(x.detach(), cam, postprocessing)
