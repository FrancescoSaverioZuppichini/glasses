import cv2
import numpy as np
import torch
from torch import nn
from glasses.utils.Tracker import Tracker
from typing import Type, Callable


def tensor2cam(image, cam):
    image_with_heatmap = image2cam(image.permute(1, 2, 0).cpu().numpy(),
                                   cam.detach().cpu().numpy())

    return torch.from_numpy(image_with_heatmap)


def image2cam(image, cam):
    h, w, c = image.shape

    cam -= np.min(cam)
    cam /= np.max(cam)  # Normalize between 0-1
    cam = cv2.resize(cam, (h, w))
    cam = np.uint8(cam * 255.0)

    img_with_cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    img_with_cam = cv2.cvtColor(img_with_cam, cv2.COLOR_BGR2RGB)
    img_with_cam = img_with_cam + (image * 255)
    img_with_cam /= np.max(img_with_cam)

    return img_with_cam


def find_last_layer(x: torch.Tensor, module: nn.Module, of_type: Type) -> nn.Module:
    """Utility function that return the last layer of a given type


    :Example:

    >>> x = torch.rand((1,3,224,224))
    >>> model = ResNet.resnet18()
    >>> find_last_layer(x, module, nn.Conv2d) 

    Args:
        x (torch.Tensor): [description]
        module (nn.Module): [description]
        of_type (Type): [description]

    Returns:
        nn.Module: [description]
    """
    tr = Tracker(module)
    tr(x)

    layer = None
    # iterate backward so we save time!
    for m in tr.traced[::-1]:
        if isinstance(m, of_type):
            layer = m
            break
    assert layer != None, f'layer of type {of_type} not found in {module.__name__}'

    return layer

def find_first_layer(x: torch.Tensor, module: nn.Module, of_type: Type) -> nn.Module:
    """Utility function that return the first layer of a given type


    :Example:

    >>> x = torch.rand((1,3,224,224))
    >>> model = ResNet.resnet18()
    >>> find_last_layer(x, module, nn.Conv2d) 

    Args:
        x (torch.Tensor): [description]
        module (nn.Module): [description]
        of_type (Type): [description]

    Returns:
        nn.Module: [description]
    """
    tr = Tracker(module)
    tr(x)

    layer = None
    for m in tr.traced:
        if isinstance(m, of_type):
            layer = m
            break
    assert layer != None, f'layer of type {of_type} not found in {module.__name__}'

    return layer


def convert_to_grayscale(cv2im):
    """
        Converts 3d image to grayscale
    Args:
        cv2im (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    credits to https://github.com/utkuozbulak/pytorch-cnn-visualizations
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im
