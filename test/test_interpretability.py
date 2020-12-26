import torch
from torch import nn
from glasses.interpretability import GradCam, SaliencyMap
from glasses.interpretability.utils import *
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, ToPILImage, ToTensor, Compose

def test_gradcam():
    x = torch.rand((1, 3, 224, 224))

    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3),
        nn.Conv2d(32, 32, kernel_size=3),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(32, 10)
    )

    cam = GradCam()

    cam_res = cam(x, model)


    assert type(cam_res.cam) == torch.Tensor
    assert cam_res.show()


    cam_res = cam(x, model, layer=model[0])
    assert type(cam_res.cam) == torch.Tensor

    assert cam_res.show()

    cam_res = cam(x, model, postprocessing=Compose([lambda x: x.squeeze(0), ToPILImage(), Resize(224), ToTensor()]))

    assert type(cam_res.cam) == torch.Tensor
    assert cam_res.show()

    cam = GradCam()
    target = 1
    ctx = torch.zeros(10).unsqueeze(0)
    ctx[0][int(target)] = 1
    cam_res = cam(x, model, target=target, ctx=ctx)

    assert type(cam_res.cam) == torch.Tensor
    assert cam_res.show()


def test_saliency_map():
    x = torch.rand((1, 3, 224, 224))

    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3),
        nn.Conv2d(32, 32, kernel_size=3),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(32, 10)
    )

    saliency = SaliencyMap()

    saliency_res = saliency(x, model)

    assert saliency_res.show()
    assert len(saliency_res.saliency_map.squeeze(0).shape) == 2
    assert type(saliency_res.saliency_map) == torch.Tensor

    saliency_res = saliency(x, model, layer=model[0])

    assert len(saliency_res.saliency_map.squeeze(0).shape) == 2
    assert type(saliency_res.saliency_map) == torch.Tensor


    saliency_res = saliency(x, model, guide=False)

    assert saliency_res.show()
    assert len(saliency_res.saliency_map.squeeze(0).shape) == 2
    assert type(saliency_res.saliency_map) == torch.Tensor

    saliency = SaliencyMap()
    target = 1
    ctx = torch.zeros(10).unsqueeze(0)
    ctx[0][int(target)] = 1
    saliency_res = saliency(x, model, target=target, ctx=ctx)
    
    assert saliency_res.show()
    assert len(saliency_res.saliency_map.squeeze(0).shape) == 2
    assert type(saliency_res.saliency_map) == torch.Tensor

def test_find_last_layer():
    x = torch.rand((1, 3, 224, 224))

    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3),
        nn.Conv2d(32, 32, kernel_size=3)
    )

    last  = find_last_layer(x, model, nn.Conv2d)

    assert last is model[1]


def test_find_first_layer():
    x = torch.rand((1, 3, 224, 224))

    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3),
        nn.Conv2d(32, 32, kernel_size=3)
    )

    last  = find_first_layer(x, model, nn.Conv2d)

    assert last is model[0]

    
        