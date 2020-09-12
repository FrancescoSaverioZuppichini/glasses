import torch
from glasses.nn.models.segmentation.unet import UNet, UNetEncoder
from glasses.nn.models.classification.senet import SENetBasicBlock
from glasses.nn.models.classification import ResNet
from glasses.nn.blocks import Conv2dPad
from functools import partial

def test_UNet():
    x = torch.rand((1, 1, 32 * 12, 32*12))
    # custom encoder
    resnet = ResNet.resnet18(in_channels=1)
    unet = UNet(1, n_classes=2, widths=resnet.encoder.widths)
    unet.encoder = resnet.encoder
    unet(x)

    unet = UNet(encoder=partial(UNetEncoder, block=SENetBasicBlock))
    unet(x)