import torch
from glasses.nn.models.segmentation.unet import UNet, UNetEncoder, UNetDecoder
from glasses.nn.models.classification.senet import SENetBasicBlock
from glasses.nn.models.classification.resnet import ResNet, ResNetBasicBlock, ResNetEncoder, ResNetBottleneckBlock
from glasses.nn.blocks import Conv2dPad
from functools import partial

def test_UNet():
    x = torch.rand((1, 1, 32 * 12, 32*12))
    # custom encoder
    unet = UNet(encoder=partial(ResNetEncoder, block=ResNetBasicBlock, depths=[1,1,2,2]))
    unet(x)
    unet = UNet(encoder=partial(ResNetEncoder, block=ResNetBottleneckBlock, depths=[1,1,2,2]))
    unet(x)
    # custom block
    unet = UNet(encoder=partial(UNetEncoder, block=SENetBasicBlock))
    unet(x)