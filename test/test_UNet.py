import torch
from glasses.nn.models.segmentation.unet import UNet, UNetEncoder, UNetDecoder
from glasses.nn.models.classification.senet import SENetBasicBlock
from glasses.nn.models.classification.resnet import ResNet, ResNetBasicBlock, ResNetEncoder, ResNetBottleneckBlock
from glasses.nn.models.classification import EfficientNet
from glasses.nn.models.classification import EfficientNetLite
from glasses.nn.models import AutoModel
from glasses.nn.blocks import Conv2dPad
from functools import partial
import pytest

def test_UNet():
    x = torch.rand((1, 1, 32 * 12, 32*12))
    # unet
    unet = UNet()
    unet(x)
    # custom encoder
    unet = UNet(encoder=lambda *args, **
                kwargs: ResNet.resnet26(*args, **kwargs).encoder,)
    unet(x)
    # change decoder
    unet = UNet(decoder=partial(
        UNetDecoder, widths=[256, 128, 64, 32, 16]))
    unet(x)
    # using efficienet net
    unet = UNet(encoder=lambda *args, **
                kwargs: EfficientNet.efficientnet_b2(*args, **kwargs).encoder)
    unet(x)
    # combine them
    unet = UNet(encoder=lambda *args, **kwargs: EfficientNet.efficientnet_b2(*args, **kwargs).encoder,
                decoder=partial(UNetDecoder, widths=[256, 128, 64, 32, 16]))
    unet(x)
    unet = UNet(encoder=lambda *args, **kwargs: EfficientNetLite.efficientnet_lite3(*args, **kwargs).encoder,)
    unet(x)
    # customize the encoder
    unet = UNet(encoder=partial(ResNetEncoder,
                                block=ResNetBasicBlock, depths=[1, 1, 2, 2]))
    unet(x)
    unet = UNet(encoder=partial(ResNetEncoder,
                                block=ResNetBottleneckBlock, depths=[1, 1, 2, 2]))
    unet(x)
    # custom block
    unet = UNet(encoder=partial(UNetEncoder, block=SENetBasicBlock))
    unet(x)

    # using .from_encoder
    unet = UNet.from_encoder(lambda *args, **kwargs: ResNet.resnet26(*args, **kwargs))
    unet(x)
    # with AutoModel
    unet = UNet.from_encoder(partial(AutoModel.from_name, 'resnet18'))
    unet(x)
    with pytest.raises(AttributeError):
        unet = UNet.from_encoder(lambda *args, **kwargs: None)
