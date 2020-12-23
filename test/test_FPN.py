from glasses.models.segmentation.fpn import *
from glasses.models.segmentation.unet import *
from functools import partial
from glasses.models import *
from glasses.models.classification.resnet import ResNetBottleneckBlock
from glasses.models.classification.senet import SENetBasicBlock
import torch


def test_FPN():
    x = torch.rand((1, 1, 224, 224))
    model = FPN(activation=nn.SELU)
    # change number of classes (default is 2 )
    model = FPN(n_classes=3)
    out = model(x)
    assert len(out) == 4
    # change encoder
    fpn = FPN(encoder=lambda *args, **
              kwargs: ResNet.resnet26(*args, **kwargs).encoder,)
    fpn = FPN(encoder=lambda *args, **
              kwargs: EfficientNet.efficientnet_b0(*args, **kwargs).encoder,)
    # change decoder
    FPN(decoder=partial(FPNDecoder, pyramid_width=64, prediction_width=32))
    # pass a different block to decoder
    FPN(encoder=partial(ResNetEncoder, block=SENetBasicBlock))
    # all *Decoder class can be directly used
    fpn = FPN(encoder=partial(ResNetEncoder,
                              block=ResNetBottleneckBlock, depths=[2, 2, 2, 2]))


def test_PFPN():
    x = torch.rand((1, 1, 224, 224))
    # change activation
    model = PFPN(activation=nn.SELU)
    out = model(x)
    assert out.shape[1] == 2
    assert out.shape[2] == out.shape[3] == 224
    # change number of classes (default is 2 )
    model = PFPN(n_classes=3)
    out = model(x)
    assert out.shape[1] == 3
    # change encoder
    model = PFPN(encoder=lambda *args, **
                 kwargs: ResNet.resnet26(*args, **kwargs).encoder,)
    out = model(x)
    assert out.shape[1] == 2
    assert out.shape[2] == out.shape[3] == 224

    model = PFPN(encoder=lambda *args, **kwargs: EfficientNet.efficientnet_b0(*args, **kwargs).encoder,)
    out = model(x)
    assert out.shape[1] == 2
    assert out.shape[2] == out.shape[3] == 224
    # change decoder
    model = PFPN(decoder=partial(
        PFPNDecoder, pyramid_width=64, prediction_width=32))
    out = model(x)
    assert out.shape[1] == 2
    assert out.shape[2] == out.shape[3] == 224
    # pass a different block to decoder
    model = PFPN(encoder=partial(ResNetEncoder, block=SENetBasicBlock))
    out = model(x)
    assert out.shape[1] == 2
    assert out.shape[2] == out.shape[3] == 224
    # all *Decoder class can be directly used
    model = PFPN(encoder=partial(
        ResNetEncoder, block=ResNetBottleneckBlock, depths=[2, 2, 2, 2]))
    out = model(x)
    assert out.shape[1] == 2
    assert out.shape[2] == out.shape[3] == 224
