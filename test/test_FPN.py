from glasses.models.segmentation.fpn import *
from glasses.models.segmentation.unet import *
from functools import partial
from glasses.models import *
from glasses.models.classification.resnet import ResNetBottleneckBlock
from glasses.models.classification.senet import SENetBasicBlock

def test_FPN():
    FPN(activation=nn.SELU)
    # change number of classes (default is 2 )
    FPN(n_classes=2)
    # change encoder
    fpn = FPN(encoder=lambda *args, **kwargs: ResNet.resnet26(*args, **kwargs).encoder,)
    fpn = FPN(encoder=lambda *args, **kwargs: EfficientNet.efficientnet_b0(*args, **kwargs).encoder,)
    # change decoder
    FPN(decoder=partial(FPNDecoder, pyramid_width=64, prediction_width=32))
    # pass a different block to decoder
    FPN(encoder=partial(ResNetEncoder, block=SENetBasicBlock))
    # all *Decoder class can be directly used
    fpn = FPN(encoder=partial(ResNetEncoder, block=ResNetBottleneckBlock, depths=[2,2,2,2]))

def test_PFPN():
    # change activation
    PFPN(activation=nn.SELU)
    # change number of classes (default is 2 )
    PFPN(n_classes=2)
    # change encoder
    pfpn = PFPN(encoder=lambda *args, **kwargs: ResNet.resnet26(*args, **kwargs).encoder,)
    pfpn = PFPN(encoder=lambda *args, **kwargs: EfficientNet.efficientnet_b0(*args, **kwargs).encoder,)
    # change decoder
    PFPN(decoder=partial(PFPNDecoder, pyramid_width=64, prediction_width=32))
    # pass a different block to decoder
    PFPN(encoder=partial(ResNetEncoder, block=SENetBasicBlock))
    # all *Decoder class can be directly used
    pfpn = PFPN(encoder=partial(ResNetEncoder, block=ResNetBottleneckBlock, depths=[2,2,2,2]))