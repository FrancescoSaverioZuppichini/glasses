import torch
from glasses.nn.models.segmentation.unet import UNet
from glasses.nn.models.classification.senet import SENetBasicBlock
from glasses.nn.models.classification import ResNet
from glasses.nn.blocks import Conv2dPad

def test_UNet():
    x = torch.rand((1, 1, 32 * 12, 32*12))
    # custom encoder
    resnet = ResNet.resnet18()
    # we need to change the first conv in order to accept a gray image
    resnet.encoder.blocks[0].block[0].block.block.conv1 = Conv2dPad(1, 64, kernel_size=1)
    unet = UNet(1, n_classes=2, blocks_sizes=resnet.encoder.blocks_sizes)
    unet.encoder.blocks = resnet.encoder.blocks
    x = torch.rand((1, 1, 32 * 12, 32*12))
    unet(x)
    # custom block
    unet = UNet(down_block=SENetBasicBlock, up_block=SENetBasicBlock)
    unet(x)