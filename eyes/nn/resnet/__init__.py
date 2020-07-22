from torch import nn
from torch import Tensor
from ..blocks.residuals import ResidualAdd
from collections import OrderedDict
from typing import List


class ResNetShorcut(nn.Module):
    """Shorcut function applied by ResNet to upsample the channel
    when residual and output features do not match

    Args:
        in_features (int): features (channels) of the input
        out_features (int): features (channels) of the desidered output
    """

    def __init__(self, in_features: int, out_features: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features,
                              kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_features: int, out_features: int, downsampling: int = 1, activation: nn.Module = nn.ReLU):
        """Basic ResNet block composed by two 3x3 convs.

        Args:
            in_features (int): features (channels) of the input
            out_features (int): features (channels) of the desidered output
            activation (nn.Module, optional): Activation applied between the weights. Defaults to nn.ReLU(inplace=True).
        """
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.block = ResidualAdd(
            nn.Sequential(
                OrderedDict(
                    {
                        'conv1': nn.Conv2d(in_features, out_features, kernel_size=3, stride=downsampling, padding=1, bias=False),
                        'bn1': nn.BatchNorm2d(out_features),
                        'act1': activation(),
                        'conv2': nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, bias=False),
                        'bn2': nn.BatchNorm2d(out_features),
                    }
                )),
            shortcut=ResNetShorcut(in_features, out_features, downsampling) if self.should_apply_shortcut else None)

        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        # activation is applied after the residual
        x = self.act(x)
        return x

    @property
    def expanded_channels(self):
        return self.out_features * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_features != self.expanded_channels


class ResNetBottleNeckBlock(ResNetBasicBlock):
    expansion: int = 4

    def __init__(self, in_features: int, out_features: int, downsampling: int = 1, expansion: int = 4, activation: nn.Module = nn.ReLU):
        """Basic ResNet block composed by two 3x3 convs.

        Args:
            in_features (int): features (channels) of the input
            out_features (int): features (channels) of the desidered output
            expansion (int): expansion factor of the output features (channels)
            activation (nn.Module, optional): Activation applied between the weights. Defaults to nn.ReLU(inplace=True).
        """
        super().__init__(in_features, out_features, activation)
        self.block = ResidualAdd(
            nn.Sequential(
                OrderedDict(
                    {
                        'conv1': nn.Conv2d(in_features, out_features, kernel_size=1, bias=False),
                        'bn1': nn.BatchNorm2d(out_features),
                        'act1': activation(),
                        'conv2': nn.Conv2d(out_features, out_features, kernel_size=3, stride=downsampling, padding=1, bias=False),
                        'bn2': nn.BatchNorm2d(out_features),
                        'act2': activation(),
                        'conv3': nn.Conv2d(out_features, out_features * expansion, kernel_size=1, bias=False),
                        'bn3': nn.BatchNorm2d(out_features * expansion),
                    }
                )),
            shortcut=ResNetShorcut(in_features, out_features * expansion, downsampling) if self.should_apply_shortcut else None)


class ResNetLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block: nn.Module = ResNetBasicBlock, n: int = 1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, downsampling=downsampling, *args, **kwargs),
            *[block(out_channels * block.expansion,
                    out_channels, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels: int = 3, blocks_sizes: List[int] = [64, 128, 256, 512], deepths: List[int] = [2, 2, 2, 2],
                 activation: nn.Module = nn.ReLU, block: nn.Module = ResNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            OrderedDict(
                {
                    'conv': nn.Conv2d(
                        in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
                    'bn': nn.BatchNorm2d(self.blocks_sizes[0]),
                    'act': activation(),
                    'pool': nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                }
            )
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet(nn.Module):
    """Scalable implementation of ResNet proposed in "Deep Residual Learning for Image Recognition"(https://arxiv.org/abs/1512.03385)

    Args:
        in_channels (int, optional): Number of channels in the input Image (3 for RGB and 1 for Gray). Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1000, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(
            self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def resnet18(in_channels: int, n_classes: int) -> ResNet:
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[2, 2, 2, 2])


def resnet34(in_channels: int, n_classes: int) -> ResNet:
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[3, 4, 6, 3])


def resnet50(in_channels: int, n_classes: int) -> ResNet:
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3])


def resnet101(in_channels: int, n_classes: int) -> ResNet:
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 23, 3])


def resnet152(in_channels: int, n_classes: int) -> ResNet:
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 8, 36, 3])
