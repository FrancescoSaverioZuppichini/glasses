import torch.nn as nn
from torch import Tensor
from ..blocks.residuals import ResidualAdd
from collections import OrderedDict


class ResNetShorcut(nn.Module):
    """Shorcut function applied by ResNet to upsample the channel 
    when they do not match between the residual and the output

    Args:
        in_features (int): features (channels) of the input
        out_features (int): features (channels) of the desidered output
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features,
                              kernel_size=1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBottleNeckConvs(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module = nn.ReLU(inplace=True), stride: int = 1, expansion: int = 4):
        """BottleNeck ResNet convs composed by two 3x3 convs. 

        Args:
            in_features (int): features (channels) of the input
            out_features (int): features (channels) of the desidered output
            activation (nn.Module, optional): Activation applied between the weights. Defaults to nn.ReLU(inplace=True).
        """
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.expansion = expansion
        self.block = nn.Sequential(
            OrderedDict(
                {
                    'conv1': nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, bias=False),
                    'bn1': nn.BatchNorm2d(out_features),
                    'act1': activation,
                    'conv2': nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, bias=False),
                    'bn2': nn.BatchNorm2d(out_features),
                    'act2': activation,
                    'conv3': nn.Conv2d(out_features, out_features * expansion, kernel_size=1, bias=False),
                    'bn3': nn.BatchNorm2d(out_features * expansion),
                }
            ))

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class ResNetBasicBlockConvs(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module = nn.ReLU(inplace=True),  stride: int = 1,):
        """Basic ResNet convs composed by two 3x3 convs. 

        Args:
            in_features (int): features (channels) of the input
            out_features (int): features (channels) of the desidered output
            activation (nn.Module, optional): Activation applied between the weights. Defaults to nn.ReLU(inplace=True).
        """
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.block = nn.Sequential(
            OrderedDict(
                {
                    'conv1': nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=1, bias=False),
                    'bn1': nn.BatchNorm2d(out_features),
                    'act1': activation,
                    'conv2': nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, bias=False),
                    'bn2': nn.BatchNorm2d(out_features),
                }
            ))

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class ResNetBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_features: int, out_features: int, activation: nn.Module = nn.ReLU(inplace=True)):
        """Basic ResNet block composed by two 3x3 convs. 

        Args:
            in_features (int): features (channels) of the input
            out_features (int): features (channels) of the desidered output
            activation (nn.Module, optional): Activation applied between the weights. Defaults to nn.ReLU(inplace=True).
        """
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.block = ResidualAdd(
            ResNetBasicBlockConvs(in_features, out_features, activation,
                                  stride=2 if self.should_apply_shortcut else 1),
            shortcut=ResNetShorcut(
                in_features, out_features * self.expansion) if self.should_apply_shortcut else None
        )
        self.act = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        # activation is applied after the residual
        x = self.act(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_features != self.out_features


class ResNetBottleNeckBlock(ResNetBasicBlock):
    expansion: int = 4

    def __init__(self, in_features: int, out_features: int, activation: nn.Module = nn.ReLU(inplace=True)):
        """Basic ResNet block composed by two 3x3 convs. 

        Args:
            in_features (int): features (channels) of the input
            out_features (int): features (channels) of the desidered output
            activation (nn.Module, optional): Activation applied between the weights. Defaults to nn.ReLU(inplace=True).
        """
        super().__init__(in_features, out_features, activation)
        self.block = ResidualAdd(
            ResNetBottleNeckConvs(in_features, out_features, activation,
                                  stride=2 if self.should_apply_shortcut else 1),
            shortcut=ResNetShorcut(
                in_features, out_features * self.expansion) if self.should_apply_shortcut else None
        )


class ResNetLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block: nn.Module = ResNetBasicBlock, n: int = 1, *args, **kwargs):
        super().__init__()

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs),
            *[block(out_channels * block.expansion,
                    out_channels, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x
