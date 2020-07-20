import torch.nn as nn
from torch import Tensor
from ..blocks.residuals import ResidualAdd
from collections import OrderedDict


class Shorcut(nn.Module):
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


class BasicBlock(nn.Module):
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
            nn.Sequential(
                OrderedDict(
                    {
                        'conv1': nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False),
                        'bn1': nn.BatchNorm2d(out_features),
                        'act1': activation,
                        'conv2': nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, bias=False),
                        'bn2': nn.BatchNorm2d(out_features),
                    }
                )),
            shortcut=Shorcut(
                in_features, out_features) if self.should_apply_shortcut else None
        )
        self.act = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        # TODO should decide how to activate in the end
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_features != self.out_features
