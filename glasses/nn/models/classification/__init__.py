"""Classification models"""

from .resnet import ResNet
from .resnetxt import ResNetXt
from .wide_resnet import WideResNet
from .densenet import DenseNet
from .senet import SEResNet
from .vgg import VGG
from .se import SpatialSE, ChannelSE, SpatialChannelSE
from .mobilenet import MobileNetV2
from .alexnet import AlexNet
from .efficientnet import EfficientNet
from .fishnet import FishNet

__all__ = ['ResNet', 'DenseNet', 'ResNetXt', 'WideResNet', 'FishNet', 'SpatialSE', 'ChannelSE', 'SpatialChannelSE', 'SEResNet', 'VGG', 'MobileNetV2', 'AlexNet',  'EfficientNet']
