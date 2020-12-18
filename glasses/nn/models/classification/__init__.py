"""Classification models"""

from .resnet import ResNet
from .resnetxt import ResNetXt
from .wide_resnet import WideResNet
from .densenet import DenseNet
from .senet import SEResNet
from .regnet import RegNet
from .vgg import VGG
from .mobilenet import MobileNet
from .alexnet import AlexNet
from .efficientnet import EfficientNet, EfficientNetLite
from .fishnet import FishNet

__all__ = ['ResNet', 'DenseNet', 'ResNetXt', 'RegNet', 'WideResNet', 'FishNet',
 'SEResNet', 'VGG', 'MobileNet', 'AlexNet',  'EfficientNet', 'EfficientNetLite']
