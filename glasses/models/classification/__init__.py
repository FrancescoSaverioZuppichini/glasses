"""Classification models"""

from .resnet import ResNet
from .resnetxt import ResNetXt
from .wide_resnet import WideResNet
from .densenet import DenseNet
from .senet import SEResNet
from .regnet import RegNet
from .resnest import ResNeSt
from .vgg import VGG
from .mobilenet import MobileNet
from .alexnet import AlexNet
from .efficientnet import EfficientNet, EfficientNetLite
from .fishnet import FishNet
from .vit import ViT
from .deit import DeiT

__all__ = ['ResNet', 'DenseNet', 'ResNetXt', 'RegNet', 'ResNeSt', 'WideResNet', 'FishNet',
 'SEResNet', 'VGG', 'MobileNet', 'AlexNet',  'EfficientNet', 'EfficientNetLite', 'ViT', 'DeiT']
