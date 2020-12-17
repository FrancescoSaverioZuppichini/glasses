from torch import nn
from glasses.utils.PretrainedWeightsProvider import PretrainedWeightsProvider, Config
from glasses.nn.models import *
import difflib

class AutoConfig:

    zoo = {
        'default': Config(),
        'resnet26': Config(interpolation='bicubic'),
        'resnet26d': Config(interpolation='bicubic'),
        'resnet50d':  Config(interpolation='bicubic'),
        'cse_resnet50':  Config(interpolation='bicubic'),
        'efficientnet_b0':  Config(resize=224, input_size=224, interpolation='bicubic'),
        'efficientnet_b1':  Config(resize=240, input_size=240, interpolation='bicubic'),
        'efficientnet_b2':  Config(resize=260, input_size=260, interpolation='bicubic'),
        'efficientnet_b3':  Config(resize=300, input_size=300, interpolation='bicubic'),
        'efficientnet_b4':  Config(resize=380, input_size=380, interpolation='bicubic'),
        'efficientnet_b5':  Config(resize=456, input_size=456, interpolation='bicubic'),
        'efficientnet_b6':  Config(resize=528, input_size=528, interpolation='bicubic'),
        'efficientnet_b7':  Config(resize=600, input_size=600, interpolation='bicubic'),
        'efficientnet_b8':  Config(resize=672, input_size=672, interpolation='bicubic'),
        'efficientnet_l2':  Config(resize=800, input_size=800, interpolation='bicubic'),
        'efficientnet_lite0':  Config(resize=224, input_size=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), interpolation='bicubic'),
        'efficientnet_lite1':  Config(resize=240, input_size=240, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), interpolation='bicubic'),
        'efficientnet_lite2':  Config(resize=260, input_size=260, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), interpolation='bicubic'),
        'efficientnet_lite3':  Config(resize=280, input_size=280, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), interpolation='bicubic'),
        'efficientnet_lite4':  Config(resize=300, input_size=300, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), interpolation='bicubic'),
        'unet': Config(resize=384, input_size=384, mean=None, std=None),

    }

    @staticmethod
    def from_name(name: str, *args, **kwargs) -> nn.Module:
        cfg = AutoConfig.zoo.get(name, AutoConfig.zoo['default'])
        return cfg

    @staticmethod
    def names(self):
        return self.zoo.keys()
