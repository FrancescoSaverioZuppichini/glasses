from torch import nn
import difflib
from glasses.utils.PretrainedWeightsProvider import PretrainedWeightsProvider
from .classification import *
from .segmentation import *


class AutoModel:
    """This class returns a model based on its name

    Examples:

        >>> AutoConfig.from_name('resnet18')
        >>> AutoConfig.from_name('resnet18', activation=nn.SELU)
        >>> AutoConfig.from_pretrained('resnet18')


    Raises:
        KeyError: [description]
        KeyError: [description]

    Returns:
        [type]: [description]
    """

    zoo = {
        'resnet18': ResNet.resnet18,
        'resnet26': ResNet.resnet26,
        'resnet26d': ResNet.resnet26d,
        'resnet34': ResNet.resnet34,
        'resnet50':  ResNet.resnet50,
        'resnet50d':  ResNet.resnet50d,
        'resnet101': ResNet.resnet101,
        'resnet152': ResNet.resnet152,
        'resnet200': ResNet.resnet200,
        'se_resnet18' : SEResNet.se_resnet18,
        'se_resnet34' : SEResNet.se_resnet34,
        'se_resnet50' : SEResNet.se_resnet50,
        'se_resnet101' : SEResNet.se_resnet101,
        'se_resnet152' : SEResNet.se_resnet152,
        'cse_resnet18' : SEResNet.cse_resnet18,
        'cse_resnet34' : SEResNet.cse_resnet34,
        'cse_resnet50' : SEResNet.cse_resnet50,
        'cse_resnet101' : SEResNet.cse_resnet101,
        'cse_resnet152' : SEResNet.cse_resnet152,
        'resnext50_32x4d': ResNetXt.resnext50_32x4d,
        'resnext101_32x8d': ResNetXt.resnext101_32x8d,
        'resnext101_32x16d': ResNetXt.resnext101_32x16d,
        'resnext101_32x32d': ResNetXt.resnext101_32x32d,
        'resnext101_32x48d': ResNetXt.resnext101_32x48d,
        'resnext101_32x32d': ResNetXt.resnext101_32x32d,
        'resnext101_32x48d': ResNetXt.resnext101_32x48d,
        'resnext101_32x32d': ResNetXt.resnext101_32x32d,
        'resnext101_32x48d': ResNetXt.resnext101_32x48d,
        'wide_resnet50_2': WideResNet.wide_resnet50_2,
        'wide_resnet101_2': WideResNet.wide_resnet101_2,
        'densenet121': DenseNet.densenet121,
        'densenet169': DenseNet.densenet169,
        'densenet201': DenseNet.densenet201,
        'densenet161': DenseNet.densenet161,
        'fishnet99' :  FishNet.fishnet99,
        'fishnet150' : FishNet.fishnet150,
        'vgg11': VGG.vgg11,
        'vgg13': VGG.vgg13,
        'vgg16':  VGG.vgg16,
        'vgg19': VGG.vgg19,
        'vgg11_bn': VGG.vgg11_bn,
        'vgg13_bn': VGG.vgg13_bn,
        'vgg16_bn': VGG.vgg16_bn,
        'vgg19_bn': VGG.vgg19_bn,
        'efficientnet_b0': EfficientNet.efficientnet_b0,
        'efficientnet_b1': EfficientNet.efficientnet_b1,
        'efficientnet_b2': EfficientNet.efficientnet_b2,
        'efficientnet_b3': EfficientNet.efficientnet_b3,
        'efficientnet_b4': EfficientNet.efficientnet_b4,
        'efficientnet_b5': EfficientNet.efficientnet_b5,
        'efficientnet_b6': EfficientNet.efficientnet_b6,
        'efficientnet_b7': EfficientNet.efficientnet_b7,
        'efficientnet_b8': EfficientNet.efficientnet_b8,
        'efficientnet_l2': EfficientNet.efficientnet_l2,
        'efficientnet_lite0': EfficientNetLite.efficientnet_lite0,
        'efficientnet_lite1': EfficientNetLite.efficientnet_lite1,
        'efficientnet_lite2': EfficientNetLite.efficientnet_lite2,
        'efficientnet_lite3': EfficientNetLite.efficientnet_lite3,
        'efficientnet_lite4': EfficientNetLite.efficientnet_lite4,
        'mobilenetv2': MobileNet.mobilenet_v2,
        'unet': UNet
    }

    @staticmethod
    def from_name(name: str, *args, **kwargs) -> nn.Module:
        if name not in AutoModel.zoo:
            suggestions = difflib.get_close_matches(name, AutoModel.zoo.keys())
            msg = f"Model \"{name}\" does not exists."
            if len(suggestions) > 0:
                msg += f' Did you mean "{suggestions[0]}?"'
            raise KeyError(msg)

        model = AutoModel.zoo[name](*args, **kwargs)
        return model

    @staticmethod
    def from_pretrained(name: str, *args, **kwargs) -> nn.Module:
        # check if key is valid
        if name not in PretrainedWeightsProvider.weights_zoo:
            suggestions = difflib.get_close_matches(name, AutoModel.zoo.keys())
            msg = f"Model \"{name}\" does not exists."
            if len(suggestions) > 0:
                msg += f' Did you mean "{suggestions[0]}?"'

            msg += f'Available models are {",".join(list(PretrainedWeightsProvider.weights_zoo.keys()))}'
            raise KeyError(msg)

        model = AutoModel.from_name(name, pretrained=True, *args, **kwargs)
        return model

    @property
    def models(self):
        return self.zoo.keys()
