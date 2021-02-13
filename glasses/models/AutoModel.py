import difflib
from functools import wraps
from typing import Any, Callable, List, OrderedDict
from glasses.utils.PretrainedWeightsProvider import PretrainedWeightsProvider
from torch import nn

from .classification import *
from .segmentation import *


class AutoModel:
    """This class returns a model based on its name.

    Examples:
        >>> AutoModel.models() # odict_keys(['resnet18', 'resnet26', .... ])
        >>> AutoModel.from_name('resnet18')
        >>> AutoModel.from_name('resnet18', activation=nn.SELU)
        >>> AutoModel.from_pretrained('resnet18')



    Raises:
        KeyError: Raised if the name of the model is not found.

    """

    zoo = OrderedDict({
        'resnet18': ResNet.resnet18,
        'resnet26': ResNet.resnet26,
        'resnet26d': ResNet.resnet26d,
        'resnet34': ResNet.resnet34,
        'resnet34d':  ResNet.resnet34d,
        'resnet50':  ResNet.resnet50,
        'resnet50d':  ResNet.resnet50d,
        'resnet101': ResNet.resnet101,
        'resnet152': ResNet.resnet152,
        'resnet200': ResNet.resnet200,
        'se_resnet18': SEResNet.se_resnet18,
        'se_resnet34': SEResNet.se_resnet34,
        'se_resnet50': SEResNet.se_resnet50,
        'se_resnet101': SEResNet.se_resnet101,
        'se_resnet152': SEResNet.se_resnet152,
        'cse_resnet18': SEResNet.cse_resnet18,
        'cse_resnet34': SEResNet.cse_resnet34,
        'cse_resnet50': SEResNet.cse_resnet50,
        'cse_resnet101': SEResNet.cse_resnet101,
        'cse_resnet152': SEResNet.cse_resnet152,
        'resnext50_32x4d': ResNetXt.resnext50_32x4d,
        'resnext101_32x8d': ResNetXt.resnext101_32x8d,
        'resnext101_32x16d': ResNetXt.resnext101_32x16d,
        'resnext101_32x32d': ResNetXt.resnext101_32x32d,
        'resnext101_32x48d': ResNetXt.resnext101_32x48d,
        'regnetx_002': RegNet.regnetx_002,
        'regnetx_004': RegNet.regnetx_004,
        'regnetx_006': RegNet.regnetx_006,
        'regnetx_008': RegNet.regnetx_008,
        'regnetx_016': RegNet.regnetx_016,
        'regnetx_032': RegNet.regnetx_032,
        'regnety_002': RegNet.regnety_002,
        'regnety_004': RegNet.regnety_004,
        'regnety_006': RegNet.regnety_006,
        'regnety_008': RegNet.regnety_008,
        'regnety_016': RegNet.regnety_016,
        'regnety_032': RegNet.regnety_032,
        'resnest14d' : ResNeSt.resnest14d,
        'resnest26d' : ResNeSt.resnest26d,
        'resnest50d' : ResNeSt.resnest50d,
        'resnest50d_1s4x24d' : ResNeSt.resnest50d_1s4x24d,
        'resnest50d_4s2x40d' : ResNeSt.resnest50d_4s2x40d,
        'resnest101e' : ResNeSt.resnest101e,
        'resnest200e' : ResNeSt.resnest200e,
        'resnest269e' : ResNeSt.resnest269e,
        'wide_resnet50_2': WideResNet.wide_resnet50_2,
        'wide_resnet101_2': WideResNet.wide_resnet101_2,
        'densenet121': DenseNet.densenet121,
        'densenet169': DenseNet.densenet169,
        'densenet201': DenseNet.densenet201,
        'densenet161': DenseNet.densenet161,
        'fishnet99':  FishNet.fishnet99,
        'fishnet150': FishNet.fishnet150,
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
        'vit_small_patch16_224': ViT.vit_small_patch16_224,
        'vit_base_patch16_224':  ViT.vit_base_patch16_224,
        'vit_base_patch16_384': ViT.vit_base_patch16_384,
        'vit_base_patch32_384': ViT.vit_base_patch32_384,
        'vit_huge_patch16_224':  ViT.vit_huge_patch16_224,
        'vit_huge_patch32_384': ViT.vit_huge_patch32_384,
        'vit_large_patch16_224':  ViT.vit_large_patch16_224,
        'vit_large_patch16_384': ViT.vit_large_patch16_384,
        'vit_large_patch32_384': ViT.vit_large_patch32_384,
        'deit_tiny_patch16_224': DeiT.deit_tiny_patch16_224,
        'deit_small_patch16_224': DeiT.deit_small_patch16_224,
        'deit_base_patch16_224': DeiT.deit_base_patch16_224,
        'deit_base_patch16_384': DeiT.deit_base_patch16_384,
        'mobilenetv2': MobileNet.mobilenet_v2,
        'unet': UNet
    })

    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name)` or the `AutoModel.from_name(model_name)`  method."
        )

    @staticmethod
    def from_name(name: str, *args, **kwargs) -> nn.Module:
        """Instantiates one of the model classes of the library.

        Examples:
            >>> AutoModel.models() # odict_keys(['resnet18', 'resnet26', .... ])
            >>> AutoModel.from_name('resnet18')
            >>> AutoModel.from_name('resnet18', activation=nn.SELU)


        Args:
            name (str): Name of the model, e.g. 'resnet18'

        Raises:
            KeyError: Raised if the name of the model is not found.

        Returns:
            nn.Module: A fully instantiated model
        """
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
        """Instantiates one of the pretrained model classes of the library.

        Examples:
            >>> AutoModel.pretrained_models() # odict_keys(['resnet18', 'resnet26', .... ])
            >>> AutoModel.from_pretrained('resnet18')

        Args:
            name (str): Name of the model, e.g. 'resnet18'

        Raises:
            KeyError: Raised if the name of the model is not found.

        Returns:
            nn.Module: A fully instantiated pretrained model
        """
        # check if key is valid
        if name not in PretrainedWeightsProvider.weights_zoo:
            suggestions = difflib.get_close_matches(name, AutoModel.zoo.keys())
            msg = f"Model \"{name}\" does not exists."
            if len(suggestions) > 0:
                msg += f' Did you mean "{suggestions[0]}?"'

            msg += f'Available pretrained models are {",".join(list(PretrainedWeightsProvider.weights_zoo.keys()))}'
            raise KeyError(msg)

        model = AutoModel.from_name(name, pretrained=True, *args, **kwargs)
        return model

    @staticmethod
    def models() -> List[str]:
        """List the available models name

        Returns:
            List[str]: [description]
        """
        return AutoModel.zoo.keys()

    @staticmethod
    def pretrained_models() -> List[str]:
        """List the available pretrained models name

        Returns:
            List[str]: [description]
        """
        return PretrainedWeightsProvider.weights_zoo.keys()