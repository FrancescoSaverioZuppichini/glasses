from glasses.models import *
from glasses.utils.weights.PretrainedWeightsProvider import Config
from torch import nn
import torchvision.transforms as T


# class Transform()

class AutoTransform:

    zoo = {
        "default": {},
        "resnet26": {'interpolation' :"bicubic"},
        "resnet26d": {'interpolation' :"bicubic"},
        "resnet50d": {'interpolation' :"bicubic"},
        "cse_resnet50": {'interpolation' :"bicubic"},
        "resnest200e": {'resize' : 320, 'input_size' : 320, 'interpolation' :"bicubic"},
        "resnest269e": {'resize' : 461, 'input_size' : 416, 'interpolation' :"bicubic"},
        "efficientnet_b0": {'resize' : 224, 'input_size' : 224, 'interpolation' :"bicubic"},
        "efficientnet_b1": {'resize' : 240, 'input_size' : 240, 'interpolation' :"bicubic"},
        "efficientnet_b2": {'resize' : 260, 'input_size' : 260, 'interpolation' :"bicubic"},
        "efficientnet_b3": {'resize' : 300, 'input_size' : 300, 'interpolation' :"bicubic"},
        "efficientnet_b4": {'resize' : 380, 'input_size' : 380, 'interpolation' :"bicubic"},
        "efficientnet_b5": {'resize' : 456, 'input_size' : 456, 'interpolation' :"bicubic"},
        "efficientnet_b6": {'resize' : 528, 'input_size' : 528, 'interpolation' :"bicubic"},
        "efficientnet_b7": {'resize' : 600, 'input_size' : 600, 'interpolation' :"bicubic"},
        "efficientnet_b8": {'resize' : 672, 'input_size' : 672, 'interpolation' :"bicubic"},
        "efficientnet_l2": {'resize' : 800, 'input_size' : 800, 'interpolation' :"bicubic"},
        "efficientnet_lite0": {
            'resize' : 224,
            'input_size' : 224,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "efficientnet_lite1": {
            'resize' : 240,
            'input_size' : 240,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "efficientnet_lite2": {
            'resize' : 260,
            'input_size' : 260,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
       },
        "efficientnet_lite3": {
            'resize' : 280,
            'input_size' : 280,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "efficientnet_lite4": {
            'resize' : 300,
            'input_size' : 300,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "unet": {'resize' : 384, 'input_size' : 384, 'mean' :None, 'std':None),
        "vit_base_patch16_224": {
            'resize' : 224,
            'input_size' : 224,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "vit_base_patch16_384": {
            'resize' : 384,
            'input_size' : 384,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "vit_base_patch32_384": {
            'resize' : 384,
            'input_size' : 384,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "vit_huge_patch16_224": {
            'resize' : 224,
            'input_size' : 224,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "vit_huge_patch32_384": {
            'resize' : 384,
            'input_size' : 384,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "vit_large_patch16_224": {
            'resize' : 224,
            'input_size' : 224,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "vit_large_patch16_384": {
            'resize' : 384,
            'input_size' : 384,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "vit_large_patch32_384": {
            'resize' : 384,
            'input_size' : 384,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "vit_small_patch16_224": {
            'resize' : 224,
            'input_size' : 224,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "deit_tiny_patch16_224": {
            'resize' : 224,
            'input_size' : 224,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "deit_small_patch16_224": {
            'resize' : 224,
            'input_size' : 224,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "deit_base_patch16_224": {
            'resize' : 224,
            'input_size' : 224,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        },
        "deit_base_patch16_384": {
            'resize' : 384,
            'input_size' : 384,
            'mean' :(0.5, 0.5, 0.5),
            'std':(0.5, 0.5, 0.5),
            'interpolation' :"bicubic",
        ),
    }

    def __init__(self):
        raise EnvironmentError(
            "AutoTransform is designed to be instantiated "
            "using the `AutoModel.from_name(model_name)` method."
        )

    @staticmethod
    def from_name(name: str) -> Config:
        """Returns a configuration from a given model name.
        If the name is not found, it returns a default one.


        Examples:
            >>> AutoTransform.from_name('resnet18')

            You can access the preprocess `transformation`, you should use it
            to preprocess your inputs.

            >>> cfg =  AutoTransform.from_name('resnet18')
            >>> cfg.transform


        Args:
            name (str): [description]

        Returns:
            Config: The model's config
        """
        cfg = AutoTransform.zoo.get(name, AutoTransform.zoo["default"])
        return cfg

    @staticmethod
    def names():
        return AutoTransform.zoo.keys()
