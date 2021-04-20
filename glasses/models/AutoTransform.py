import torch
import torchvision.transforms as T
from glasses.models import *
from typing import Tuple, List, Callable
from torchvision.transforms import InterpolationMode
from functools import partial

IMAGENET_DEFAULT_MEAN = torch.Tensor([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = torch.Tensor([0.229, 0.224, 0.225])


class Transform(T.Compose):
    interpolations = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }

    def __init__(
        self,
        input_size: int,
        resize: int,
        std: Tuple[float],
        mean: Tuple[float],
        interpolation: str = "bilinear",
        transforms: List[Callable] = list(),
    ):
        base_transforms = [
            T.Resize(resize, Transform.interpolations[interpolation]),
            T.CenterCrop(input_size),
            T.ToTensor(),
        ]

        if mean != None or std != None:
            base_transforms.append(T.Normalize(mean=mean, std=std))

        super().__init__([*base_transforms, *transforms])


ImageNetTransform = partial(
    Transform,
    input_size=224,
    resize=256,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
)


class AutoTransform:

    zoo = {
        "default": ImageNetTransform(),
        "resnet26": ImageNetTransform(interpolation="bicubic"),
        "resnet26d": ImageNetTransform(interpolation="bicubic"),
        "resnet50d": ImageNetTransform(interpolation="bicubic"),
        "cse_resnet50": ImageNetTransform(interpolation="bicubic"),
        "resnest200e": ImageNetTransform(
            resize=320, input_size=320, interpolation="bicubic"
        ),
        "resnest269e": ImageNetTransform(
            resize=461, input_size=416, interpolation="bicubic"
        ),
        "efficientnet_b0": ImageNetTransform(
            resize=224, input_size=224, interpolation="bicubic"
        ),
        "efficientnet_b1": ImageNetTransform(
            resize=240, input_size=240, interpolation="bicubic"
        ),
        "efficientnet_b2": ImageNetTransform(
            resize=260, input_size=260, interpolation="bicubic"
        ),
        "efficientnet_b3": ImageNetTransform(
            resize=300, input_size=300, interpolation="bicubic"
        ),
        "efficientnet_b4": ImageNetTransform(
            resize=380, input_size=380, interpolation="bicubic"
        ),
        "efficientnet_b5": ImageNetTransform(
            resize=456, input_size=456, interpolation="bicubic"
        ),
        "efficientnet_b6": ImageNetTransform(
            resize=528, input_size=528, interpolation="bicubic"
        ),
        "efficientnet_b7": ImageNetTransform(
            resize=600, input_size=600, interpolation="bicubic"
        ),
        "efficientnet_b8": ImageNetTransform(
            resize=672, input_size=672, interpolation="bicubic"
        ),
        "efficientnet_l2": ImageNetTransform(
            resize=800, input_size=800, interpolation="bicubic"
        ),
        "efficientnet_lite0": ImageNetTransform(
            resize=224,
            input_size=224,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "efficientnet_lite1": ImageNetTransform(
            resize=240,
            input_size=240,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "efficientnet_lite2": ImageNetTransform(
            resize=260,
            input_size=260,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "efficientnet_lite3": ImageNetTransform(
            resize=280,
            input_size=280,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "efficientnet_lite4": ImageNetTransform(
            resize=300,
            input_size=300,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "unet": ImageNetTransform(resize=384, input_size=384, mean=None, std=None),
        "vit_base_patch16_224": ImageNetTransform(
            resize=224,
            input_size=224,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "vit_base_patch16_384": ImageNetTransform(
            resize=384,
            input_size=384,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "vit_base_patch32_384": ImageNetTransform(
            resize=384,
            input_size=384,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "vit_huge_patch16_224": ImageNetTransform(
            resize=224,
            input_size=224,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "vit_huge_patch32_384": ImageNetTransform(
            resize=384,
            input_size=384,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "vit_large_patch16_224": ImageNetTransform(
            resize=224,
            input_size=224,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "vit_large_patch16_384": ImageNetTransform(
            resize=384,
            input_size=384,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "vit_large_patch32_384": ImageNetTransform(
            resize=384,
            input_size=384,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "vit_small_patch16_224": ImageNetTransform(
            resize=224,
            input_size=224,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "deit_tiny_patch16_224": ImageNetTransform(
            resize=224,
            input_size=224,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "deit_small_patch16_224": ImageNetTransform(
            resize=224,
            input_size=224,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "deit_base_patch16_224": ImageNetTransform(
            resize=224,
            input_size=224,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        "deit_base_patch16_384": ImageNetTransform(
            resize=384,
            input_size=384,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
    }

    def __init__(self):
        raise EnvironmentError(
            "AutoTransform is designed to be instantiated "
            "using the `AutoTransform.from_name(model_name)` method."
        )

    @staticmethod
    def from_name(name: str) -> ImageNetTransform:
        """Returns a ImageNetTransformuration from a given model name.
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
