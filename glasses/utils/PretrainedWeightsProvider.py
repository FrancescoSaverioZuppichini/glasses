import torch
import requests
import sys
from torch import nn
from dataclasses import dataclass
from functools import partial
from typing import Dict
from torch import Tensor
from .ModuleTransfer import ModuleTransfer
# from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
# from torchvision.models import densenet121, densenet161, densenet169, densenet201
# from torchvision.models import vgg11, vgg13, vgg16, vgg19
# from torchvision.models import mobilenet_v2
# from ..nn.models.classification.resnet import ResNet
# from ..nn.models.classification.densenet import DenseNet
# from ..nn.models.classification.vgg import VGG
# from ..nn.models.classification import MobileNetV2, ResNetXt, WideResNet
# from ..nn.models.classification import EfficientNet
from tqdm.autonotebook import tqdm
from pathlib import Path
from efficientnet_pytorch import EfficientNet as EfficientNetPytorch
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image
from typing import Tuple
import logging

logging.basicConfig( level=logging.INFO)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass
class Config:
    input_size: int = 224
    resize: int = 256
    mean: Tuple[float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float] = IMAGENET_DEFAULT_STD
    interpolation: str = 'bilinear'

    @property
    def transform(self):
        interpolations = {
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC
        }
        return T.Compose([
            T.Resize(self.resize, interpolations[self.interpolation]),
            T.CenterCrop(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])


StateDict = Dict[str, Tensor]


def pretrained(func):
    name = func.__name__
    provider = PretrainedWeightsProvider()
    def wrapper(*args,  pretrained=False, **kwargs):
        model = func(*args, **kwargs)
        if pretrained:
            model.load_state_dict(provider[name])
        return model
    return wrapper

@dataclass
class PretrainedWeightsProvider:
    """
    This class allows to retrieve pretrained models.

    Example:
        >>> provider = PretrainedWeightsProvider()
        >>> provider['resnet18'] # get a pre-trained resnet18 model
        >>> provider = PretrainedWeightsProvider(verbose=1) # see all the outputs
        >>> provider = PretrainedWeightsProvider(save_dir=Path('./awesome/')) # change save dir
        >>> provider = PretrainedWeightsProvider(override=True) # override model even if already downloaded
    """

    # zoo = {
    # 'resnet18':ResNet.resnet18,
    # 'resnet34':ResNet.resnet34,
    # 'resnet50':ResNet.resnet50,
    # 'resnet101': ResNet.resnet101,
    # 'resnet152': ResNet.resnet152,

    # 'resnext50_32x4d': ResNetXt.resnext50_32x4d,
    # 'resnext101_32x8d': ResNetXt.resnext101_32x8d,
    # 'wide_resnet50_2': WideResNet.wide_resnet50_2,
    # 'wide_resnet101_2': WideResNet.wide_resnet101_2,

    # 'densenet121': DenseNet.densenet121,
    # 'densenet169': DenseNet.densenet169,
    # 'densenet201': DenseNet.densenet201,
    # 'densenet161': DenseNet.densenet161,
    # 'vgg11': VGG.vgg11,
    # 'vgg13':  VGG.vgg13,
    # 'vgg16': VGG.vgg16,
    # 'vgg19':  VGG.vgg19,

    # 'mobilenet_v2': MobileNetV2,

    # 'efficientnet-b0': EfficientNet.b0,
    # 'efficientnet-b1': EfficientNet.b1,
    # 'efficientnet-b2': EfficientNet.b2,
    # 'efficientnet-b3': EfficientNet.b3,
    # 'efficientnet-b4': EfficientNet.b4,
    # 'efficientnet-b5': EfficientNet.b5,
    # 'efficientnet-b6': EfficientNet.b6,
    # 'efficientnet-b7': EfficientNet.b7,

    # }

    BASE_URL = 'https://cv-glasses.s3.eu-central-1.amazonaws.com'
    BASE_DIR = Path('/home/zuppif/Documents/models_weights')
    save_dir: Path = BASE_DIR
    chunk_size: int = 1024
    verbose: int = 0
    override: bool = False

    def __post_init__(self):
        self.save_dir.mkdir(exist_ok=True)

    def download_weight(self, url: str, save_path: Path) -> Path:
        r = requests.get(url, stream=True)

        with open(save_path, 'wb') as f:
            total_length = sys.getsizeof(r.content)
            bar = tqdm(r.iter_content(chunk_size=self.chunk_size),
                       total=total_length // self.chunk_size)
            for chunk in bar:
                if chunk:
                    f.write(chunk)
                    f.flush()

    def __getitem__(self, key: str) -> nn.Module:
        # if key not in self.zoo:
        #     raise KeyError(
        #         f'No weights for model "{key}". Available models are {",".join(list(self.zoo_models_mapping.keys()))}')

        save_path = self.save_dir / f'{key}.pth'

        should_download = not save_path.exists()

        if should_download or self.override:
            url = f'{self.BASE_URL}/{key}.pth'
            self.download_weight(url, save_path)

        weights = torch.load(save_path)
        logging.info(f'Loaded {key} pretrained weights.')
        return weights

    # def __contains__(self, key: str) -> bool:
    #     return key in self.zoo.keys()
