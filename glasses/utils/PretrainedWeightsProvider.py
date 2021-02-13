import torch
import requests
import sys
import os
import logging
import torchvision.transforms as T
import torch.nn as nn
import numpy as np
from torch import nn
from dataclasses import dataclass
from functools import partial
from typing import Dict
from torch import Tensor
from .ModuleTransfer import ModuleTransfer
from tqdm.autonotebook import tqdm
from pathlib import Path
from PIL import Image
from typing import Tuple
from typing import Callable
from functools import wraps
logging.basicConfig(level=logging.INFO)

IMAGENET_DEFAULT_MEAN = torch.Tensor([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = torch.Tensor([0.229, 0.224, 0.225])


@dataclass
class Config:
    """Describe one configuration for a pretrained model.

    Returns:
        [type]: [description]
    """
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
        tr = T.Compose([
            T.Resize(self.resize, interpolations[self.interpolation]),
            T.CenterCrop(self.input_size),
            T.ToTensor(),
        ])

        if self.mean != None or self.std != None:
            tr.transforms.append(T.Normalize(mean=self.mean, std=self.std))

        return tr


StateDict = Dict[str, Tensor]


def pretrained(name: str = None) -> Callable:
    _name = name

    def decorator(func: Callable) -> Callable:
        """Decorator to fetch the pretrained model.

        Args:
            func ([Callable]): [description]

        Returns:
            [Callable]: [description]
        """
        name = func.__name__ if _name is None else _name
        provider = PretrainedWeightsProvider()
        @wraps(func)
        def wrapper(*args,  pretrained: bool = False, **kwargs) -> Callable:
            model = func(*args, **kwargs)
            if pretrained:
                model.load_state_dict(provider[name])
                model.eval()
            return model
        return wrapper
    return decorator


@dataclass
class BasicUrlHandler:
    url: str

    def get_response(self) -> requests.Request:
        r = requests.get(self.url, stream=True)
        return r

    def __call__(self, save_path: Path, chunk_size: int = 1024) -> Path:
        r = self.get_response()

        with open(save_path, 'wb') as f:
            total_length = sys.getsizeof(r.content)
            bar = tqdm(r.iter_content(chunk_size=chunk_size),
                       total=total_length // chunk_size)
            for chunk in bar:
                if chunk:
                    f.write(chunk)
                    f.flush()


class GoogleDriveUrlHandler(BasicUrlHandler):

    def __init__(self, url: str, file_id: str):
        super().__init__(url)
        self.file_id = file_id

    def get_confirm_token(self, response: requests.Request) -> object:
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def get_response(self) -> requests.Request:
        session = requests.Session()

        response = session.get(
            self.url, params={'id': self.file_id}, stream=True)
        token = self.get_confirm_token(response)

        if token:
            params = {'id': self.file_id, 'confirm': token}
            response = session.get(self.url, params=params, stream=True)

        return response


@dataclass
class PretrainedWeightsProvider:
    """
    This class allows to retrieve pretrained models weights (state dict).

    Example:
        >>> provider = PretrainedWeightsProvider()
        >>> provider['resnet18'] # get a pre-trained resnet18 model
        # see all the outputs
        >>> provider = PretrainedWeightsProvider(verbose=1)
        # change save dir
        >>> provider = PretrainedWeightsProvider(save_dir=Path('./awesome/'))
        # override model even if already downloaded
        >>> provider = PretrainedWeightsProvider(override=True)
    """

    BASE_URL: str = 'https://cv-glasses.s3.eu-central-1.amazonaws.com'
    BASE_DIR: Path = Path(torch.hub.get_dir()) / Path('glasses')
    save_dir: Path = BASE_DIR
    chunk_size: int = 1024 * 1
    verbose: int = 0
    override: bool = False

    weights_zoo = {
        'densenet121': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/densenet121.pth?raw=true'),
        'densenet169': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/densenet169.pth?raw=true'),
        'densenet201': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/densenet201.pth?raw=true'),
        'dummy': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/dummy.pth?raw=true'),
        'efficientnet_b0': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/efficientnet_b0.pth?raw=true'),
        'efficientnet_b1': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/efficientnet_b1.pth?raw=true'),
        'efficientnet_b2': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/efficientnet_b2.pth?raw=true'),
        'efficientnet_b3': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/efficientnet_b3.pth?raw=true'),
        'regnetx_002': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/regnetx_002.pth?raw=true'),
        'regnetx_004': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/regnetx_004.pth?raw=true'),
        'regnetx_006': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/regnetx_006.pth?raw=true'),
        'regnetx_008': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/regnetx_008.pth?raw=true'),
        'regnetx_016': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/regnetx_016.pth?raw=true'),
        'regnetx_032': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/regnetx_032.pth?raw=true'),
        'regnety_002': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/regnety_002.pth?raw=true'),
        'regnety_004': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/regnety_004.pth?raw=true'),
        'regnety_006': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/regnety_006.pth?raw=true'),
        'regnety_008': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/regnety_008.pth?raw=true'),
        'regnety_016': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/regnety_016.pth?raw=true'),
        'regnety_032': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/regnety_032.pth?raw=true'),
        'resnet18': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/resnet18.pth?raw=true'),
        'resnet26': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/resnet26.pth?raw=true'),
        'resnet26d': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/resnet26d.pth?raw=true'),
        'resnet34': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/resnet34.pth?raw=true'),
        'resnet34d': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/resnet34d.pth?raw=true'),
        'deit_tiny_patch16_224': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/deit_tiny_patch16_224.pth?raw=true'),
        'deit_small_patch16_224': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/deit_small_patch16_224.pth?raw=true'),
        # aws stored
        'cse_resnet50': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/cse_resnet50.pth'),
        'densenet161': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/densenet161.pth'),
        'resnet101': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/resnet101.pth'),
        'resnet152': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/resnet152.pth'),
        'resnet50': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/resnet50.pth'),
        'resnet50d': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/resnet50d.pth'),
        'resnext101_32x8d': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/resnext101_32x8d.pth'),
        'resnext50_32x4d': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/resnext50_32x4d.pth'),
        'vgg11': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/vgg11.pth'),
        'vgg11_bn': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/vgg11_bn.pth'),
        'vgg13': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/vgg13.pth'),
        'vgg13_bn': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/vgg13_bn.pth'),
        'vgg16': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/vgg16.pth'),
        'vgg16_bn': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/vgg16_bn.pth'),
        'vgg19': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/vgg19.pth'),
        'vgg19_bn': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/vgg19_bn.pth'),
        'wide_resnet101_2': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/wide_resnet101_2.pth'),
        'wide_resnet50_2': BasicUrlHandler('https://glasses-weights.s3.eu-central-1.amazonaws.com/wide_resnet50_2.pth'),
        
    }

    def __post_init__(self):
        try:
            self.save_dir.mkdir(exist_ok=True)
        except FileNotFoundError:
            default_dir = str(Path(__file__).resolve().parent)
            self.save_dir = Path(os.environ.get(
                'HOME', default_dir)) / Path('.glasses/')
            self.save_dir.mkdir(exist_ok=True)
        os.environ['GLASSES_HOME'] = str(self.save_dir)

    def __getitem__(self, key: str) -> dict:
        if key not in self.weights_zoo:
            raise KeyError(
                f'No weights for model "{key}". Available models are {",".join(list(self.weights_zoo.keys()))}')

        save_path = self.save_dir / f'{key}.pth'

        should_download = not save_path.exists()

        if should_download or self.override:
            handler = self.weights_zoo[key]
            handler(save_path, self.chunk_size)

        weights = torch.load(save_path)
        logging.info(f'Loaded {key} pretrained weights.')
        return weights

    # def __contains__(self, key: str) -> bool:
    #     return key in self.zoo.keys()
