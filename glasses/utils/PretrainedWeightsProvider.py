import torch
import requests
import sys
import os
import logging
import torchvision.transforms as T
import torch.nn as nn
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

        response = session.get(self.url, params = { 'id' : self.file_id }, stream = True)
        token = self.get_confirm_token(response)

        if token:
            params = { 'id' : self.file_id, 'confirm' : token }
            response = session.get(self.url, params = params, stream = True)  

        return response

@dataclass
class PretrainedWeightsProvider:
    """
    This class allows to retrieve pretrained models weights (state dict).

    Example:
        >>> provider = PretrainedWeightsProvider()
        >>> provider['resnet18'] # get a pre-trained resnet18 model
        >>> provider = PretrainedWeightsProvider(verbose=1) # see all the outputs
        >>> provider = PretrainedWeightsProvider(save_dir=Path('./awesome/')) # change save dir
        >>> provider = PretrainedWeightsProvider(override=True) # override model even if already downloaded
    """

    BASE_URL: str = 'https://cv-glasses.s3.eu-central-1.amazonaws.com'
    BASE_DIR: Path = Path(torch.hub.get_dir()) / Path('glasses')
    save_dir: Path = BASE_DIR
    chunk_size: int = 1024 * 1
    verbose: int = 0
    override: bool = False

    weights_zoo = {
        'resnet18': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses/blob/feature/weights/weights/resnet18.pth?raw=true'),
        'resnet26': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses/blob/feature/weights/weights/resnet26.pth?raw=true'),
        'resnet34': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses/blob/feature/weights/weights/resnet34.pth?raw=true'),
        'mobilenet_v2': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses/blob/feature/weights/weights/mobilenet_v2.pth?raw=true'),
        'efficientnet_b0': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses/blob/feature/weights/weights/efficientnet_b0.pth?raw=true'),
        'efficientnet_b1': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses/blob/feature/weights/weights/efficientnet_b1.pth?raw=true'),
        'efficientnet_b2': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses/blob/feature/weights/weights/efficientnet_b2.pth?raw=true'),
        'efficientnet_b3': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses/blob/feature/weights/weights/efficientnet_b3.pth?raw=true'),
        'densenet121': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses/blob/feature/weights/weights/densenet121.pth?raw=true'),
        'densenet169': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses/blob/feature/weights/weights/densenet169.pth?raw=true'),
        'densenet201': BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses/blob/feature/weights/weights/densenet201.pth?raw=true'),
        # from google drive
        'resnet50': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1DYXJ12tLb-W687Wa9MWfvyarlz52cyD3'),
        'cse_resnet50': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1CMyib_ACsWUIbXa7KjX3NXKkfAQyNnLd'),
        'resnet101': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '14q5m53eYqQOPb1ZQYHButFW_g9Ec5pmR'),
        'resnet152': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1d-EGQi-HGFNXEdQE7cVzXvmFl9iZAd-F'),
        'resnext50_32x4d': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1lvV5v-WT0YBLSB9j3beGs8cV7Qc3ecEg'),
        'resnext101_32x8d': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1y4GfcknrznFhMdMsbZwdZBYPN6UUNP2H'),
        'wide_resnet50_2': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1or9L8aO7QDU0haP1pdbwGPrSLiTRQkqa'),
        'wide_resnet101_2': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1VUvWd6MF7ySDx7kQH3siJjtpmxxw5LE8'),
        'vgg11': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1dnlUB4ew8EdLMTVa9xS0CpyXVICpEls2'),
        'vgg13': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1X87UaYvENTuLRD94TP8PJE0h-7su3lJL'),
        'vgg16': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1yER36sIvoYZXRY_QHgk9sX6ecMP6t2h7'),
        'vgg19': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1VWBABqCyrlqNXlacS5lHjDCWGkbwSIYZ'),
        'vgg11_bn': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1HCqOnxN2RCyRvUy8pQ_5XnAqH3zI1bTp'),
        'vgg13_bn': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1YlttLo-9VDgXq03gdnkJ8NIBEMpYwejN'),
        'vgg16_bn': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1X6dvcZYPQcwTGlQ1S87pOUuCmTUhP1zj'),
        'vgg19_bn': GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '1rHNKV8MgES-7PXYdzarI23MklRMpxaOm'),
        'densenet161':  GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id = '153fMUorCUGSl4pKSA4tzduaI6BFG7hu5'),

    }

    def __post_init__(self):
        try:
            self.save_dir.mkdir(exist_ok=True)
        except FileNotFoundError:
            self.save_dir = Path(os.environ['HOME']) / Path('.glasses/')
            self.save_dir.mkdir(exist_ok=True)

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
