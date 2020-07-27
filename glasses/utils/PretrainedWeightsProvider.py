import torch
import requests
from torch import nn
from dataclasses import dataclass
from functools import partial
from typing import Dict
from torch import Tensor
from .ModuleTransfer import ModuleTransfer
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from ..nn.models.classification.resnet import ResNet
from tqdm.autonotebook import tqdm
from pathlib import Path

StateDict = Dict[str, Tensor]


@dataclass
class PretrainedWeightsProvider:

    zoo = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    zoo_models_mapping = {
        'resnet18': [partial(resnet18, pretrained=False), ResNet.resnet18],
        'resnet34': [partial(resnet34, pretrained=False), ResNet.resnet34],
        'resnet50': [partial(resnet50, pretrained=False), ResNet.resnet50],
        'resnet101': [partial(resnet101, pretrained=False), ResNet.resnet101],
        'resnet152': [partial(resnet152, pretrained=False), ResNet.resnet152],
    }

    save_dir: Path = Path('./')
    chunk_size: int = 1024 

    def download_weight(self, url: str, save_path: Path) -> Path:
        r = requests.get(url, stream=True)

        with open(save_path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            bar = tqdm(r.iter_content(chunk_size=self.chunk_size),
                       total=total_length/self.chunk_size)
            for chunk in bar:
                if chunk:
                    f.write(chunk)
                    f.flush()

    def clone_model(self, key: str, save_path: Path) -> nn.Module:
        src_def, dst_def = self.zoo_models_mapping[key]
        src = src_def().eval()
        dst = dst_def().eval()
        src.load_state_dict(torch.load(save_path))

        x = torch.rand((1, 3, 224, 224))
        a = src(x)
        b = dst(x)

        assert not torch.equal(a, b)

        ModuleTransfer(src, dst)(x)

        a = src(x)
        b = dst(x)

        assert torch.equal(a, b)

        return dst

    def __getitem__(self, key: str) -> nn.Module:
        if key not in self:
            raise ValueError(f'Available models are {",".join(list(self.zoo.keys()))}')
        url = self.zoo[key]
        save_path = self.save_dir / Path(key + '.pth')
        self.download_weight(url, save_path)
        model = self.clone_model(key, save_path)
        return model

    def __contains__(self, key: str) -> bool:
        return key in self.zoo.keys()
