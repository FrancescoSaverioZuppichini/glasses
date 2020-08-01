import torch
import requests
from torch import nn
from dataclasses import dataclass
from functools import partial
from typing import Dict
from torch import Tensor
from .ModuleTransfer import ModuleTransfer
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, densenet121, densenet161, densenet169, densenet201
from ..nn.models.classification.resnet import ResNet
from ..nn.models.classification.densenet import DenseNet
from tqdm.autonotebook import tqdm
from pathlib import Path

StateDict = Dict[str, Tensor]


@dataclass
class PretrainedWeightsProvider:
    """
    This class allows to retrieve pretrained models.

    Example:
        >>> provider = PretrainedWeightsProvider()
        >>> provider['resnet18'] # get a pre-trained resnet18 model

    """

    zoo = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

        'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
        'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
        'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
        'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    }

    zoo_models_mapping = {
        'resnet18': [partial(resnet18, pretrained=True), ResNet.resnet18],
        'resnet34': [partial(resnet34, pretrained=True), ResNet.resnet34],
        'resnet50': [partial(resnet50, pretrained=True), ResNet.resnet50],
        'resnet101': [partial(resnet101, pretrained=True), ResNet.resnet101],
        'resnet152': [partial(resnet152, pretrained=True), ResNet.resnet152],
        'densenet121': [partial(densenet121, pretrained=True), DenseNet.densenet121],
        'densenet169': [partial(densenet169, pretrained=True), DenseNet.densenet169],
        'densenet201': [partial(densenet201, pretrained=True), DenseNet.densenet201],
        'densenet161': [partial(densenet161, pretrained=True), DenseNet.densenet161],
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
        # src.load_state_dict(torch.load(save_path))

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
            raise KeyError(
                f'No weights for model "{key}". Available models are {",".join(list(self.zoo.keys()))}')
        url = self.zoo[key]
        save_path = self.save_dir / Path(key + '.pth')
        # should_download = not save_path.exists()

        # if should_download: 
        #     self.download_weight(url, save_path)
        model = self.clone_model(key, save_path)
        return model

    def __contains__(self, key: str) -> bool:
        return key in self.zoo.keys()
