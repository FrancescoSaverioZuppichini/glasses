import torch
import requests
from argparse import ArgumentParser
from torch import nn
from dataclasses import dataclass
from functools import partial
from typing import Dict
from torch import Tensor
from glasses.utils.ModuleTransfer import ModuleTransfer
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from torchvision.models import densenet121, densenet161, densenet169, densenet201
from torchvision.models import vgg11, vgg13, vgg16, vgg19
from torchvision.models import mobilenet_v2
from glasses.nn.models.classification.resnet import ResNet
from glasses.nn.models.classification.densenet import DenseNet
from glasses.nn.models.classification.vgg import VGG
from glasses.nn.models.classification import MobileNetV2, ResNetXt, WideResNet
from glasses.nn.models.classification import EfficientNet
from tqdm.autonotebook import tqdm
from pathlib import Path
from efficientnet_pytorch import EfficientNet as EfficientNetPytorch
import boto3
from boto3.s3.transfer import TransferConfig
from io import BytesIO
import logging


zoo_models_mapping = {
    'resnet18': [partial(resnet18, pretrained=True), ResNet.resnet18],
    'resnet34': [partial(resnet34, pretrained=True), ResNet.resnet34],
    'resnet50': [partial(resnet50, pretrained=True), ResNet.resnet50],
    'resnet101': [partial(resnet101, pretrained=True), ResNet.resnet101],
    'resnet152': [partial(resnet152, pretrained=True), ResNet.resnet152],


    'resnext50_32x4d': [partial(resnext50_32x4d, pretrained=True), ResNetXt.resnext50_32x4d],
    'resnext101_32x8d': [partial(resnext101_32x8d, pretrained=True), ResNetXt.resnext101_32x8d],
    'wide_resnet50_2': [partial(wide_resnet50_2, pretrained=True), WideResNet.wide_resnet50_2],
    'wide_resnet101_2': [partial(wide_resnet101_2, pretrained=True), WideResNet.wide_resnet101_2],

    'densenet121': [partial(densenet121, pretrained=True), DenseNet.densenet121],
    'densenet169': [partial(densenet169, pretrained=True), DenseNet.densenet169],
    'densenet201': [partial(densenet201, pretrained=True), DenseNet.densenet201],
    'densenet161': [partial(densenet161, pretrained=True), DenseNet.densenet161],
    'vgg11': [partial(vgg11, pretrained=True), VGG.vgg11],
    'vgg13': [partial(vgg13, pretrained=True), VGG.vgg13],
    'vgg16': [partial(vgg16, pretrained=True), VGG.vgg16],
    'vgg19': [partial(vgg19, pretrained=True), VGG.vgg19],

    'mobilenet_v2': [partial(mobilenet_v2, pretrained=True), MobileNetV2],

    'efficientnet-b0': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b0'), EfficientNet.b0],
    'efficientnet-b1': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b1'), EfficientNet.b1],
    'efficientnet-b2': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b2'), EfficientNet.b2],
    'efficientnet-b3': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b3'), EfficientNet.b3],
    'efficientnet-b4': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b4'), EfficientNet.b4],
    'efficientnet-b5': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b5'), EfficientNet.b5],
    'efficientnet-b6': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b6'), EfficientNet.b6],
    'efficientnet-b7': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b7'), EfficientNet.b7],

}


def clone_model(src: nn.Module, dst: nn.Module) -> nn.Module:
    src = src.eval()
    dst = dst.eval()

    x = torch.rand((1, 3, 224, 224))
    a = src(x)
    b = dst(x)

    # assert not torch.equal(a, b)

    ModuleTransfer(src, dst)(x)

    # a = src(x)
    # b = dst(x)

    # assert torch.equal(a, b)

    return dst


@dataclass
class LocalStorage:
    root: Path = Path('~/.glasses/models/')

    def __post_init__(self):
        self.root.mkdir(exist_ok=True)

    def __call__(self, key: str, model: nn.Module):
        save_path = self.root / Path(f'{key}.pt')

        torch.save(model.state_dict(), save_path)
        assert save_path.exists()
        model.load_state_dict(torch.load(save_path))


class AWSSTorage:

    def __init__(self):
        self.s3 = boto3.resource('s3')

    def __call__(self, key: str, model: nn.Module, bar: tqdm):
        buffer = BytesIO()
        torch.save(cloned.state_dict(), buffer)
        buffer.seek(0)

        bar.reset(total=buffer.getbuffer().nbytes)
        bar.set_description('📤')
        obj = self.s3.Object('cv-glasses', f'{key}.pt')

        obj.upload_fileobj(buffer, ExtraArgs={'ACL': 'public-read'}, Callback=lambda x: bar.update(x))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--storage', type=str, choices=['local', 'aws'], default='aws')
    parser.add_argument('-o', type=Path)
 
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.info(f'Using {args.storage} storage 💾')

    if args.o is not None:
        save_dir = args.o
        save_dir.mkdir(exist_ok=True)

    storage =  LocalStorage(root=Path('./models')) if args.storage == 'local' else AWSSTorage()

    bar = tqdm(zoo_models_mapping.items())
    uploading_bar = tqdm()
    for key, mapping in bar:
        bar.set_description(key)

        src_def, dst_def = mapping
        cloned = clone_model(src_def(), dst_def())

        storage(key, cloned, uploading_bar)
        # uploading_bar.update(0)
