import torch
import requests
from argparse import ArgumentParser
from torch import nn
from dataclasses import dataclass
from functools import partial
from typing import Dict
from torch import Tensor
from glasses.utils.ModuleTransfer import ModuleTransfer
from glasses.utils.PretrainedWeightsProvider import PretrainedWeightsProvider
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from torchvision.models import densenet121, densenet161, densenet169, densenet201
from torchvision.models import vgg11, vgg13, vgg16, vgg19
from torchvision.models import mobilenet_v2
from glasses.nn.models import *
from tqdm.autonotebook import tqdm
from pathlib import Path
import boto3
from boto3.s3.transfer import TransferConfig
from io import BytesIO
import logging
import timm
import pretrainedmodels


zoo_models_mapping = {
    'resnet18': [partial(resnet18, pretrained=True), ResNet.resnet18],
    'resnet26': [partial(timm.create_model, 'resnet26', pretrained=True), ResNet.resnet26],
    'resnet26d': [partial(timm.create_model, 'resnet26d', pretrained=True), ResNet.resnet26d],

    'resnet34': [partial(timm.create_model, 'resnet34', pretrained=True), ResNet.resnet34],
    'resnet50': [partial(resnet50, pretrained=True), ResNet.resnet50],
    'resnet50d': [partial(timm.create_model, 'resnet50d', pretrained=True), ResNet.resnet50d],

    'resnet101': [partial(resnet101, pretrained=True), ResNet.resnet101],
    'resnet152': [partial(resnet152, pretrained=True), ResNet.resnet152],
    'cse_resnet50': [partial(timm.create_model, 'seresnet50', pretrained=True), SEResNet.cse_resnet50],
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
    'vgg11_bn':[pretrainedmodels.__dict__['vgg11_bn'], VGG.vgg11_bn],
    'vgg13_bn':[pretrainedmodels.__dict__['vgg13_bn'], VGG.vgg13_bn],
    'vgg16_bn':[pretrainedmodels.__dict__['vgg16_bn'], VGG.vgg16_bn],
    'vgg19_bn':[pretrainedmodels.__dict__['vgg19_bn'], VGG.vgg19_bn],

    'mobilenet_v2': [partial(mobilenet_v2, pretrained=True), MobileNetV2],

    'efficientnet_b0': [partial(timm.create_model, 'efficientnet_b0', pretrained=True), EfficientNet.efficientnet_b0],
    'efficientnet_b1': [partial(timm.create_model, 'efficientnet_b1', pretrained=True), EfficientNet.efficientnet_b1],
    'efficientnet_b2': [partial(timm.create_model, 'efficientnet_b2', pretrained=True), EfficientNet.efficientnet_b2],
    'efficientnet_b3': [partial(timm.create_model, 'efficientnet_b3', pretrained=True), EfficientNet.efficientnet_b3],

}


def clone_model(src: nn.Module, dst: nn.Module) -> nn.Module:
    src = src.eval()
    dst = dst.eval()

    x = torch.rand((1, 3, 224, 224))
    a = src(x)
    b = dst(x)

    ModuleTransfer(src, dst)(x)

    return dst


@dataclass
class LocalStorage:
    root: Path = PretrainedWeightsProvider.BASE_DIR
    override: bool = False

    def __post_init__(self):
        self.root.mkdir(exist_ok=True)
        self.models_files = list(self.root.glob('*.pth'))

    def __call__(self, key: str, model: nn.Module, bar: tqdm):
        save_path = self.root / Path(f'{key}.pth')

        torch.save(model.state_dict(), save_path)
        assert save_path.exists()
        model.load_state_dict(torch.load(save_path))

    def __contains__(self, el: 'str') -> bool:
        return el in [file.stem for file in self.models_files]


class AWSSTorage:

    def __init__(self):
        self.s3 = boto3.resource('s3')

    def __call__(self, key: str, model: nn.Module, bar: tqdm):
        buffer = BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)

        bar.reset(total=buffer.getbuffer().nbytes)
        bar.set_description('📤')
        obj = self.s3.Object('cv-glasses', f'{key}.pth')

        obj.upload_fileobj(buffer, ExtraArgs={
                           'ACL': 'public-read'}, Callback=lambda x: bar.update(x))

    def __contains__(self, el: 'str') -> bool:
        return False

    parser = ArgumentParser()
    parser.add_argument('--storage', type=str,
                        choices=['local', 'aws'], default='local')
    parser.add_argument('-o', type=Path)

    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.info(f'Using {args.storage} storage 💾')

    if args.o is not None:
        save_dir = args.o
        save_dir.mkdir(exist_ok=True)

    storage = LocalStorage() if args.storage == 'local' else AWSSTorage()

    override = True

    bar = tqdm(zoo_models_mapping.items())
    uploading_bar = tqdm()
    for key, mapping in bar:
        bar.set_description(key)

        if key not in storage or override:
            src_def, dst_def = mapping
            cloned = clone_model(src_def(), dst_def())
            storage(key, cloned, uploading_bar)

        # uploading_bar.update(0)
