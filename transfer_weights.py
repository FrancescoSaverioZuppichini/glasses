from glasses.models.AutoConfig import AutoConfig
import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Dict

import boto3
import pretrainedmodels
import timm
import torch
from torch import Tensor, nn
from torchvision.models import (densenet121, densenet161, densenet169,
                                densenet201, resnet18,
                                resnet50, resnet101, resnet152,
                                resnext50_32x4d, resnext101_32x8d, vgg11,
                                vgg13, vgg16, vgg19, wide_resnet50_2,
                                wide_resnet101_2)
from tqdm.autonotebook import tqdm

from glasses.models.AutoModel import AutoModel
from glasses.models import *
from glasses.utils.ModuleTransfer import ModuleTransfer
from glasses.utils.PretrainedWeightsProvider import PretrainedWeightsProvider
from glasses.models.classification.vit import ViTTokens
from glasses.models.classification.deit import DeiTTokens


def vit_clone(key: str):
    src = timm.create_model(key, pretrained='True')
    dst = AutoModel.from_name(key)

    cfg = AutoConfig.from_name(key)

    dst = clone_model(src, dst, torch.randn(
        (1, 3, cfg.input_size, cfg.input_size)), dest_skip=[ViTTokens])

    dst.embedding.positions.data.copy_(src.pos_embed.data.squeeze(0))
    dst.embedding.tokens.cls.data.copy_(src.cls_token.data)

    return dst


def deit_clone(key: str):
    k_split = key.split('_')
    hub_key = "_".join(k_split[:2]) + '_distilled_' + "_".join(k_split[2:])
    src = torch.hub.load('facebookresearch/deit:main',
                         hub_key, pretrained=True)

    dst = AutoModel.from_name(key)

    cfg = AutoConfig.from_name(f"vit_{'_'.join(key.split('_')[1:])}")

    dst = clone_model(src, dst, torch.randn(
        (1, 3, cfg.input_size, cfg.input_size)), dest_skip=[DeiTTokens])

    dst.embedding.positions.data.copy_(src.pos_embed.data.squeeze(0))
    dst.embedding.tokens.cls.data.copy_(src.cls_token.data)
    dst.embedding.tokens.dist.data.copy_(src.dist_token.data)

    return dst


zoo_source = {
    'resnet18': partial(resnet18, pretrained=True),
    'resnet26': partial(timm.create_model, 'resnet26', pretrained=True),
    'resnet26d': partial(timm.create_model, 'resnet26d', pretrained=True),
    'resnet34': partial(timm.create_model, 'resnet34', pretrained=True),
    'resnet34d': partial(timm.create_model, 'resnet34d', pretrained=True),
    'resnet50': partial(resnet50, pretrained=True),
    'resnet50d': partial(timm.create_model, 'resnet50d', pretrained=True),
    'resnet101': partial(resnet101, pretrained=True),
    'resnet152': partial(resnet152, pretrained=True),
    'cse_resnet50': partial(timm.create_model, 'seresnet50', pretrained=True),
    'resnext50_32x4d': partial(resnext50_32x4d, pretrained=True),
    'resnext101_32x8d': partial(resnext101_32x8d, pretrained=True),
    'wide_resnet50_2': partial(wide_resnet50_2, pretrained=True),
    'wide_resnet101_2': partial(wide_resnet101_2, pretrained=True),

    'regnetx_002': None,
    'regnetx_004': None,
    'regnetx_006': None,
    'regnetx_008': None,
    'regnetx_016': None,
    'regnetx_032': None,
    'regnety_002': None,
    'regnety_004': None,
    'regnety_006': None,
    'regnety_008': None,
    'regnety_016': None,
    'regnety_032': None,

    'densenet121': partial(densenet121, pretrained=True),
    'densenet169': partial(densenet169, pretrained=True),
    'densenet201': partial(densenet201, pretrained=True),
    'densenet161': partial(densenet161, pretrained=True),

    'vgg11': partial(vgg11, pretrained=True),
    'vgg13': partial(vgg13, pretrained=True),
    'vgg16': partial(vgg16, pretrained=True),
    'vgg19': partial(vgg19, pretrained=True),
    'vgg11_bn': pretrainedmodels.__dict__['vgg11_bn'],
    'vgg13_bn': pretrainedmodels.__dict__['vgg13_bn'],
    'vgg16_bn': pretrainedmodels.__dict__['vgg16_bn'],
    'vgg19_bn': pretrainedmodels.__dict__['vgg19_bn'],

    # 'mobilenet_v2': [partial(mobilenet_v2, pretrained=True), MobileNetV2],

    'efficientnet_b0': partial(timm.create_model, 'efficientnet_b0', pretrained=True),
    'efficientnet_b1': partial(timm.create_model, 'efficientnet_b1', pretrained=True),
    'efficientnet_b2': partial(timm.create_model, 'efficientnet_b2', pretrained=True),
    'efficientnet_b3': partial(timm.create_model, 'efficientnet_b3', pretrained=True),
    # 'efficientnet_b5': partial(timm.create_model, 'efficientnet_b5', pretrained=True),
    # 'efficientnet_b6': partial(timm.create_model, 'efficientnet_b6', pretrained=True),

    'vit_base_patch16_224': (vit_clone, True),
    'vit_base_patch16_384': (vit_clone, True),
    'vit_base_patch32_384': (vit_clone, True),
    'vit_huge_patch16_224': (vit_clone, True),
    'vit_huge_patch32_384': (vit_clone, True),
    'vit_large_patch16_224': (vit_clone, True),
    'vit_large_patch16_384': (vit_clone, True),
    'vit_large_patch32_384': (vit_clone, True),

    'deit_tiny_patch16_224': (deit_clone, True),
    'deit_small_patch16_224': (deit_clone, True),
    'deit_base_patch16_224': (deit_clone, True),
    'deit_base_patch16_384': (deit_clone, True),

}


def clone_model(src: nn.Module, dst: nn.Module, x: Tensor = torch.rand((1, 3, 224, 224)), **kwargs) -> nn.Module:
    src = src.eval()
    dst = dst.eval()

    a = src(x)
    b = dst(x)

    ModuleTransfer(src, dst, **kwargs)(x)

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
        obj = self.s3.Object('glasses-weights', f'{key}.pth')

        obj.upload_fileobj(buffer, ExtraArgs={
                           'ACL': 'public-read'}, Callback=lambda x: bar.update(x))

    def __contains__(self, el: 'str') -> bool:
        return False


if __name__ == '__main__':
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
    if args.storage == 'local':
        logging.info(f'Store root={storage.root}')

    override = False

    bar = tqdm(zoo_source.items())
    uploading_bar = tqdm()
    for key, src_def in bar:
        bar.set_description(key)
        if src_def is None:
            # it means I was lazy and I meant to use timm
            src_def = partial(timm.create_model, key, pretrained=True)
        if key not in storage or override:
            if type(src_def) is tuple:
                # I have a custom clone func -> not the most elegant way, but it works!
                clone_func, flag = src_def
                cloned = clone_func(key)
            else:
                src, dst = src_def(), AutoModel.from_name(key)
                cloned = clone_model(src, dst)
            storage(key, cloned, uploading_bar)
