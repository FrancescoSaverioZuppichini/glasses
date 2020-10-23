import torch
import torch.nn as nn
import math
import numpy as np
import time
import pandas as pd
from pprint import pprint
from torchvision.datasets import ImageNet
from PIL import Image
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from torchvision.models import densenet121, densenet161, densenet169, densenet201
from torchvision.models import vgg11, vgg13, vgg16, vgg19
from torchvision.models import mobilenet_v2
from glasses.nn.models import *
from tqdm.autonotebook import tqdm
from glasses.utils.PretrainedWeightsProvider import PretrainedWeightsProvider
from functools import partial
from pathlib import Path
from efficientnet_pytorch import EfficientNet as EfficientNetPytorch
from glasses.utils.ModuleTransfer import ModuleTransfer
from sotabencheval.image_classification import ImageNetEvaluator


models = {
    'resnet18':  ResNet.resnet18,
    'resnet26':  ResNet.resnet26,
    'resnet34':ResNet.resnet34,
    'resnet50': ResNet.resnet50,
    'resnet101': ResNet.resnet101,
    'resnet152': ResNet.resnet152,
}

# zoo_models_mapping = {
#     # 'resnet18': [partial(resnet18, pretrained=True), ResNet.resnet18],
#     # 'resnet34': [partial(resnet34, pretrained=True), ResNet.resnet34],
#     # 'resnet50': [partial(resnet50, pretrained=True), ResNet.resnet50],
#     # 'resnet101': [partial(resnet101, pretrained=True), ResNet.resnet101],
#     # 'resnet152': [partial(resnet152, pretrained=True), ResNet.resnet152],


#     # 'resnext50_32x4d': [partial(resnext50_32x4d, pretrained=True), ResNetXt.resnext50_32x4d],
#     # 'resnext101_32x8d': [partial(resnext101_32x8d, pretrained=True), ResNetXt.resnext101_32x8d],
#     # 'wide_resnet50_2': [partial(wide_resnet50_2, pretrained=True), WideResNet.wide_resnet50_2],
#     # 'wide_resnet101_2': [partial(wide_resnet101_2, pretrained=True), WideResNet.wide_resnet101_2],

#     # 'densenet121': [partial(densenet121, pretrained=True), DenseNet.densenet121],
#     # 'densenet169': [partial(densenet169, pretrained=True), DenseNet.densenet169],
#     # 'densenet201': [partial(densenet201, pretrained=True), DenseNet.densenet201],
#     # 'densenet161': [partial(densenet161, pretrained=True), DenseNet.densenet161],
#     # 'vgg11': [partial(vgg11, pretrained=True), VGG.vgg11],
#     # 'vgg13': [partial(vgg13, pretrained=True), VGG.vgg13],
#     # 'vgg16': [partial(vgg16, pretrained=True), VGG.vgg16],
#     # 'vgg19': [partial(vgg19, pretrained=True), VGG.vgg19],

#     # 'mobilenet_v2': [partial(mobilenet_v2, pretrained=True), MobileNetV2],

#     'efficientnet_b0': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet_b0'), EfficientNet.efficientnet_b0],
#     'efficientnet_b1': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet_b1'), EfficientNet.efficientnet_b1],
#     'efficientnet_b2': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet_b2'), EfficientNet.efficientnet_b2],
#     'efficientnet_b3': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet_b3'), EfficientNet.efficientnet_b3],
#     'efficientnet_b4': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet_b4'), EfficientNet.efficientnet_b4],
#     'efficientnet_b5': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet_b5'), EfficientNet.efficientnet_b5],
#     'efficientnet_b6': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet_b6'), EfficientNet.efficientnet_b6],
#     'efficientnet_b7': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet_b7'), EfficientNet.efficientnet_b7],

# }

resize_size = {
    # 'resnet18': [partial(resnet18, pretrained=True), ResNet.resnet18],
    # 'resnet34': [partial(resnet34, pretrained=True), ResNet.resnet34],
    # 'resnet50': [partial(resnet50, pretrained=True), ResNet.resnet50],
    # 'resnet101': [partial(resnet101, pretrained=True), ResNet.resnet101],
    # 'resnet152': [partial(resnet152, pretrained=True), ResNet.resnet152],


    # 'resnext50_32x4d': [partial(resnext50_32x4d, pretrained=True), ResNetXt.resnext50_32x4d],
    # 'resnext101_32x8d': [partial(resnext101_32x8d, pretrained=True), ResNetXt.resnext101_32x8d],
    # 'wide_resnet50_2': [partial(wide_resnet50_2, pretrained=True), WideResNet.wide_resnet50_2],
    # 'wide_resnet101_2': [partial(wide_resnet101_2, pretrained=True), WideResNet.wide_resnet101_2],

    # 'densenet121': [partial(densenet121, pretrained=True), DenseNet.densenet121],
    # 'densenet169': [partial(densenet169, pretrained=True), DenseNet.densenet169],
    # 'densenet201': [partial(densenet201, pretrained=True), DenseNet.densenet201],
    # 'densenet161': [partial(densenet161, pretrained=True), DenseNet.densenet161],
    # 'vgg11': [partial(vgg11, pretrained=True), VGG.vgg11],
    # 'vgg13': [partial(vgg13, pretrained=True), VGG.vgg13],
    # 'vgg16': [partial(vgg16, pretrained=True), VGG.vgg16],
    # 'vgg19': [partial(vgg19, pretrained=True), VGG.vgg19],

    # 'mobilenet_v2': [partial(mobilenet_v2, pretrained=True), MobileNetV2],

    'efficientnet-b0': 224,
    'efficientnet-b1': 240,
    'efficientnet-b2': 260,
    'efficientnet-b3': 300,
    'efficientnet-b4': 380,
    'efficientnet-b5': 456,
    'efficientnet-b6': 528,
    'efficientnet-b7': 600,

}

batch_sizes = {
     'efficientnet-b0': 256,
    'efficientnet-b1': 128,
    'efficientnet-b2': 64,
    'efficientnet-b3': 64,
    'efficientnet-b4': 64,
    'efficientnet-b5': 32,
    'efficientnet-b6': 32,
    'efficientnet-b7': 16,
}


provider = PretrainedWeightsProvider()
# code stolen from https://github.com/ansleliu/EfficientNet.PyTorch/blob/master/eval.py
# if you are using it, show some love an star his repo!``


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_img_id(image_name):
    return image_name.split('/')[-1].replace('.JPEG', '')

def benchmark(model: nn.Module, transform, batch_size=64):

    valid_dataset = ImageNet(
        root='/home/zuppif/Downloads/ImageNet', split='val', transform=transform)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=12, pin_memory=True)
                                 
    evaluator = ImageNetEvaluator(model_name='test',
                              paper_arxiv_id='1905.11946')
    model.eval()
    num_batches = int(math.ceil(len(valid_loader.dataset) /
                            float(valid_loader.batch_size)))

    start = time.time()

    with torch.no_grad():
        pbar = tqdm(np.arange(num_batches), leave=False)
        for i_val, (images, labels) in enumerate(valid_loader):

            images = images.to(device)
            labels = torch.squeeze(labels.to(device))

            net_out = model(images)

            image_ids = [get_img_id(img[0]) for img in valid_loader.dataset.imgs[i_val*valid_loader.batch_size:(i_val+1)*valid_loader.batch_size]]
            evaluator.add(dict(zip(image_ids, list(net_out.cpu().numpy()))))
          
            pbar.update(1)
        pbar.close()
    stop = time.time()
    res = evaluator.get_results()
    return res['Top 1 Accuracy'], res['Top 5 Accuracy'], stop - start


def benchmark_all() -> pd.DataFrame:
    save_path = Path('./benchmark.pkl')
    index = []
    records = []

    bar = tqdm(models.items())

    for key, model_def in bar:
        model = model_def()
        cfg = model.configs[key]
        tr = cfg.transform

        batch_size = 64

        if key in batch_sizes:
            batch_size = batch_sizes[key] 

        bar.set_description(f'{key}, size={cfg.input_size}, batch_size={batch_size}')
                                   
        data = {}

        # original_model = original_model_func()
        # glasses_model = glasses_model_func()

        model.load_state_dict(provider[key])

        glasses_top1, glasses_top5, glasses_time = benchmark(model.to(device), tr, batch_size)
        # original_top1, original_top5, original_time = benchmark(original_model.to(device), valid_loader)
        index.append(key)

        data = {
            # 'original_top1': original_top1,
            # 'original_top5': original_top5,
            # 'original_time': original_time,
            'glasses_top1': glasses_top1,
            'glasses_top5': glasses_top5,
            'glasses_time': glasses_time
        }

        records.append(data)

        df = pd.DataFrame.from_records(records, index=index)
        df.to_pickle(str(save_path))
        pprint(records)
    # pd.DataFrame.

    print(df)

if __name__ == '__name__':
    sbenchmark_all()
