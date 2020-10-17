import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
from pprint import pprint
from torchvision.datasets import ImageNet
from PIL import Image
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from glasses.nn.models import *
from tqdm import tqdm
from glasses.utils.PretrainedWeightsProvider import PretrainedWeightsProvider
from functools import partial

zoo_models_mapping = {
    'resnet18': [partial(resnet18, pretrained=True), ResNet.resnet18],
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

    # 'efficientnet-b0': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b0'), EfficientNet.b0],
    # 'efficientnet-b1': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b1'), EfficientNet.b1],
    # 'efficientnet-b2': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b2'), EfficientNet.b2],
    # 'efficientnet-b3': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b3'), EfficientNet.b3],
    # 'efficientnet-b4': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b4'), EfficientNet.b4],
    # 'efficientnet-b5': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b5'), EfficientNet.b5],
    # 'efficientnet-b6': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b6'), EfficientNet.b6],
    # 'efficientnet-b7': [partial(EfficientNetPytorch.from_pretrained, 'efficientnet-b7'), EfficientNet.b7],

}

provider = PretrainedWeightsProvider()
# code stolen from https://github.com/ansleliu/EfficientNet.PyTorch/blob/master/eval.py
# if you are using it, show some love an star his repo!``


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

size = 224

transform = Compose([Resize(256),
                     CenterCrop(size),
                     ToTensor(),
                     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

valid_dataset = ImageNet(
    root='/home/zuppif/Downloads/ImageNet', split='val', transform=transform)


valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False,
                                           num_workers=12, pin_memory=False)
num_batches = int(math.ceil(len(valid_loader.dataset) /
                            float(valid_loader.batch_size)))


def benchmark(model: nn.Module):
    model.eval()

    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        pbar = tqdm(np.arange(num_batches))
        for i_val, (images, labels) in enumerate(valid_loader):

            images = images.to(device)
            labels = torch.squeeze(labels.to(device))

            net_out = model(images)

            prec1, prec5 = accuracy(net_out, labels, topk=(1, 5))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

            pbar.update(1)
            pbar.set_description("> Eval")
            pbar.set_postfix(Top1=top1.avg, Top5=top5.avg)
        pbar.set_postfix(Top1=top1.avg, Top5=top5.avg)
        pbar.update(1)
        pbar.close()
    return top1.avg, top5.avg


def benchmark_all() -> pd.DataFrame:
    save_path = Path('./benchmark.pkl')
    
    records = []
    for key, (original_model_func, glasses_model_func) in zoo_models_mapping.items():
        data = {}
        original_top1, original_top5 = benchmark(original_model_func().to(device))
        glasses_model = glasses_model_func()
        glasses_model.load_state_dict(provider[key])
        glasses_top1, glasses_top5 = benchmark(glasses_model.to(device))

        data[key] = {'original_top1': original_top1,
                     'original_top5': original_top5,
                     'glasses_top1': glasses_top1,
                     'glasses_top5': glasses_top5

                     }

        records.append(data)
    pprint(records)


    df = pd.DataFrame(records)
    print(df)
    df.to_pickle(save_path)
    # pd.DataFrame.

    # # model = provider['resnet34'].to(device)
    # model = densenet121(True, memory_efficient=True).to(device)

benchmark_all()