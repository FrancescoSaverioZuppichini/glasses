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
from glasses.utils.ModuleTransfer import ModuleTransfer
from sotabencheval.image_classification import ImageNetEvaluator


models = {
     'vgg11':   VGG.vgg11,
    'vgg13': VGG.vgg13,
    'vgg16': VGG.vgg16,
    'vgg19': VGG.vgg19,
    'vgg11_bn':   VGG.vgg11_bn,
    'vgg13_bn': VGG.vgg13_bn,
    'vgg16_bn': VGG.vgg16_bn,
    'vgg19_bn': VGG.vgg19_bn,
    'resnet18':  ResNet.resnet18,
    'resnet26':  ResNet.resnet26,
    'resnet34':ResNet.resnet34,
    'resnet50': ResNet.resnet50,
    'cse_resnet50': SEResNet.cse_resnet50,
    'resnet101': ResNet.resnet101,
    'resnet152': ResNet.resnet152,


    'resnext50_32x4d': ResNetXt.resnext50_32x4d,
    'resnext101_32x8d':ResNetXt.resnext101_32x8d,
    'wide_resnet50_2': WideResNet.wide_resnet50_2,
    'wide_resnet101_2': WideResNet.wide_resnet101_2,

    'densenet121': DenseNet.densenet121,
    'densenet169': DenseNet.densenet169,
    'densenet201': DenseNet.densenet201,
    'densenet161': DenseNet.densenet161,
    # 'mobilenet_v2': MobileNetV2,

    'efficientnet_b0': EfficientNet.efficientnet_b0,
    'efficientnet_b1': EfficientNet.efficientnet_b1,
    'efficientnet_b2':EfficientNet.efficientnet_b2,
    'efficientnet_b3': EfficientNet.efficientnet_b3,

}


batch_sizes = {
     'efficientnet-b0': 256,
    'efficientnet-b1': 128,
    'efficientnet-b2': 64,
    'efficientnet-b3': 64
}



provider = PretrainedWeightsProvider()
# code stolen from https://github.com/ansleliu/EfficientNet.PyTorch/blob/master/eval.py
# if you are using it, show some love an star his repo!``


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_img_id(image_name):
    return image_name.split('/')[-1].replace('.JPEG', '')

def benchmark(model: nn.Module, transform, batch_size=64, device = device):

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
    df = pd.DataFrame()
    if save_path.exists():
        df = pd.read_pickle(str(save_path))
    
    index = []
    records = []

    bar = tqdm(models.items())

    for key, model_def in bar:
        
        if key not in df.index:
            model = model_def()
            try:
                cfg = model.configs[key]
            except KeyError:
                # default one
                cfg = ResNet.configs['resnet18']
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
                'top1': glasses_top1,
                'top5': glasses_top5,
                'time': glasses_time
            }
            print(data)
            records.append(data)

            new_df = pd.DataFrame.from_records(records, index=index)
            
            if df is not None:
                df = pd.concat([df, new_df])
            else: 
                df = new_df
            # df.to_pickle(str(save_path))
            # pprint(records)
    # pd.DataFrame.

    print(df)

if __name__ == '__name__':
    benchmark_all()
