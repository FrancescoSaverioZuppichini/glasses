import math
import time
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sotabencheval.image_classification import ImageNetEvaluator
from torchvision.datasets import ImageNet

from tqdm.autonotebook import tqdm

from glasses.models import *
from glasses.models.AutoModel import AutoModel
from glasses.models.AutoConfig import AutoConfig
from glasses.utils.PretrainedWeightsProvider import PretrainedWeightsProvider

models =list(PretrainedWeightsProvider.weights_zoo.keys())
    
batch_sizes = {
    'efficientnet_b0': 256,
    'efficientnet_b1': 128,
}


provider = PretrainedWeightsProvider()
# code stolen from https://github.com/ansleliu/EfficientNet.PyTorch/blob/master/eval.py
# if you are using it, show some love an star his repo!``


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_img_id(image_name):
    return image_name.split('/')[-1].replace('.JPEG', '')


def benchmark(model: nn.Module, transform, batch_size=64, device=device, fast: bool = False):

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

            image_ids = [get_img_id(img[0]) for img in valid_loader.dataset.imgs[i_val *
                                                                                 valid_loader.batch_size:(i_val+1)*valid_loader.batch_size]]
            evaluator.add(dict(zip(image_ids, list(net_out.cpu().numpy()))))
            pbar.set_description(f'f1={evaluator.top1.avg:.2f}')
            pbar.update(1)
            if fast:
                break
        pbar.close()
    stop = time.time()
    if fast:
        return evaluator.top1.avg, None, None
    else:
        res = evaluator.get_results()
        return res['Top 1 Accuracy'], res['Top 5 Accuracy'], stop - start

def benchmark_all() -> pd.DataFrame:
    save_path = Path('./benchmark.csv')
    df = pd.DataFrame()
    if save_path.exists():
        df = pd.read_csv(str(save_path), index_col=0)
    index = []
    records = []

    bar = tqdm(models)
    try:
        for key in bar:
            if key not in df.index:
                try:
                    model = AutoModel.from_pretrained(key)
                    cfg = AutoConfig.from_name(key)
                    tr = cfg.transform

                    batch_size = 64

                    # if key in batch_sizes:
                    #     batch_size = batch_sizes[key]

                    bar.set_description(
                        f'{key}, size={cfg.input_size}, batch_size={batch_size}')

                    top1, top5, time = benchmark(model.to(device), tr, batch_size)

                    index.append(key)

                    data = {
                        'top1': top1,
                        'top5': top5,
                        'time': time,
                        'batch_size': batch_size
                    }

                    pprint(data)
                    records.append(data)
                except KeyError:
                    continue
    except Exception as e:
        print(e)
        pass
    
    if len(records) > 0: 
        new_df = pd.DataFrame.from_records(records, index=index)

        if df is not None:
            df = pd.concat([df, new_df])
        else:
            df = new_df

        df.to_csv('./benchmark.csv')
        mk = df.sort_values('top1', ascending=False).to_markdown()

        with open('./benchmark.md', 'w') as f:
            f.write(mk)

    return df


if __name__ == '__main__':
    benchmark_all()
