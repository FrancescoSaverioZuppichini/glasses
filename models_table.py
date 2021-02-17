from argparse import ArgumentParser

import pandas as pd
import torch
from torchinfo import summary
import tqdm
from pathlib import Path
from glasses.models import AutoModel, AutoConfig

# from torchvision.models import *

parser = ArgumentParser()
parser.add_argument('-o', default='./table.md', type=Path)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device}')
print(f'out={args.o}')


def row(item):
    key, model_factory = item
    model = model_factory()
    input_size = AutoConfig.from_name(key).input_size
    n_classes = 1 if key == 'unet' else 3
    total_params, _, param_size, total_size = summary(
        model.to(device), (n_classes, input_size, input_size))

    del model

    return {
        'name': key,
        'Parameters': f"{total_params.item():,}",
        'Size (MB)': f"{param_size.item():.2f}",
        # 'Total Size (MB)': int(total_size.item())
    }

df = pd.DataFrame()
if Path('./table.csv').exists():
    df = pd.read_csv('./table.csv', index_col=0)

res = []
bar = tqdm.tqdm(AutoModel.zoo.items())
for item in bar:
    if item[0] not in df.index.values:
        bar.set_description(item[0])
        try:
            out = row(item)
            res.append(out)
        except RuntimeError as e:
            res.append({'name': item[0],
                        'Parameters': '😥',
                        'Size (MB)': '😥',
                        })
            continue

new_df = pd.DataFrame.from_records(res)
new_df = new_df.set_index('name', drop=True)

df = pd.concat([df, new_df])
df.to_csv('./table.csv')

mk = df.to_markdown()

with open(args.o, 'w') as f:
    f.write(mk)
