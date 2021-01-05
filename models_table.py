from argparse import ArgumentParser

import pandas as pd
import torch
from torchsummary import summary
import tqdm

from glasses.models import AutoModel, AutoConfig

# from torchvision.models import *

parser = ArgumentParser()
parser.add_argument('-o', default='./table.md')

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


res = []
bar = tqdm.tqdm(AutoModel.zoo.items())
for item in bar:
    bar.set_description(item[0])
    try:
        out = row(item)
        res.append(out)
    except RuntimeError:
        res.append({'name': item[0],
                    'Parameters': '😥',
                    'Size (MB)': '😥',
                    })
        continue
# res = list(map(row, tqdm.tqdm(AutoModel.zoo.items())))

df = pd.DataFrame.from_records(res)
print(df['name'].values)
print(df)

mk = df.set_index('name', drop=True).to_markdown()

with open(args.o, 'w') as f:
    f.write(mk)
