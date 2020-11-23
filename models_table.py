import torch
import pandas as pd
from argparse import ArgumentParser
from torchsummary import summary
from glasses import AutoModel

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
    input_size = (1, 224, 224) if key == 'unet' else (3, 224, 224)
    total_params, _, param_size, total_size = summary(
        model.to(device), input_size)

    del model

    return {
        'name': key,
        'Parameters': f"{total_params.item():,}",
        'Size (MB)': f"{param_size.item():.2f}",
        # 'Total Size (MB)': int(total_size.item())
    }


res = list(map(row, AutoModel.zoo.items()))

df = pd.DataFrame.from_records(res)
print(df['name'].values)
print(df)

mk = df.set_index('name', drop=True).to_markdown()

with open(args.o, 'w') as f:
    f.write(mk)
