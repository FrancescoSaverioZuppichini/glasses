```python
%load_ext autoreload
%autoreload 2
```

# Glasses 😎

![alt](https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/background.png?raw=true)

[![codecov](https://codecov.io/gh/FrancescoSaverioZuppichini/glasses/branch/develop/graph/badge.svg)](https://codecov.io/gh/FrancescoSaverioZuppichini/glasses)

Compact, concise and customizable 
deep learning computer vision library

**This is an early beta, code will change and pretrained weights are not available (I need to find a place to store them online, any advice?)**

Doc is [here](https://francescosaveriozuppichini.github.io/glasses/index.html)

## Installation

You can install `glasses` using pip by running

```
pip install git+https://github.com/FrancescoSaverioZuppichini/glasses
```

### Motivation

All the existing implementation of the most famous model are written with very bad coding practices, what today is called *research code*. I struggled myself to understand some of the implementation that in the end were just few lines of code. 

Most of them are missing a global structure, they used tons of code repetition, they are not easily customizable and not tested. Since I do computer vision for living, so I needed a way to make my life easier.

## Getting started

The API are shared across **all** models!


```python
import torch
from glasses import AutoModel, AutoConfig
from torch import nn
# load one model
model = AutoModel.from_pretrained('resnet18')
cfg = AutoConfig.from_name('resnet18')
model.summary(device='cpu') # thanks to torchsummary
```

### Interpretability


```python
import requests
from PIL import Image
from io import BytesIO
from glasses.interpretability import GradCam, SaliencyMap
from torchvision.transforms import Normalize
r = requests.get('https://i.insider.com/5df126b679d7570ad2044f3e?width=700&format=jpeg&auto=webp')
im = Image.open(BytesIO(r.content))
# un normalize when done
postprocessing = Normalize(-cfg.mean / cfg.std, (1.0 / cfg.std))
# apply preprocessing
x =  cfg.transform(im).unsqueeze(0)
_ = model.interpret(x, using=GradCam(), postprocessing=postprocessing).show()
```

## Classification

```python
from glasses.models import ResNet

# change activation
ResNet.resnet18(activation=nn.SELU)
# change number of classes
ResNet.resnet18(n_classes=100)
# freeze only the convolution weights
model = ResNet.resnet18(pretrained=True)
model.freeze(who=model.encoder)
# get the last layer, usuful to hook to it if you want to get the embeeded vector
model.encoder.layers[-1]
# what about resnet with inverted residuals?
from glasses.models.classification import InvertedResidualBlock

ResNet.resnet18(block=InvertedResidualBlock)
```

## Segmentation

```python
from functools import partial
from glasses.models import UNet, UNetDecoder

# vanilla Unet
unet = UNet()
# let's change the encoder
unet = UNet.from_encoder(partial(AutoModel.from_name, 'efficientnet_b1'))
# mmm I want more layers in the decoder!
unet = UNet(decoder=partial(UNetDecoder, widths=[256, 128, 64, 32, 16]))
# maybe resnet was better
unet = UNet(encoder=lambda **kwargs: ResNet.resnet26(**kwargs).encoder)
# same API
unet.summary(input_shape=(1, 224, 224))
```

### More examples


```python
# change the decoder part
model = ResNet.resnet18(pretrained=True)
my_head = nn.Sequential(
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(model.encoder.widths[-1], 512),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.Linear(512, 1000))

model.head = my_head

x = torch.rand((1,3,224,224))
model(x).shape #torch.Size([1, 1000])
```

![alt](https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/grad_cam.png?raw=true)

## Pretrained Models

This is a list of all the pretrained models available so far!. They are all trained on *ImageNet*

|                  |    top1 |    top5 |     time |
|:-----------------|--------:|--------:|---------:|
| efficientnet_b3  | 0.8204  | 0.96044 | 233.535  |
| cse_resnet50     | 0.80236 | 0.9507  | 103.796  |
| efficientnet_b2  | 0.8011  | 0.95118 | 143.739  |
| resnext101_32x8d | 0.79312 | 0.94526 | 332.005  |
| wide_resnet101_2 | 0.78848 | 0.94284 | 234.597  |
| wide_resnet50_2  | 0.78468 | 0.94086 | 146.662  |
| efficientnet_b1  | 0.78338 | 0.94078 | 109.463  |
| resnet152        | 0.78312 | 0.94046 | 207.622  |
| resnext50_32x4d  | 0.77618 | 0.93698 | 135.172  |
| resnet101        | 0.77374 | 0.93546 | 151.992  |
| efficientnet_b0  | 0.77364 | 0.9356  |  74.3195 |
| densenet161      | 0.77138 | 0.9356  | 201.173  |
| densenet201      | 0.76896 | 0.9337  | 143.988  |
| resnet50         | 0.7613  | 0.92862 |  92.408  |
| densenet169      | 0.756   | 0.92806 | 115.986  |
| resnet26         | 0.75292 | 0.9257  |  65.2226 |
| resnet34         | 0.75112 | 0.92288 |  61.9156 |
| densenet121      | 0.74434 | 0.91972 |  95.5099 |
| vgg19_bn         | 0.74218 | 0.91842 | 172.343  |
| vgg16_bn         | 0.7336  | 0.91516 | 152.662  |
| vgg19            | 0.72376 | 0.90876 | 160.982  |
| mobilenet_v2     | 0.71878 | 0.90286 |  53.3237 |
| vgg16            | 0.71592 | 0.90382 | 141.572  |
| vgg13_bn         | 0.71586 | 0.90374 | 129.88   |
| vgg11_bn         | 0.7037  | 0.8981  |  91.5699 |
| vgg13            | 0.69928 | 0.89246 | 119.631  |
| resnet18         | 0.69758 | 0.89078 |  46.7778 |
| vgg11            | 0.6902  | 0.88628 |  84.9438 |

Assuming you want to load `efficientnet_b1`, you can also grab it from its class

```python
from glasses.models import EfficientNet

model = EfficientNet.efficientnet_b1(pretrained=True)
# you may also need to get the correct transformation that must be applied on the input
cfg = AutoConfig.from_name('efficientnet_b1')
transform = cfg.transform
```

In this case, `transform` is 

```
Compose(
    Resize(size=240, interpolation=PIL.Image.BICUBIC)
    CenterCrop(size=(240, 240))
    ToTensor()
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
)
```

## Deep Customization

All models are composed by sharable parts:
- `Block`
- `Layer`
- `Encoder`
- `Head`
- `Decoder`

### Block

Each model has its building block, they are noted by `*Block`. In each block, all the weights are in the `.block` field. This makes it very easy to customize one specific model.

```python
from glasses.models.classification import VGGBasicBlock
from glasses.models.classification import ResNetBasicBlock, ResNetBottleneckBlock, ResNetBasicPreActBlock,
    ResNetBottleneckPreActBlock
from glasses.models.classification import SENetBasicBlock, SENetBottleneckBlock
from glasses.models.classification import ResNetXtBottleNeckBlock
from glasses.models.classification import DenseBottleNeckBlock
from glasses.models.classification import WideResNetBottleNeckBlock
from glasses.models.classification import EfficientNetBasicBlock
```

For example, if we want to add Squeeze and Excitation to the resnet bottleneck block, we can just

```python
from glasses.nn.att import SpatialSE
from glasses.models.classification import ResNetBottleneckBlock


class SEResNetBottleneckBlock(ResNetBottleneckBlock):
    def __init__(self, in_features: int, out_features: int, squeeze: int = 16, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)
        # all the weights are in block, we want to apply se after the weights
        self.block.add_module('se', SpatialSE(out_features, reduction=squeeze))


SEResNetBottleneckBlock(32, 64)
```

Then, we can use the class methods to create the new models following the existing architecture blueprint, for example, to create `se_resnet50`


```python
ResNet.resnet50(block=ResNetBottleneckBlock)
```

The cool thing is each model has the same api, if I want to create a vgg13 with the `ResNetBottleneckBlock` I can just

```python
from glasses.models import VGG

model = VGG.vgg13(block=SEResNetBottleneckBlock)
model.summary()
```

Some specific model can require additional parameter to the block, for example `MobileNetV2` also required a `expansion` parameter so our `SEResNetBottleneckBlock` won't work. 

### Layer

A `Layer` is a collection of blocks, it is used to stack multiple blocks together following some logic. For example, `ResNetLayer`

```python
from glasses.models.classification import ResNetLayer

ResNetLayer(64, 128, depth=2)
```

### Encoder

The encoder is what encoders a vector, so the convolution layers. It has always two very important parameters.

- widths
- depths


**widths** is the wide at each layer, so how much features there are
**depths** is the depth at each layer, so how many blocks there are

For example, `ResNetEncoder` will creates multiple `ResNetLayer` based on the len of `widths` and `depths`. Let's see some example.

```python
from glasses.models.classification import ResNetEncoder

# 3 layers, with 32,64,128 features and 1,2,3 block each
ResNetEncoder(
    widths=[32, 64, 128],
    depths=[1, 2, 3])

```

All encoders are subclass of `Encoder` that allows us to hook on specific stages to get the featuers. All you have to do is first call `.features` to notify the model you want to receive the features, and then pass an input.


```python
enc = ResNetEncoder()
enc.features
enc(torch.randn((1,3,224,224)))
print([f.shape for f in enc.features])
```

**Remember** each model has always a `.decoder` field

```python
from glasses.models import ResNet

model = ResNet.resnet18()
model.encoder.widths[-1]
```

The encoder knows the number of output features, you can access them by

#### Features

Each encoder can return a list of features accessable by the `.features` field. You need to call it once before in order to notify the encoder we wish to also store the features

```python
from glasses.models.classification import ResNetEncoder

x = torch.randn(1, 3, 224, 224)
enc = ResNetEncoder()
enc.features  # call it once
enc(x)
features = enc.features  # now we have all the features from each layer (stage)
[print(f.shape) for f in features]
# torch.Size([1, 64, 112, 112])
# torch.Size([1, 64, 56, 56])
# torch.Size([1, 128, 28, 28])
# torch.Size([1, 256, 14, 14])
```

### Head

Head is the last part of the model, it usually perform the classification

```python
from glasses.models.classification import ResNetHead

ResNetHead(512, n_classes=1000)
```

### Decoder

The decoder takes the last feature from the `.encoder` and decode it. This is usually done in `segmentation` models, such as Unet.

```python
from glasses.models import UNetDecoder

x = torch.randn(1, 3, 224, 224)
enc = ResNetEncoder()
enc.features  # call it once
x = enc(x)
features = enc.features
# we need to tell the decoder the first feature size and the size of the lateral features
dec = UNetDecoder(start_features=enc.widths[-1],
                  lateral_widths=enc.features_widths[::-1])
out = dec(x, features[::-1])
out.shape
```

**This object oriented structure allows to reuse most of the code across the models**

### Models

The models so far

| name               | Parameters   |   Size (MB) |
|:-------------------|:-------------|------------:|
| resnet18           | 11,689,512   |       44.59 |
| resnet26           | 15,995,176   |       61.02 |
| resnet26d          | 16,014,408   |       61.09 |
| resnet34           | 21,797,672   |       83.15 |
| resnet50           | 25,557,032   |       97.49 |
| resnet50d          | 25,576,264   |       97.57 |
| resnet101          | 44,549,160   |      169.94 |
| resnet152          | 60,192,808   |      229.62 |
| resnet200          | 64,673,832   |      246.71 |
| se_resnet18        | 11,776,552   |       44.92 |
| se_resnet34        | 21,954,856   |       83.75 |
| se_resnet50        | 28,071,976   |      107.09 |
| se_resnet101       | 49,292,328   |      188.04 |
| se_resnet152       | 66,770,984   |      254.71 |
| cse_resnet18       | 11,778,592   |       44.93 |
| cse_resnet34       | 21,958,868   |       83.77 |
| cse_resnet50       | 28,088,024   |      107.15 |
| cse_resnet101      | 49,326,872   |      188.17 |
| cse_resnet152      | 66,821,848   |      254.91 |
| resnext50_32x4d    | 25,028,904   |       95.48 |
| resnext101_32x8d   | 88,791,336   |      338.71 |
| resnext101_32x16d  | 194,026,792  |      740.15 |
| resnext101_32x32d  | 468,530,472  |     1787.3  |
| resnext101_32x48d  | 828,411,176  |     3160.14 |
| wide_resnet50_2    | 68,883,240   |      262.77 |
| wide_resnet101_2   | 126,886,696  |      484.03 |
| densenet121        | 7,978,856    |       30.44 |
| densenet169        | 14,149,480   |       53.98 |
| densenet201        | 20,013,928   |       76.35 |
| densenet161        | 28,681,000   |      109.41 |
| fishnet99          | 16,630,312   |       63.44 |
| fishnet150         | 24,960,808   |       95.22 |
| vgg11              | 132,863,336  |      506.83 |
| vgg13              | 133,047,848  |      507.54 |
| vgg16              | 138,357,544  |      527.79 |
| vgg19              | 143,667,240  |      548.05 |
| vgg11_bn           | 132,868,840  |      506.85 |
| vgg13_bn           | 133,053,736  |      507.56 |
| vgg16_bn           | 138,365,992  |      527.82 |
| vgg19_bn           | 143,678,248  |      548.09 |
| efficientnet_b0    | 5,288,548    |       20.17 |
| efficientnet_b1    | 7,794,184    |       29.73 |
| efficientnet_b2    | 9,109,994    |       34.75 |
| efficientnet_b3    | 12,233,232   |       46.67 |
| efficientnet_b4    | 19,341,616   |       73.78 |
| efficientnet_b5    | 30,389,784   |      115.93 |
| efficientnet_b6    | 43,040,704   |      164.19 |
| efficientnet_b7    | 66,347,960   |      253.1  |
| efficientnet_b8    | 87,413,142   |      333.45 |
| efficientnet_l2    | 480,309,308  |     1832.23 |
| efficientnet_lite0 | 4,652,008    |       17.75 |
| efficientnet_lite1 | 5,416,680    |       20.66 |
| efficientnet_lite2 | 6,092,072    |       23.24 |
| efficientnet_lite3 | 8,197,096    |       31.27 |
| efficientnet_lite4 | 13,006,568   |       49.62 |
| mobilenetv2        | 3,504,872    |       13.37 |
| unet               | 23,202,530   |       88.51 |

## Credits

Most of the weights were trained by other people and adapted to glasses. It is worth cite

- [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- [torchvision](hhttps://github.com/pytorch/vision)

