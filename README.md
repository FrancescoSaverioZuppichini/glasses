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
pip install glasses
```

### Motivation

All the existing implementation of the most famous model are written with very bad coding practices, what today is called *research code*. I struggled myself to understand some of the implementation that in the end were just few lines of code. 

Most of them are missing a global structure, they used tons of code repetition, they are not easily customizable and not tested. Since I do computer vision for living, so I needed a way to make my life easier.

## Getting started

The API are shared across **all** models!


```python
import torch
from glasses.nn.models import *
from torch import nn

model = ResNet.resnet18(pretrained=True)
model.summary() #thanks to torchsummary
# change activation
ResNet.resnet18(activation = nn.SELU)
# change number of classes
ResNet.resnet18(n_classes=100)
# freeze only the convolution weights
model = ResNet.resnet18(pretrained=True)
model.freeze(who=model.encoder)
# get the last layer, usuful to hook to it if you want to get the embeeded vector
model.encoder.blocks[-1]
# what about resnet with inverted residuals?
from glasses.nn.models.classification.mobilenet import InvertedResidualBlock
ResNet.resnet18(block = InvertedResidualBlock)
```


```python
# change the decoder part
model = ResNet.resnet18(pretrained=True)
my_decoder = nn.Sequential(
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(model.encoder.widths[-1], 512),
    nn.Dropout(0.2),
    nn.ReLU(),
    nn.Linear(512, 1000))

model.decoder = my_decoder

x = torch.rand((1,3,224,224))
model(x).shape #torch.Size([1, 1000])
```

## Deep Customization

All models is composed by 4 parts:
- `Block`
- `Layer`
- `Encoder`
- `Decoder`

### Block

Each model has its building block, they are noted by `*Block`. In each block, all the weights are in the `.block` field. This makes it very easy to customize one specific model. 


```python
from glasses.nn.models.classification.vgg import VGGBasicBlock
from glasses.nn.models.classification.resnet import ResNetBasicBlock, ResNetBottleneckBlock, ResNetBasicPreActBlock, ResNetBottleneckPreActBlock
from glasses.nn.models.classification.senet import SENetBasicBlock, SENetBottleneckBlock
from glasses.nn.models.classification.resnetxt import ResNetXtBottleNeckBlock
from glasses.nn.models.classification.densenet import DenseBottleNeckBlock
from glasses.nn.models.classification.wide_resnet import WideResNetBottleNeckBlock
from glasses.nn.models.classification.mobilenet import MobileNetBasicBlock
from glasses.nn.models.classification.efficientnet import EfficientNetBasicBlock
```

For example, if we want to add Squeeze and Excitation to the resnet bottleneck block, we can just


```python
from glasses.nn.models.classification.se import SpatialSE
from  glasses.nn.models.classification.resnet import ResNetBottleneckBlock

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
model = VGG.vgg13(block=SEResNetBottleneckBlock)
model.summary()
```

Some specific model can require additional parameter to the block, for example `MobileNetV2` also required a `expansion` parameter so our `SEResNetBottleneckBlock` won't work. 

### Layer

A `Layer` is a collection of blocks, it is used to stack multiple blocks together following some logic. For example, `ResNetLayer`


```python
from glasses.nn.models.classification.resnet import ResNetLayer

ResNetLayer(64, 128, n=2)
```

### Encoder

The encoder is what encoders a vector, so the convolution layers. It has always two very important parameters.

- widths
- depths


**widths** is the wide at each layer, so how much features there are
**depths** is the depth at each layer, so how many blocks there are

For example, `ResNetEncoder` will creates multiple `ResNetLayer` based on the len of `widths` and `depths`. Let's see some example.


```python
from glasses.nn.models.classification.resnet import ResNetEncoder
# 3 layers, with 32,64,128 features and 1,2,3 block each
ResNetEncoder(
    widths=[32,64,128],
    depths=[1,2,3])

```

**Remember** each model has always a `.decoder` field


```python
from glasses.nn.models import ResNet

model = ResNet.resnet18()
model.encoder.widths[-1]
```

The encoder knows the number of output features, you can access them by

### Decoder

The decoder takes the last feature from the `.encoder` and decode it. Usually it is just a linear layer. The `ResNetDecoder` looks like


```python
from glasses.nn.models.classification.resnet import ResNetDecoder


ResNetDecoder(512, n_classes=1000)
```

**This object oriented structure allows to reuse most of the code across the models**

### Models

The models so far


| name             | Parameters   |   Size (MB) |
|:-----------------|:-------------|------------:|
| resnet18         | 11,689,512   |       44.59 |
| resnet26         | 15,995,176   |       61.02 |
| resnet34         | 21,797,672   |       83.15 |
| resnet50         | 25,557,032   |       97.49 |
| resnet101        | 44,549,160   |      169.94 |
| resnet152        | 60,192,808   |      229.62 |
| resnet200        | 64,673,832   |      246.71 |
| resnext50_32x4d  | 25,028,904   |       95.48 |
| resnext101_32x8d | 88,791,336   |      338.71 |
| wide_resnet50_2  | 68,883,240   |      262.77 |
| wide_resnet101_2 | 126,886,696  |      484.03 |
| se_resnet18      | 11,776,552   |       44.92 |
| se_resnet34      | 21,954,856   |       83.75 |
| se_resnet50      | 28,071,976   |      107.09 |
| se_resnet101     | 49,292,328   |      188.04 |
| se_resnet152     | 66,770,984   |      254.71 |
| densenet121      | 7,978,856    |       30.44 |
| densenet161      | 28,681,000   |      109.41 |
| densenet169      | 14,149,480   |       53.98 |
| densenet201      | 20,013,928   |       76.35 |
| MobileNetV2      | 3,504,872    |       13.37 |
| fishnet99        | 16,630,312   |       63.44 |
| fishnet150       | 24,960,808   |       95.22 |
| efficientnet_b0  | 5,288,548    |       20.17 |
| efficientnet_b1  | 7,794,184    |       29.73 |
| efficientnet_b2  | 9,109,994    |       34.75 |
| efficientnet_b3  | 12,233,232   |       46.67 |
| efficientnet_b4  | 19,341,616   |       73.78 |
| efficientnet_b5  | 30,389,784   |      115.93 |
| efficientnet_b6  | 43,040,704   |      164.19 |
| efficientnet_b7  | 66,347,960   |      253.1  |
| efficientnet_b8  | 87,413,142   |      333.45 |
| efficientnet_l2  | 480,309,308  |     1832.23 |

## Credits

Most of the weights were trained by other people and adapted to glasses. It is worth cite

- [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- [torchvision](hhttps://github.com/pytorch/vision)

