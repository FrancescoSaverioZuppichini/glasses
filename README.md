# Glasses 😎

![alt](https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/background.png?raw=true)

[![codecov](https://codecov.io/gh/FrancescoSaverioZuppichini/glasses/branch/develop/graph/badge.svg)](https://codecov.io/gh/FrancescoSaverioZuppichini/glasses)

Compact, concise and customizable 
deep learning computer vision library 

**This is a early beta, code will change and pretrained weights are not available (I need to find a place to store them online, any advice?)**

Doc is [here](https://francescosaveriozuppichini.github.io/glasses/index.html)

## Installation

You can install `glasses` using pip by runing

```
pip install git+https://github.com/FrancescoSaverioZuppichini/glasses
```

### Motivation

All the existing implementation of the most famous model are writting with very bad cod practice, what today is called *research code*. I struggled myself to understand some of the implementation that in the end were just few lines of code. 

Most of them are missing a global structure, they used tons of code repetition, they are not easily customizable and not tested. Since I do computer vision for living, so I needed a way to make my life easier.

## Getting started


```python
import torch
from glasses.nn.models import *
from torch import nn

model = ResNet.resnet18(pretrained=True)
model.summary() #thanks to torchsummary
```

    INFO:root:Loaded resnet18 pretrained weights.


    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
             Conv2dPad-1         [-1, 64, 112, 112]           9,408
           BatchNorm2d-2         [-1, 64, 112, 112]             128
                  ReLU-3         [-1, 64, 112, 112]               0
             MaxPool2d-4           [-1, 64, 56, 56]               0
             Conv2dPad-5           [-1, 64, 56, 56]          36,864
           BatchNorm2d-6           [-1, 64, 56, 56]             128
                  ReLU-7           [-1, 64, 56, 56]               0
             Conv2dPad-8           [-1, 64, 56, 56]          36,864
           BatchNorm2d-9           [-1, 64, 56, 56]             128
             Identity-10           [-1, 64, 56, 56]               0
                 ReLU-11           [-1, 64, 56, 56]               0
            Conv2dPad-12           [-1, 64, 56, 56]          36,864
          BatchNorm2d-13           [-1, 64, 56, 56]             128
                 ReLU-14           [-1, 64, 56, 56]               0
            Conv2dPad-15           [-1, 64, 56, 56]          36,864
          BatchNorm2d-16           [-1, 64, 56, 56]             128
             Identity-17           [-1, 64, 56, 56]               0
                 ReLU-18           [-1, 64, 56, 56]               0
          ResNetLayer-19           [-1, 64, 56, 56]               0
            Conv2dPad-20          [-1, 128, 28, 28]          73,728
          BatchNorm2d-21          [-1, 128, 28, 28]             256
                 ReLU-22          [-1, 128, 28, 28]               0
            Conv2dPad-23          [-1, 128, 28, 28]         147,456
          BatchNorm2d-24          [-1, 128, 28, 28]             256
            Conv2dPad-25          [-1, 128, 28, 28]           8,192
          BatchNorm2d-26          [-1, 128, 28, 28]             256
        ResNetShorcut-27          [-1, 128, 28, 28]               0
                 ReLU-28          [-1, 128, 28, 28]               0
            Conv2dPad-29          [-1, 128, 28, 28]         147,456
          BatchNorm2d-30          [-1, 128, 28, 28]             256
                 ReLU-31          [-1, 128, 28, 28]               0
            Conv2dPad-32          [-1, 128, 28, 28]         147,456
          BatchNorm2d-33          [-1, 128, 28, 28]             256
             Identity-34          [-1, 128, 28, 28]               0
                 ReLU-35          [-1, 128, 28, 28]               0
          ResNetLayer-36          [-1, 128, 28, 28]               0
            Conv2dPad-37          [-1, 256, 14, 14]         294,912
          BatchNorm2d-38          [-1, 256, 14, 14]             512
                 ReLU-39          [-1, 256, 14, 14]               0
            Conv2dPad-40          [-1, 256, 14, 14]         589,824
          BatchNorm2d-41          [-1, 256, 14, 14]             512
            Conv2dPad-42          [-1, 256, 14, 14]          32,768
          BatchNorm2d-43          [-1, 256, 14, 14]             512
        ResNetShorcut-44          [-1, 256, 14, 14]               0
                 ReLU-45          [-1, 256, 14, 14]               0
            Conv2dPad-46          [-1, 256, 14, 14]         589,824
          BatchNorm2d-47          [-1, 256, 14, 14]             512
                 ReLU-48          [-1, 256, 14, 14]               0
            Conv2dPad-49          [-1, 256, 14, 14]         589,824
          BatchNorm2d-50          [-1, 256, 14, 14]             512
             Identity-51          [-1, 256, 14, 14]               0
                 ReLU-52          [-1, 256, 14, 14]               0
          ResNetLayer-53          [-1, 256, 14, 14]               0
            Conv2dPad-54            [-1, 512, 7, 7]       1,179,648
          BatchNorm2d-55            [-1, 512, 7, 7]           1,024
                 ReLU-56            [-1, 512, 7, 7]               0
            Conv2dPad-57            [-1, 512, 7, 7]       2,359,296
          BatchNorm2d-58            [-1, 512, 7, 7]           1,024
            Conv2dPad-59            [-1, 512, 7, 7]         131,072
          BatchNorm2d-60            [-1, 512, 7, 7]           1,024
        ResNetShorcut-61            [-1, 512, 7, 7]               0
                 ReLU-62            [-1, 512, 7, 7]               0
            Conv2dPad-63            [-1, 512, 7, 7]       2,359,296
          BatchNorm2d-64            [-1, 512, 7, 7]           1,024
                 ReLU-65            [-1, 512, 7, 7]               0
            Conv2dPad-66            [-1, 512, 7, 7]       2,359,296
          BatchNorm2d-67            [-1, 512, 7, 7]           1,024
             Identity-68            [-1, 512, 7, 7]               0
                 ReLU-69            [-1, 512, 7, 7]               0
          ResNetLayer-70            [-1, 512, 7, 7]               0
        ResNetEncoder-71            [-1, 512, 7, 7]               0
    AdaptiveAvgPool2d-72            [-1, 512, 1, 1]               0
              Flatten-73                  [-1, 512]               0
               Linear-74                 [-1, 1000]         513,000
               ResNet-75                 [-1, 1000]               0
    ================================================================
    Total params: 11,689,512
    Trainable params: 11,689,512
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 65.87
    Params size (MB): 44.59
    Estimated Total Size (MB): 111.03
    ----------------------------------------------------------------
    





    (tensor(11689512), tensor(11689512), tensor(44.5919), tensor(111.0330))




```python
# change activation
ResNet.resnet18(activation = nn.SELU)
# change number of classes
ResNet.resnet18(n_classes=100)
# freeze only the convolution weights
model = ResNet.resnet18(pretrained=True)
for param in model.decoder.parameters():
     param.requires_grad = False
```

    INFO:root:Loaded resnet18 pretrained weights.



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

    INFO:root:Loaded resnet18 pretrained weights.





    torch.Size([1, 1000])



## Deep Customization

All models is composed by 4 parts:
- `Block`
- `Layer`
- `Decoder`
- `Encoder`

### Block

Each model has its building block, they are noted by `*Block`. In each block, all the weights are in the `.block` field. This make very easy to customize one specific model. 


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




    SEResNetBottleneckBlock(
      (block): Sequential(
        (conv1): Conv2dPad(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act1): ReLU(inplace=True)
        (conv2): Conv2dPad(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act2): ReLU(inplace=True)
        (conv3): Conv2dPad(16, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (se): SpatialSE(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (att): Sequential(
            (fc1): Linear(in_features=64, out_features=4, bias=False)
            (act1): ReLU(inplace=True)
            (fc2): Linear(in_features=4, out_features=64, bias=False)
            (act2): Sigmoid()
          )
        )
      )
      (shortcut): ResNetShorcut(
        (conv): Conv2dPad(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (act): ReLU(inplace=True)
    )



Then, we can use the class methods to create the new models following the existing architecture blueprint, for example, to create `se_resnet50`


```python
ResNet.resnet50(block=ResNetBottleneckBlock)
```




    ResNet(
      (encoder): ResNetEncoder(
        (gate): Sequential(
          (conv): Conv2dPad(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): ReLU(inplace=True)
          (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        (blocks): ModuleList(
          (0): ResNetLayer(
            (block): Sequential(
              (0): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): ResNetShorcut(
                  (conv): Conv2dPad(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (act): ReLU(inplace=True)
              )
              (1): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): Identity()
                (act): ReLU(inplace=True)
              )
              (2): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): Identity()
                (act): ReLU(inplace=True)
              )
            )
          )
          (1): ResNetLayer(
            (block): Sequential(
              (0): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): ResNetShorcut(
                  (conv): Conv2dPad(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (act): ReLU(inplace=True)
              )
              (1): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): Identity()
                (act): ReLU(inplace=True)
              )
              (2): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): Identity()
                (act): ReLU(inplace=True)
              )
              (3): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): Identity()
                (act): ReLU(inplace=True)
              )
            )
          )
          (2): ResNetLayer(
            (block): Sequential(
              (0): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): ResNetShorcut(
                  (conv): Conv2dPad(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (act): ReLU(inplace=True)
              )
              (1): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): Identity()
                (act): ReLU(inplace=True)
              )
              (2): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): Identity()
                (act): ReLU(inplace=True)
              )
              (3): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): Identity()
                (act): ReLU(inplace=True)
              )
              (4): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): Identity()
                (act): ReLU(inplace=True)
              )
              (5): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): Identity()
                (act): ReLU(inplace=True)
              )
            )
          )
          (3): ResNetLayer(
            (block): Sequential(
              (0): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): ResNetShorcut(
                  (conv): Conv2dPad(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (act): ReLU(inplace=True)
              )
              (1): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): Identity()
                (act): ReLU(inplace=True)
              )
              (2): ResNetBottleneckBlock(
                (block): Sequential(
                  (conv1): Conv2dPad(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act1): ReLU(inplace=True)
                  (conv2): Conv2dPad(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (act2): ReLU(inplace=True)
                  (conv3): Conv2dPad(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                (shortcut): Identity()
                (act): ReLU(inplace=True)
              )
            )
          )
        )
      )
      (decoder): ResNetDecoder(
        (pool): AdaptiveAvgPool2d(output_size=(1, 1))
        (flat): Flatten()
        (fc): Linear(in_features=2048, out_features=1000, bias=True)
      )
    )



The cool thing is each model has the same api, if I want to create a vgg13 with the `ResNetBottleneckBlock` I can just


```python
model = VGG.vgg13(block=SEResNetBottleneckBlock)
model.summary()
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
             Conv2dPad-1         [-1, 16, 224, 224]              48
           BatchNorm2d-2         [-1, 16, 224, 224]              32
                  ReLU-3         [-1, 16, 224, 224]               0
             Conv2dPad-4         [-1, 16, 224, 224]           2,304
           BatchNorm2d-5         [-1, 16, 224, 224]              32
                  ReLU-6         [-1, 16, 224, 224]               0
             Conv2dPad-7         [-1, 64, 224, 224]           1,024
           BatchNorm2d-8         [-1, 64, 224, 224]             128
     AdaptiveAvgPool2d-9             [-1, 64, 1, 1]               0
               Linear-10                    [-1, 4]             256
                 ReLU-11                    [-1, 4]               0
               Linear-12                   [-1, 64]             256
              Sigmoid-13                   [-1, 64]               0
            SpatialSE-14         [-1, 64, 224, 224]               0
            Conv2dPad-15         [-1, 64, 224, 224]             192
          BatchNorm2d-16         [-1, 64, 224, 224]             128
        ResNetShorcut-17         [-1, 64, 224, 224]               0
                 ReLU-18         [-1, 64, 224, 224]               0
            Conv2dPad-19         [-1, 16, 224, 224]           1,024
          BatchNorm2d-20         [-1, 16, 224, 224]              32
                 ReLU-21         [-1, 16, 224, 224]               0
            Conv2dPad-22         [-1, 16, 224, 224]           2,304
          BatchNorm2d-23         [-1, 16, 224, 224]              32
                 ReLU-24         [-1, 16, 224, 224]               0
            Conv2dPad-25         [-1, 64, 224, 224]           1,024
          BatchNorm2d-26         [-1, 64, 224, 224]             128
    AdaptiveAvgPool2d-27             [-1, 64, 1, 1]               0
               Linear-28                    [-1, 4]             256
                 ReLU-29                    [-1, 4]               0
               Linear-30                   [-1, 64]             256
              Sigmoid-31                   [-1, 64]               0
            SpatialSE-32         [-1, 64, 224, 224]               0
             Identity-33         [-1, 64, 224, 224]               0
                 ReLU-34         [-1, 64, 224, 224]               0
            MaxPool2d-35         [-1, 64, 112, 112]               0
             VGGLayer-36         [-1, 64, 112, 112]               0
            Conv2dPad-37         [-1, 32, 112, 112]           2,048
          BatchNorm2d-38         [-1, 32, 112, 112]              64
                 ReLU-39         [-1, 32, 112, 112]               0
            Conv2dPad-40         [-1, 32, 112, 112]           9,216
          BatchNorm2d-41         [-1, 32, 112, 112]              64
                 ReLU-42         [-1, 32, 112, 112]               0
            Conv2dPad-43        [-1, 128, 112, 112]           4,096
          BatchNorm2d-44        [-1, 128, 112, 112]             256
    AdaptiveAvgPool2d-45            [-1, 128, 1, 1]               0
               Linear-46                    [-1, 8]           1,024
                 ReLU-47                    [-1, 8]               0
               Linear-48                  [-1, 128]           1,024
              Sigmoid-49                  [-1, 128]               0
            SpatialSE-50        [-1, 128, 112, 112]               0
            Conv2dPad-51        [-1, 128, 112, 112]           8,192
          BatchNorm2d-52        [-1, 128, 112, 112]             256
        ResNetShorcut-53        [-1, 128, 112, 112]               0
                 ReLU-54        [-1, 128, 112, 112]               0
            Conv2dPad-55         [-1, 32, 112, 112]           4,096
          BatchNorm2d-56         [-1, 32, 112, 112]              64
                 ReLU-57         [-1, 32, 112, 112]               0
            Conv2dPad-58         [-1, 32, 112, 112]           9,216
          BatchNorm2d-59         [-1, 32, 112, 112]              64
                 ReLU-60         [-1, 32, 112, 112]               0
            Conv2dPad-61        [-1, 128, 112, 112]           4,096
          BatchNorm2d-62        [-1, 128, 112, 112]             256
    AdaptiveAvgPool2d-63            [-1, 128, 1, 1]               0
               Linear-64                    [-1, 8]           1,024
                 ReLU-65                    [-1, 8]               0
               Linear-66                  [-1, 128]           1,024
              Sigmoid-67                  [-1, 128]               0
            SpatialSE-68        [-1, 128, 112, 112]               0
             Identity-69        [-1, 128, 112, 112]               0
                 ReLU-70        [-1, 128, 112, 112]               0
            MaxPool2d-71          [-1, 128, 56, 56]               0
             VGGLayer-72          [-1, 128, 56, 56]               0
            Conv2dPad-73           [-1, 64, 56, 56]           8,192
          BatchNorm2d-74           [-1, 64, 56, 56]             128
                 ReLU-75           [-1, 64, 56, 56]               0
            Conv2dPad-76           [-1, 64, 56, 56]          36,864
          BatchNorm2d-77           [-1, 64, 56, 56]             128
                 ReLU-78           [-1, 64, 56, 56]               0
            Conv2dPad-79          [-1, 256, 56, 56]          16,384
          BatchNorm2d-80          [-1, 256, 56, 56]             512
    AdaptiveAvgPool2d-81            [-1, 256, 1, 1]               0
               Linear-82                   [-1, 16]           4,096
                 ReLU-83                   [-1, 16]               0
               Linear-84                  [-1, 256]           4,096
              Sigmoid-85                  [-1, 256]               0
            SpatialSE-86          [-1, 256, 56, 56]               0
            Conv2dPad-87          [-1, 256, 56, 56]          32,768
          BatchNorm2d-88          [-1, 256, 56, 56]             512
        ResNetShorcut-89          [-1, 256, 56, 56]               0
                 ReLU-90          [-1, 256, 56, 56]               0
            Conv2dPad-91           [-1, 64, 56, 56]          16,384
          BatchNorm2d-92           [-1, 64, 56, 56]             128
                 ReLU-93           [-1, 64, 56, 56]               0
            Conv2dPad-94           [-1, 64, 56, 56]          36,864
          BatchNorm2d-95           [-1, 64, 56, 56]             128
                 ReLU-96           [-1, 64, 56, 56]               0
            Conv2dPad-97          [-1, 256, 56, 56]          16,384
          BatchNorm2d-98          [-1, 256, 56, 56]             512
    AdaptiveAvgPool2d-99            [-1, 256, 1, 1]               0
              Linear-100                   [-1, 16]           4,096
                ReLU-101                   [-1, 16]               0
              Linear-102                  [-1, 256]           4,096
             Sigmoid-103                  [-1, 256]               0
           SpatialSE-104          [-1, 256, 56, 56]               0
            Identity-105          [-1, 256, 56, 56]               0
                ReLU-106          [-1, 256, 56, 56]               0
           MaxPool2d-107          [-1, 256, 28, 28]               0
            VGGLayer-108          [-1, 256, 28, 28]               0
           Conv2dPad-109          [-1, 128, 28, 28]          32,768
         BatchNorm2d-110          [-1, 128, 28, 28]             256
                ReLU-111          [-1, 128, 28, 28]               0
           Conv2dPad-112          [-1, 128, 28, 28]         147,456
         BatchNorm2d-113          [-1, 128, 28, 28]             256
                ReLU-114          [-1, 128, 28, 28]               0
           Conv2dPad-115          [-1, 512, 28, 28]          65,536
         BatchNorm2d-116          [-1, 512, 28, 28]           1,024
    AdaptiveAvgPool2d-117            [-1, 512, 1, 1]               0
              Linear-118                   [-1, 32]          16,384
                ReLU-119                   [-1, 32]               0
              Linear-120                  [-1, 512]          16,384
             Sigmoid-121                  [-1, 512]               0
           SpatialSE-122          [-1, 512, 28, 28]               0
           Conv2dPad-123          [-1, 512, 28, 28]         131,072
         BatchNorm2d-124          [-1, 512, 28, 28]           1,024
       ResNetShorcut-125          [-1, 512, 28, 28]               0
                ReLU-126          [-1, 512, 28, 28]               0
           Conv2dPad-127          [-1, 128, 28, 28]          65,536
         BatchNorm2d-128          [-1, 128, 28, 28]             256
                ReLU-129          [-1, 128, 28, 28]               0
           Conv2dPad-130          [-1, 128, 28, 28]         147,456
         BatchNorm2d-131          [-1, 128, 28, 28]             256
                ReLU-132          [-1, 128, 28, 28]               0
           Conv2dPad-133          [-1, 512, 28, 28]          65,536
         BatchNorm2d-134          [-1, 512, 28, 28]           1,024
    AdaptiveAvgPool2d-135            [-1, 512, 1, 1]               0
              Linear-136                   [-1, 32]          16,384
                ReLU-137                   [-1, 32]               0
              Linear-138                  [-1, 512]          16,384
             Sigmoid-139                  [-1, 512]               0
           SpatialSE-140          [-1, 512, 28, 28]               0
            Identity-141          [-1, 512, 28, 28]               0
                ReLU-142          [-1, 512, 28, 28]               0
           MaxPool2d-143          [-1, 512, 14, 14]               0
            VGGLayer-144          [-1, 512, 14, 14]               0
           Conv2dPad-145          [-1, 128, 14, 14]          65,536
         BatchNorm2d-146          [-1, 128, 14, 14]             256
                ReLU-147          [-1, 128, 14, 14]               0
           Conv2dPad-148          [-1, 128, 14, 14]         147,456
         BatchNorm2d-149          [-1, 128, 14, 14]             256
                ReLU-150          [-1, 128, 14, 14]               0
           Conv2dPad-151          [-1, 512, 14, 14]          65,536
         BatchNorm2d-152          [-1, 512, 14, 14]           1,024
    AdaptiveAvgPool2d-153            [-1, 512, 1, 1]               0
              Linear-154                   [-1, 32]          16,384
                ReLU-155                   [-1, 32]               0
              Linear-156                  [-1, 512]          16,384
             Sigmoid-157                  [-1, 512]               0
           SpatialSE-158          [-1, 512, 14, 14]               0
            Identity-159          [-1, 512, 14, 14]               0
                ReLU-160          [-1, 512, 14, 14]               0
           Conv2dPad-161          [-1, 128, 14, 14]          65,536
         BatchNorm2d-162          [-1, 128, 14, 14]             256
                ReLU-163          [-1, 128, 14, 14]               0
           Conv2dPad-164          [-1, 128, 14, 14]         147,456
         BatchNorm2d-165          [-1, 128, 14, 14]             256
                ReLU-166          [-1, 128, 14, 14]               0
           Conv2dPad-167          [-1, 512, 14, 14]          65,536
         BatchNorm2d-168          [-1, 512, 14, 14]           1,024
    AdaptiveAvgPool2d-169            [-1, 512, 1, 1]               0
              Linear-170                   [-1, 32]          16,384
                ReLU-171                   [-1, 32]               0
              Linear-172                  [-1, 512]          16,384
             Sigmoid-173                  [-1, 512]               0
           SpatialSE-174          [-1, 512, 14, 14]               0
            Identity-175          [-1, 512, 14, 14]               0
                ReLU-176          [-1, 512, 14, 14]               0
           MaxPool2d-177            [-1, 512, 7, 7]               0
            VGGLayer-178            [-1, 512, 7, 7]               0
          VGGEncoder-179            [-1, 512, 7, 7]               0
    AdaptiveAvgPool2d-180            [-1, 512, 7, 7]               0
             Flatten-181                [-1, 25088]               0
              Linear-182                 [-1, 4096]     102,764,544
                ReLU-183                 [-1, 4096]               0
             Dropout-184                 [-1, 4096]               0
              Linear-185                 [-1, 4096]      16,781,312
                ReLU-186                 [-1, 4096]               0
             Dropout-187                 [-1, 4096]               0
              Linear-188                 [-1, 1000]       4,097,000
                 VGG-189                 [-1, 1000]               0
    ================================================================
    Total params: 125,231,320
    Trainable params: 125,231,320
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 723.21
    Params size (MB): 477.72
    Estimated Total Size (MB): 1201.51
    ----------------------------------------------------------------
    





    (tensor(125231320), tensor(125231320), tensor(477.7196), tensor(1201.5082))



Some specific models require additional parameter to the block, for example `MobileNetV2` also required a `expansion` parameter so our `SEResNetBottleneckBlock` won't work. 

### Block

Layer is a collection of blocks, it is used to stack multiple block together following some logic. For example, `ResNetLayer`


```python
from glasses.nn.models.classification.resnet import ResNetLayer

ResNetLayer(64, 128, n=2)
```




    ResNetLayer(
      (block): Sequential(
        (0): ResNetBasicBlock(
          (block): Sequential(
            (conv1): Conv2dPad(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act1): ReLU(inplace=True)
            (conv2): Conv2dPad(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): ResNetShorcut(
            (conv): Conv2dPad(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (act): ReLU(inplace=True)
        )
        (1): ResNetBasicBlock(
          (block): Sequential(
            (conv1): Conv2dPad(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act1): ReLU(inplace=True)
            (conv2): Conv2dPad(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (shortcut): Identity()
          (act): ReLU(inplace=True)
        )
      )
    )



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




    ResNetEncoder(
      (gate): Sequential(
        (conv): Conv2dPad(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): ReLU(inplace=True)
        (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
      (blocks): ModuleList(
        (0): ResNetLayer(
          (block): Sequential(
            (0): ResNetBasicBlock(
              (block): Sequential(
                (conv1): Conv2dPad(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act1): ReLU(inplace=True)
                (conv2): Conv2dPad(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (shortcut): ResNetShorcut(
                (conv): Conv2dPad(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (act): ReLU(inplace=True)
            )
          )
        )
        (1): ResNetLayer(
          (block): Sequential(
            (0): ResNetBasicBlock(
              (block): Sequential(
                (conv1): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act1): ReLU(inplace=True)
                (conv2): Conv2dPad(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (shortcut): ResNetShorcut(
                (conv): Conv2dPad(32, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (act): ReLU(inplace=True)
            )
            (1): ResNetBasicBlock(
              (block): Sequential(
                (conv1): Conv2dPad(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act1): ReLU(inplace=True)
                (conv2): Conv2dPad(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (shortcut): Identity()
              (act): ReLU(inplace=True)
            )
          )
        )
        (2): ResNetLayer(
          (block): Sequential(
            (0): ResNetBasicBlock(
              (block): Sequential(
                (conv1): Conv2dPad(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act1): ReLU(inplace=True)
                (conv2): Conv2dPad(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (shortcut): ResNetShorcut(
                (conv): Conv2dPad(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (act): ReLU(inplace=True)
            )
            (1): ResNetBasicBlock(
              (block): Sequential(
                (conv1): Conv2dPad(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act1): ReLU(inplace=True)
                (conv2): Conv2dPad(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (shortcut): Identity()
              (act): ReLU(inplace=True)
            )
            (2): ResNetBasicBlock(
              (block): Sequential(
                (conv1): Conv2dPad(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act1): ReLU(inplace=True)
                (conv2): Conv2dPad(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (shortcut): Identity()
              (act): ReLU(inplace=True)
            )
          )
        )
      )
    )



**Remember** each model has always a `.decoder` field


```python
from glasses.nn.models import ResNet

model = ResNet.resnet18()
model.encoder.widths[-1]
```




    512



The encoder knows the number of output features, you can access them by

### Decoder

The decoder takes the last feature from the `.encoder` and decode it. Usually it is just a linear layer. The `ResNetDecoder` looks like


```python
from glasses.nn.models.classification.resnet import ResNetDecoder


ResNetDecoder(512, n_classes=1000)
```




    ResNetDecoder(
      (pool): AdaptiveAvgPool2d(output_size=(1, 1))
      (flat): Flatten()
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )



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

