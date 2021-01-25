```python
%load_ext autoreload
%autoreload 2
```

# Glasses 😎

![alt](https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/background.png?raw=true)

[![codecov](https://codecov.io/gh/FrancescoSaverioZuppichini/glasses/branch/develop/graph/badge.svg)](https://codecov.io/gh/FrancescoSaverioZuppichini/glasses)

Compact, concise and customizable 
deep learning computer vision library

**So far I have the [following](#pretrained-models) pretrainde weights. I am working on porting more. They are hosted on GitHub if < 100MB and on AWS (thaks to Francis Ukpeh) if > 100MB.***

Doc is [here](https://francescosaveriozuppichini.github.io/glasses/index.html)

## TL;TR

This library has

- human readable code, no *research code*
- common component are shared across [models](#Models)
- [same APIs](#classification) for all models (you learn them once and they are always the same)
- clear and easy to use model constomization (see [here](#block))
- [classification](#classification) and [segmentation](#segmentation) 
- emoji in the name ;)

Architectures implemented so far:

- [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2010.11929.pdf)
- [Vision Transformer -  An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale](https://arxiv.org/pdf/2010.11929.pdf)
- [ResNeSt: Split-Attention Networks](https://arxiv.org/abs/2004.08955) 
- [AlexNet-  ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [DenseNet - Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- [EfficientNet - EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [EfficientNetLite - Higher accuracy on vision models with EfficientNet-Lite](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html)
- [FishNet - FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction
](https://arxiv.org/abs/1901.03495)
- [MobileNet - MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)
- [RegNet - Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
- [ResNet - Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [ResNetD - Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)
- [ResNetXt - Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)
- [SEResNet - Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
- [VGG - Very Deep Convolutional Networks For Large-scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
- [WideResNet - Wide Residual Networks](https://arxiv.org/pdf/1605.07146.pdf)
- [FPN - Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
- [PFPN - Panoptic Feature Pyramid Networks](https://arxiv.org/pdf/1901.02446.pdf)
- [UNet - U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Squeeze and Excitation - Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
- [ECA - ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/pdf/1910.03151.pdf)

## Installation

You can install `glasses` using pip by running

```
pip install git+https://github.com/FrancescoSaverioZuppichini/glasses
```

### Motivations

Almost all existing implementations of the most famous model are written with very bad coding practices, what today is called *research code*. I struggled myself to understand some of the implementations that in the end were just a few lines of code. 

Most of them are missing a global structure, they used tons of code repetition, they are not easily customizable and not tested. Since I do computer vision for living, so I needed a way to make my life easier.

## Getting started

The API are shared across **all** models!


```python
import torch
from glasses.models import AutoModel, AutoConfig
from torch import nn
# load one model
model = AutoModel.from_pretrained('resnet18')
cfg = AutoConfig.from_name('resnet18')
model.summary(device='cpu' ) # thanks to torchsummary
AutoModel.models() # 'resnet18', 'resnet26', 'resnet26d', 'resnet34', 'resnet50', ...
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

![alt](https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/grad_cam.png?raw=true)

## Classification


```python
from glasses.models import ResNet
# change activation
ResNet.resnet18(activation = nn.SELU)
# change number of classes
ResNet.resnet18(n_classes=100)
# freeze only the convolution weights
model = ResNet.resnet18(pretrained=True)
model.freeze(who=model.encoder)
# get the last layer, usuful to hook to it if you want to get the embeeded vector
model.encoder.layers[-1]
# what about resnet with inverted residuals?
from glasses.models.classification.efficientnet import InvertedResidualBlock
ResNet.resnet18(block = InvertedResidualBlock)
```

## Segmentation


```python
from functools import partial
from glasses.models.segmentation.unet import UNet, UNetDecoder
# vanilla Unet
unet = UNet()
# let's change the encoder
unet = UNet.from_encoder(partial(AutoModel.from_name, 'efficientnet_b1'))
# mmm I want more layers in the decoder!
unet = UNet(decoder=partial(UNetDecoder, widths=[256, 128, 64, 32, 16]))
# maybe resnet was better
unet = UNet(encoder=lambda **kwargs: ResNet.resnet26(**kwargs).encoder)
# same API
unet.summary(input_shape=(1,224,224))
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

## Pretrained Models

**I am currently working on the pretrained models and the best way to make them available**

This is a list of all the pretrained models available so far!. They are all trained on *ImageNet*.

I used a `batch_size=64` and a GTX 1080ti to evaluale the models.

|                        |    top1 |    top5 |     time |   batch_size |
|:-----------------------|--------:|--------:|---------:|-------------:|
| efficientnet_b3        | 0.82034 | 0.9603  | 199.599  |           64 |
| regnety_032            | 0.81958 | 0.95964 | 136.518  |           64 |
| deit_small_patch16_224 | 0.81082 | 0.95316 | 132.868  |           64 |
| resnet50d              | 0.80492 | 0.95128 |  97.5827 |           64 |
| cse_resnet50           | 0.80292 | 0.95048 | 108.765  |           64 |
| efficientnet_b2        | 0.80126 | 0.95124 | 127.177  |           64 |
| resnext101_32x8d       | 0.7921  | 0.94556 | 290.38   |           64 |
| wide_resnet101_2       | 0.7891  | 0.94344 | 277.755  |           64 |
| wide_resnet50_2        | 0.78464 | 0.94064 | 201.634  |           64 |
| efficientnet_b1        | 0.7831  | 0.94096 |  98.7143 |           64 |
| resnet152              | 0.7825  | 0.93982 | 186.191  |           64 |
| regnetx_032            | 0.7792  | 0.93996 | 319.558  |           64 |
| resnext50_32x4d        | 0.77628 | 0.9368  | 114.325  |           64 |
| regnety_016            | 0.77604 | 0.93702 |  96.547  |           64 |
| efficientnet_b0        | 0.77332 | 0.93566 |  67.2147 |           64 |
| resnet101              | 0.77314 | 0.93556 | 134.148  |           64 |
| densenet161            | 0.77146 | 0.93602 | 239.388  |           64 |
| resnet34d              | 0.77118 | 0.93418 |  59.9938 |           64 |
| densenet201            | 0.76932 | 0.9339  | 158.514  |           64 |
| regnetx_016            | 0.76684 | 0.9328  |  91.7536 |           64 |
| resnet26d              | 0.766   | 0.93188 |  70.6453 |           64 |
| regnety_008            | 0.76238 | 0.93026 |  54.1286 |           64 |
| resnet50               | 0.76012 | 0.92934 |  89.7976 |           64 |
| densenet169            | 0.75628 | 0.9281  | 127.077  |           64 |
| resnet26               | 0.75394 | 0.92584 |  65.5801 |           64 |
| resnet34               | 0.75096 | 0.92246 |  56.8985 |           64 |
| regnety_006            | 0.75068 | 0.92474 |  55.5611 |           64 |
| regnetx_008            | 0.74788 | 0.92194 |  57.9559 |           64 |
| densenet121            | 0.74472 | 0.91974 | 104.13   |           64 |
| deit_tiny_patch16_224  | 0.7437  | 0.91898 |  66.662  |           64 |
| vgg19_bn               | 0.74216 | 0.91848 | 169.357  |           64 |
| regnety_004            | 0.73766 | 0.91638 |  68.4893 |           64 |
| regnetx_006            | 0.73682 | 0.91568 |  81.4703 |           64 |
| vgg16_bn               | 0.73476 | 0.91536 | 150.317  |           64 |
| vgg19                  | 0.7236  | 0.9085  | 155.851  |           64 |
| regnetx_004            | 0.72298 | 0.90644 |  58.0049 |           64 |
| vgg16                  | 0.71628 | 0.90368 | 135.398  |           64 |
| vgg13_bn               | 0.71618 | 0.9036  | 129.077  |           64 |
| vgg11_bn               | 0.70408 | 0.89724 |  86.9459 |           64 |
| vgg13                  | 0.69984 | 0.89306 | 116.052  |           64 |
| regnety_002            | 0.6998  | 0.89422 |  46.804  |           64 |
| resnet18               | 0.69644 | 0.88982 |  46.2029 |           64 |
| vgg11                  | 0.68872 | 0.88658 |  79.4136 |           64 |
| regnetx_002            | 0.68658 | 0.88244 |  45.9211 |           64 |

Assuming you want to load `efficientnet_b1`:


```python
from glasses.models import EfficientNet, AutoModel, AutoConfig

# load it using AutoModel
model = AutoModel.from_pretrained('efficientnet_b1')
# or from its own class
model = EfficientNet.efficientnet_b1(pretrained=True)
# you may also need to get the correct transformation that must be applied on the input
cfg = AutoConfig.from_name('efficientnet_b1')
transform = cfg.transform
```

    INFO:root:Loaded efficientnet_b1 pretrained weights.
    INFO:root:Loaded efficientnet_b1 pretrained weights.


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
from glasses.models.classification.vgg import VGGBasicBlock
from glasses.models.classification.resnet import ResNetBasicBlock, ResNetBottleneckBlock, ResNetBasicPreActBlock, ResNetBottleneckPreActBlock
from glasses.models.classification.senet import SENetBasicBlock, SENetBottleneckBlock
from glasses.models.classification.resnetxt import ResNetXtBottleNeckBlock
from glasses.models.classification.densenet import DenseBottleNeckBlock
from glasses.models.classification.wide_resnet import WideResNetBottleNeckBlock
from glasses.models.classification.efficientnet import EfficientNetBasicBlock
```

For example, if we want to add Squeeze and Excitation to the resnet bottleneck block, we can just


```python
from glasses.nn.att import SpatialSE
from  glasses.models.classification.resnet import ResNetBottleneckBlock

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
from glasses.models.classification.resnet import ResNetLayer

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
from glasses.models.classification.resnet import ResNetEncoder
# 3 layers, with 32,64,128 features and 1,2,3 block each
ResNetEncoder(
    widths=[32,64,128],
    depths=[1,2,3])

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
from glasses.models.classification.resnet import ResNetEncoder

x = torch.randn(1,3,224,224)
enc = ResNetEncoder()
enc.features # call it once
enc(x)
features = enc.features # now we have all the features from each layer (stage)
[print(f.shape) for f in features]
# torch.Size([1, 64, 112, 112])
# torch.Size([1, 64, 56, 56])
# torch.Size([1, 128, 28, 28])
# torch.Size([1, 256, 14, 14])
```

### Head

Head is the last part of the model, it usually perform the classification


```python
from glasses.models.classification.resnet import ResNetHead


ResNetHead(512, n_classes=1000)
```

### Decoder

The decoder takes the last feature from the `.encoder` and decode it. This is usually done in `segmentation` models, such as Unet.


```python
from glasses.models.segmentation.unet import UNetDecoder
x = torch.randn(1,3,224,224)
enc = ResNetEncoder()
enc.features # call it once
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

😥 = I don't have enough GPU RAM 

| name                   | Parameters   | Size (MB)   |
|:-----------------------|:-------------|:------------|
| resnet18               | 11,689,512   | 44.59       |
| resnet26               | 15,995,176   | 61.02       |
| resnet26d              | 16,014,408   | 61.09       |
| resnet34               | 21,797,672   | 83.15       |
| resnet34d              | 21,816,904   | 83.22       |
| resnet50               | 25,557,032   | 97.49       |
| resnet50d              | 25,576,264   | 97.57       |
| resnet101              | 44,549,160   | 169.94      |
| resnet152              | 60,192,808   | 229.62      |
| resnet200              | 64,673,832   | 246.71      |
| se_resnet18            | 11,776,552   | 44.92       |
| se_resnet34            | 21,954,856   | 83.75       |
| se_resnet50            | 28,071,976   | 107.09      |
| se_resnet101           | 49,292,328   | 188.04      |
| se_resnet152           | 66,770,984   | 254.71      |
| cse_resnet18           | 11,778,592   | 44.93       |
| cse_resnet34           | 21,958,868   | 83.77       |
| cse_resnet50           | 28,088,024   | 107.15      |
| cse_resnet101          | 49,326,872   | 188.17      |
| cse_resnet152          | 66,821,848   | 254.91      |
| resnext50_32x4d        | 25,028,904   | 95.48       |
| resnext101_32x8d       | 88,791,336   | 338.71      |
| resnext101_32x16d      | 194,026,792  | 740.15      |
| resnext101_32x32d      | 468,530,472  | 1787.30     |
| resnext101_32x48d      | 828,411,176  | 3160.14     |
| regnetx_002            | 2,684,792    | 10.24       |
| regnetx_004            | 5,157,512    | 19.67       |
| regnetx_006            | 6,196,040    | 23.64       |
| regnetx_008            | 7,259,656    | 27.69       |
| regnetx_016            | 9,190,136    | 35.06       |
| regnetx_032            | 15,296,552   | 58.35       |
| regnety_002            | 3,162,996    | 12.07       |
| regnety_004            | 4,344,144    | 16.57       |
| regnety_006            | 6,055,160    | 23.10       |
| regnety_008            | 6,263,168    | 23.89       |
| regnety_016            | 11,202,430   | 42.73       |
| regnety_032            | 19,436,338   | 74.14       |
| resnest14d             | 10,611,688   | 40.48       |
| resnest26d             | 17,069,448   | 65.11       |
| resnest50d             | 27,483,240   | 104.84      |
| resnest50d_1s4x24d     | 25,677,000   | 97.95       |
| resnest50d_4s2x40d     | 30,417,592   | 116.03      |
| resnest101e            | 48,275,016   | 184.15      |
| resnest200e            | 70,201,544   | 267.80      |
| resnest269e            | 7,551,112    | 28.81       |
| wide_resnet50_2        | 68,883,240   | 262.77      |
| wide_resnet101_2       | 126,886,696  | 484.03      |
| densenet121            | 7,978,856    | 30.44       |
| densenet169            | 14,149,480   | 53.98       |
| densenet201            | 20,013,928   | 76.35       |
| densenet161            | 28,681,000   | 109.41      |
| fishnet99              | 16,630,312   | 63.44       |
| fishnet150             | 24,960,808   | 95.22       |
| vgg11                  | 132,863,336  | 506.83      |
| vgg13                  | 133,047,848  | 507.54      |
| vgg16                  | 138,357,544  | 527.79      |
| vgg19                  | 143,667,240  | 548.05      |
| vgg11_bn               | 132,868,840  | 506.85      |
| vgg13_bn               | 133,053,736  | 507.56      |
| vgg16_bn               | 138,365,992  | 527.82      |
| vgg19_bn               | 143,678,248  | 548.09      |
| efficientnet_b0        | 5,288,548    | 20.17       |
| efficientnet_b1        | 7,794,184    | 29.73       |
| efficientnet_b2        | 9,109,994    | 34.75       |
| efficientnet_b3        | 12,233,232   | 46.67       |
| efficientnet_b4        | 19,341,616   | 73.78       |
| efficientnet_b5        | 30,389,784   | 115.93      |
| efficientnet_b6        | 43,040,704   | 164.19      |
| efficientnet_b7        | 66,347,960   | 253.10      |
| efficientnet_b8        | 😥           | 😥          |
| efficientnet_l2        | 😥           | 😥          |
| efficientnet_lite0     | 4,652,008    | 17.75       |
| efficientnet_lite1     | 5,416,680    | 20.66       |
| efficientnet_lite2     | 6,092,072    | 23.24       |
| efficientnet_lite3     | 8,197,096    | 31.27       |
| efficientnet_lite4     | 13,006,568   | 49.62       |
| vit_small_patch16_224  | 48,602,344   | 185.40      |
| vit_base_patch16_224   | 86,415,592   | 329.65      |
| vit_base_patch16_384   | 86,415,592   | 329.65      |
| vit_base_patch32_384   | 88,185,064   | 336.40      |
| vit_huge_patch16_224   | 631,823,080  | 2410.21     |
| vit_huge_patch32_384   | 634,772,200  | 2421.46     |
| vit_large_patch16_224  | 304,123,880  | 1160.14     |
| vit_large_patch16_384  | 304,123,880  | 1160.14     |
| vit_large_patch32_384  | 306,483,176  | 1169.14     |
| deit_tiny_patch16_224  | 5,872,400    | 22.40       |
| deit_small_patch16_224 | 22,359,632   | 85.30       |
| deit_base_patch16_224  | 87,184,592   | 332.58      |
| mobilenetv2            | 3,504,872    | 13.37       |
| unet                   | 23,202,530   | 88.51       |
| deit_base_patch16_384  | 87,184,592   | 332.58      |

## Credits

Most of the weights were trained by other people and adapted to glasses. It is worth cite

- [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- [torchvision](hhttps://github.com/pytorch/vision)

