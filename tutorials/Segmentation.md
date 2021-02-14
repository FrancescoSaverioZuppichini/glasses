# Segmentation
### Preambula
To get started you need to install glasses, this can be done through `pip`

```bash
pip install git+https://github.com/FrancescoSaverioZuppichini/glasses
```

## Segmentation

Segmentation models can be found in `glasses.models.segmentation`. Easily


```python
import torch
from glasses.models.segmentation import UNet

x = torch.randn((1,1, 384, 384))
model = UNet(n_classes=2)
out = model(x)
out.shape
```




    torch.Size([1, 2, 384, 384])



### Change Encoder

 In glasses you can on the fly change the encoder of each segmentation model. Each segmentation model inherits from `SegmentationModule` and expects a `Encoder` instance. 
 
All glasses classification models are composed of an **encoder** and a **head**, thus changing the encoder is as easy as pass it as a parameter


```python
from glasses.models.classification.resnet import ResNetEncoder

x = torch.randn((1,1, 384, 384))
model = UNet(encoder=ResNetEncoder, n_classes=2)
out = model(x)
out.shape
```




    torch.Size([1, 2, 192, 192])



Notice how the output is twice as small as in the standard u-net, this is why resnet has 4 stages, a.k.a four layers that reduce by half the input resolution. To match the correct input shape we have to upsample one more. In other words, we need to increase the widths of the net decoder. 

Similar to classification models, each segmentation model is composed by three sub-modules: **encoder**, **decoder** and **head**. So, we can easily compose them to create any custom model.


```python
from glasses.models.segmentation.unet import UNetDecoder
from functools import partial

x = torch.randn((1,1, 384, 384))
model = UNet(encoder=ResNetEncoder, decoder=partial(UNetDecoder, widths=[512, 256, 128, 64, 32]), n_classes=2)
out = model(x)
out.shape
```




    torch.Size([1, 2, 384, 384])



We used `partial` to change the `widths` parameter of the decoder to match the encoder's stages. Each segmentation model has the `.from_encoder` method that takes a model as input and automatically the model with that model's encoder.


```python
from glasses.models import AutoModel

x = torch.randn((1,1, 384, 384))
model = UNet.from_encoder(model=partial(AutoModel.from_name, 'efficientnet_b1'), n_classes=2)
out = model(x)
out.shape
```




    torch.Size([1, 2, 192, 192])



### Pretrained encoders
Easily, we can pass a pretrained network using `AutoModel`. In this case, pretrained models on ImageNet expects an input with 3 channels


```python
from glasses.models import AutoModel

x = torch.randn((1,3, 384, 384))
model = UNet.from_encoder(model=partial(AutoModel.from_pretrained, 'efficientnet_b1'), in_channels=3, n_classes=2)
out = model(x)
out.shape
```

    INFO:root:Loaded efficientnet_b1 pretrained weights.





    torch.Size([1, 2, 192, 192])



What if we would like to use an input with different channels than 3? We need to replace the stem. 

**I [am working](https://github.com/FrancescoSaverioZuppichini/glasses/issues/179) on a way to load only a specific subset of weights, so we can directly create a model with a different stem but with all the rest of the weights pretrained**


```python
from glasses.models.classification.resnet import ResNetStem
    
def get_encoder(*args, **kwargs):
    model = AutoModel.from_pretrained('resnet50')
    # replace the stem
    model.encoder.stem = ResNetStem(1, model.encoder.start_features)
    return model.encoder
    
x = torch.randn((1,1, 384, 384))
model = UNet(encoder=get_encoder, in_channels=1, n_classes=2)
out = model(x)
out.shape
```

    INFO:root:Loaded resnet50 pretrained weights.





    torch.Size([1, 2, 192, 192])



The APIs are shared from all segmentation models. For example, we can also import PFPN ([Panoptic Feature Pyramid Networks](https://arxiv.org/pdf/1901.02446.pdf)) and keep the same code


```python
from glasses.models import AutoModel
from glasses.models.segmentation import PFPN

x = torch.randn((1,1, 384, 384))
model = PFPN.from_encoder(model=partial(AutoModel.from_name, 'efficientnet_b1'), n_classes=2)
out = model(x)
out.shape
```




    torch.Size([1, 2, 384, 384])



In this case the output always match the input, this is due to how PFPN works.
