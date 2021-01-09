# Transfer Learning

### Preambula
To get started you need to install glasses, this can be done through `pip`

```bash
pip install git+https://github.com/FrancescoSaverioZuppichini/glasses
```

## Transfer Learning

Train a deep convolutional neural network may take a lot of time, **transfer learning**, as the name suggests, uses models already trained on a huge image dataset, such as ImageNet, to speed up the learning procedure. 

Even if your dataset may be different than ImageNet, the pre-trained models have learned useful weights that can be easily adapt to your new dataset.

### Loading a Model

You can use `AutoModel` and `AutoConfig` to load your model and your preprocessing function. In this tutorial, we are going to use `resnet34`.


```python
from glasses.models import AutoModel, AutoConfig

resnet34 = AutoModel.from_pretrained('resnet34') 
cfg = AutoConfig.from_name('resnet34')
```

You can also call `.summary()` to see your models parameters


```python
resnet34.summary()
```

`AutoConfig` returns the correct configuration for a specific model. This is crucial because you need to properly preprocess your input in the same way it was done when the model was originally trained. `cfg` returns a `Config` object that contains the correct PyTorch transformation. 


```python
tr = cfg.transform
tr
```




    Compose(
        Resize(size=256, interpolation=PIL.Image.BILINEAR)
        CenterCrop(size=(224, 224))
        ToTensor()
        Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
    )



A list of available models can be obtained using `AutoModel.models()`

### Freeze model layers and replace the classification head

Cool, we have our model. Now we need to **freeze** the convolution layers and change the classification head. In glasses, each classification model is composed by a `Encoder` (where the convs are) and a `Head` (usually a linear layer) that performs the final classification. Each `Encoder` has the `.widths` field that tells the number of output features at each layer.


```python
from glasses.models.classification.resnet import ResNetHead

resnet34.freeze()
# you can also freeze a specific layer e.g. resnet34.freeze(who=resnet34.encoder.layers[0])
# head will need to know how many features we are passing into
resnet34.head = ResNetHead(in_features=resnet34.encoder.widths[-1], n_classes=2)
# just to show you
resnet34.encoder.widths
```




    [64, 128, 256, 512]



Just to be sure :)


```python
# no grad in the encoder
for param in resnet34.encoder.parameters():
    assert not param.requires_grad
# grad in the head
for param in resnet34.head.parameters():
    assert param.requires_grad
```

Now your model is ready to train it!
