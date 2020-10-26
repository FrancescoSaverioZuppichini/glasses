import torch
from glasses.nn.models.classification.vgg import VGG
from glasses.utils.PretrainedWeightsProvider import PretrainedWeightsProvider


def test_vgg():
    x = torch.rand(1, 3, 224, 224)
    model = VGG().eval()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = VGG().vgg11()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = VGG().vgg13()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = VGG().vgg16()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = VGG().vgg19()
    pred = model(x)
    assert pred.shape[-1] == 1000

    # model = VGG().vgg11()
    # model.load_state_dict(provider['vgg11'])
    # pred = model(x)
    # assert pred.shape[-1] == 1000


    model = VGG().vgg11_bn()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = VGG().vgg13_bn()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = VGG().vgg16_bn()
    pred = model(x)
    assert pred.shape[-1] == 1000

    model = VGG().vgg19_bn()
    pred = model(x)
    assert pred.shape[-1] == 1000


    model = VGG().vgg13_bn()
    pred = model(x)
    assert pred.shape[-1] == 1000
