from glasses.nn.models import AutoConfig, AutoModel
from glasses.utils.PretrainedWeightsProvider import Config, pretrained, BasicUrlHandler, PretrainedWeightsProvider
from glasses.nn import ResNet
import pytest
from torch import nn

class Dummy(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3, 32, kernel_size=3))

    @classmethod
    @pretrained()
    def dummy(cls, *args, **kwargs):
        return cls()


AutoModel.zoo['dummy'] = Dummy.dummy

PretrainedWeightsProvider.weights_zoo['dummy'] = BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/dummy.pth?raw=true')

def test_AutoModel():
    model = AutoModel.from_name('dummy')
    assert isinstance(model, Dummy)
    model = AutoModel.from_pretrained('dummy')
    assert isinstance(model, Dummy)
    with pytest.raises(KeyError):
        model = AutoModel.from_name('resn')
    with pytest.raises(KeyError):
        model = AutoModel.from_name('resnetasddsadas')
    with pytest.raises(KeyError):
        model = AutoModel.from_pretrained('resnetasddsadas')
    with pytest.raises(KeyError):
        model = AutoModel.from_pretrained('resn')
    assert len(list(AutoModel.models())) > 0



def test_AutoConfig():
    cfg = AutoConfig.from_name('resnet18')
    assert isinstance(cfg, Config)
    cfg = AutoConfig.from_name('resnetasddsadas')
    assert len(list(AutoConfig.names())) > 0
