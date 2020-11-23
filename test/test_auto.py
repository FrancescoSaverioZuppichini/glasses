from glasses import AutoConfig, AutoModel
from glasses.utils.PretrainedWeightsProvider import Config
from glasses.nn import ResNet
import pytest

def test_AutoModel():
    model = AutoModel.from_name('resnet18')
    assert isinstance(model, ResNet)
    model = AutoModel.from_pretrained('resnet18')
    assert isinstance(model, ResNet)
    with pytest.raises(KeyError):
        model = AutoModel.from_name('resnetasddsadas')
    with pytest.raises(KeyError):
        model = AutoModel.from_pretrained('resnetasddsadas')

    AutoModel.models



def test_AutoConfig():
    cfg = AutoConfig.from_name('resnet18')
    assert isinstance(cfg, Config)
    cfg = AutoConfig.from_name('resnetasddsadas')

