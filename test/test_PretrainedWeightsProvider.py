import torch
import torch.nn as nn
from glasses.utils.PretrainedWeightsProvider import PretrainedWeightsProvider
from glasses.nn.models.classification.resnet import ResNet
import pytest
from pathlib import Path
import os
def test_PretrainedWeightsProvider():
    provider = PretrainedWeightsProvider()

    with pytest.raises(KeyError):
        provider['does_not_exist']

    resnet18 = ResNet.resnet18()
    resnet18_prov = provider['resnet18']

    for mod, mod_prov in zip(resnet18.modules(), resnet18_prov.modules()):
        assert str(mod) ==  str(mod_prov)

    # this test will take too much time!
    # for key, models in PretrainedWeightsProvider.zoo_models_mapping.items():
    # _, model_def = models
    # model = model_def()
    # model_prov = provider[key]

    # for mod, mod_prov in zip(model.children(), model_prov.children()):
    #     assert str(mod) ==  str(mod_prov)

    


def test_PretrainedWeightsProvider_download_weight():
    provider = PretrainedWeightsProvider()

    save_path = Path('./resnet.pth')
    assert not save_path.exists()
    
    provider.download_weight(PretrainedWeightsProvider.zoo['resnet18'], save_path=save_path)
    assert save_path.exists()

    os.remove(save_path)

    