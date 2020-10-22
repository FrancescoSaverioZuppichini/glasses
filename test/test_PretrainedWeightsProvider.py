import torch
import torch.nn as nn
from glasses.utils.PretrainedWeightsProvider import PretrainedWeightsProvider, Config, pretrained
from glasses.nn.models.classification.resnet import ResNet
import pytest
from torchvision.transforms import Compose
from pathlib import Path
import os

def test_PretrainedWeightsProvider():
    provider = PretrainedWeightsProvider()

    config = Config()
    tr = config.transform

    assert type(tr) is Compose

    resnet18_state = ResNet.resnet18().state_dict()
    resnet18_prov_state = provider['resnet18']

    for mod, mod_prov in zip(resnet18_state.keys(), resnet18_prov_state.keys()):
        assert str(mod) == str(mod_prov)



# need to find another place to host the weights!
# def test_PretrainedWeightsProvider_download_weight():
#     provider = PretrainedWeightsProvider()

#     save_path = Path('/tmp/resnet18.pt')
#     assert not save_path.exists()
    
#     provider = PretrainedWeightsProvider()
#     provider.download_weight(PretrainedWeightsProvider.BASE_URL + '/resnet18.pt', save_path=save_path)
#     assert save_path.exists()

#     os.remove(save_path)

def test_pretrained():
    # not really a great test!
    x = torch.rand(1, 3,224,224)
    model_pretrained = ResNet.resnet18(pretrained=True)
    pred = model_pretrained(x)
    assert pred.shape[-1] == 1000


    
    