import torch
import torch.nn as nn
from glasses.utils.PretrainedWeightsProvider import PretrainedWeightsProvider, Config, GoogleDriveUrlHandler, BasicUrlHandler, pretrained    
from glasses.models.classification import ResNet
import pytest
from pathlib import Path
import os
from PIL import Image
from pytest import raises


PretrainedWeightsProvider.weights_zoo['dummy'] = BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/dummy.pth?raw=true')
PretrainedWeightsProvider.weights_zoo['dummy1'] = BasicUrlHandler('https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/dummy.pth?raw=true')

class Dummy(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3, 32, kernel_size=3))

    @classmethod
    @pretrained()
    def dummy(cls, *args, **kwargs):
        return cls()

    @classmethod
    @pretrained(name='dummy1')
    def dummy1(cls, *args, **kwargs):
        return cls()   

def test_PretrainedWeightsProvider():
    google_handler = GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id='19wLg526wenvhJMPSLPYMlnMgCS6n6jVA')
    save_path = Path('./test.jpg')
    google_handler(save_path=Path('./test.jpg'))
    assert save_path.exists()

    provider = PretrainedWeightsProvider(BASE_DIR=Path('.'))
    with pytest.raises(KeyError):
        provider['IDontExists']           


    dummy_state = Dummy().state_dict()
    dummy_prov_state = provider['dummy']

    for mod, mod_prov in zip(dummy_state.keys(), dummy_prov_state.keys()):
        assert str(mod) == str(mod_prov)
    
    model = Dummy.dummy(pretrained=True)
    assert type(model) is Dummy

    model = Dummy.dummy1(pretrained=True)

    assert type(model) is Dummy

    cfg = Config(interpolation='bilinear')
    
    assert cfg.transform.transforms[0].interpolation == Image.BILINEAR

    cfg = Config(interpolation='bicubic')

    assert cfg.transform.transforms[0].interpolation == Image.BICUBIC

