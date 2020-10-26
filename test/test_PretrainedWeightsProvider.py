import torch
import torch.nn as nn
from glasses.utils.PretrainedWeightsProvider import PretrainedWeightsProvider, Config, GoogleDriveUrlHandler    
from glasses.nn.models.classification.resnet import ResNet
import pytest
from pathlib import Path
import os
from PIL import Image

def test_PretrainedWeightsProvider():
    google_handler = GoogleDriveUrlHandler('https://docs.google.com/uc?export=download', file_id='19wLg526wenvhJMPSLPYMlnMgCS6n6jVA')
    save_path = Path('./test.jpg')
    google_handler(save_path=Path('./test.jpg'))
    assert save_path.exists()

    provider = PretrainedWeightsProvider(BASE_DIR=Path('.'))
                        
    # with pytest.raises(KeyError):
    #     provider['does_not_exist']

    resnet18_state = ResNet.resnet18().state_dict()
    resnet18_prov_state = provider['resnet18']

    for mod, mod_prov in zip(resnet18_state.keys(), resnet18_prov_state.keys()):
        assert str(mod) == str(mod_prov)
    
    model = ResNet.resnet18(pretrained=True)

    assert type(model) is ResNet

    cfg = Config(interpolation='bilinear')
    
    assert cfg.transform.transforms[0].interpolation == Image.BILINEAR

    cfg = Config(interpolation='bicubic')

    assert cfg.transform.transforms[0].interpolation == Image.BICUBIC

    # this test will take too much time!
    # for key, models in PretrainedWeightsProvider.zoo_models_mapping.items():
    # _, model_def = models
    # model = model_def()
    # model_prov = provider[key]

    # for mod, mod_prov in zip(model.children(), model_prov.children()):
    #     assert str(mod) ==  str(mod_prov)

