import torch
import torch.nn as nn
from glasses.utils.PretrainedWeightsProvider import (
    PretrainedWeightsProvider,
    Config,
    GoogleDriveUrlHandler,
    BasicUrlHandler,
    pretrained,
    load_pretrained_model,
)
import pytest
from pathlib import Path
import os
from pytest import raises
from torchvision.transforms import InterpolationMode


PretrainedWeightsProvider.weights_zoo["dummy"] = BasicUrlHandler(
    "https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/dummy.pth?raw=true"
)
PretrainedWeightsProvider.weights_zoo["dummy1"] = BasicUrlHandler(
    "https://github.com/FrancescoSaverioZuppichini/glasses-weights/blob/main/dummy.pth?raw=true"
)


class Dummy(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3, 32, kernel_size=3))

    @classmethod
    @pretrained()
    def dummy(cls, *args, **kwargs):
        return cls()

    @classmethod
    @pretrained(name="dummy1")
    def dummy1(cls, *args, **kwargs):
        return cls()


def test_PretrainedWeightsProvider():
    google_handler = GoogleDriveUrlHandler(
        "https://docs.google.com/uc?export=download",
        file_id="19wLg526wenvhJMPSLPYMlnMgCS6n6jVA",
    )
    save_path = Path("./test.jpg")
    google_handler(save_path=Path("./test.jpg"))
    assert save_path.exists()

    provider = PretrainedWeightsProvider(BASE_DIR=Path("."))
    with pytest.raises(KeyError):
        provider["IDontExists"]

    dummy_state = Dummy().state_dict()
    dummy_prov_state = provider["dummy"]

    for mod, mod_prov in zip(dummy_state.keys(), dummy_prov_state.keys()):
        assert str(mod) == str(mod_prov)

    model = Dummy.dummy(pretrained=True)
    assert type(model) is Dummy

    model = Dummy.dummy1(pretrained=True)

    assert type(model) is Dummy

    cfg = Config(interpolation="bilinear")

    assert cfg.transform.transforms[0].interpolation == InterpolationMode.BILINEAR

    cfg = Config(interpolation="bicubic")

    assert cfg.transform.transforms[0].interpolation == InterpolationMode.BICUBIC


def test_pretrained():
    def get_params():
        dummy = Dummy()
        temp = Dummy()
        nn.init.zeros_(temp[0].weight)

        state_dict = dummy.state_dict()
        new_state_dict = temp.state_dict()

        return dummy, state_dict, new_state_dict

    model, state_dict, new_state_dict = get_params()

    load_pretrained_model(model, new_state_dict, excluding=None)

    assert torch.equal(model[0].weight, torch.zeros(model[0].weight.shape))

    model, state_dict, new_state_dict = get_params()

    load_pretrained_model(model, state_dict, excluding=None)

    assert not torch.equal(model[0].weight, torch.zeros(model[0].weight.shape))

    model, state_dict, new_state_dict = get_params()

    load_pretrained_model(model, new_state_dict, excluding=lambda x: x[0])

    assert not torch.equal(model[0].weight, torch.zeros(model[0].weight.shape))

    with pytest.raises(AttributeError):
        load_pretrained_model(model, new_state_dict, excluding=lambda x: x.no_exist)
