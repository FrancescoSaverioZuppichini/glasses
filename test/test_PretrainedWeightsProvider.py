import torch
import torch.nn as nn
from glasses.utils.weights.PretrainedWeightsProvider import (
    PretrainedWeightsProvider,
    pretrained,
    load_pretrained_model,
)
from glasses.models.AutoTransform import Transform
import pytest
import os
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
from requests.exceptions import HTTPError


class Dummy(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3, 32, kernel_size=3))

    @classmethod
    @pretrained()
    def dummy(cls, *args, **kwargs):
        return cls()


def test_PretrainedWeightsProvider():
    provider = PretrainedWeightsProvider()
    dummy_state = Dummy().state_dict()
    dummy_prov_state = provider["dummy"]

    for mod, mod_prov in zip(dummy_state.keys(), dummy_prov_state.keys()):
        assert str(mod) == str(mod_prov)

    model = Dummy.dummy(pretrained=True)
    assert type(model) is Dummy
    x = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
    tr = Transform(mean=1, std=1, input_size=224, resize=256, interpolation="bilinear")
    x_ = tr(x)
    assert x_.shape[-1] == 224
    assert x_.shape[-2] == 224

    assert tr.transforms[0].interpolation == InterpolationMode.BILINEAR

    tr = Transform(mean=1, std=1, input_size=224, resize=256, interpolation="bicubic")

    assert tr.transforms[0].interpolation == InterpolationMode.BICUBIC

    with pytest.raises(HTTPError):
        tmp = provider["foo"]


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
