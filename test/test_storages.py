from glasses.models import AutoModel, AutoTransform
from pathlib import Path
from glasses.utils.weights.storage import HuggingFaceStorage, LocalStorage, Storage
from torch import nn
import pytest

key = "dummy"


class Dummy(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3, 32, kernel_size=3))


model = Dummy()


def test_storage_api():
    class MyStorage(Storage):
        pass

    with pytest.raises(TypeError):
        MyStorage()


def test_hf_storage():
    storage = HuggingFaceStorage()
    state_dict = storage.get(key)
    model.load_state_dict(state_dict)


def test_local_storage():
    storage = LocalStorage(root=Path("/tmp/"))
    storage.put(key, model)
    state_dict = storage.get(key)
    model.load_state_dict(state_dict)
