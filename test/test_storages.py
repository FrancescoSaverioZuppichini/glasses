import pytest
from glasses.models import AutoModel, AutoTransform
from pathlib import Path
from glasses.utils.weights.storage import HuggingFaceStorage, LocalStorage, Storage
from torch import nn
from fixtures import glasses_path

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


@pytest.mark.parametrize("fmt", ["pth", "foo"])
def test_local_storage(glasses_path: Path, fmt: str):
    storage = LocalStorage(root=glasses_path, fmt=fmt)
    storage.put(key, model)
    state_dict = storage.get(key)
    model.load_state_dict(state_dict)

    assert key in storage
    assert len(storage.models) == 1
    assert key in storage.models
