import pytest
import os
from pathlib import Path
from glasses.utils.storage import HuggingFaceStorage, LocalStorage, Storage
from torch import nn
from fixtures import glasses_path
from huggingface_hub import HfFolder

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

    key = "resnet18"

    storage = HuggingFaceStorage()

    has_token = HfFolder().get_token()
    if not has_token:
        token = os.environ["HUGGING_FACE_TOKEN"]
        HfFolder().save_token(token)

    state_dict = storage.get(key)
    model.load_state_dict(state_dict)

    assert key in storage
    assert key in storage.models


@pytest.mark.parametrize("fmt", ["pth", "foo"])
def test_local_storage(glasses_path: Path, fmt: str):
    storage = LocalStorage(root=glasses_path, fmt=fmt)
    storage.put(key, model)
    state_dict = storage.get(key)
    model.load_state_dict(state_dict)

    assert key in storage
    assert len(storage.models) == 1
    assert key in storage.models
