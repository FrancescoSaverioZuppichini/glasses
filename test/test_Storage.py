import torch
from torch import nn
from glasses.utils.Storage import ForwardModuleStorage, BackwardModuleStorage, MutipleKeysDict
import pytest

def test_storage():

    d = MutipleKeysDict({ 'a' : 1, 'b' : 2, 'c' : 3})
    out = d[['a', 'b']]

    assert len(out) == 2
    x = torch.rand(1,3,224,224)
    y = torch.rand(1,3,224,224)

    cnn = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3),
        nn.Conv2d(32, 32, kernel_size=3),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(32, 10)
    )

    layer = cnn[0]
    
    storage = ForwardModuleStorage(cnn, [layer])

    print(storage)
    # check if layer is a correct key
    assert layer in storage.state
    storage(x)
    # it must be a tensor
    assert type(storage[layer][0]) is torch.Tensor
    assert len(storage[layer]) == 1

    layer1 = cnn[2]
    storage = ForwardModuleStorage(cnn, { 'a' : [layer], 'b': [layer, layer1] })
    storage(x, 'a')

    assert type(storage['a'][layer]) is torch.Tensor
    assert layer not in storage['b']

    storage(y, 'b')
    assert layer in storage['b']
    assert layer1 in storage['b']
    assert type(storage['b'][layer]) is torch.Tensor

    storage.clear()


    storage = BackwardModuleStorage([layer])
    x = torch.rand(1,3,224,224).requires_grad_(True)
    loss = nn.CrossEntropyLoss()

    output = loss(cnn(x), torch.tensor([1]))
    storage(output)

    assert type(storage[layer][0][0]) is torch.Tensor
    assert len(storage[layer]) == 1

    with pytest.raises(ValueError):
        storage.register_hooks('wrong')