import torch
from glasses.models.classification import SEResNet


def test_seresnet():
    with torch.no_grad():
        x = torch.rand(1, 3, 224, 224)
        model = SEResNet.se_resnet18().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000

        model = SEResNet.se_resnet34().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000

        model = SEResNet.se_resnet50().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000

        model = SEResNet.se_resnet101().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000

        model = SEResNet.se_resnet152().eval()
        pred = model(x)
        assert pred.shape[-1] == 1000
