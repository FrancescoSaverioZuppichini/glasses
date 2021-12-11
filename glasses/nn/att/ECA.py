import math
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce


class ECA(nn.Module):
    """Implementation of Efficient Channel Attention proposed in `ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks <https://arxiv.org/pdf/1910.03151.pdf>`_

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/ECA.png?raw=true

    Examples:

        >>> # create ecaresnet50
        >>> from glasses.models.classification.resnet import ResNet, ResNetBottleneckBlock
        >>> from glasses.nn.att import ECA, WithAtt
        >>> eca_resnet50 = ResNet.resnet50(block=WithAtt(ResNetBottleneckBlock, att=ECA))
        >>> eca_resnet50.summary()

    Args:
        features (int, optional): Number of features features. Defaults to None.
        kernel_size (int, optional): [description]. Defaults to 3.
        gamma (int, optional): [description]. Defaults to 2.
        beta (int, optional): [description]. Defaults to 1.
    """

    def __init__(
        self, features: int, kernel_size: int = 3, gamma: int = 2, beta: int = 1
    ):

        super().__init__()
        assert kernel_size % 2 == 1

        t = int(abs(math.log(features, 2) + beta) / gamma)
        k = t if t % 2 else t + 1

        self.att = nn.Sequential(
            Reduce("b c h w -> b 1 c", reduction="mean"),
            nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False),
            Rearrange("b 1 c -> b c 1 1"),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.att(x)
        return x * y
