import torch
from torch import nn
from torch import Tensor


class SEModule(nn.Module):
    """Implementation of Squeeze and Excitation Module proposed in `Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_
    The idea is to apply learned an channel-wise attention. 
    
    The authors reported a bigger performance increase where the number of features are higher.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/se.png?raw=true

    Examples:

        To add `SeModule` to your own model is very simple. 
        
        >>> nn.Sequantial(
        >>>    nn.Conv2d(32, 64, kernel_size=3),
        >>>    nn.SEModule(64, reduction=4)
        >>>    nn.ReLU(),
        >>> )

        
        The following example shows a more advance scenarion where we add Squeeze ad Excitation to a `ResNetBasicBlock`. 

        >>> class SENetBasicBlock(ResNetBasicBlock):
        >>>     def __init__(self, in_features: int, out_features: int, reduction: int =16, *args, **kwargs):
        >>>        super().__init__(in_features, out_features, *args, **kwargs)
        >>>        # add se to the `.block`
        >>>        self.block.add_module('se', SEModule(out_features))


    Args:
        features (int): Number of features
        reduction (int, optional): Reduction ratio used to downsample the input. Defaults to 16.
    """

    def __init__(self, features: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(
            nn.Linear(features, features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(features // reduction, features, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y has shape [B, C]
        y = self.att(y)
        # resphape to [B, C, 1, 1]  to match the space dims of x
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)
