import torch
from torch import nn
from torch import Tensor
from ..resnet import ReLUInPlace

class SEModule(nn.Module):
    """Implementation of Squeeze and Excitation Module proposed in `Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_
    The idea is to apply learned an channel-wise attention. 
    
    The authors reported a bigger performance increase where the number of features is higher.

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

    def __init__(self, features: int, reduction: int = 16, activation: nn.Module = ReLUInPlace):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(
            nn.Linear(features, features // reduction),
            activation(),
            nn.Linear(features // reduction, features),
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


class SEModuleConv(SEModule):
    """Modified implement of Squeeze and Excitation Module proposed in `Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_
    The idea is to apply learned an channel-wise attention. 
    
    Here we use two 1x1 convs. The first reduce the inputs' channels, then  the second learns the scaling funtion.

    The authors reported a bigger performance increase where the number of features is higher.

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/se.png?raw=true

    Examples:

        To add `SeModule` to your own model is very simple. 
        
        >>> nn.Sequantial(
        >>>    nn.Conv2d(32, 64, kernel_size=3),
        >>>    nn.SEModuleConv(64, reduction=4)
        >>>    nn.ReLU(),
        >>> )

        You can also direcly specify the number of features inside the module


        >>> nn.Sequantial(
        >>>    nn.Conv2d(32, 64, kernel_size=3),
        >>>    nn.SEModuleConv(64, reduced_features=10)
        >>>    nn.ReLU(),
        >>> )


    Args:
        features (int): Number of features
        reduction (int, optional): Reduction ratio used to downsample the input. Defaults to 16.
        reduced_features (int, optional): If passed, use it instead of calculating the reduced features using `reduction`. Defaults to None.

    """

    def __init__(self, features: int, reduction: int = 16, reduced_features: int = None, activation: nn.Module = ReLUInPlace):
        super().__init__(features, reduction, activation)
        reduced_features = features // reduction if reduced_features is None else reduced_features
        self.att = nn.Sequential(
            nn.Conv2d(features, reduced_features, kernel_size=1),
            activation(),
            nn.Conv2d(reduced_features, features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x)
        y = self.att(y)
       
        return x * y