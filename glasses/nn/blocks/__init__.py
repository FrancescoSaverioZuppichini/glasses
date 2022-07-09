import math
from typing import Callable, Tuple, Optional
from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import StochasticDepth


class Lambda(nn.Module):
    """An utility Module, it allows custom function to be passed

    Args:
        lambd (Callable[Tensor]): A function that does something on a tensor

    Examples:
        >>> add_two = Lambda(lambd x: x + 2)
        >>> add_two(Tensor([0])) // 2
    """

    def __init__(self, lambd: Callable[[Tensor], Tensor]):

        super().__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        return self.lambd(x)


class Conv2dPad(nn.Conv2d):
    """2D Convolutions with different padding modes.

    'auto' will use the kernel_size to calculate the padding
    'same' same padding as TensorFLow. It will dynamically pad the image based on its size

    Args:
        mode (str, optional): [description]. Defaults to 'auto'.
    """

    def __init__(self, *args, mode: str = "auto", padding: int = 0, **kwargs):

        super().__init__(*args, **kwargs)
        self.mode = mode
        # dynamic add padding based on the kernel_size
        if self.mode == "auto":
            self.padding = (
                self._get_padding(padding)
                if padding != 0
                else (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
            )

    def _get_padding(self, padding: int) -> Tuple[int, int]:
        return (padding, padding)

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "same":
            ih, iw = x.size()[-2:]
            kh, kw = self.weight.size()[-2:]
            sh, sw = self.stride
            # change the output size according to stride
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max(
                (oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0
            )
            pad_w = max(
                (ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0
            )
            if pad_h > 0 or pad_w > 0:
                x = F.pad(
                    x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
                )
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        else:
            return super().forward(x)


class ConvNormAct(nn.Sequential):
    """Utility module that stacks one convolution layer, a normalization layer and an activation function.

    Example:
        >>> ConvNormAct(32, 64, kernel_size=3)
            ConvNormAct(
                (conv): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act): ReLU()
            )

        >>> ConvNormAct(32, 64, kernel_size=3, normalization = None )
            ConvNormAct(
                (conv): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (act): ReLU()
            )

        >>> ConvNormAct(32, 64, kernel_size=3, activation = None )
            ConvNormAct(
                (conv): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )

    We also provide additional modules built on top of this one: `ConvBn`, `ConvAct`, `Conv3x3BnAct`
    Args:
            out_features (int): Number of input features
            out_features (int): Number of output features
            conv (nn.Module, optional): Convolution layer. Defaults to Conv2dPad.
            normalization (nn.Module, optional): Normalization layer. Defaults to nn.BatchNorm2d.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        conv: nn.Module = Conv2dPad,
        activation: Optional[nn.Module] = nn.ReLU,
        normalization: Optional[nn.Module] = nn.BatchNorm2d,
        bias: bool = False,
        **kwargs
    ):
        super().__init__()
        self.add_module("conv", conv(in_features, out_features, **kwargs, bias=bias))
        if normalization:
            self.add_module("norm", normalization(out_features))
        if activation:
            self.add_module("act", activation())


class ConvNormRegAct(nn.Sequential):
    """Utility module that stacks one convolution layer, a normalization layer, a regularization layer and an activation function.

    Example:
        >>> ConvNormDropAct(32, 64, kernel_size=3)
            ConvNormDropAct(
                (conv): Conv2dPad(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (reg): StochasticDepth(p=0.2)
                (act): ReLU()
            )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        conv: nn.Module = Conv2dPad,
        activation: Optional[nn.Module] = nn.ReLU,
        normalization: Optional[nn.Module] = nn.BatchNorm2d,
        regularization: Optional[nn.Module] = partial(StochasticDepth, mode="batch"),
        p: float = 0.2,
        bias: bool = False,
        **kwargs
    ):
        super().__init__()
        self.add_module("conv", conv(in_features, out_features, **kwargs, bias=bias))
        if normalization:
            self.add_module("norm", normalization(out_features))
        if regularization:
            self.add_module("reg", regularization(p=p))
        if activation:
            self.add_module("act", activation())


ReLUInPlace = partial(nn.ReLU, inplace=True)


class NormActConv(nn.Sequential):
    """A Sequential layer composed by a normalization, an activation and a convolution layer. This is usually known as a 'Preactivation Block'

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        conv (nn.Module, optional): [description]. Defaults to Conv2dPad.
        normalization (nn.Module, optional): [description]. Defaults to nn.BatchNorm2d.
        activation (nn.Module, optional): [description]. Defaults to nn.ReLU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        conv: nn.Module = Conv2dPad,
        normalization: Optional[nn.Module] = nn.BatchNorm2d,
        activation: Optional[nn.Module] = ReLUInPlace,
        *args,
        **kwargs
    ):
        super().__init__()
        if normalization:
            self.add_module("norm", normalization(in_features))
        if activation:
            self.add_module("act", activation())
        self.add_module("conv", conv(in_features, out_features, *args, **kwargs))


ConvBnAct = partial(ConvNormAct, normalization=nn.BatchNorm2d)
ConvBn = partial(ConvBnAct, activation=None)
ConvAct = partial(ConvBnAct, normalization=None, bias=True)
Conv3x3BnAct = partial(ConvBnAct, kernel_size=3)
BnActConv = partial(NormActConv, normalization=nn.BatchNorm2d)
ConvBnDropAct = partial(
    ConvNormRegAct, normalization=nn.BatchNorm2d, regularization=nn.Dropout2d
)
